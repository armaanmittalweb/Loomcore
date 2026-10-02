// Connecting to the runtime. It runs on an always-on VM, so the normal case
// is checking -> live within one request. When it is down or restarting
// (a redeploy, a reboot), requests fail until it is back, so the console keeps
// polling /health for a while:
//
//   checking -> live                        (the normal case)
//   checking -> waking -> loading -> live   (it was restarting)
//   ...      -> asleep                      (no answer within the limit: down)
//
// `loading` means the server answered but the runtime is still loading
// models and warming up. The reducer is pure; `wake()` drives it.
import type { Client } from './api';
import type { Health } from './types';

export type Phase = 'checking' | 'waking' | 'loading' | 'live' | 'asleep';

export interface WakeState {
  phase: Phase;
  startedAt: number;
  attempts: number;
  health: Health | null;
}

export type WakeEvent =
  | { type: 'start'; now: number }
  | { type: 'answer'; health: Health }
  | { type: 'no-answer'; now: number; limitMs: number };

export const initialWake = (now = 0): WakeState => ({ phase: 'checking', startedAt: now, attempts: 0, health: null });

export function wakeReducer(state: WakeState, event: WakeEvent): WakeState {
  switch (event.type) {
    case 'start':
      return initialWake(event.now);
    case 'answer':
      return {
        ...state,
        attempts: state.attempts + 1,
        health: event.health,
        phase: event.health.ready ? 'live' : 'loading',
      };
    case 'no-answer': {
      const attempts = state.attempts + 1;
      if (state.phase === 'live') return { ...state, attempts }; // a blip after it was live: keep going
      const phase = event.now - state.startedAt >= event.limitMs ? 'asleep' : 'waking';
      return { ...state, attempts, phase };
    }
  }
}

export const WAKE_LIMIT_MS = 60_000;

export interface WakeOptions {
  limitMs?: number;
  intervalMs?: number;
  now?: () => number;
  sleep?: (ms: number, signal: AbortSignal) => Promise<void>;
}

const defaultSleep = (ms: number, signal: AbortSignal) =>
  new Promise<void>((resolve) => {
    const t = setTimeout(resolve, ms);
    signal.addEventListener('abort', () => {
      clearTimeout(t);
      resolve();
    });
  });

/** Polls until live or asleep, reporting each state. Abort to stop. */
export async function wake(client: Pick<Client, 'health'>, onState: (s: WakeState) => void, signal: AbortSignal, opts: WakeOptions = {}): Promise<WakeState> {
  const now = opts.now ?? Date.now;
  const limitMs = opts.limitMs ?? WAKE_LIMIT_MS;
  const intervalMs = opts.intervalMs ?? 4000;
  const sleep = opts.sleep ?? defaultSleep;
  let state = wakeReducer(initialWake(), { type: 'start', now: now() });
  onState(state);
  while (!signal.aborted) {
    try {
      const health = await client.health(signal);
      state = wakeReducer(state, { type: 'answer', health });
    } catch {
      if (signal.aborted) break;
      state = wakeReducer(state, { type: 'no-answer', now: now(), limitMs });
    }
    onState(state);
    if (state.phase === 'live' || state.phase === 'asleep') return state;
    await sleep(state.phase === 'loading' ? 2000 : intervalMs, signal);
  }
  return state;
}
