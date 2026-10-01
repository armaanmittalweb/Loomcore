// The Space client. Every failure becomes a typed SpaceError the UI can word
// precisely: unreachable (asleep, waking or offline), busy (429 with a retry
// time), warming (the server is up but still loading models), timeout, or a
// request the server refused (with its own message).
import type { Bench, GraphInfo, Health, LoadResult, PolicyName, ReloadResult, RunResult } from './types';

export const SPACE_URL: string = (import.meta.env?.VITE_SPACE_URL as string | undefined) || 'https://armaanmittalweb-loomcore.hf.space';

export type Failure =
  | { kind: 'unreachable' }
  | { kind: 'timeout' }
  | { kind: 'busy'; retryAfter: number; message: string }
  | { kind: 'warming'; retryAfter: number }
  | { kind: 'refused'; status: number; code: string; message: string };

export class SpaceError extends Error {
  constructor(public failure: Failure) {
    super(failure.kind === 'refused' || failure.kind === 'busy' ? failure.message : failure.kind);
  }
}

export interface RunInput {
  sample?: string;
  file?: Blob;
  timeBudgetMs?: number | null;
  policies: PolicyName[];
}

export interface LoadInput {
  jobs: number;
  concurrency: number;
  timeBudgetMs?: number | null;
  policies: PolicyName[];
  sample?: string;
}

export interface ReloadInput {
  jobs: number;
  swaps: number;
}

type Fetch = typeof fetch;

export function createClient(base: string = SPACE_URL, fetchImpl: Fetch = (...a) => fetch(...a)) {
  const root = base.replace(/\/+$/, '');

  async function request<T>(path: string, init: RequestInit = {}, timeoutMs = 30_000, outer?: AbortSignal): Promise<T> {
    const ctl = new AbortController();
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      ctl.abort();
    }, timeoutMs);
    const onOuter = () => ctl.abort();
    outer?.addEventListener('abort', onOuter);
    let res: Response;
    try {
      res = await fetchImpl(root + path, { ...init, signal: ctl.signal, credentials: 'omit' });
    } catch (err) {
      if (outer?.aborted) throw err;
      throw new SpaceError(timedOut ? { kind: 'timeout' } : { kind: 'unreachable' });
    } finally {
      clearTimeout(timer);
      outer?.removeEventListener('abort', onOuter);
    }
    let body: unknown = null;
    try {
      body = await res.json();
    } catch {
      body = null; // a sleeping Space answers with an HTML page, not JSON
    }
    if (res.ok && body && typeof body === 'object') return body as T;
    const b = (body && typeof body === 'object' ? body : {}) as { error?: string; message?: string; retryAfter?: number };
    const header = Number(res.headers.get('retry-after'));
    const retryAfter = typeof b.retryAfter === 'number' ? b.retryAfter : Number.isFinite(header) && header > 0 ? header : 5;
    if (res.status === 429) throw new SpaceError({ kind: 'busy', retryAfter, message: b.message ?? 'The runtime is busy.' });
    if (res.status === 503 && b.error === 'warming_up') throw new SpaceError({ kind: 'warming', retryAfter });
    if (!b.error) throw new SpaceError({ kind: 'unreachable' }); // a proxy page, not the server
    throw new SpaceError({ kind: 'refused', status: res.status, code: b.error, message: b.message ?? `HTTP ${res.status}` });
  }

  const post = <T>(path: string, body: unknown, timeoutMs: number) =>
    request<T>(path, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) }, timeoutMs);

  return {
    base: root,
    health: (signal?: AbortSignal) => request<Health>('/health', {}, 10_000, signal),
    graph: () => request<GraphInfo>('/graph', {}, 15_000),
    bench: () => request<Bench>('/bench', {}, 15_000),
    run(input: RunInput): Promise<RunResult> {
      if (input.file) {
        const form = new FormData();
        form.append('image', input.file, 'upload');
        form.append('policies', JSON.stringify(input.policies));
        if (input.timeBudgetMs != null) form.append('timeBudgetMs', String(input.timeBudgetMs));
        return request<RunResult>('/run', { method: 'POST', body: form }, 40_000);
      }
      const body: Record<string, unknown> = { sample: input.sample, policies: input.policies };
      if (input.timeBudgetMs != null) body.timeBudgetMs = input.timeBudgetMs;
      return post<RunResult>('/run', body, 40_000);
    },
    load(input: LoadInput): Promise<LoadResult> {
      const body: Record<string, unknown> = { ...input };
      if (input.timeBudgetMs == null) delete body.timeBudgetMs;
      return post<LoadResult>('/load', body, 100_000);
    },
    reload: (input: ReloadInput) => post<ReloadResult>('/reload', input, 100_000),
  };
}

export type Client = ReturnType<typeof createClient>;

/** One line for a person, from any failure. */
export function describeFailure(err: unknown): string {
  if (!(err instanceof SpaceError)) return 'Something went wrong in the console. Reload the page and try again.';
  const f = err.failure;
  switch (f.kind) {
    case 'unreachable':
      return 'Could not reach the live runtime. It may be asleep or restarting.';
    case 'timeout':
      return 'The runtime took too long to answer. It may be waking up; try again in a moment.';
    case 'busy':
      return `${f.message.replace(/\.$/, '')}. Try again in ${Math.ceil(f.retryAfter)} s.`;
    case 'warming':
      return `The runtime is still loading its models. Try again in ${Math.ceil(f.retryAfter)} s.`;
    case 'refused':
      return f.message.charAt(0).toUpperCase() + f.message.slice(1).replace(/\.?$/, '.');
  }
}
