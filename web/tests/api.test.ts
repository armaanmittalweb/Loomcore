import { describe, expect, it, vi } from 'vitest';
import { createClient, describeFailure, SpaceError } from '../src/api';
import { initialWake, wake, wakeReducer } from '../src/wake';
import type { Health } from '../src/types';

const json = (body: unknown, status = 200, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json', ...headers } });

describe('createClient', () => {
  it('posts runs as JSON to the configured Space, trimming slashes', async () => {
    const fetch = vi.fn(async () => json({ status: 'ok' }));
    const c = createClient('https://space.example/', fetch as unknown as typeof globalThis.fetch);
    await c.run({ sample: 'fox', timeBudgetMs: 40, policies: ['load-aware'] });
    const [url, init] = fetch.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe('https://space.example/run');
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body as string)).toEqual({ sample: 'fox', timeBudgetMs: 40, policies: ['load-aware'] });
    expect(init.credentials).toBe('omit');
  });

  it('omits an unset budget, and sends uploads as multipart', async () => {
    const fetch = vi.fn(async () => json({ status: 'ok' }));
    const c = createClient('https://s', fetch as unknown as typeof globalThis.fetch);
    await c.run({ sample: 'fox', timeBudgetMs: null, policies: [] });
    expect(JSON.parse((fetch.mock.calls[0] as unknown as [string, RequestInit])[1].body as string)).toEqual({ sample: 'fox', policies: [] });
    await c.run({ file: new Blob([new Uint8Array([0xff, 0xd8, 0xff])], { type: 'image/jpeg' }), policies: ['bulkhead'], timeBudgetMs: 60 });
    const form = (fetch.mock.calls[1] as unknown as [string, RequestInit])[1].body as FormData;
    expect(form).toBeInstanceOf(FormData);
    expect(form.get('policies')).toBe('["bulkhead"]');
    expect(form.get('timeBudgetMs')).toBe('60');
    expect(form.get('image')).toBeInstanceOf(Blob);
    await c.load({ jobs: 8, concurrency: 2, timeBudgetMs: null, policies: [] });
    expect(JSON.parse((fetch.mock.calls[2] as unknown as [string, RequestInit])[1].body as string)).toEqual({ jobs: 8, concurrency: 2, policies: [] });
  });

  it('turns a 429 into busy with the retry time', async () => {
    const c = createClient('https://s', (async () => json({ error: 'busy', message: 'the runtime is busy with another run', retryAfter: 4.2 }, 429)) as unknown as typeof fetch);
    const err = await c.load({ jobs: 8, concurrency: 2, policies: [] }).catch((e) => e);
    expect(err).toBeInstanceOf(SpaceError);
    expect(err.failure).toEqual({ kind: 'busy', retryAfter: 4.2, message: 'the runtime is busy with another run' });
    expect(describeFailure(err)).toBe('the runtime is busy with another run. Try again in 5 s.');
  });

  it('falls back to the Retry-After header', async () => {
    const c = createClient('https://s', (async () => json({ error: 'rate_limited', message: 'slow down' }, 429, { 'retry-after': '12' })) as unknown as typeof fetch);
    const err = await c.run({ policies: [] }).catch((e) => e);
    expect(err.failure.retryAfter).toBe(12);
  });

  it('tells warming apart from a refusal', async () => {
    const warming = createClient('https://s', (async () => json({ error: 'warming_up', message: 'loading', retryAfter: 5 }, 503)) as unknown as typeof fetch);
    expect((await warming.graph().catch((e) => e)).failure).toEqual({ kind: 'warming', retryAfter: 5 });
    const refused = createClient('https://s', (async () => json({ error: 'too_large', message: 'the image must be 2 MB or smaller' }, 413)) as unknown as typeof fetch);
    const err = await refused.run({ policies: [] }).catch((e) => e);
    expect(err.failure).toEqual({ kind: 'refused', status: 413, code: 'too_large', message: 'the image must be 2 MB or smaller' });
    expect(describeFailure(err)).toBe('The image must be 2 MB or smaller.');
  });

  it('reads a network failure or a non-JSON page as unreachable', async () => {
    const down = createClient('https://s', (async () => {
      throw new TypeError('Failed to fetch');
    }) as unknown as typeof fetch);
    expect((await down.health().catch((e) => e)).failure).toEqual({ kind: 'unreachable' });
    const html = createClient('https://s', (async () => new Response('<html>Building</html>', { status: 503 })) as unknown as typeof fetch);
    expect((await html.health().catch((e) => e)).failure).toEqual({ kind: 'unreachable' });
  });

  it('times out', async () => {
    vi.useFakeTimers();
    const hang = createClient('https://s', ((_: string, init: RequestInit) =>
      new Promise((_r, reject) => init.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError'))))) as unknown as typeof fetch);
    const p = hang.health().catch((e) => e);
    await vi.advanceTimersByTimeAsync(10_001);
    expect((await p).failure).toEqual({ kind: 'timeout' });
    vi.useRealTimers();
  });
});

const health = (ready: boolean): Health => ({ ok: true, ready, version: 'v', models: {}, cpu: 'cpu', uptimeS: 1, busy: false, bench: 'ready' });

describe('wakeReducer', () => {
  it('goes live on a ready answer, loading on an unready one', () => {
    expect(wakeReducer(initialWake(0), { type: 'answer', health: health(true) }).phase).toBe('live');
    expect(wakeReducer(initialWake(0), { type: 'answer', health: health(false) }).phase).toBe('loading');
  });
  it('waits while the Space wakes, then gives up after the limit', () => {
    const s = wakeReducer(initialWake(1000), { type: 'no-answer', now: 2000, limitMs: 60_000 });
    expect(s.phase).toBe('waking');
    expect(wakeReducer(s, { type: 'no-answer', now: 61_000, limitMs: 60_000 }).phase).toBe('asleep');
  });
  it('stays live through a blip', () => {
    const live = wakeReducer(initialWake(0), { type: 'answer', health: health(true) });
    expect(wakeReducer(live, { type: 'no-answer', now: 999_999, limitMs: 1 }).phase).toBe('live');
  });
});

describe('wake', () => {
  const noSleep = async () => {};
  it('polls through waking and loading to live', async () => {
    const answers: (Health | Error)[] = [new Error('down'), new Error('down'), health(false), health(true)];
    const client = { health: vi.fn(async () => {
      const a = answers.shift()!;
      if (a instanceof Error) throw a;
      return a;
    }) };
    const phases: string[] = [];
    const end = await wake(client, (s) => phases.push(s.phase), new AbortController().signal, { sleep: noSleep, now: () => 0 });
    expect(end.phase).toBe('live');
    expect(phases).toEqual(['checking', 'waking', 'waking', 'loading', 'live']);
  });
  it('reports asleep once the limit passes', async () => {
    let t = 0;
    const client = { health: vi.fn(async () => {
      throw new Error('down');
    }) };
    const end = await wake(client, () => {}, new AbortController().signal, { sleep: async () => void (t += 5000), now: () => t, limitMs: 12_000 });
    expect(end.phase).toBe('asleep');
    expect(client.health).toHaveBeenCalledTimes(4);
  });
  it('stops when aborted', async () => {
    const ctl = new AbortController();
    const client = { health: vi.fn(async () => {
      ctl.abort();
      throw new Error('down');
    }) };
    const end = await wake(client, () => {}, ctl.signal, { sleep: noSleep });
    expect(end.phase).toBe('checking');
    expect(client.health).toHaveBeenCalledTimes(1);
  });
});
