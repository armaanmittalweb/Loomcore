// `npm run record -- [--from <space url>] [--os "<os label>"]`
//
// Captures real results from a running Loomcore Space (the HF Space, or
// `python space/server.py` locally) into src/recorded/, which the console
// renders on open while it wakes the live runtime. Also writes the extra runs
// the screenshot script serves for its error states to scripts/fixtures/.
// Nothing here is synthesised: every file is a response body, as returned.
import { mkdirSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const args = process.argv.slice(2);
const opt = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : fallback;
};
const BASE = opt('from', 'http://127.0.0.1:7860').replace(/\/$/, '');
const OS = opt('os', '');
const OUT = fileURLToPath(new URL('../src/recorded/', import.meta.url));
const FIX = fileURLToPath(new URL('./fixtures/', import.meta.url));
mkdirSync(OUT, { recursive: true });
mkdirSync(FIX, { recursive: true });

async function call(path, body) {
  const res = await fetch(BASE + path, body ? { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) } : {});
  const json = await res.json();
  if (!res.ok) throw new Error(`${path}: ${res.status} ${JSON.stringify(json)}`);
  return json;
}
const save = (dir, name, data) => writeFileSync(dir + name, JSON.stringify(data) + '\n');

let health;
for (let i = 0; i < 120; i++) {
  health = await call('/health').catch(() => null);
  if (health?.ready && health.bench !== 'running' && health.bench !== 'pending') break;
  await new Promise((r) => setTimeout(r, 1000));
}
if (!health?.ready) throw new Error(`no ready runtime at ${BASE}`);

const graph = await call('/graph');
const bench = await call('/bench');
// Warm the default chain so the recorded run doesn't include a graph swap.
await call('/run', { sample: 'fox' });
// The featured run: a budget tight enough that the router downgrades a node, loose
// enough that the job completes (timing varies by a few ms run to run).
let run;
for (const budget of [40, 42, 45, 50, 55, 60]) {
  run = await call('/run', { sample: 'fox', timeBudgetMs: budget });
  if (run.status === 'ok' && run.nodes.length === 2 && run.decisions.length) break;
}

// Extra real runs for the screenshot script's states, taken before the load test
// so its queueing does not inflate the latency the planner judges budgets by.
save(FIX, 'run-skip.json', await call('/run', { sample: 'lighthouse' }));
save(FIX, 'run-rejected.json', await call('/run', { sample: 'tabby', timeBudgetMs: 3 }));
let cancelled;
for (const budget of [25, 22, 28, 20, 30]) {
  cancelled = await call('/run', { sample: 'tabby', timeBudgetMs: budget });
  if (cancelled.status === 'cancelled') break;
}
save(FIX, 'run-cancelled.json', cancelled);
const load = await call('/load', { jobs: 48, concurrency: 12, timeBudgetMs: 120 });
const reload = await call('/reload', { jobs: 48, swaps: 4 });

const meta = {
  recordedAt: new Date().toISOString(),
  source: BASE.includes('hf.space') ? 'space' : 'local',
  machine: health.cpu,
  os: OS || undefined,
  version: health.version,
};
save(OUT, 'meta.json', meta);
save(OUT, 'health.json', health);
save(OUT, 'graph.json', graph);
save(OUT, 'bench.json', bench);
save(OUT, 'run.json', run);
save(OUT, 'load.json', load);
save(OUT, 'reload.json', reload);


save(FIX, 'load-shed.json', await call('/load', { jobs: 32, concurrency: 12, policies: ['bulkhead', 'load-aware'] }));

console.log(`recorded from ${BASE} (${health.cpu}) at ${meta.recordedAt}`);
console.log(`run: ${run.status} ${run.label} ${run.totalMs} ms; load: ${load.completed}/${load.submitted}; reload lost ${reload.lost}; cancelled fixture: ${cancelled.status}`);
