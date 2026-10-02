// loomcore.amittal.dev/: what Loomcore is, how to drive the console, how to call it from code, how a job
// runs, and the numbers. Every number here is read from the recorded run the console opens on
// (src/recorded/*.json), so this page and the console never disagree.
import { useState } from 'preact/hooks';
import { ThemeToggle } from '../App';
import { Timeline } from '../timeline/Timeline';
import { count, day, ms, pct, ratio, shortCpu } from '../format';
import type { Bench, GraphInfo, LoadResult, RecordingMeta, ReloadResult, RunResult } from '../types';
import recordedMeta from '../recorded/meta.json';
import recordedRun from '../recorded/run.json';
import recordedLoad from '../recorded/load.json';
import recordedReload from '../recorded/reload.json';
import recordedBench from '../recorded/bench.json';
import recordedGraph from '../recorded/graph.json';

const META = recordedMeta as RecordingMeta;
const RUN = recordedRun as unknown as RunResult;
const LOAD = recordedLoad as unknown as LoadResult;
const SWAP = recordedReload as unknown as ReloadResult;
const BENCH = recordedBench as unknown as Bench;
const GRAPH = recordedGraph as unknown as GraphInfo;
const REPO = 'https://github.com/armaanmittalweb/loomcore';
const API = 'https://loomcore-api.amittal.dev';

const GitHub = () => (
  <svg viewBox="0 0 16 16" aria-hidden="true">
    <path d="M8 .2a8 8 0 0 0-2.5 15.6c.4 0 .5-.2.5-.4v-1.5c-2.2.5-2.7-1-2.7-1-.4-.9-.9-1.2-.9-1.2-.7-.5.1-.5.1-.5.8.1 1.2.8 1.2.8.7 1.3 1.9.9 2.3.7.1-.5.3-.9.5-1.1-1.8-.2-3.6-.9-3.6-4 0-.9.3-1.6.8-2.1-.1-.2-.4-1 .1-2.1 0 0 .7-.2 2.2.8a7.5 7.5 0 0 1 4 0c1.5-1 2.2-.8 2.2-.8.4 1.1.2 1.9.1 2.1.5.6.8 1.3.8 2.1 0 3.1-1.9 3.8-3.6 4 .3.3.6.8.6 1.5v2.2c0 .2.1.5.6.4A8 8 0 0 0 8 .2Z" />
  </svg>
);

const mobilenet = LOAD.nodes.find((n) => n.id === 'mobilenet');
const slowestSwap = Math.max(...SWAP.swapTimings.map((s) => s.buildMs));
const benchRow = (model: string, p: string) => BENCH.rows?.find((r) => r.model === model && r.precision === p);

const STEPS = [
  {
    id: 'run',
    size: [784, 1222],
    title: 'Run one job',
    body: 'Pick a sample photo, or drop in your own JPEG or PNG (up to 2 MB), and press Run. MobileNetV2 labels it; unless it is already 85% sure, bert_tiny turns the label into a 128-number text embedding.',
    look: 'The router’s decisions sit above the lanes with their reasons. Hover any bar for the node, precision, batch and time.',
    result: `Recorded: ${RUN.label}, ${pct(RUN.confidence)} sure, so bert_tiny ran. The job took ${ms(RUN.totalMs)} ms.`,
  },
  {
    id: 'load',
    size: [784, 1056],
    title: 'Run a load test',
    body: 'Fire up to 64 jobs, up to 16 at a time, with an optional time budget per job. Toggle router policies on the graph panel to see what each one changes.',
    look: 'Calls stack into batches of up to 4 per model, split between the CPU and GPU_SIM lanes by queue depth. Rejected and cancelled jobs are drawn in red.',
    result: `Recorded: ${LOAD.completed} of ${LOAD.submitted} jobs done, ${LOAD.concurrency} at a time, job p50 ${ms(LOAD.jobP50)} ms and p95 ${ms(LOAD.jobP95)} ms${mobilenet ? `; MobileNetV2 ran ${mobilenet.lanes.CPU ?? 0} times on CPU and ${mobilenet.lanes.GPU_SIM ?? 0} on GPU_SIM` : ''}.`,
  },
  {
    id: 'swap',
    size: [784, 764],
    title: 'Hot-swap the graph under load',
    body: 'Rebuild the whole graph, models and scheduler up to 6 times while jobs keep arriving. Each swap loads fresh ONNX Runtime sessions beside the old ones.',
    look: 'The swap markers cut across the lanes; jobs that started before a swap finish on the graph they started with. The count to check is “lost”.',
    result: `Recorded: ${SWAP.swaps} swaps during ${SWAP.submitted} jobs, ${SWAP.completed} completed, ${SWAP.lost} lost. The slowest swap took ${ms(slowestSwap)} ms to build.`,
  },
] as const;

const CODE: Record<'http' | 'python' | 'build', { label: string; note: string; text: string }> = {
  http: {
    label: 'HTTP',
    note: `The live runtime behind the console. ${'/health'}, /graph, /run, /load, /reload and /bench; 20 POSTs a minute per address.`,
    text: `curl -X POST ${API}/run \\
  -H 'content-type: application/json' \\
  -d '{"sample": "retriever"}'

# {"label": "Golden Retriever", "confidence": 0.9141,
#  "skipped": ["bert_tiny"], "totalMs": 31.2,
#  "decisions": [...], "nodes": [...], "trace": [...]}`,
  },
  python: {
    label: 'Python',
    note: 'The pybind11 bindings. Binders map graph inputs and upstream outputs to each model’s inputs; a routing policy can be written in Python too. Full version: examples/run_example.py.',
    text: `import loomcore

router = loomcore.CompositeRouter()
router.add(loomcore.LatencyBudgetPolicy(30.0))
router.add(loomcore.LoadAwareBackendPolicy())

runtime = loomcore.Runtime()
runtime.load_graph(
    "examples/graph_config.json",
    binders={"mobilenet": mobilenet_binder, "bert_tiny": bert_binder},
    confidence_extractors={"bert_tiny": confidence_from_logits},
    router=router,
)

result = runtime.run({"data": image})      # float32 [1, 3, 224, 224]
print(runtime.node_stats("mobilenet"))     # p50_ms, p95_ms, count`,
  },
  build: {
    label: 'Build',
    note: 'C++17 and CMake on Linux or Windows; the build fetches ONNX Runtime itself. Installs as a CMake package: find_package(Loomcore), target loomcore::loomcore_core.',
    text: `git clone ${REPO}.git && cd loomcore
pip install -r scripts/requirements.txt
python scripts/prepare_all_models.py      # export and quantize both models

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --parallel
ctest --test-dir build --output-on-failure

./build/bin/loomcore_example examples/sample.jpg
./build/bin/loomcore_reload_demo           # 6 hot-swaps under load`,
  },
};

function CodeTabs() {
  const [tab, setTab] = useState<keyof typeof CODE>('http');
  const c = CODE[tab];
  return (
    <div class="lp-code">
      <div class="tabs" role="tablist" aria-label="Ways to use Loomcore">
        {(Object.keys(CODE) as (keyof typeof CODE)[]).map((k) => (
          <button key={k} type="button" role="tab" id={`code-${k}`} class="tab" aria-selected={k === tab} aria-controls="code-panel" onClick={() => setTab(k)}>
            {CODE[k].label}
          </button>
        ))}
      </div>
      <div role="tabpanel" id="code-panel" aria-labelledby={`code-${tab}`}>
        <pre class="lp-pre" tabIndex={0}><code>{c.text}</code></pre>
        <p class="fine">{c.note}</p>
      </div>
    </div>
  );
}

/** The graph a job flows through: the router decides per node, the lanes batch across jobs. */
function Flow() {
  return (
    <svg class="lp-flow" viewBox="0 0 760 248" role="img" aria-label="A job enters the router, which decides per node; MobileNetV2 runs on the CPU lane, then bert_tiny on the GPU_SIM lane unless MobileNetV2 is already confident; both lanes batch calls from concurrent jobs.">
      <defs>
        <marker id="lp-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 8 4 0 8Z" class="lp-flow-head" />
        </marker>
      </defs>
      <g class="lp-flow-box"><rect x="8" y="88" width="104" height="56" rx="4" /><text x="60" y="113">job</text><text x="60" y="131" class="sub">image</text></g>
      <g class="lp-flow-box lp-flow-router"><rect x="150" y="24" width="460" height="34" rx="4" /><text x="380" y="46">router: skip? precision? lane?   (asked before every node)</text></g>
      <g class="lp-flow-box"><rect x="150" y="88" width="190" height="56" rx="4" /><text x="245" y="113">MobileNetV2</text><text x="245" y="131" class="sub">CPU lane · FP32 | INT8</text></g>
      <g class="lp-flow-box"><rect x="420" y="88" width="190" height="56" rx="4" /><text x="515" y="113">bert_tiny</text><text x="515" y="131" class="sub">GPU_SIM lane · FP32 | INT8</text></g>
      <g class="lp-flow-box lp-flow-out"><rect x="648" y="88" width="104" height="56" rx="4" /><text x="700" y="113">label +</text><text x="700" y="131" class="sub">embedding</text></g>
      <path d="M112 116H146" class="lp-flow-line" marker-end="url(#lp-arrow)" />
      <path d="M340 116H416" class="lp-flow-line" marker-end="url(#lp-arrow)" />
      <path d="M610 116H644" class="lp-flow-line" marker-end="url(#lp-arrow)" />
      <path d="M245 88V62M515 88V62" class="lp-flow-dash" />
      <path d="M245 144c0 36 455 36 455 4" class="lp-flow-dash" marker-end="url(#lp-arrow)" />
      <text x="472" y="198" class="lp-flow-note">bert_tiny is skipped when MobileNetV2 is ≥ 85% sure</text>
      <text x="380" y="236" class="lp-flow-note">each lane batches same-model calls from concurrent jobs (up to 4, within 8–12 ms)</text>
    </svg>
  );
}

export function Landing() {
  const fp = benchRow('mobilenetv2', 'FP32');
  const i8 = benchRow('mobilenetv2', 'INT8');
  const bfp = benchRow('bert_tiny', 'FP32');
  const bi8 = benchRow('bert_tiny', 'INT8');
  const flags = BENCH.cpu?.flags;
  return (
    <div class="lp">
      <a class="skip" href="#main">Skip to the content</a>
      <header class="lp-top">
        <a class="wordmark lp-wm" href="/" aria-label="Loomcore home">
          Loomcore<span class="wm-dot" aria-hidden="true" />
        </a>
        <nav class="lp-nav" aria-label="On this page">
          <a href="#try">Try it</a>
          <a href="#code">Use it</a>
          <a href="#how">How it works</a>
          <a href="#numbers">Numbers</a>
        </nav>
        <div class="top-tools">
          <ThemeToggle />
          <a class="icon-btn" href={REPO} aria-label="Source on GitHub" title="Source on GitHub"><GitHub /></a>
          <a class="btn" href="/console">Open the console</a>
        </div>
      </header>

      <main id="main">
        <section class="lp-hero" aria-labelledby="hero-h">
          <p class="kicker">C++17 · ONNX Runtime · Python bindings · MIT</p>
          <h1 id="hero-h">Run several ONNX models as one scheduled graph</h1>
          <p class="lp-lede">
            Loomcore is a C++ runtime for multi-model inference. Each job flows through a dependency graph of models. Before every node a router decides whether to
            skip it, which precision to run and which lane to use. Calls are batched across jobs, deadlines are enforced, and the whole graph can be swapped under
            load without losing a job.
          </p>
          <div class="lp-actions">
            <a class="btn lp-btn" href="/console">Open the console</a>
            <a class="btn btn-quiet lp-btn" href={REPO}>Read the code</a>
          </div>
          <figure class="panel lp-hero-tl">
            <figcaption class="lp-cap">
              <span class="kicker">Load test · recorded {day(META.recordedAt)}</span>
              <span class="fine">
                {LOAD.submitted} jobs, {LOAD.concurrency} at a time, on the live runtime ({shortCpu(META.machine)}). Each bar is one ONNX Runtime call, placed where the
                scheduler ran it.
              </span>
            </figcaption>
            <Timeline trace={LOAD.trace} animate label={`Recorded load test: ${LOAD.submitted} jobs`} />
          </figure>
        </section>

        <section class="lp-sec" id="try" aria-labelledby="try-h">
          <h2 id="try-h" class="lp-h2">Try it in the console</h2>
          <p class="lp-sub">
            The console opens on the recorded run above and connects to the live runtime in the background. The runtime sleeps when nobody is using it, so the first
            request can take up to half a minute while it starts.
          </p>
          <ol class="lp-steps">
            {STEPS.map((s, i) => (
              <li key={s.id} class="lp-step">
                <div class="lp-step-text">
                  <p class="kicker">Step {i + 1}</p>
                  <h3>{s.title}</h3>
                  <p>{s.body}</p>
                  <p><b>Look for:</b> {s.look}</p>
                  <p class="lp-result mono">{s.result}</p>
                </div>
                <picture class="lp-shot">
                  <source media="(prefers-color-scheme: dark)" srcSet={`/landing/${s.id}-dark.webp`} />
                  <img src={`/landing/${s.id}-light.webp`} width={s.size[0]} height={s.size[1]} alt={`The console’s Controls panel for “${s.title}”, with the settings used for the recorded run`} loading="lazy" decoding="async" />
                </picture>
              </li>
            ))}
          </ol>
          <a class="btn lp-btn" href="/console">Open the console</a>
        </section>

        <section class="lp-sec" id="code" aria-labelledby="code-h">
          <h2 id="code-h" class="lp-h2">Use it from code</h2>
          <p class="lp-sub">Call the live runtime over HTTP, or build the library and drive it from C++ or Python.</p>
          <CodeTabs />
        </section>

        <section class="lp-sec" id="how" aria-labelledby="how-h">
          <h2 id="how-h" class="lp-h2">How a job runs</h2>
          <div class="panel lp-flow-wrap" tabIndex={0} role="region" aria-label="Diagram: how a job flows through the graph"><Flow /></div>
          <div class="lp-cols">
            <div>
              <h3 class="lp-h3">The router, in order</h3>
              <ol class="lp-policies">
                {GRAPH.policies.map((p) => (
                  <li key={p.name}>
                    <code>{p.name}</code>
                    <span>{p.decides}</span>
                    {!GRAPH.activePolicies.includes(p.name) && <em class="fine">off by default in the console</em>}
                  </li>
                ))}
              </ol>
            </div>
            <div>
              <h3 class="lp-h3">Deadlines that bind</h3>
              <p>Give a job a time budget and the planner solves a 0/1 knapsack over the critical path to choose which nodes run INT8. A job it judges undeliverable is rejected before anything runs; one that runs out of time is cancelled inside the ONNX Runtime call.</p>
              <h3 class="lp-h3">Hot-swap without a lost job</h3>
              <p>A reload builds a new graph, models and scheduler as one snapshot and swaps a pointer to it atomically. Jobs in flight keep the snapshot they started with; new jobs get the new one. No locks on the hot path, nothing torn.</p>
              <h3 class="lp-h3">Two lanes, one of them simulated</h3>
              <p>CPU and GPU_SIM are two worker pools. There is no GPU here: GPU_SIM adds a modelled transfer cost, so the load-aware policy has a real trade-off to make.</p>
            </div>
          </div>
        </section>

        <section class="lp-sec" id="numbers" aria-labelledby="numbers-h">
          <h2 id="numbers-h" class="lp-h2">Numbers</h2>
          <p class="lp-sub">
            FP32 against INT8, {BENCH.warmup} warm-up runs then {BENCH.iters} timed, measured by <code>loomcore_bench</code> on the live runtime ({shortCpu(META.machine)}) when it started.
          </p>
          <div class="panel table-scroll">
            <table class="tbl">
              <thead>
                <tr><th scope="col">Model</th><th scope="col" class="num">FP32 p50</th><th scope="col" class="num">INT8 p50</th><th scope="col" class="num">INT8 speed-up</th></tr>
              </thead>
              <tbody>
                {fp && i8 && <tr><th scope="row">MobileNetV2</th><td class="num">{ms(fp.p50)} ms</td><td class="num">{ms(i8.p50)} ms</td><td class="num">{ratio(fp.p50 / i8.p50)}</td></tr>}
                {bfp && bi8 && <tr><th scope="row">bert_tiny</th><td class="num">{ms(bfp.p50)} ms</td><td class="num">{ms(bi8.p50)} ms</td><td class="num">{ratio(bfp.p50 / bi8.p50)}</td></tr>}
              </tbody>
            </table>
          </div>
          <p class="fine lp-after">
            {flags && !flags.avx512_vnni && !flags.avx_vnni
              ? 'This CPU has AVX2 but no VNNI, so INT8 convolutions have no fast path and MobileNetV2 gains nothing from quantizing; bert_tiny’s matrix multiplies still do. On a CPU with VNNI the gap opens up. '
              : ''}
            The console’s benchmark panel shows the same run, live from whichever machine the runtime is on.
          </p>
        </section>

        <section class="lp-sec" aria-labelledby="limits-h">
          <h2 id="limits-h" class="lp-h2">What it is not</h2>
          <ul class="lp-limits">
            <li>Not a model: MobileNetV2 and bert_tiny are untrained, off-the-shelf exports chosen for their different shapes. The work is the scheduling around them.</li>
            <li>Not a GPU server: everything runs on CPU, and GPU_SIM is simulated.</li>
            <li>The live runtime is small: Modal’s free tier, 2 cores, one job-running request at a time, started on demand. For real numbers, build it on your own machine.</li>
          </ul>
        </section>
      </main>

      <footer class="foot">
        <p>
          <a href={REPO}>github.com/armaanmittalweb/loomcore</a> · MIT licence · <a href={`${REPO}/blob/main/docs/CLAIMS.md`}>every claim with the command that checks it</a>
        </p>
        <p>Part of the Lab at <a href="https://www.amittal.dev">amittal.dev</a> · {count(LOAD.trace.length)} trace events on this page, recorded {day(META.recordedAt)}</p>
      </footer>
    </div>
  );
}
