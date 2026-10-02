import { useEffect, useMemo, useRef, useState } from 'preact/hooks';
import { type Client, createClient, describeFailure, SpaceError } from './api';
import { clock, ms, shortCpu } from './format';
import type { Bench, GraphInfo, LoadResult, PolicyName, ReloadResult, RecordingMeta, RunResult } from './types';
import { type Phase, wake, type WakeState } from './wake';
import { Timeline } from './timeline/Timeline';
import { GraphPanel, type NodeActivity } from './panels/GraphPanel';
import { Controls, type Mode, type Settings } from './panels/Controls';
import { LoadView, RunView, SwapView } from './panels/Results';
import { BenchPanel } from './panels/BenchPanel';
import { Claims } from './panels/Claims';
import { type Origin, Panel, recordedLine, SourceTag } from './ui';
import samplesJson from './samples.json';
import recordedMeta from './recorded/meta.json';
import recordedRun from './recorded/run.json';
import recordedLoad from './recorded/load.json';
import recordedReload from './recorded/reload.json';
import recordedBench from './recorded/bench.json';
import recordedGraph from './recorded/graph.json';

const META = recordedMeta as RecordingMeta;
const REPO = 'https://github.com/armaanmittalweb/loomcore';

type Held<T> = { data: T; origin: Origin; image?: string | null; caption?: string; fresh?: number };

interface Results {
  run: Held<RunResult>;
  load: Held<LoadResult>;
  swap: Held<ReloadResult>;
}

const SAMPLES = samplesJson as { id: string; caption: string; author: string; licence: string; source: string }[];
const captionOf = (id: string) => SAMPLES.find((s) => s.id === id)?.caption ?? id;
const recorded: Origin = { kind: 'recorded', meta: META };

function initialResults(): Results {
  const run = recordedRun as unknown as RunResult;
  return {
    run: { data: run, origin: recorded, image: `/samples/${run.source}-448.jpg`, caption: captionOf(run.source) },
    load: { data: recordedLoad as unknown as LoadResult, origin: recorded },
    swap: { data: recordedReload as unknown as ReloadResult, origin: recorded },
  };
}

function activityOf(mode: Mode, r: Results): Record<string, NodeActivity> {
  const out: Record<string, NodeActivity> = {};
  const get = (id: string) => (out[id] ??= { precisions: {}, lanes: [], skipped: 0, ran: 0 });
  if (mode === 'run') {
    for (const n of r.run.data.nodes) {
      const a = get(n.id);
      a.precisions[n.precision] = (a.precisions[n.precision] ?? 0) + 1;
      a.lanes.push(n.backend);
      a.ran += 1;
    }
    for (const id of r.run.data.skipped) get(id).skipped += 1;
  } else {
    const d = mode === 'load' ? r.load.data : r.swap.data;
    for (const n of d.nodes) {
      const a = get(n.id);
      a.precisions = { ...n.precisions };
      a.ran = n.count;
    }
    if (mode === 'load') for (const [id, c] of Object.entries(r.load.data.skipped)) get(id).skipped = c;
  }
  return out;
}

function StatusBar({ state, onRetry }: { state: WakeState; onRetry: () => void }) {
  const [now, setNow] = useState(state.startedAt);
  useEffect(() => {
    if (state.phase !== 'waking' && state.phase !== 'loading') return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [state.phase]);
  const secs = Math.max(0, Math.round((now - state.startedAt) / 1000));
  const h = state.health;
  const text: Record<Phase, string> = {
    checking: 'Live runtime: checking',
    waking: `Live runtime: starting (${secs} s)`,
    loading: 'Live runtime: loading models',
    live: h ? `Live · ${shortCpu(h.cpu)}` : 'Live',
    asleep: 'Live runtime unreachable',
  };
  return (
    <div class={`status st-${state.phase}`} role="status" title={h ? `Runtime ${h.version}, ${h.cpu}` : undefined}>
      <i class="lamp" aria-hidden="true" />
      <span>{text[state.phase]}</span>
      {state.phase === 'asleep' && (
        <button type="button" class="link" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}

export function ThemeToggle() {
  const [theme, setTheme] = useState<'light' | 'dark' | null>(null);
  useEffect(() => {
    let saved: string | null = null;
    try {
      saved = localStorage.getItem('loomcore-theme');
    } catch {
      saved = null;
    }
    const sys = window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    const t = saved === 'light' || saved === 'dark' ? saved : sys;
    setTheme(t);
    if (saved) document.documentElement.dataset.theme = saved;
  }, []);
  const flip = () => {
    const next = theme === 'dark' ? 'light' : 'dark';
    setTheme(next);
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem('loomcore-theme', next);
    } catch {
      /* per-viewer nicety only */
    }
  };
  return (
    <button type="button" class="icon-btn" onClick={flip} aria-label={theme === 'dark' ? 'Use the light theme' : 'Use the dark theme'} title="Theme">
      <svg viewBox="0 0 16 16" aria-hidden="true">
        <circle cx="8" cy="8" r="6" />
        <path d="M8 2a6 6 0 0 1 0 12Z" />
      </svg>
    </button>
  );
}

export function App({ client: injected }: { client?: Client }) {
  const client = useMemo(() => injected ?? createClient(), [injected]);
  const [wakeState, setWakeState] = useState<WakeState>({ phase: 'checking', startedAt: 0, attempts: 0, health: null });
  const [wakeRun, setWakeRun] = useState(0);
  const [graph, setGraph] = useState<{ data: GraphInfo; origin: Origin }>({ data: recordedGraph as GraphInfo, origin: recorded });
  const [bench, setBench] = useState<{ data: Bench; origin: Origin }>({ data: recordedBench as Bench, origin: recorded });
  const [results, setResults] = useState<Results>(initialResults);
  const [mode, setMode] = useState<Mode>('run');
  const [pending, setPending] = useState<Mode | null>(null);
  const [errors, setErrors] = useState<Partial<Record<Mode, string>>>({});
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [policies, setPolicies] = useState<PolicyName[]>((recordedGraph as GraphInfo).activePolicies);
  const [settings, setSettings] = useState<Settings>({
    sample: 'fox',
    upload: null,
    budgetOn: false,
    budget: 40,
    loadJobs: 48,
    concurrency: 12,
    loadBudgetOn: true,
    loadBudget: 200,
    swapJobs: 48,
    swaps: 4,
  });
  const liveCpu = wakeState.health?.cpu;
  const phaseRef = useRef<Phase>('checking');
  phaseRef.current = wakeState.phase;

  // Connect to the runtime in the background; the recorded run is on screen meanwhile.
  useEffect(() => {
    const ctl = new AbortController();
    const q = new URLSearchParams(location.search);
    const knobs = import.meta.env.MODE === 'shots' ? { limitMs: Number(q.get('wakeLimit')) || undefined, intervalMs: Number(q.get('poll')) || undefined } : {};
    wake(client, setWakeState, ctl.signal, knobs).then(async (s) => {
      if (s.phase !== 'live' || ctl.signal.aborted) return;
      const at = Date.now();
      const [g, b] = await Promise.allSettled([client.graph(), client.bench()]);
      if (ctl.signal.aborted) return;
      if (g.status === 'fulfilled') setGraph({ data: g.value, origin: { kind: 'live', at, cpu: s.health?.cpu } });
      if (b.status === 'fulfilled' && b.value.status === 'ready') setBench({ data: b.value, origin: { kind: 'live', at, cpu: s.health?.cpu } });
    });
    return () => ctl.abort();
  }, [client, wakeRun]);

  const patch = (p: Partial<Settings>) => setSettings((s) => ({ ...s, ...p }));
  const togglePolicy = (name: PolicyName) => setPolicies((ps) => (ps.includes(name) ? ps.filter((p) => p !== name) : [...ps, name]));

  async function go() {
    const m = mode;
    setPending(m);
    setErrors((e) => ({ ...e, [m]: undefined }));
    const live = (): Origin => ({ kind: 'live', at: Date.now(), cpu: liveCpu });
    try {
      if (m === 'run') {
        const s = settings;
        const data = await client.run({
          sample: s.upload ? undefined : s.sample,
          file: s.upload?.file,
          timeBudgetMs: s.budgetOn ? s.budget : null,
          policies,
        });
        setResults((r) => ({
          ...r,
          run: { data, origin: live(), image: s.upload ? s.upload.url : `/samples/${s.sample}-448.jpg`, caption: s.upload ? 'Your image' : captionOf(s.sample), fresh: Date.now() },
        }));
      } else if (m === 'load') {
        const data = await client.load({
          jobs: settings.loadJobs,
          concurrency: settings.concurrency,
          timeBudgetMs: settings.loadBudgetOn ? settings.loadBudget : null,
          policies,
          sample: settings.sample,
        });
        setResults((r) => ({ ...r, load: { data, origin: live(), fresh: Date.now() } }));
      } else {
        const data = await client.reload({ jobs: settings.swapJobs, swaps: settings.swaps });
        setResults((r) => ({ ...r, swap: { data, origin: live(), fresh: Date.now() } }));
      }
    } catch (err) {
      setErrors((e) => ({ ...e, [m]: describeFailure(err) }));
      if (err instanceof SpaceError && err.failure.kind === 'unreachable') setWakeRun((n) => n + 1);
    } finally {
      setPending(null);
    }
  }

  const held = results[mode];
  const trace = held.data.trace;
  const swaps = mode === 'swap' ? results.swap.data.swapTimings : undefined;
  const run = results.run.data;
  const load = results.load.data;
  const swap = results.swap.data;
  const title =
    mode === 'run'
      ? `${run.jobId ?? 'one job'} · ${results.run.caption ?? run.source} · ${ms(run.totalMs)} ms`
      : mode === 'load'
        ? `${load.submitted} jobs, ${load.concurrency} at a time · ${ms(load.wallMs)} ms`
        : `${swap.submitted} jobs across ${swap.swaps} graph swaps · ${ms(swap.wallMs)} ms`;
  const what = mode === 'run' ? 'One job' : mode === 'load' ? 'Load test' : 'Hot-swap under load';
  const busyText = pending === mode ? (mode === 'run' ? 'Running one job on the live runtime…' : mode === 'load' ? `Running ${settings.loadJobs} jobs on the live runtime…` : `Swapping the graph ${settings.swaps} times under load…`) : null;
  const activity = activityOf(mode, results);

  return (
    <div class="app">
      <a class="skip" href="#timeline">
        Skip to the timeline
      </a>
      <header class="top">
        <div class="brand">
          <h1 class="wordmark">
            Loomcore<span class="wm-dot" aria-hidden="true" />
          </h1>
          <p class="tagline">A C++ runtime that schedules ONNX models as a graph: a policy router, dynamic batching on two lanes, INT8 paths, binding deadlines, zero-downtime hot-swap.</p>
        </div>
        <div class="top-tools">
          <StatusBar state={wakeState} onRetry={() => setWakeRun((n) => n + 1)} />
          <ThemeToggle />
          <a class="icon-btn" href={REPO} aria-label="Source on GitHub" title="Source on GitHub">
            <svg viewBox="0 0 16 16" aria-hidden="true">
              <path d="M8 .2a8 8 0 0 0-2.5 15.6c.4 0 .5-.2.5-.4v-1.5c-2.2.5-2.7-1-2.7-1-.4-.9-.9-1.2-.9-1.2-.7-.5.1-.5.1-.5.8.1 1.2.8 1.2.8.7 1.3 1.9.9 2.3.7.1-.5.3-.9.5-1.1-1.8-.2-3.6-.9-3.6-4 0-.9.3-1.6.8-2.1-.1-.2-.4-1 .1-2.1 0 0 .7-.2 2.2.8a7.5 7.5 0 0 1 4 0c1.5-1 2.2-.8 2.2-.8.4 1.1.2 1.9.1 2.1.5.6.8 1.3.8 2.1 0 3.1-1.9 3.8-3.6 4 .3.3.6.8.6 1.5v2.2c0 .2.1.5.6.4A8 8 0 0 0 8 .2Z" />
            </svg>
          </a>
        </div>
      </header>

      <main class="grid">
        <div class="col col-main">
          <Panel
            id="timeline"
            kicker={`Timeline · ${what}`}
            title={title}
            tag={<SourceTag origin={held.origin} />}
            class="p-timeline"
          >
            <p class="panel-sub">
              {held.origin.kind === 'recorded'
                ? `${recordedLine(META)} ${phaseRef.current === 'live' ? 'Run it yourself from Controls.' : 'Connecting to the live runtime.'}`
                : `Live, ${clock(held.origin.at)}. Each bar is one real ONNX Runtime call, placed where the scheduler ran it.`}
            </p>
            <Timeline key={`${mode}-${held.fresh ?? 0}`} trace={trace} swaps={swaps} animate={!!held.fresh} busy={busyText} label={`${what}: ${title}`} />
          </Panel>

          <Panel id="result" kicker="Result" title={mode === 'run' ? 'What the job produced' : mode === 'load' ? 'What the scheduler did under load' : 'Jobs across the swaps'} tag={<SourceTag origin={held.origin} />} class="p-result">
            {mode === 'run' && <RunView r={run} image={results.run.image ?? null} caption={results.run.caption ?? run.source} />}
            {mode === 'load' && <LoadView r={load} />}
            {mode === 'swap' && <SwapView r={swap} />}
          </Panel>
          <BenchPanel bench={bench.data} origin={bench.origin} />
        </div>

        <div class="col col-side">
          <Controls
            mode={mode}
            onMode={setMode}
            settings={settings}
            onChange={patch}
            samples={SAMPLES}
            phase={wakeState.phase}
            pending={pending}
            error={errors[mode] ?? null}
            uploadError={uploadError}
            onUploadError={setUploadError}
            onGo={go}
          />
          <GraphPanel graph={graph.data} origin={graph.origin} activity={activity} showCounts={mode !== 'run'} policies={policies} onToggle={togglePolicy} gateThreshold={0.85} />
        </div>

        <div class="col col-wide">
          <Claims />
        </div>
      </main>

      <footer class="foot">
        <p>
          <a href={REPO}>github.com/armaanmittalweb/loomcore</a> · MIT licence · runtime on Modal's free tier, started on demand
        </p>
        <p>
          Part of the Lab at <a href="https://www.amittal.dev">amittal.dev</a> · sample photos from Wikimedia Commons, credited in <a href={`${REPO}/blob/main/space/samples/LICENSES.md`}>LICENSES.md</a>
        </p>
      </footer>
    </div>
  );
}
