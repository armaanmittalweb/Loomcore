// Trace -> timeline: pure functions that turn the runtime's Chrome Trace
// Event JSON (loomcore::exportPerfettoTrace) into lanes, bars, router markers
// and DAG edges, and do the view arithmetic (scale, ticks, zoom, pan). The
// renderer (Timeline.tsx) only draws what these return.
import type { Lane, PolicyOpinion, Precision, SwapTiming, TraceEvent } from '../types';

export const LANES: Lane[] = ['CPU', 'GPU_SIM'];
const TID_LANE: Record<number, Lane> = { 1: 'CPU', 2: 'GPU_SIM' };

export interface Bar {
  id: number;
  node: string;
  lane: Lane;
  start: number; // µs from the trace's zero
  end: number;
  precision: Precision;
  batch: number;
  jobs: string[];
  row: number; // sub-row within the lane: overlapping executions (several lane workers) stack
  cutShort: boolean; // ran past its job's deadline cancel (the ONNX Runtime call was terminated)
}

export type MarkerKind = 'decision' | 'skip' | 'rejected' | 'cancelled' | 'shed' | 'error';

export interface Marker {
  id: number;
  ts: number;
  kind: MarkerKind;
  node: string;
  job: string;
  message: string;
  policies: PolicyOpinion[];
}

export interface Edge {
  from: number; // bar id
  to: number;
  job: string;
}

/** One job's life: submitted to settled (the Jobs track). */
export interface JobSpan {
  job: string;
  start: number;
  end: number;
  status: 'ok' | 'cancelled' | 'failed';
  row: number;
}

export interface SwapMark {
  index: number;
  ts: number;
  buildMs: number;
  inFlight: number;
}

export interface TimelineModel {
  bars: Bar[];
  jobs: JobSpan[];
  jobRows: number;
  markers: Marker[];
  edges: Edge[];
  swaps: SwapMark[];
  rows: Record<Lane, number>; // sub-rows per lane (at least 1)
  end: number; // µs: the last thing that happened
}

export interface View {
  t0: number;
  t1: number;
}

/** "CompositeRouter: A: why | B: why" or "CompositeRouter -> A: why" -> opinions. */
export function parseRouterMessage(message: string): PolicyOpinion[] {
  const body = message.replace(/^CompositeRouter(?::\s*| -> )/, '');
  return body
    .split(' | ')
    .filter(Boolean)
    .map((part) => {
      const i = part.indexOf(': ');
      return i < 0 ? { policy: part.trim(), reason: '' } : { policy: part.slice(0, i).trim(), reason: part.slice(i + 2).trim() };
    });
}

function markerKind(ev: TraceEvent, message: string): MarkerKind {
  if (ev.name === 'job_rejected') return 'rejected';
  if (ev.name === 'error') {
    if (message.includes('exceeded its time budget')) return 'cancelled';
    if (message.includes('shed upstream')) return 'shed';
    return 'error';
  }
  return message.startsWith('CompositeRouter -> ') ? 'skip' : 'decision';
}

/** Greedy interval packing: each bar goes in the first sub-row that is free by its start. */
export function packRows(bars: Bar[]): Record<Lane, number> {
  const rows: Record<Lane, number> = { CPU: 1, GPU_SIM: 1 };
  for (const lane of LANES) {
    const ends: number[] = [];
    for (const bar of bars.filter((b) => b.lane === lane).sort((a, b) => a.start - b.start || a.id - b.id)) {
      let row = ends.findIndex((e) => e <= bar.start);
      if (row < 0) {
        row = ends.length;
        ends.push(bar.end);
      } else ends[row] = bar.end;
      bar.row = row;
    }
    rows[lane] = Math.max(1, ends.length);
  }
  return rows;
}

/** The same packing for job spans; returns the number of rows (0 when there are no spans). */
export function packSpans(spans: JobSpan[]): number {
  const ends: number[] = [];
  for (const s of [...spans].sort((a, b) => a.start - b.start)) {
    let row = ends.findIndex((e) => e <= s.start);
    if (row < 0) {
      row = ends.length;
      ends.push(s.end);
    } else ends[row] = s.end;
    s.row = row;
  }
  return ends.length;
}

function shortPolicy(name: string): string {
  return name.replace(/Policy$/, '').replace(/^CompositeRouter -> /, '');
}

/** A compact direct label for a router marker: "LatencyBudget: bert_tiny → INT8". */
export function markerLabel(m: Marker): string {
  if (m.kind === 'cancelled') return `cancelled ${m.job}`.trim();
  if (m.kind === 'rejected') return 'rejected by admission control';
  if (m.kind === 'shed') return `${m.job} failed: shed upstream`;
  if (m.kind === 'error') return `${m.job} failed`;
  const parts = m.policies.map((p) => {
    const who = shortPolicy(p.policy);
    const node = p.reason.match(/node '([^']+)'/)?.[1] ?? m.node;
    if (m.kind === 'skip') return `${who}: skip ${node}`;
    if (/INT8/.test(p.reason)) return `${who}: ${node} → INT8`;
    const q = p.reason.match(/cpu_queue=(\d+) gpu_sim_queue=(\d+)/);
    if (q) return `${who}: ${node} → ${Number(q[1]) < Number(q[2]) ? 'CPU' : 'GPU_SIM'}`;
    return who;
  });
  return parts.join(' · ');
}

function barAt(bars: Bar[], lane: Lane | undefined, ts: number): Bar | undefined {
  const onLane = bars.filter((b) => b.lane === lane);
  return (
    onLane.find((b) => b.start <= ts && ts <= b.end) ??
    onLane.reduce<Bar | undefined>((best, b) => {
      const d = Math.min(Math.abs(b.start - ts), Math.abs(b.end - ts));
      return !best || d < Math.min(Math.abs(best.start - ts), Math.abs(best.end - ts)) ? b : best;
    }, undefined)
  );
}

export function buildModel(trace: TraceEvent[], swaps: SwapTiming[] = []): TimelineModel {
  const bars: Bar[] = [];
  const markers: Marker[] = [];
  const jobs: JobSpan[] = [];
  const flowStarts = new Map<number, TraceEvent>();
  const flowEnds = new Map<number, TraceEvent>();

  for (const ev of trace) {
    if (ev.ph === 'X' && ev.cat === 'job' && typeof ev.ts === 'number') {
      const status = ev.args?.status;
      jobs.push({ job: ev.name, start: ev.ts, end: ev.ts + Math.max(0, ev.dur ?? 0), status: status === 'cancelled' || status === 'failed' ? status : 'ok', row: 0 });
    } else if (ev.ph === 'X' && typeof ev.ts === 'number') {
      const lane = TID_LANE[ev.tid ?? 0];
      if (!lane) continue;
      const args = ev.args ?? {};
      const jobs = Array.isArray(args.jobs) ? (args.jobs as string[]) : [];
      bars.push({
        id: bars.length,
        node: ev.name,
        lane,
        start: ev.ts,
        end: ev.ts + Math.max(0, ev.dur ?? 0),
        precision: args.precision === 'INT8' ? 'INT8' : 'FP32',
        batch: Number(args.batch_size ?? jobs.length) || 1,
        jobs,
        row: 0,
        cutShort: false,
      });
    } else if (ev.ph === 'i' && typeof ev.ts === 'number') {
      const args = ev.args ?? {};
      const message = String(args.message ?? '');
      const kind = markerKind(ev, message);
      markers.push({
        id: markers.length,
        ts: ev.ts,
        kind,
        node: String(args.node_id ?? ''),
        job: String(args.job_id ?? ''),
        message,
        policies: kind === 'decision' || kind === 'skip' ? parseRouterMessage(message) : [],
      });
    } else if (ev.cat === 'dag_edge' && typeof ev.id === 'number') {
      (ev.ph === 's' ? flowStarts : ev.ph === 'f' ? flowEnds : undefined)?.set(ev.id, ev);
    }
  }

  const cancelAt = new Map<string, number>();
  for (const m of markers) if (m.kind === 'cancelled' && m.job) cancelAt.set(m.job, m.ts);
  for (const bar of bars) {
    bar.cutShort = bar.jobs.some((j) => cancelAt.has(j) && bar.end >= (cancelAt.get(j) as number));
  }

  const edges: Edge[] = [];
  for (const [id, s] of flowStarts) {
    const f = flowEnds.get(id);
    if (!f || s.ts === undefined || f.ts === undefined) continue;
    const from = barAt(bars, TID_LANE[s.tid ?? 0], s.ts);
    const to = barAt(bars, TID_LANE[f.tid ?? 0], f.ts);
    if (from && to && from !== to) edges.push({ from: from.id, to: to.id, job: String(s.args?.job_id ?? '') });
  }

  const rows = packRows(bars);
  const jobRows = packSpans(jobs);
  const swapMarks = swaps.map((s) => ({ index: s.index, ts: s.atMs * 1000, buildMs: s.buildMs, inFlight: s.inFlight }));
  const end = Math.max(1, ...bars.map((b) => b.end), ...jobs.map((j) => j.end), ...markers.map((m) => m.ts), ...swapMarks.map((s) => s.ts));
  return { bars, jobs, jobRows, markers, edges, swaps: swapMarks, rows, end };
}

// -- view arithmetic --------------------------------------------------------

export const MIN_SPAN_US = 200; // never zoom in past 0.2 ms across the whole width

export function fitView(model: TimelineModel): View {
  const pad = model.end * 0.03;
  return { t0: 0, t1: model.end + pad };
}

export function xOf(ts: number, view: View, width: number): number {
  return ((ts - view.t0) / (view.t1 - view.t0)) * width;
}

export function tsOf(x: number, view: View, width: number): number {
  return view.t0 + (x / width) * (view.t1 - view.t0);
}

function clampView(t0: number, t1: number, bounds: View): View {
  const span = Math.min(Math.max(t1 - t0, MIN_SPAN_US), bounds.t1 - bounds.t0);
  let a = t0;
  if (a < bounds.t0) a = bounds.t0;
  if (a + span > bounds.t1) a = bounds.t1 - span;
  return { t0: a, t1: a + span };
}

/** Zoom by `factor` (<1 zooms in) keeping `anchor` (µs) at the same screen position. */
export function zoomAt(view: View, factor: number, anchor: number, bounds: View): View {
  const t0 = anchor - (anchor - view.t0) * factor;
  const t1 = anchor + (view.t1 - anchor) * factor;
  const span = Math.min(Math.max(t1 - t0, MIN_SPAN_US), bounds.t1 - bounds.t0);
  // Re-derive t0 so the anchor stays put even after the span was clamped.
  const ratio = (anchor - view.t0) / (view.t1 - view.t0);
  return clampView(anchor - ratio * span, anchor - ratio * span + span, bounds);
}

export function panBy(view: View, delta: number, bounds: View): View {
  return clampView(view.t0 + delta, view.t1 + delta, bounds);
}

/** Round tick positions (1/2/5 x 10^n microseconds) about `targetPx` apart. */
export function ticks(view: View, width: number, targetPx = 90): number[] {
  const span = view.t1 - view.t0;
  const raw = (span / Math.max(1, width)) * targetPx;
  const pow = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * pow).find((s) => s >= raw) ?? 10 * pow;
  const out: number[] = [];
  for (let t = Math.ceil(view.t0 / step) * step; t <= view.t1 + 1e-9; t += step) out.push(Math.round(t * 1000) / 1000);
  return out;
}

export function formatTick(us: number, stepUs: number): string {
  const ms = us / 1000;
  const decimals = stepUs >= 1000 ? 0 : stepUs >= 100 ? 1 : 2;
  return `${ms.toFixed(decimals)}`;
}

/** A DAG edge from the end of one bar to the start of another, as a soft S-curve. */
export function edgePath(x1: number, y1: number, x2: number, y2: number): string {
  const dx = Math.max(12, Math.abs(x2 - x1) * 0.45);
  return `M${x1.toFixed(1)} ${y1.toFixed(1)} C${(x1 + dx).toFixed(1)} ${y1.toFixed(1)} ${(x2 - dx).toFixed(1)} ${y2.toFixed(1)} ${x2.toFixed(1)} ${y2.toFixed(1)}`;
}

/** Everything in time order, for keyboard stepping and the table view. */
export type Item = { type: 'bar'; bar: Bar } | { type: 'marker'; marker: Marker } | { type: 'job'; job: JobSpan };

export function itemTime(i: Item): number {
  return i.type === 'bar' ? i.bar.start : i.type === 'marker' ? i.marker.ts : i.job.start;
}

export function itemsInOrder(model: TimelineModel): Item[] {
  const items: Item[] = [
    ...model.jobs.map((job) => ({ type: 'job' as const, job })),
    ...model.bars.map((bar) => ({ type: 'bar' as const, bar })),
    ...model.markers.map((marker) => ({ type: 'marker' as const, marker })),
  ];
  return items.sort((a, b) => itemTime(a) - itemTime(b));
}

export function itemKey(item: Item): string {
  return item.type === 'bar' ? `b${item.bar.id}` : item.type === 'marker' ? `m${item.marker.id}` : `j${item.job.job}`;
}
