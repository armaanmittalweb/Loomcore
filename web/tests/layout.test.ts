import { describe, expect, it } from 'vitest';
import { buildModel, edgePath, fitView, formatTick, itemsInOrder, markerLabel, MIN_SPAN_US, packRows, panBy, parseRouterMessage, ticks, tsOf, xOf, zoomAt, type Bar } from '../src/timeline/layout';
import type { TraceEvent } from '../src/types';
import run from '../src/recorded/run.json';
import load from '../src/recorded/load.json';
import reload from '../src/recorded/reload.json';
import cancelled from '../scripts/fixtures/run-cancelled.json';

const X = (name: string, tid: number, ts: number, dur: number, args: Record<string, unknown> = {}): TraceEvent => ({ name, cat: 'execution', ph: 'X', ts, dur, pid: 1, tid, args: { precision: 'FP32', batch_size: 1, jobs: ['job-1'], ...args } });

describe('parseRouterMessage', () => {
  it('splits a merged decision into its policies', () => {
    expect(parseRouterMessage("CompositeRouter: LatencyBudgetPolicy: remaining budget 19ms < threshold 30ms | LoadAwareBackendPolicy: cpu_queue=2 gpu_sim_queue=0; routing to the shallower lane")).toEqual([
      { policy: 'LatencyBudgetPolicy', reason: 'remaining budget 19ms < threshold 30ms' },
      { policy: 'LoadAwareBackendPolicy', reason: 'cpu_queue=2 gpu_sim_queue=0; routing to the shallower lane' },
    ]);
  });
  it('reads a terminal skip', () => {
    expect(parseRouterMessage("CompositeRouter -> ConfidenceGatePolicy: upstream confidence 0.99 >= threshold 0.85; skipping node 'bert_tiny'")).toEqual([
      { policy: 'ConfidenceGatePolicy', reason: "upstream confidence 0.99 >= threshold 0.85; skipping node 'bert_tiny'" },
    ]);
  });
});

describe('buildModel', () => {
  it('maps tracks to lanes, keeps precision and batch, and finds the run end', () => {
    const m = buildModel([X('mobilenet', 1, 8000, 12000), X('bert_tiny', 2, 30000, 4000, { precision: 'INT8', batch_size: 3, jobs: ['a', 'b', 'c'] })]);
    expect(m.bars.map((b) => [b.node, b.lane, b.precision, b.batch])).toEqual([
      ['mobilenet', 'CPU', 'FP32', 1],
      ['bert_tiny', 'GPU_SIM', 'INT8', 3],
    ]);
    expect(m.end).toBe(34000);
  });

  it('reads the recorded run: one job, two executions, one decision, one DAG edge', () => {
    const m = buildModel(run.trace as TraceEvent[]);
    expect(m.jobs).toHaveLength(1);
    expect(m.bars.map((b) => b.node)).toEqual(['mobilenet', 'bert_tiny']);
    expect(m.markers.map((x) => x.kind)).toEqual(['decision']);
    expect(m.edges).toHaveLength(1);
    const [from, to] = [m.bars[m.edges[0].from], m.bars[m.edges[0].to]];
    expect(from.node).toBe('mobilenet');
    expect(to.node).toBe('bert_tiny');
    expect(from.end).toBeLessThanOrEqual(to.start); // the dependent starts after its upstream ends
    expect(m.jobs[0].start).toBeLessThanOrEqual(from.start);
    expect(m.jobs[0].end).toBeGreaterThanOrEqual(to.end);
  });

  it('marks a deadline cancel and the execution it cut short', () => {
    const m = buildModel(cancelled.trace as TraceEvent[]);
    const cancel = m.markers.find((x) => x.kind === 'cancelled');
    expect(cancel).toBeDefined();
    expect(m.jobs[0].status).toBe('cancelled');
    const cut = m.bars.filter((b) => b.cutShort);
    expect(cut.length).toBeGreaterThan(0);
    for (const b of cut) expect(b.end).toBeGreaterThanOrEqual(cancel!.ts);
  });

  it('places swaps at their trace time', () => {
    const m = buildModel(reload.trace as TraceEvent[], reload.swapTimings);
    expect(m.swaps.map((s) => s.index)).toEqual(reload.swapTimings.map((s) => s.index));
    expect(m.swaps[0].ts).toBeCloseTo(reload.swapTimings[0].atMs * 1000);
    expect(m.end).toBeGreaterThanOrEqual(m.swaps[m.swaps.length - 1].ts);
  });

  it('packs a load test into sub-rows with no overlap within a row', () => {
    const m = buildModel(load.trace as TraceEvent[]);
    for (const lane of ['CPU', 'GPU_SIM'] as const) {
      const bars = m.bars.filter((b) => b.lane === lane);
      for (let r = 0; r < m.rows[lane]; r++) {
        const row = bars.filter((b) => b.row === r).sort((a, b) => a.start - b.start);
        for (let i = 1; i < row.length; i++) expect(row[i].start).toBeGreaterThanOrEqual(row[i - 1].end);
      }
    }
    expect(m.bars.length).toBe(load.batches.reduce((n, b) => n + b.count, 0));
    expect(m.jobs.length).toBe(load.submitted - load.rejected);
  });
});

describe('packRows', () => {
  it('reuses a row as soon as it frees up and opens a new one on overlap', () => {
    const bar = (id: number, start: number, end: number): Bar => ({ id, node: 'n', lane: 'CPU', start, end, precision: 'FP32', batch: 1, jobs: [], row: -1, cutShort: false });
    const bars = [bar(0, 0, 10), bar(1, 5, 15), bar(2, 10, 20), bar(3, 12, 14)];
    const rows = packRows(bars);
    expect(rows.CPU).toBe(3);
    expect(bars.map((b) => b.row)).toEqual([0, 1, 0, 2]);
    expect(rows.GPU_SIM).toBe(1);
  });
});

describe('markerLabel', () => {
  const m = buildModel(run.trace as TraceEvent[]);
  it('says what the decision did', () => {
    expect(markerLabel(m.markers[0])).toBe('LatencyBudget: bert_tiny → INT8');
  });
  it('names the lane a load-aware decision chose', () => {
    const lb = buildModel([{ name: 'routing_decision', ph: 'i', ts: 5, tid: 3, args: { node_id: 'mobilenet', job_id: 'job-2', message: 'CompositeRouter: LoadAwareBackendPolicy: cpu_queue=3 gpu_sim_queue=1; routing to the shallower lane' } }]);
    expect(markerLabel(lb.markers[0])).toBe('LoadAwareBackend: mobilenet → GPU_SIM');
  });
});

describe('view arithmetic', () => {
  const bounds = { t0: 0, t1: 100_000 };
  it('maps time to x and back', () => {
    expect(xOf(25_000, bounds, 800)).toBe(200);
    expect(tsOf(200, bounds, 800)).toBe(25_000);
  });
  it('zooms around the anchor and keeps it in place', () => {
    const v = zoomAt(bounds, 0.5, 50_000, bounds);
    expect(v).toEqual({ t0: 25_000, t1: 75_000 });
    const near = zoomAt(bounds, 0.5, 20_000, bounds);
    expect((20_000 - near.t0) / (near.t1 - near.t0)).toBeCloseTo(0.2);
  });
  it('never zooms out past the run or in past the minimum span', () => {
    expect(zoomAt(bounds, 4, 50_000, bounds)).toEqual(bounds);
    const tight = zoomAt(bounds, 1e-6, 50_000, bounds);
    expect(tight.t1 - tight.t0).toBe(MIN_SPAN_US);
  });
  it('pans within bounds', () => {
    expect(panBy({ t0: 10_000, t1: 20_000 }, 5_000, bounds)).toEqual({ t0: 15_000, t1: 25_000 });
    expect(panBy({ t0: 10_000, t1: 20_000 }, -50_000, bounds)).toEqual({ t0: 0, t1: 10_000 });
    expect(panBy({ t0: 80_000, t1: 95_000 }, 50_000, bounds)).toEqual({ t0: 85_000, t1: 100_000 });
  });
  it('fits with a little air after the last event', () => {
    const v = fitView(buildModel(run.trace as TraceEvent[]));
    expect(v.t0).toBe(0);
    expect(v.t1).toBeGreaterThan(buildModel(run.trace as TraceEvent[]).end);
  });
  it('puts ticks on round 1/2/5 steps', () => {
    const t = ticks({ t0: 0, t1: 40_000 }, 800);
    expect(t[0]).toBe(0);
    const step = t[1] - t[0];
    expect([1000, 2000, 5000, 10000]).toContain(step);
    expect(t.every((v) => v % step === 0)).toBe(true);
    expect(formatTick(5000, 5000)).toBe('5');
    expect(formatTick(500, 500)).toBe('0.5');
  });
  it('draws edges as a curve from end to start', () => {
    expect(edgePath(10, 20, 110, 60)).toMatch(/^M10\.0 20\.0 C.* 110\.0 60\.0$/);
  });
  it('orders every item in time for keyboard stepping', () => {
    const items = itemsInOrder(buildModel(run.trace as TraceEvent[]));
    const times = items.map((i) => (i.type === 'bar' ? i.bar.start : i.type === 'marker' ? i.marker.ts : i.job.start));
    expect([...times].sort((a, b) => a - b)).toEqual(times);
    expect(items[0].type).toBe('job');
  });
});
