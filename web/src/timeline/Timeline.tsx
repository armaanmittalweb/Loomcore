import { useEffect, useMemo, useRef, useState } from 'preact/hooks';
import type { SwapTiming, TraceEvent } from '../types';
import { ms, tidyReason } from '../format';
import {
  type Bar,
  type Item,
  type JobSpan,
  type Marker,
  type TimelineModel,
  type View,
  LANES,
  buildModel,
  edgePath,
  fitView,
  formatTick,
  itemKey,
  itemsInOrder,
  itemTime,
  markerLabel,
  panBy,
  ticks,
  tsOf,
  xOf,
  zoomAt,
} from './layout';

const LABEL_W = 108;
const RULER_H = 30;
const ROW_H = 26;
const BAR_H = 18;
const LANE_PAD = 11;
const ROUTER_H = 42;
const JOB_ROW = 9;
const JOB_PAD = 12;
const MIN_W = 760;

const LANE_NAME = { CPU: 'CPU', GPU_SIM: 'GPU_SIM' } as const;
const LANE_SUB = { CPU: 'lane', GPU_SIM: 'simulated lane' } as const;

interface Props {
  trace: TraceEvent[];
  swaps?: SwapTiming[];
  animate?: boolean;
  busy?: string | null;
  label: string;
}

function laneTops(model: TimelineModel) {
  const tops: Record<string, number> = {};
  let y = RULER_H;
  tops.jobs = y;
  if (model.jobRows > 0) y += Math.max(44, model.jobRows * JOB_ROW + JOB_PAD * 2);
  for (const lane of LANES) {
    tops[lane] = y;
    y += Math.max(44, model.rows[lane] * ROW_H - (ROW_H - BAR_H) + LANE_PAD * 2);
  }
  tops.router = y;
  return { tops, height: y + ROUTER_H };
}

const barY = (bar: Bar, tops: Record<string, number>) => tops[bar.lane] + LANE_PAD + bar.row * ROW_H;

const jobY = (j: JobSpan, tops: Record<string, number>) => tops.jobs + JOB_PAD + j.row * JOB_ROW + JOB_ROW / 2;
const JOB_STATUS = { ok: 'completed', cancelled: 'cancelled at its deadline', failed: 'failed' } as const;

function describe(item: Item): string {
  if (item.type === 'job') {
    const j = item.job;
    return `${j.job}: submitted at ${ms(j.start / 1000)} ms, ${JOB_STATUS[j.status]} ${ms((j.end - j.start) / 1000)} ms later`;
  }
  if (item.type === 'bar') {
    const b = item.bar;
    const jobs = b.jobs.length ? `${b.jobs.length === 1 ? 'job' : 'jobs'} ${b.jobs.join(', ')}` : '';
    return `${b.node} on ${b.lane}, ${b.precision}, ${ms((b.end - b.start) / 1000)} ms from ${ms(b.start / 1000)} ms${b.batch > 1 ? `, batch of ${b.batch}` : ''}${jobs ? `, ${jobs}` : ''}${b.cutShort ? ', cut short by the deadline' : ''}`;
  }
  const m = item.marker;
  return `${KIND_LABEL[m.kind]} at ${ms(m.ts / 1000)} ms${m.node ? ` for ${m.node}` : ''}: ${tidyReason(m.message)}`;
}

const KIND_LABEL: Record<Marker['kind'], string> = {
  decision: 'Router decision',
  skip: 'Router skip',
  rejected: 'Rejected by admission control',
  cancelled: 'Cancelled at the deadline',
  shed: 'Shed upstream',
  error: 'Job failed',
};

function MarkerGlyph({ m, x, y }: { m: Marker; x: number; y: number }) {
  if (m.kind === 'decision') return <path class="mk mk-decision" d={`M${x} ${y - 5.5}L${x + 5.5} ${y}L${x} ${y + 5.5}L${x - 5.5} ${y}Z`} />;
  if (m.kind === 'skip') return <path class="mk mk-skip" d={`M${x} ${y - 5}L${x + 5} ${y}L${x} ${y + 5}L${x - 5} ${y}Z`} />;
  return (
    <g class={`mk mk-bad mk-${m.kind}`}>
      <circle cx={x} cy={y} r={6.5} />
      <path d={`M${x - 3} ${y - 3}L${x + 3} ${y + 3}M${x + 3} ${y - 3}L${x - 3} ${y + 3}`} />
    </g>
  );
}

function Tip({ item }: { item: Item }) {
  if (item.type === 'job') {
    const j = item.job;
    return (
      <>
        <div class="tip-head">
          <strong>{j.job}</strong>
          <span>{JOB_STATUS[j.status]}</span>
        </div>
        <dl class="tip-grid">
          <dt>life</dt>
          <dd>{ms((j.end - j.start) / 1000)} ms, submit to {j.status === 'ok' ? 'result' : 'failure'}</dd>
          <dt>from</dt>
          <dd>{ms(j.start / 1000)} → {ms(j.end / 1000)} ms</dd>
        </dl>
        <p class="tip-msg">Gaps between its bars are batch-window waits and queueing.</p>
      </>
    );
  }
  if (item.type === 'bar') {
    const b = item.bar;
    return (
      <>
        <div class="tip-head">
          <i class={`sw sw-${b.precision.toLowerCase()}`} aria-hidden="true" />
          <strong>{b.node}</strong>
          <span>{b.precision} · {b.lane}</span>
        </div>
        <dl class="tip-grid">
          <dt>ran</dt>
          <dd>{ms((b.end - b.start) / 1000)} ms</dd>
          <dt>from</dt>
          <dd>{ms(b.start / 1000)} → {ms(b.end / 1000)} ms</dd>
          <dt>batch</dt>
          <dd>{b.batch === 1 ? '1 job' : `${b.batch} jobs in one ONNX Runtime call`}</dd>
          {b.jobs.length > 0 && (
            <>
              <dt>{b.jobs.length === 1 ? 'job' : 'jobs'}</dt>
              <dd>{b.jobs.join(' ')}</dd>
            </>
          )}
        </dl>
        {b.cutShort && <p class="tip-bad">Cut short: the deadline reaper cancelled this call.</p>}
      </>
    );
  }
  const m = item.marker;
  return (
    <>
      <div class="tip-head">
        <strong>{KIND_LABEL[m.kind]}</strong>
        <span>{ms(m.ts / 1000)} ms{m.node ? ` · ${m.node}` : ''}{m.job ? ` · ${m.job}` : ''}</span>
      </div>
      {m.policies.length > 0 ? (
        <ul class="tip-reasons">
          {m.policies.map((p) => (
            <li key={p.policy}>
              <code>{p.policy}</code>
              <span>{tidyReason(p.reason)}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p class="tip-msg">{tidyReason(m.message)}</p>
      )}
    </>
  );
}

export function Timeline({ trace, swaps, animate = false, busy = null, label }: Props) {
  const model = useMemo(() => buildModel(trace, swaps), [trace, swaps]);
  const items = useMemo(() => itemsInOrder(model), [model]);
  const bounds = useMemo(() => fitView(model), [model]);
  const [view, setView] = useState<View>(bounds);
  const [width, setWidth] = useState(880);
  const [hover, setHover] = useState<string | null>(null);
  const [sel, setSel] = useState<number | null>(null);
  const [copied, setCopied] = useState<'idle' | 'ok' | 'fail'>('idle');
  const scroller = useRef<HTMLDivElement>(null);
  const plot = useRef<SVGSVGElement>(null);
  const drag = useRef<{ x: number; view: View } | null>(null);

  useEffect(() => {
    setView(bounds);
    setSel(null);
    setHover(null);
  }, [bounds]);

  useEffect(() => {
    const el = scroller.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(() => setWidth(Math.max(MIN_W, el.clientWidth)));
    ro.observe(el);
    setWidth(Math.max(MIN_W, el.clientWidth));
    return () => ro.disconnect();
  }, []);

  const plotW = width - LABEL_W;
  const { tops, height } = laneTops(model);
  const x = (ts: number) => LABEL_W + xOf(ts, view, plotW);
  const tickList = ticks(view, plotW);
  const step = tickList.length > 1 ? tickList[1] - tickList[0] : view.t1 - view.t0;
  const barById = new Map(model.bars.map((b) => [b.id, b]));

  // ctrl/cmd + wheel (and trackpad pinch) zooms at the pointer; shift + wheel pans.
  useEffect(() => {
    const el = plot.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      const rect = el.getBoundingClientRect();
      const px = e.clientX - rect.left - LABEL_W;
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        const anchor = tsOf(Math.max(0, px), view, plotW);
        setView((v) => zoomAt(v, Math.exp(e.deltaY * 0.0025), anchor, bounds));
      } else if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
        const d = e.shiftKey ? e.deltaY : e.deltaX;
        if (view.t0 <= bounds.t0 && view.t1 >= bounds.t1) return; // nothing to pan
        e.preventDefault();
        setView((v) => panBy(v, (d / plotW) * (v.t1 - v.t0), bounds));
      }
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [view, plotW, bounds]);

  const zoomBy = (f: number) => setView((v) => zoomAt(v, f, (v.t0 + v.t1) / 2, bounds));
  const fitted = view.t0 <= bounds.t0 + 1 && view.t1 >= bounds.t1 - 1;

  const selected = sel != null ? items[sel] : null;
  const active = hover ? items.find((i) => itemKey(i) === hover) ?? null : selected;
  const activeJobs = new Set(
    active?.type === 'bar' ? active.bar.jobs : active?.type === 'job' ? [active.job.job] : active?.type === 'marker' && active.marker.job ? [active.marker.job] : [],
  );

  function revealItem(i: number) {
    const ts = itemTime(items[i]);
    setView((v) => (ts < v.t0 || ts > v.t1 ? panBy(v, ts - (v.t0 + v.t1) / 2, bounds) : v));
  }

  function onKey(e: KeyboardEvent) {
    const span = view.t1 - view.t0;
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      e.preventDefault();
      if (e.shiftKey) return setView((v) => panBy(v, (e.key === 'ArrowRight' ? 0.2 : -0.2) * span, bounds));
      if (!items.length) return;
      const next = sel == null ? (e.key === 'ArrowRight' ? 0 : items.length - 1) : Math.min(items.length - 1, Math.max(0, sel + (e.key === 'ArrowRight' ? 1 : -1)));
      setSel(next);
      setHover(null);
      revealItem(next);
    } else if (e.key === '+' || e.key === '=') {
      e.preventDefault();
      zoomBy(0.7);
    } else if (e.key === '-' || e.key === '_') {
      e.preventDefault();
      zoomBy(1 / 0.7);
    } else if (e.key === '0') {
      e.preventDefault();
      setView(bounds);
    } else if (e.key === 'Escape') {
      setSel(null);
    }
  }

  function onPointerDown(e: PointerEvent) {
    if (e.button !== 0 || fitted) return;
    drag.current = { x: e.clientX, view };
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
  }
  function onPointerMove(e: PointerEvent) {
    const d = drag.current;
    if (!d) return;
    const dt = ((d.x - e.clientX) / plotW) * (d.view.t1 - d.view.t0);
    setView(panBy(d.view, dt, bounds));
  }
  const onPointerUp = () => {
    drag.current = null;
  };

  async function copyTrace() {
    try {
      await navigator.clipboard.writeText(JSON.stringify({ traceEvents: trace, displayTimeUnit: 'ms' }));
      setCopied('ok');
    } catch {
      setCopied('fail');
    }
    setTimeout(() => setCopied('idle'), 2500);
  }

  // Tooltip position: above the item, inside the panel.
  let tip: { left: number; top: number; flip: boolean } | null = null;
  if (active) {
    const span = (a: number, b: number) => x(Math.max(view.t0, a)) + Math.min(60, Math.max(0, x(Math.min(view.t1, b)) - x(Math.max(view.t0, a))) / 2);
    const ix = active.type === 'bar' ? span(active.bar.start, active.bar.end) : active.type === 'job' ? span(active.job.start, active.job.end) : x(active.marker.ts);
    const iy = active.type === 'bar' ? barY(active.bar, tops) : active.type === 'job' ? jobY(active.job, tops) - 4 : tops.router + ROUTER_H / 2 - 6;
    const scrollLeft = scroller.current?.scrollLeft ?? 0;
    const visibleW = scroller.current?.clientWidth ?? width;
    const left = Math.min(Math.max(8, ix - scrollLeft - 150), Math.max(8, visibleW - 316));
    const flip = iy < 120;
    tip = { left, top: flip ? iy + BAR_H + 10 : iy - 8, flip };
  }

  const durationMs = model.end / 1000;
  const animTotal = 480;

  return (
    <div class={`tl ${busy ? 'is-busy' : ''}`}>
      <div class="tl-toolbar">
        <div class="legend" aria-label="Legend">
          <span><i class="sw sw-fp32" aria-hidden="true" />FP32</span>
          <span><i class="sw sw-int8" aria-hidden="true" />INT8</span>
          <span><svg class="lg-glyph" viewBox="-7 -7 14 14" aria-hidden="true"><path class="mk mk-decision" d="M0 -5.5L5.5 0L0 5.5L-5.5 0Z" /></svg>router decision</span>
          <span><svg class="lg-glyph" viewBox="-7 -7 14 14" aria-hidden="true"><path class="mk mk-skip" d="M0 -5L5 0L0 5L-5 0Z" /></svg>skip</span>
          <span><svg class="lg-glyph" viewBox="-8 -8 16 16" aria-hidden="true"><g class="mk mk-bad"><circle r="6.5" /><path d="M-3 -3L3 3M3 -3L-3 3" /></g></svg>reject / cancel</span>
          <span><svg class="lg-glyph lg-edge" viewBox="0 0 22 10" aria-hidden="true"><path d="M1 8 C8 8 12 2 20 2" /></svg>DAG edge</span>
          {model.swaps.length > 0 && <span><i class="sw sw-swap" aria-hidden="true" />graph swap</span>}
        </div>
        <div class="zoom" role="group" aria-label="Zoom">
          <button type="button" class="tool" onClick={() => zoomBy(1 / 0.7)} disabled={fitted} aria-label="Zoom out">−</button>
          <span class="zoom-read" aria-live="polite">{ms((view.t1 - view.t0) / 1000)} ms</span>
          <button type="button" class="tool" onClick={() => zoomBy(0.7)} aria-label="Zoom in">+</button>
          <button type="button" class="tool tool-text" onClick={() => setView(bounds)} disabled={fitted}>Fit</button>
        </div>
      </div>

      <div class="tl-frame">
        <div
          class="tl-scroll"
          ref={scroller}
          role="application"
          aria-roledescription="timeline"
          aria-label={`${label}. ${model.jobs.length} jobs, ${model.bars.length} executions and ${model.markers.length} router events over ${ms(durationMs)} ms. Left and right arrows step through events, plus and minus zoom, 0 fits.`}
          tabIndex={0}
          onKeyDown={onKey}
        >
          <svg
            ref={plot}
            aria-hidden="true"
            class={`tl-svg ${fitted ? '' : 'can-pan'} ${animate ? 'is-animated' : ''}`}
            width={width}
            height={height}
            viewBox={`0 0 ${width} ${height}`}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerCancel={onPointerUp}
            onPointerLeave={() => setHover(null)}
          >
            <defs>
              <clipPath id="tl-plot">
                <rect x={LABEL_W} y={0} width={plotW} height={height} />
              </clipPath>
              <marker id="tl-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                <path d="M0 0.5 L7.5 4 L0 7.5 Z" class="arrowhead" />
              </marker>
            </defs>

            {model.jobRows > 0 && <rect class="lane-band jobs" x={0} y={tops.jobs} width={width} height={tops[LANES[0]] - tops.jobs} />}
            {LANES.map((lane, i) => (
              <rect key={lane} class={`lane-band ${i % 2 ? '' : 'odd'}`} x={0} y={tops[lane]} width={width} height={(i === LANES.length - 1 ? tops.router : tops[LANES[i + 1]]) - tops[lane]} />
            ))}
            <rect class="lane-band router" x={0} y={tops.router} width={width} height={ROUTER_H} />

            <g clip-path="url(#tl-plot)">
              {tickList.map((t) => (
                <line key={`g${t}`} class="gl" x1={x(t)} x2={x(t)} y1={RULER_H} y2={height} />
              ))}
            </g>
            <line class="axis" x1={LABEL_W} x2={width} y1={RULER_H - 0.5} y2={RULER_H - 0.5} />
            <g class="ruler" clip-path="url(#tl-plot)">
              {tickList.map((t) => (
                <g key={`t${t}`}>
                  <line class="tick" x1={x(t)} x2={x(t)} y1={RULER_H - 6} y2={RULER_H} />
                  <text x={x(t) + 4} y={RULER_H - 10}>{formatTick(t, step)}</text>
                </g>
              ))}
            </g>
            <text class="ruler-unit" x={LABEL_W - 10} y={RULER_H - 10} text-anchor="end">ms</text>

            {LANES.map((lane) => (
              <g key={`l${lane}`} class="lane-label">
                <text x={12} y={tops[lane] + LANE_PAD + 11}>{LANE_NAME[lane]}</text>
                <text class="sub" x={12} y={tops[lane] + LANE_PAD + 24}>{LANE_SUB[lane]}</text>
              </g>
            ))}
            <g class="lane-label">
              <text x={12} y={tops.router + ROUTER_H / 2 + 4}>Router</text>
            </g>
            {model.jobRows > 0 && (
              <g class="lane-label">
                <text x={12} y={tops.jobs + JOB_PAD + 9}>Jobs</text>
                <text class="sub" x={12} y={tops.jobs + JOB_PAD + 22}>submit → settle</text>
              </g>
            )}
            <line class="divider" x1={LABEL_W - 0.5} x2={LABEL_W - 0.5} y1={RULER_H} y2={height} />

            <g clip-path="url(#tl-plot)">
              {model.swaps.map((s) => {
                const sx = x(s.ts);
                const left = sx + 46 > width; // flip the tag left of the line at the right edge
                return (
                  <g key={`s${s.index}`} class="swap">
                    <line x1={sx} x2={sx} y1={RULER_H} y2={height} />
                    <rect x={left ? sx - 45 : sx - 1} y={RULER_H + 1} width={46} height={14} rx={2} />
                    <text x={left ? sx - 41 : sx + 4} y={RULER_H + 11}>swap {s.index}</text>
                  </g>
                );
              })}

              {model.jobs.map((j) => {
                const key = `j${j.job}`;
                const x1 = x(j.start);
                const x2 = Math.max(x1 + 2, x(j.end));
                const y = jobY(j, tops);
                const isOn = active?.type === 'job' && active.job.job === j.job;
                const dim = activeJobs.size > 0 && !activeJobs.has(j.job);
                return (
                  <g key={key} class={`job job-${j.status} ${isOn ? 'on' : ''} ${dim ? 'dim' : ''}`} onPointerEnter={() => setHover(key)}>
                    <rect class="hit" x={x1} y={y - JOB_ROW / 2} width={x2 - x1} height={JOB_ROW} />
                    <line x1={x1} x2={x2} y1={y} y2={y} />
                    <line class="cap" x1={x1} x2={x1} y1={y - 3} y2={y + 3} />
                    {j.status === 'ok' ? <circle cx={x2} cy={y} r={2.6} /> : <path class="end-bad" d={`M${x2 - 3} ${y - 3}L${x2 + 3} ${y + 3}M${x2 + 3} ${y - 3}L${x2 - 3} ${y + 3}`} />}
                    {model.jobs.length <= 3 && (
                      <text x={x1 + 6} y={y - 5}>
                        {j.job} · {ms((j.end - j.start) / 1000)} ms, submit to {j.status === 'ok' ? 'result' : JOB_STATUS[j.status]}
                      </text>
                    )}
                  </g>
                );
              })}

              <g class={`edges ${activeJobs.size ? 'has-focus' : ''}`}>
                {model.edges.map((e, i) => {
                  const a = barById.get(e.from) as Bar;
                  const b = barById.get(e.to) as Bar;
                  const on = activeJobs.has(e.job);
                  return (
                    <path
                      key={i}
                      class={`edge ${on ? 'on' : ''}`}
                      d={edgePath(x(a.end), barY(a, tops) + BAR_H / 2, x(b.start) - 1, barY(b, tops) + BAR_H / 2)}
                      marker-end="url(#tl-arrow)"
                    />
                  );
                })}
              </g>

              {model.bars.map((b) => {
                const x1 = x(b.start);
                const w = Math.max(1.5, x(b.end) - x1);
                const key = `b${b.id}`;
                const text = b.batch > 1 ? `${b.node} ×${b.batch}` : b.node;
                const fits = w > text.length * 6.4 + 10;
                const isOn = active?.type === 'bar' && active.bar.id === b.id;
                const dim = activeJobs.size > 0 && !isOn && !b.jobs.some((j) => activeJobs.has(j));
                const y = barY(b, tops);
                const delay = animate ? Math.round((b.start / Math.max(1, model.end)) * (animTotal - 120)) : 0;
                return (
                  <g
                    key={key}
                    class={`bar bar-${b.precision.toLowerCase()} ${isOn ? 'on' : ''} ${dim ? 'dim' : ''}`}
                    style={animate ? { animationDelay: `${delay}ms` } : undefined}
                    onPointerEnter={() => setHover(key)}
                  >
                    <rect x={x1} y={y} width={w} height={BAR_H} rx={2} />
                    {fits && (
                      <text x={x1 + 5} y={y + BAR_H - 4}>
                        {text}
                      </text>
                    )}
                    {b.cutShort && <line class="cut" x1={x1 + w} x2={x1 + w} y1={y - 3} y2={y + BAR_H + 3} />}
                  </g>
                );
              })}

              {(() => {
                if (model.markers.length > 8) return null;
                let lastEnd = -Infinity;
                return model.markers.map((m) => {
                  const text = markerLabel(m);
                  const lx = x(m.ts) + 10;
                  if (lx < lastEnd + 6 || !text) return null;
                  lastEnd = lx + text.length * 6.3;
                  return (
                    <text key={`ml${m.id}`} class={`mk-label ${m.kind === 'decision' || m.kind === 'skip' ? '' : 'bad'}`} x={lx} y={tops.router + ROUTER_H / 2 + 3.5}>
                      {text}
                    </text>
                  );
                });
              })()}
              {model.markers.map((m) => {
                const key = `m${m.id}`;
                const mx = x(m.ts);
                const my = tops.router + ROUTER_H / 2;
                const isOn = active?.type === 'marker' && active.marker.id === m.id;
                const dim = activeJobs.size > 0 && !isOn && !(m.job && activeJobs.has(m.job));
                return (
                  <g key={key} class={`marker ${isOn ? 'on' : ''} ${dim ? 'dim' : ''}`} onPointerEnter={() => setHover(key)}>
                    <circle class="hit" cx={mx} cy={my} r={10} />
                    <MarkerGlyph m={m} x={mx} y={my} />
                  </g>
                );
              })}
            </g>
          </svg>
        </div>

        {active && tip && (
          <div class={`tip ${tip.flip ? 'below' : 'above'}`} style={{ left: `${tip.left}px`, top: `${tip.top}px` }} role="presentation">
            <Tip item={active} />
          </div>
        )}
        {busy && (
          <div class="tl-busy" role="status">
            <span class="spinner" aria-hidden="true" />
            {busy}
          </div>
        )}
      </div>

      <p class="sr-only" aria-live="polite">{selected ? describe(selected) : ''}</p>

      <div class="tl-foot">
        <p class="fine">
          <kbd>Ctrl</kbd> + wheel zooms, drag pans, <kbd>←</kbd> <kbd>→</kbd> step through events. Open in Perfetto: copy the trace, save it as <code>trace.json</code>, drop it on <code>ui.perfetto.dev</code>.
        </p>
        <button type="button" class="btn btn-quiet" onClick={copyTrace}>
          {copied === 'ok' ? 'Trace copied' : copied === 'fail' ? 'Copy blocked by the browser' : 'Copy trace JSON'}
        </button>
      </div>

      <details class="tl-table">
        <summary>All {items.length} events in time order</summary>
        <div class="table-scroll">
          <table>
            <thead>
              <tr>
                <th scope="col">ms</th>
                <th scope="col">track</th>
                <th scope="col">what</th>
                <th scope="col">detail</th>
              </tr>
            </thead>
            <tbody>
              {items.map((it) =>
                it.type === 'job' ? (
                  <tr key={itemKey(it)}>
                    <td class="num">{ms(it.job.start / 1000)}</td>
                    <td>Jobs</td>
                    <td>{it.job.job}</td>
                    <td class="mono">
                      {JOB_STATUS[it.job.status]} after {ms((it.job.end - it.job.start) / 1000)} ms
                    </td>
                  </tr>
                ) : it.type === 'bar' ? (
                  <tr key={itemKey(it)}>
                    <td class="num">{ms(it.bar.start / 1000)}</td>
                    <td>{it.bar.lane}</td>
                    <td>
                      <i class={`sw sw-${it.bar.precision.toLowerCase()}`} aria-hidden="true" /> {it.bar.node} {it.bar.precision}
                      {it.bar.batch > 1 ? ` ×${it.bar.batch}` : ''}
                    </td>
                    <td class="mono">
                      {ms((it.bar.end - it.bar.start) / 1000)} ms{it.bar.jobs.length ? ` · ${it.bar.jobs.join(' ')}` : ''}
                      {it.bar.cutShort ? ' · cut short' : ''}
                    </td>
                  </tr>
                ) : (
                  <tr key={itemKey(it)}>
                    <td class="num">{ms(it.marker.ts / 1000)}</td>
                    <td>Router</td>
                    <td>{KIND_LABEL[it.marker.kind]}</td>
                    <td class="mono">{tidyReason(it.marker.message)}</td>
                  </tr>
                ),
              )}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  );
}
