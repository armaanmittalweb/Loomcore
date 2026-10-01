import type { ComponentChildren } from 'preact';
import { day, shortCpu } from './format';
import type { RecordingMeta } from './types';

export type Origin = { kind: 'recorded'; meta: RecordingMeta } | { kind: 'live'; at: number; cpu?: string };

/** Every panel says where its numbers came from. */
export function SourceTag({ origin }: { origin: Origin }) {
  if (origin.kind === 'live') {
    return (
      <span class="src src-live" title={origin.cpu ? `Measured on the live runtime (${origin.cpu})` : 'Measured on the live runtime'}>
        <i aria-hidden="true" />
        Live
      </span>
    );
  }
  const m = origin.meta;
  const where = m.source === 'space' ? 'on the live runtime' : `on a local build, ${shortCpu(m.machine)}${m.os ? `, ${m.os.replace(/, native build$/, '')}` : ''}`;
  return (
    <span class="src src-rec" title={`Recorded ${day(m.recordedAt)} ${where}`}>
      <i aria-hidden="true" />
      Recorded {day(m.recordedAt)}
    </span>
  );
}

export function recordedLine(meta: RecordingMeta): string {
  return meta.source === 'space'
    ? `Recorded on ${day(meta.recordedAt)} on the live runtime.`
    : `Recorded on ${day(meta.recordedAt)} on a local build of the same runtime (${shortCpu(meta.machine)}${meta.os ? `, ${meta.os.replace(/, native build$/, '')}` : ''}).`;
}

export function Panel(props: { id: string; kicker: string; title: ComponentChildren; tag?: ComponentChildren; actions?: ComponentChildren; class?: string; children: ComponentChildren }) {
  return (
    <section class={`panel ${props.class ?? ''}`} aria-labelledby={`${props.id}-title`} id={props.id}>
      <header class="panel-head">
        <div class="panel-titles">
          <p class="kicker">{props.kicker}</p>
          <h2 class="panel-title" id={`${props.id}-title`}>
            {props.title}
          </h2>
        </div>
        {(props.tag || props.actions) && (
          <div class="panel-aside">
            {props.actions}
            {props.tag}
          </div>
        )}
      </header>
      {props.children}
    </section>
  );
}

/** A status glyph that never relies on colour alone. */
export function BadGlyph() {
  return (
    <svg class="bad-glyph" viewBox="-8 -8 16 16" aria-hidden="true">
      <circle r="6.5" />
      <path d="M-3 -3L3 3M3 -3L-3 3" />
    </svg>
  );
}

export function OkGlyph() {
  return (
    <svg class="ok-glyph" viewBox="-8 -8 16 16" aria-hidden="true">
      <circle r="6.5" />
      <path d="M-3.2 0.2L-0.8 2.6L3.4 -2.4" />
    </svg>
  );
}

export function PrecisionChip({ p, count, used = true }: { p: 'FP32' | 'INT8'; count?: number; used?: boolean }) {
  return (
    <span class={`pchip pchip-${p.toLowerCase()} ${used ? 'used' : ''}`}>
      <i aria-hidden="true" />
      {p}
      {count != null && <b>{count}</b>}
    </span>
  );
}

export function Stat(props: { label: string; value: ComponentChildren; unit?: string; bad?: boolean; note?: string }) {
  return (
    <div class={`stat ${props.bad ? 'is-bad' : ''}`}>
      <dt>{props.label}</dt>
      <dd>
        <span class="stat-line">
          {props.bad && <BadGlyph />}
          <span class="stat-v">{props.value}</span>
          {props.unit && <span class="stat-u">{props.unit}</span>}
        </span>
        {props.note && <span class="stat-note">{props.note}</span>}
      </dd>
    </div>
  );
}
