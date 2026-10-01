import { useRef } from 'preact/hooks';
import type { Phase } from '../wake';
import { Panel } from '../ui';

export type Mode = 'run' | 'load' | 'swap';

export interface Settings {
  sample: string;
  upload: { file: File; url: string } | null;
  budgetOn: boolean;
  budget: number;
  loadJobs: number;
  concurrency: number;
  loadBudgetOn: boolean;
  loadBudget: number;
  swapJobs: number;
  swaps: number;
}

interface Props {
  mode: Mode;
  onMode: (m: Mode) => void;
  settings: Settings;
  onChange: (patch: Partial<Settings>) => void;
  samples: { id: string; caption: string }[];
  phase: Phase;
  pending: Mode | null;
  error: string | null;
  uploadError: string | null;
  onUploadError: (msg: string | null) => void;
  onGo: () => void;
}

const TABS: { id: Mode; label: string }[] = [
  { id: 'run', label: 'Run' },
  { id: 'load', label: 'Load test' },
  { id: 'swap', label: 'Hot-swap' },
];

const MAX_UPLOAD = 2 * 1024 * 1024;

function Range(props: { id: string; label: string; min: number; max: number; step?: number; value: number; unit?: string; disabled?: boolean; onInput: (v: number) => void; hint?: string }) {
  return (
    <div class={`field ${props.disabled ? 'is-off' : ''}`}>
      <div class="field-top">
        <label for={props.id}>{props.label}</label>
        <output for={props.id} class="field-val">
          {props.disabled ? 'none' : `${props.value}${props.unit ? ` ${props.unit}` : ''}`}
        </output>
      </div>
      <input
        id={props.id}
        type="range"
        min={props.min}
        max={props.max}
        step={props.step ?? 1}
        value={props.value}
        disabled={props.disabled}
        onInput={(e) => props.onInput(Number((e.currentTarget as HTMLInputElement).value))}
        aria-describedby={props.hint ? `${props.id}-hint` : undefined}
      />
      {props.hint && (
        <p class="field-hint" id={`${props.id}-hint`}>
          {props.hint}
        </p>
      )}
    </div>
  );
}

function Budget(props: { id: string; on: boolean; value: number; onToggle: (on: boolean) => void; onInput: (v: number) => void; hint: string }) {
  return (
    <div class="budget">
      <div class="budget-switch">
        <span id={`${props.id}-lbl`}>Deadline</span>
        <button type="button" role="switch" aria-checked={props.on} aria-labelledby={`${props.id}-lbl`} class="switch" onClick={() => props.onToggle(!props.on)}>
          <i aria-hidden="true" />
        </button>
      </div>
      <Range id={props.id} label="Budget per job" min={5} max={200} value={props.value} unit="ms" disabled={!props.on} onInput={props.onInput} hint={props.hint} />
    </div>
  );
}

function lockReason(phase: Phase): string | null {
  switch (phase) {
    case 'live':
      return null;
    case 'checking':
      return 'Checking the live runtime…';
    case 'waking':
      return 'The live runtime is waking up (about a minute). Controls unlock when it is up; the recorded run stays on screen meanwhile.';
    case 'loading':
      return 'The live runtime is loading its models. A few seconds more.';
    case 'asleep':
      return 'The live runtime did not answer. The recorded runs above are real; retry from the status bar.';
  }
}

export function Controls(p: Props) {
  const s = p.settings;
  const fileRef = useRef<HTMLInputElement>(null);
  const locked = lockReason(p.phase);
  const busy = p.pending != null;

  function pickFile(e: Event) {
    const input = e.currentTarget as HTMLInputElement;
    const file = input.files?.[0];
    input.value = '';
    if (!file) return;
    if (!/^image\/(jpeg|png)$/.test(file.type)) return p.onUploadError('Choose a JPEG or PNG image.');
    if (file.size > MAX_UPLOAD) return p.onUploadError(`That image is ${(file.size / 1048576).toFixed(1)} MB; the limit is 2 MB.`);
    p.onUploadError(null);
    if (s.upload) URL.revokeObjectURL(s.upload.url);
    p.onChange({ upload: { file, url: URL.createObjectURL(file) } });
  }

  const goLabel = p.mode === 'run' ? 'Run one job' : p.mode === 'load' ? `Run ${s.loadJobs} jobs` : `Swap ${s.swaps}× under load`;
  const pendingLabel = p.mode === 'run' ? 'Running…' : p.mode === 'load' ? 'Load test running…' : 'Swapping…';

  return (
    <Panel id="controls" kicker="Controls" title="Drive the runtime" class="p-controls">
      <div class="tabs" role="tablist" aria-label="What to run">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            role="tab"
            id={`tab-${t.id}`}
            aria-selected={p.mode === t.id}
            aria-controls={`tabpanel-${t.id}`}
            tabIndex={p.mode === t.id ? 0 : -1}
            class="tab"
            onClick={() => p.onMode(t.id)}
            onKeyDown={(e) => {
              if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
              const i = TABS.findIndex((x) => x.id === p.mode);
              const next = TABS[(i + (e.key === 'ArrowRight' ? 1 : TABS.length - 1)) % TABS.length];
              p.onMode(next.id);
              document.getElementById(`tab-${next.id}`)?.focus();
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div class="tabpanel" role="tabpanel" id={`tabpanel-${p.mode}`} aria-labelledby={`tab-${p.mode}`}>
        {p.mode === 'run' && (
          <>
            <fieldset class="samples">
              <legend>Image</legend>
              <div class="sample-grid">
                {p.samples.map((sm) => (
                  <label key={sm.id} class={`sample ${!s.upload && s.sample === sm.id ? 'on' : ''}`}>
                    <input type="radio" name="sample" value={sm.id} checked={!s.upload && s.sample === sm.id} onChange={() => p.onChange({ sample: sm.id, upload: null })} />
                    <img src={`/samples/${sm.id}-128.jpg`} alt="" width={64} height={64} loading="lazy" decoding="async" />
                    <span>{sm.caption}</span>
                  </label>
                ))}
                <label class={`sample sample-upload ${s.upload ? 'on' : ''}`}>
                  <input ref={fileRef} type="file" accept="image/jpeg,image/png" onChange={pickFile} />
                  {s.upload ? <img src={s.upload.url} alt="" width={64} height={64} /> : <span class="upload-glyph" aria-hidden="true">+</span>}
                  <span>{s.upload ? 'Your image' : 'Upload'}</span>
                </label>
              </div>
              <p class="field-hint">{s.upload ? `${s.upload.file.name}, ${(s.upload.file.size / 1024).toFixed(0)} KB. Sent once, decoded in memory, not stored.` : 'JPEG or PNG up to 2 MB, or one of six Commons photos.'}</p>
              {p.uploadError && (
                <p class="err" role="alert">
                  {p.uploadError}
                </p>
              )}
            </fieldset>
            <Budget
              id="budget"
              on={s.budgetOn}
              value={s.budget}
              onToggle={(on) => p.onChange({ budgetOn: on })}
              onInput={(v) => p.onChange({ budget: v })}
              hint="Binding: admission control rejects a job that cannot fit, and the reaper cancels one that runs out of time."
            />
          </>
        )}

        {p.mode === 'load' && (
          <>
            <Range id="jobs" label="Jobs" min={1} max={64} value={s.loadJobs} onInput={(v) => p.onChange({ loadJobs: v })} />
            <Range id="conc" label="Submitted concurrently" min={1} max={16} value={s.concurrency} onInput={(v) => p.onChange({ concurrency: v })} hint="Same-node requests that arrive together are batched into one ONNX Runtime call." />
            <Budget
              id="lbudget"
              on={s.loadBudgetOn}
              value={s.loadBudget}
              onToggle={(on) => p.onChange({ loadBudgetOn: on })}
              onInput={(v) => p.onChange({ loadBudget: v })}
              hint="Queueing counts against it, so a tight budget under load produces INT8 downgrades, rejects and cancels."
            />
            <p class="field-hint">Every job uses the image picked under Run ({s.upload ? 'uploads apply to single runs only, so the first sample' : p.samples.find((x) => x.id === s.sample)?.caption.toLowerCase()}).</p>
          </>
        )}

        {p.mode === 'swap' && (
          <>
            <Range id="sjobs" label="Jobs submitted during the swaps" min={8} max={64} value={s.swapJobs} onInput={(v) => p.onChange({ swapJobs: v })} />
            <Range id="swaps" label="Graph swaps" min={1} max={6} value={s.swaps} onInput={(v) => p.onChange({ swaps: v })} />
            <p class="field-hint">
              Four producer threads keep submitting while the runtime builds a whole new graph (four ONNX sessions) and swaps it in, {s.swaps} {s.swaps === 1 ? 'time' : 'times'}. Jobs in flight finish on the graph they started on. Lost jobs must be zero.
            </p>
          </>
        )}
      </div>

      <div class="go">
        <button type="button" class="btn btn-primary" onClick={p.onGo} disabled={!!locked || busy} aria-describedby="go-why">
          {busy && p.pending === p.mode ? (
            <>
              <span class="spinner" aria-hidden="true" />
              {pendingLabel}
            </>
          ) : (
            goLabel
          )}
        </button>
        <p id="go-why" class="go-why" role="status">
          {locked ?? (busy && p.pending !== p.mode ? 'Another run is in progress.' : 'On the live runtime.')}
        </p>
      </div>
      {p.error && (
        <p class="err" role="alert">
          {p.error}
        </p>
      )}
    </Panel>
  );
}
