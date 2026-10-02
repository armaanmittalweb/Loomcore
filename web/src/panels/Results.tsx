import type { BatchStats, LoadResult, NodeStats, ReloadResult, RunResult } from '../types';
import { count, ms, pct, tidyReason } from '../format';
import { BadGlyph, OkGlyph, PrecisionChip, Stat } from '../ui';

const STATUS_TEXT: Record<Exclude<RunResult['status'], 'ok'>, string> = {
  rejected: 'Rejected by admission control before anything ran',
  cancelled: 'Cancelled at the deadline',
  shed: 'Shed: mobilenet was skipped, so the job could not finish',
  failed: 'The job failed',
};

export function RunView({ r, image, caption }: { r: RunResult; image: string | null; caption: string }) {
  const mobilenet = r.nodes.find((n) => n.id === 'mobilenet');
  return (
    <div class="rv">
      {r.status !== 'ok' && (
        <div class="banner banner-bad" role="note">
          <BadGlyph />
          <div>
            <strong>{STATUS_TEXT[r.status]}.</strong>
            <span class="mono">{r.error ? tidyReason(r.error.replace(/^job rejected by admission control: /, '')) : ''}</span>
          </div>
        </div>
      )}

      <div class="rv-top">
        <figure class="rv-img">
          {image ? <img src={image} alt={caption} width={112} height={112} /> : <div class="rv-noimg" />}
          <figcaption>{caption}</figcaption>
        </figure>
        <div class="rv-label">
          <p class="kicker">mobilenet says</p>
          {r.label ? (
            <>
              <p class="rv-name">{r.label}</p>
              <p class="rv-conf">
                <span class="mono">{pct(r.confidence)}</span> confidence
              </p>
              <ol class="top5" aria-label="Top five classes">
                {r.top5.map((t) => (
                  <li key={t.label}>
                    <span class="top5-l">{t.label}</span>
                    <span class="top5-bar" aria-hidden="true">
                      <i style={{ width: `${Math.max(1.5, t.p * 100)}%` }} />
                    </span>
                    <span class="top5-p mono">{pct(t.p)}</span>
                  </li>
                ))}
              </ol>
            </>
          ) : (
            <p class="rv-none">No label: mobilenet never ran for this job.</p>
          )}
        </div>
        <div class="rv-nodes">
          <table class="tbl">
            <caption class="kicker">Per node</caption>
            <thead>
              <tr>
                <th scope="col">node</th>
                <th scope="col">lane</th>
                <th scope="col">precision</th>
                <th scope="col" class="num">ms</th>
              </tr>
            </thead>
            <tbody>
              {['mobilenet', 'bert_tiny'].map((id) => {
                const n = r.nodes.find((x) => x.id === id);
                return (
                  <tr key={id}>
                    <th scope="row">
                      <code>{id}</code>
                    </th>
                    {n ? (
                      <>
                        <td>{n.backend}</td>
                        <td>
                          <PrecisionChip p={n.precision} />
                        </td>
                        <td class="num">{ms(n.ms)}</td>
                      </>
                    ) : (
                      <td colSpan={3} class="muted">
                        {r.skipped.includes(id) ? 'skipped by the router' : r.status === 'ok' ? 'not run' : 'did not run'}
                      </td>
                    )}
                  </tr>
                );
              })}
            </tbody>
            <tfoot>
              <tr>
                <th scope="row">job</th>
                <td colSpan={2} class="muted">
                  {r.timeBudgetMs != null ? `budget ${ms(r.timeBudgetMs)} ms` : 'no deadline'}
                </td>
                <td class="num">{ms(r.totalMs)}</td>
              </tr>
            </tfoot>
          </table>
          {r.graphSwapMs != null && <p class="fine">New router chain: the graph was hot-swapped first ({ms(r.graphSwapMs)} ms).</p>}
        </div>
      </div>

      {mobilenet?.precision === 'INT8' && (
        <p class="note">
          mobilenet ran INT8 here. Its INT8 model was calibrated on random tensors, because this project measures latency rather than accuracy, so an INT8 label is not to be trusted. The router traded accuracy for time, which is the point of the policy.
        </p>
      )}

      <div class="rv-decisions">
        <h3 class="kicker">Router decisions</h3>
        {r.decisions.length ? (
          <ul class="decisions">
            {r.decisions.map((d, i) => (
              <li key={i}>
                <code class="d-node">{d.node}</code>
                {d.policies.map((p) => (
                  <span key={p.policy} class="d-op">
                    <code class="d-pol">{p.policy}</code>
                    <span class="d-why">{tidyReason(p.reason)}</span>
                  </span>
                ))}
              </li>
            ))}
          </ul>
        ) : (
          <p class="fine">None: every policy deferred (no deadline pressure, equal queues, confidence under the gate), so each node ran FP32 on its configured lane.</p>
        )}
      </div>

      {r.embedding && (
        <div class="emb">
          <h3 class="kicker">bert_tiny embedding</h3>
          <p class="fine">
            Embedded <q>{r.text}</q>. The first 16 of 128 values:
          </p>
          <ol class="emb-grid" aria-label="First 16 embedding values">
            {r.embedding.map((v, i) => (
              <li key={i}>
                <span class="emb-v mono">{v.toFixed(3)}</span>
                <span class="emb-axis" aria-hidden="true">
                  <i class={v >= 0 ? 'pos' : 'neg'} style={{ width: `${Math.min(50, Math.abs(v) * 50)}%` }} />
                </span>
              </li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
}

function PrecisionMix({ n }: { n: NodeStats }) {
  const fp = n.precisions.FP32 ?? 0;
  const i8 = n.precisions.INT8 ?? 0;
  const total = fp + i8 || 1;
  return (
    <div class="mix" role="img" aria-label={`${fp} FP32, ${i8} INT8`}>
      <span class="mix-bar">
        {fp > 0 && <i class="fp32" style={{ width: `${(fp / total) * 100}%` }} />}
        {i8 > 0 && <i class="int8" style={{ width: `${(i8 / total) * 100}%` }} />}
      </span>
      <span class="mix-n mono">
        {fp} / {i8}
      </span>
    </div>
  );
}

function Batches({ b }: { b: BatchStats }) {
  const sizes = [1, 2, 3, 4];
  const max = Math.max(1, ...sizes.map((s) => b.sizes[String(s)] ?? 0));
  return (
    <div class="hist">
      <p class="hist-t">
        <code>{b.node}</code> <span class="muted">{b.count} calls, mean {b.meanSize.toFixed(2)} jobs each</span>
      </p>
      <div class="hist-bars" role="img" aria-label={`${b.node} batch sizes: ${sizes.map((s) => `${b.sizes[String(s)] ?? 0} of size ${s}`).join(', ')}`}>
        {sizes.map((s) => {
          const v = b.sizes[String(s)] ?? 0;
          return (
            <div key={s} class="hist-col">
              <span class="hist-v mono">{v}</span>
              <span class="hist-track">
                <i style={{ height: `${(v / max) * 100}%` }} />
              </span>
              <span class="hist-x mono">{s}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function LoadView({ r }: { r: LoadResult }) {
  return (
    <div class="lv">
      <dl class="stats">
        <Stat label="Jobs" value={count(r.submitted)} note={`${r.concurrency} at a time`} />
        <Stat label="Completed" value={count(r.completed)} />
        <Stat label="Rejected" value={count(r.rejected)} bad={r.rejected > 0} note="by admission control" />
        <Stat label="Cancelled" value={count(r.cancelled)} bad={r.cancelled > 0} note="at the deadline" />
        {(r.shed > 0 || r.failed > 0) && <Stat label="Shed" value={count(r.shed + r.failed)} bad note="mobilenet skipped" />}
        <Stat label="Throughput" value={r.throughput.toFixed(1)} unit="jobs/s" note={`${ms(r.wallMs)} ms wall`} />
        <Stat label="Job p50 / p95" value={`${ms(r.jobP50)} / ${ms(r.jobP95)}`} unit="ms" note={r.timeBudgetMs != null ? `budget ${ms(r.timeBudgetMs)} ms` : 'no deadline'} />
      </dl>

      <div class="lv-grid">
        <table class="tbl">
          <caption class="kicker">Per node, every execution</caption>
          <thead>
            <tr>
              <th scope="col">node</th>
              <th scope="col" class="num">runs</th>
              <th scope="col" class="num">p50 ms</th>
              <th scope="col" class="num">p95 ms</th>
              <th scope="col" class="mix-col">FP32 / INT8</th>
              <th scope="col" class="lanes-col">CPU / GPU_SIM</th>
            </tr>
          </thead>
          <tbody>
            {r.nodes.map((n) => (
              <tr key={n.id}>
                <th scope="row">
                  <code>{n.id}</code>
                </th>
                <td class="num">{n.count}</td>
                <td class="num">{ms(n.p50)}</td>
                <td class="num">{ms(n.p95)}</td>
                <td>
                  <PrecisionMix n={n} />
                </td>
                <td class="mono lanes-col">
                  {n.lanes.CPU ?? 0} / {n.lanes.GPU_SIM ?? 0}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p class="fine">
          Latency is per execution and includes waiting for a batch window and queueing behind other batches; a batched call's time counts once for each job in it.
          {Object.keys(r.skipped).length > 0 && ` Skipped by the router: ${Object.entries(r.skipped).map(([k, v]) => `${k} ${v}×`).join(', ')}.`}
        </p>
      </div>

      <div class="lv-row">
        <div>
          <h3 class="kicker">Batches formed (jobs per ONNX Runtime call)</h3>
          <div class="hists">
            {r.batches.map((b) => (
              <Batches key={b.node} b={b} />
            ))}
          </div>
        </div>
        <div>
          <h3 class="kicker">Router decisions by policy</h3>
          {r.decisions.length ? (
            <ul class="pcounts">
              {r.decisions.map((d) => (
                <li key={d.policy}>
                  <code>{d.policy}</code>
                  <span class="mono">{d.count}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p class="fine">No policy had an opinion during this test.</p>
          )}
        </div>
      </div>
    </div>
  );
}

export function SwapView({ r }: { r: ReloadResult }) {
  const lostBad = r.lost > 0;
  return (
    <div class="sv">
      <dl class="swap-count">
        <div>
          <dt>Submitted</dt>
          <dd>{count(r.submitted)}</dd>
        </div>
        <div>
          <dt>Completed</dt>
          <dd>{count(r.completed)}</dd>
        </div>
        <div class={lostBad ? 'is-bad' : 'is-ok'}>
          <dt>Lost</dt>
          <dd>
            {lostBad ? <BadGlyph /> : <OkGlyph />}
            {count(r.lost)}
          </dd>
        </div>
      </dl>
      <p class="sv-line">
        {lostBad
          ? `${r.lost} of ${r.submitted} jobs did not complete across ${r.swaps} swaps.`
          : `Every job completed across ${r.swaps} whole-graph ${r.swaps === 1 ? 'swap' : 'swaps'} in ${ms(r.wallMs)} ms. Jobs in flight at each swap finished on the graph they started on; the next submissions ran on the new one.`}
      </p>
      <table class="tbl">
        <caption class="kicker">Swaps, as marked on the timeline</caption>
        <thead>
          <tr>
            <th scope="col">swap</th>
            <th scope="col" class="num">at ms</th>
            <th scope="col" class="num">build + swap ms</th>
            <th scope="col" class="num">jobs in flight</th>
            <th scope="col" class="num">completed before</th>
          </tr>
        </thead>
        <tbody>
          {r.swapTimings.map((s) => (
            <tr key={s.index}>
              <th scope="row">{s.index}</th>
              <td class="num">{ms(s.atMs)}</td>
              <td class="num">{ms(s.buildMs)}</td>
              <td class="num">{s.inFlight}</td>
              <td class="num">{s.completedBefore}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p class="fine">Build time is parsing the config and loading four ONNX Runtime sessions; traffic is not paused while it happens. The swap itself is one atomic pointer store (<code>Runtime::reloadGraph</code>, RCU-style).</p>
    </div>
  );
}
