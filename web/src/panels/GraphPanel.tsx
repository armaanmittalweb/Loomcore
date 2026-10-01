import type { GraphInfo, GraphNode, PolicyName, Precision } from '../types';
import { Panel, PrecisionChip, type Origin, SourceTag } from '../ui';

/** What the shown result did at each node: precisions used (with counts for
 * a load test), lanes, and whether the node was skipped. */
export interface NodeActivity {
  precisions: Partial<Record<Precision, number>>;
  lanes: string[];
  skipped: number;
  ran: number;
}

interface Props {
  graph: GraphInfo;
  origin: Origin;
  activity: Record<string, NodeActivity>;
  showCounts: boolean;
  policies: PolicyName[];
  onToggle: (name: PolicyName) => void;
  gateThreshold: number;
}

const MODEL: Record<string, string> = { mobilenet: 'MobileNetV2 · image classifier', bert_tiny: 'BERT-tiny · text encoder' };

function NodeBox({ node, act, showCounts }: { node: GraphNode; act?: NodeActivity; showCounts: boolean }) {
  const skippedOnly = act && act.ran === 0 && act.skipped > 0;
  return (
    <div class={`dag-node ${skippedOnly ? 'is-skipped' : ''} ${act && act.ran > 0 ? 'is-ran' : ''}`}>
      <div class="dag-node-top">
        <code class="dag-id">{node.id}</code>
        <span class="dag-lane">{node.backend}</span>
      </div>
      <p class="dag-sub">{MODEL[node.id] ?? 'model'}</p>
      <div class="dag-variants">
        {node.variants.map((p) => (
          <PrecisionChip key={p} p={p} used={!!act?.precisions[p]} count={showCounts && act?.precisions[p] ? act.precisions[p] : undefined} />
        ))}
        <span class="dag-batch">batch ≤ {node.maxBatchSize} · {node.batchWindowMs} ms window</span>
      </div>
      {act && act.skipped > 0 && (
        <p class="dag-skip">
          {showCounts ? `skipped ${act.skipped}×` : 'skipped on this run'}
        </p>
      )}
    </div>
  );
}

export function GraphPanel({ graph, origin, activity, showCounts, policies, onToggle, gateThreshold }: Props) {
  const byId = new Map(graph.nodes.map((n) => [n.id, n]));
  const first = byId.get('mobilenet') ?? graph.nodes[0];
  const second = byId.get('bert_tiny') ?? graph.nodes[1];
  const gateOn = policies.includes('confidence-gate');
  return (
    <Panel id="graph" kicker="Graph" title={`${first.id} → ${second.id}`} tag={<SourceTag origin={origin} />} class="p-graph">
      <div class="dag" role="img" aria-label={`The graph: an image goes to ${first.id} on the ${first.backend} lane; if the confidence gate is open, ${second.id} on the ${second.backend} lane embeds a description of ${first.id}'s label.`}>
        <div class="dag-io">image · 224×224 RGB</div>
        <span class="dag-arrow" aria-hidden="true" />
        <NodeBox node={first} act={activity[first.id]} showCounts={showCounts} />
        <span class="dag-arrow" aria-hidden="true" />
        <div class={`dag-gate ${gateOn ? '' : 'is-off'}`}>
          <span class="dag-diamond" aria-hidden="true" />
          <span>
            {gateOn ? (
              <>
                confidence ≥ {gateThreshold.toFixed(2)}? <b>skip</b> bert_tiny
              </>
            ) : (
              'confidence gate off: bert_tiny always runs'
            )}
          </span>
        </div>
        <span class="dag-arrow" aria-hidden="true" />
        <NodeBox node={second} act={activity[second.id]} showCounts={showCounts} />
        <span class="dag-arrow" aria-hidden="true" />
        <div class="dag-io">"a photo of a {'{label}'}" → 128-d embedding</div>
      </div>

      <div class="chain">
        <div class="chain-head">
          <h3>Router chain</h3>
          <p class="fine">In order. A skip ends the chain; otherwise the first policy to set precision or lane wins. Changes apply to the next run, which hot-swaps the graph to the new router first.</p>
        </div>
        <ol class="chain-list">
          {graph.policies.map((p, i) => {
            const on = policies.includes(p.name);
            return (
              <li key={p.name} class={on ? 'on' : 'off'}>
                <span class="chain-n">{i + 1}</span>
                <div class="chain-body">
                  <div class="chain-row">
                    <code class="chain-name">{p.name}</code>
                    <button type="button" role="switch" aria-checked={on} class="switch" onClick={() => onToggle(p.name)} aria-label={`${p.name} policy`}>
                      <i aria-hidden="true" />
                    </button>
                  </div>
                  <p class="chain-cpp">{p.cpp}</p>
                  <p class="chain-what">
                    Reads {p.reads}; {p.decides.charAt(0).toLowerCase() + p.decides.slice(1)}.
                  </p>
                </div>
              </li>
            );
          })}
        </ol>
        <p class="flags">
          <span>Scheduler</span>
          {graph.scheduler.admissionControl && <code>admission control</code>}
          {graph.scheduler.deadlineCancellation && <code>deadline cancellation</code>}
          {graph.scheduler.precisionPlanning && <code>precision planning</code>}
          {graph.scheduler.edfScoring && <code>EDF lanes</code>}
        </p>
      </div>
    </Panel>
  );
}
