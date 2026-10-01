import claims from '../claims.json';
import { codeSpans } from '../format';
import { Panel } from '../ui';

function Rich({ text }: { text: string }) {
  return (
    <>
      {codeSpans(text).map((p, i) => (typeof p === 'string' ? <span key={i}>{p}</span> : <code key={i}>{p.code}</code>))}
    </>
  );
}

export function Claims() {
  return (
    <Panel id="claims" kicker="Claims" title="Every claim, next to what checks it" class="p-claims">
      <p class="fine claims-intro">
        From <a href="https://github.com/armaanmittalweb/loomcore/blob/main/docs/CLAIMS.md">docs/CLAIMS.md</a>. Each links to the test case or program that asserts it.
      </p>
      <div class="table-scroll">
        <table class="tbl claims-tbl">
          <thead>
            <tr>
              <th scope="col" class="num">#</th>
              <th scope="col">claim</th>
              <th scope="col">checked by</th>
            </tr>
          </thead>
          <tbody>
            {claims.map((c) => (
              <tr key={c.n}>
                <td class="num">{c.n}</td>
                <td>
                  <Rich text={c.claim} />
                </td>
                <td class="claims-link">
                  <a href={c.url}>
                    <code>
                      {c.file.replace(/^tests\//, '')}
                      {c.line ? `:${c.line}` : ''}
                    </code>
                  </a>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}
