// `npm run claims`: reads docs/CLAIMS.md and writes src/claims.json, linking
// each claim to the exact test (file and line of its TEST_CASE) on GitHub, or
// to the file that asserts it when the check is a program rather than a test
// case. Re-run whenever CLAIMS.md or the tests move.
import { readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const REPO = 'https://github.com/armaanmittalweb/loomcore/blob/main/';
const md = readFileSync(ROOT + 'docs/CLAIMS.md', 'utf8');

const testFiles = readdirSync(ROOT + 'tests').filter((f) => f.endsWith('.cpp'));
function findTest(name) {
  for (const f of testFiles) {
    const lines = readFileSync(ROOT + 'tests/' + f, 'utf8').split('\n');
    const i = lines.findIndex((l) => l.includes(`TEST_CASE("${name}"`));
    if (i >= 0) return { file: 'tests/' + f, line: i + 1 };
  }
  return null;
}

const plain = (s) => s.replace(/\*\*/g, '').trim();
const claims = [];
for (const row of md.split('\n')) {
  const m = row.match(/^\|\s*(\d+)\s*\|(.+?)\|(.+?)\|(.+)\|\s*$/);
  if (!m) continue;
  const [, n, claim, command, where] = m;
  const testName = command.match(/--test-case="([^"]+)"/)?.[1];
  let target = testName ? findTest(testName) : null;
  if (!target) {
    const path = where.match(/`(tests\/[^`\s]+\.cpp)`/)?.[1] ?? where.match(/`((?:tests|examples|benchmarks|bindings|src|tools|\.github)\/[^`\s]+)`/)?.[1] ?? command.match(/`((?:tests|examples|benchmarks|bindings|src|tools)\/[^`\s]+)`/)?.[1];
    target = path ? { file: path.replace(/\/$/, ''), line: null } : { file: 'tests/', line: null };
  }
  const commandText = command.match(/`([^`]+)`/)?.[1] ?? plain(command);
  claims.push({
    n: Number(n),
    claim: plain(claim),
    command: commandText,
    test: testName ?? null,
    file: target.file,
    line: target.line,
    url: REPO + target.file + (target.line ? `#L${target.line}` : ''),
  });
}
if (claims.length < 10) throw new Error(`only parsed ${claims.length} claims from docs/CLAIMS.md`);
writeFileSync(fileURLToPath(new URL('../src/claims.json', import.meta.url)), JSON.stringify(claims, null, 2) + '\n');
console.log(`${claims.length} claims, ${claims.filter((c) => c.line).length} linked to a TEST_CASE line`);
