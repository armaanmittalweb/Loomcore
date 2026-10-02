// `npm run shots`: builds the console (mode "shots", which only adds two
// wake-up timing knobs), serves it with the production headers from
// vercel.json (CSP included), and photographs every state at 1440x900 and
// 390x844 in light and dark, running axe on each. The Space is mocked at the
// network layer with real responses recorded from the runtime
// (src/recorded/*.json and scripts/fixtures/*.json).
// Output: web/shots/*.png and web/shots/report.json (gitignored).
// ONLY=name,name limits scenarios; VARIANTS=desktop-light,... limits variants;
// NOBUILD=1 reuses dist/.
import { spawn, spawnSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import AxeBuilder from '@axe-core/playwright';

const WEB = fileURLToPath(new URL('..', import.meta.url));
const OUT = fileURLToPath(new URL('../shots/', import.meta.url));
const PORT = 5178;
const BASE = `http://localhost:${PORT}`;
const SPACE = 'https://loomcore-api.amittal.dev';
mkdirSync(OUT, { recursive: true });
const win = process.platform === 'win32';
const npx = win ? 'npx.cmd' : 'npx';
const node = process.execPath;

if (!process.env.NOBUILD) {
  for (const args of [['vite', 'build', '--mode', 'shots'], ['vite', 'build', '--mode', 'shots', '--ssr', 'src/entry-server.tsx', '--outDir', 'dist-ssr']]) {
    const b = spawnSync(npx, args, { cwd: WEB, stdio: 'inherit', shell: win });
    if (b.status !== 0) process.exit(1);
  }
  const p = spawnSync(node, ['scripts/prerender.mjs'], { cwd: WEB, stdio: 'inherit' });
  if (p.status !== 0) process.exit(1);
}
const server = spawn(npx, ['vite', 'preview', '--port', String(PORT), '--strictPort'], { cwd: WEB, stdio: 'ignore', shell: win });
const cleanup = () => {
  try {
    if (win) spawnSync('taskkill', ['/pid', String(server.pid), '/T', '/F']);
    else server.kill();
  } catch {
    /* already gone */
  }
};
process.on('exit', cleanup);
for (let i = 0; i < 80; i++) {
  try {
    if ((await fetch(BASE)).ok) break;
  } catch {
    /* not up yet */
  }
  await new Promise((r) => setTimeout(r, 250));
}

const json = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url), 'utf8'));
const R = {
  health: json('../src/recorded/health.json'),
  graph: json('../src/recorded/graph.json'),
  bench: json('../src/recorded/bench.json'),
  run: json('../src/recorded/run.json'),
  load: json('../src/recorded/load.json'),
  reload: json('../src/recorded/reload.json'),
  skip: json('./fixtures/run-skip.json'),
  cancelled: json('./fixtures/run-cancelled.json'),
  rejected: json('./fixtures/run-rejected.json'),
  shed: json('./fixtures/load-shed.json'),
};
// "Live" in these shots serves the recorded responses as they are, machine included.
const live = R.health;

/** Mocks the Space. `routes` maps "METHOD /path" to a body, {status, body, delay}, 'hang' or 'down'. */
async function mock(page, routes) {
  await page.route(`${SPACE}/**`, async (route) => {
    const req = route.request();
    const path = new URL(req.url()).pathname;
    const key = `${req.method()} ${path}`;
    if (req.method() === 'OPTIONS') return route.fulfill({ status: 204, headers: cors() });
    let spec = routes[key];
    if (spec === undefined) spec = 'down';
    if (spec === 'hang') return; // never answer
    if (spec === 'down') return route.abort('connectionrefused');
    const { status = 200, body = spec, delay = 0 } = spec && spec.__spec ? spec : { body: spec };
    if (delay) await new Promise((r) => setTimeout(r, delay));
    try {
      await route.fulfill({ status, headers: { ...cors(), 'content-type': 'application/json' }, body: JSON.stringify(body) });
    } catch {
      /* page closed */
    }
  });
}
const cors = () => ({ 'access-control-allow-origin': BASE, 'access-control-allow-headers': 'content-type', 'access-control-allow-methods': 'GET, POST' });
const spec = (o) => ({ __spec: true, ...o });
const liveRoutes = (extra = {}) => ({ 'GET /health': live, 'GET /graph': R.graph, 'GET /bench': R.bench, ...extra });
const waitLive = (p) => p.waitForSelector('.status.st-live');
const wait = (p, ms = 350) => p.waitForTimeout(ms);
const clickGo = (p) => p.click('.go .btn');

const scenarios = [
  ['01-recorded', async (p, shot) => {
    await mock(p, { 'GET /health': 'hang' });
    await p.goto(BASE);
    await p.waitForSelector('.p-timeline .bar');
    await wait(p, 600);
    await shot();
    await shot({ suffix: 'full', full: true });
  }],
  ['02-waking', async (p, shot) => {
    await mock(p, {});
    await p.goto(BASE);
    await p.waitForSelector('.status.st-waking');
    await wait(p, 1200);
    await shot();
  }],
  ['03-loading', async (p, shot) => {
    await mock(p, { 'GET /health': { ...live, ready: false, bench: 'pending' } });
    await p.goto(BASE);
    await p.waitForSelector('.status.st-loading');
    await wait(p);
    await shot();
  }],
  ['04-live', async (p, shot) => {
    await mock(p, liveRoutes());
    await p.goto(BASE);
    await waitLive(p);
    await wait(p, 500);
    await shot();
    await shot({ suffix: 'full', full: true });
  }],
  ['05-running', async (p, shot) => {
    await mock(p, liveRoutes({ 'POST /run': spec({ body: R.skip, delay: 8000 }) }));
    await p.goto(BASE);
    await waitLive(p);
    await p.click('label.sample:has-text("Lighthouse")');
    await clickGo(p);
    await p.waitForSelector('.tl-busy');
    await wait(p, 400);
    await shot();
  }],
  ['06-run-live', async (p, shot) => {
    await mock(p, liveRoutes({ 'POST /run': R.skip }));
    await p.goto(BASE);
    await waitLive(p);
    await p.click('label.sample:has-text("Lighthouse")');
    await clickGo(p);
    await p.waitForSelector('.p-timeline .src-live');
    await wait(p, 800);
    await shot();
  }],
  ['07-run-cancelled', async (p, shot) => {
    await mock(p, liveRoutes({ 'POST /run': R.cancelled }));
    await p.goto(BASE);
    await waitLive(p);
    await p.click('label.sample:has-text("Tabby")');
    await clickGo(p);
    await p.waitForSelector('.banner-bad');
    await wait(p, 800);
    await shot();
  }],
  ['08-run-rejected', async (p, shot) => {
    await mock(p, liveRoutes({ 'POST /run': R.rejected }));
    await p.goto(BASE);
    await waitLive(p);
    await clickGo(p);
    await p.waitForSelector('.banner-bad');
    await wait(p, 800);
    await shot();
  }],
  ['09-tooltip', async (p, shot) => {
    await mock(p, { 'GET /health': 'hang' });
    await p.goto(BASE);
    await p.waitForSelector('.p-timeline .marker');
    await p.locator('.p-timeline .marker .hit').first().hover({ force: true });
    await p.waitForSelector('.tip');
    await wait(p);
    await shot();
    await p.locator('.p-timeline .bar').nth(1).hover({ force: true });
    await wait(p);
    await shot({ suffix: 'bar' });
  }],
  ['10-load', async (p, shot) => {
    await mock(p, liveRoutes({ 'POST /load': R.load }));
    await p.goto(BASE);
    await waitLive(p);
    await p.click('#tab-load');
    await clickGo(p);
    await p.waitForSelector('.p-timeline .src-live');
    await wait(p, 800);
    await shot();
    await shot({ suffix: 'full', full: true });
  }],
  ['11-load-shed', async (p, shot) => {
    await mock(p, liveRoutes({ 'POST /load': R.shed }));
    await p.goto(BASE);
    await waitLive(p);
    await p.click('#tab-load');
    await clickGo(p);
    await p.waitForSelector('.p-timeline .src-live');
    await wait(p, 800);
    await shot();
  }],
  ['12-hotswap', async (p, shot) => {
    await mock(p, liveRoutes({ 'POST /reload': R.reload }));
    await p.goto(BASE);
    await waitLive(p);
    await p.click('#tab-swap');
    await clickGo(p);
    await p.waitForSelector('.p-timeline .src-live');
    await wait(p, 800);
    await shot();
    await shot({ suffix: 'full', full: true });
  }],
  ['13-error-busy', async (p, shot) => {
    await mock(p, liveRoutes({ 'POST /load': spec({ status: 429, body: { error: 'busy', message: 'the runtime is busy with another run; try again shortly', retryAfter: 4.2 } }) }));
    await p.goto(BASE);
    await waitLive(p);
    await p.click('#tab-load');
    await clickGo(p);
    await p.waitForSelector('.p-controls .err');
    await wait(p);
    await shot();
  }],
  ['14-error-upload', async (p, shot) => {
    await mock(p, liveRoutes({ 'POST /run': spec({ status: 415, body: { error: 'unreadable_image', message: 'the image could not be decoded' } }) }));
    await p.goto(BASE);
    await waitLive(p);
    await p.setInputFiles('.sample-upload input', { name: 'huge.png', mimeType: 'image/png', buffer: Buffer.alloc(3 * 1024 * 1024 + 7) });
    await p.waitForSelector('.samples .err');
    await p.setInputFiles('.sample-upload input', { name: 'broken.png', mimeType: 'image/png', buffer: Buffer.from('89504e470d0a1a0a0000', 'hex') });
    await clickGo(p);
    await p.waitForSelector('.p-controls > .err');
    await wait(p);
    await shot();
  }],
  ['15-asleep', async (p, shot) => {
    await mock(p, {});
    await p.goto(`${BASE}/?wakeLimit=1200&poll=250`);
    await p.waitForSelector('.status.st-asleep');
    await wait(p);
    await shot();
  }],
];

const variants = [
  { name: 'desktop-light', viewport: { width: 1440, height: 900 }, scheme: 'light' },
  { name: 'desktop-dark', viewport: { width: 1440, height: 900 }, scheme: 'dark' },
  { name: 'phone-light', viewport: { width: 390, height: 844 }, scheme: 'light', mobile: true },
  { name: 'phone-dark', viewport: { width: 390, height: 844 }, scheme: 'dark', mobile: true },
];

const only = process.env.ONLY?.split(',');
const onlyV = process.env.VARIANTS?.split(',');
const browser = await chromium.launch();
const report = { axe: {}, console: [], overflow: [] };

for (const v of variants.filter((x) => !onlyV || onlyV.includes(x.name))) {
  const ctx = await browser.newContext({ viewport: v.viewport, colorScheme: v.scheme, deviceScaleFactor: v.mobile ? 2 : 1, isMobile: !!v.mobile, hasTouch: !!v.mobile, reducedMotion: 'no-preference' });
  for (const [name, fn] of scenarios.filter(([n]) => !only || only.some((o) => n.includes(o)))) {
    const page = await ctx.newPage();
    page.on('console', (m) => {
      // Failed requests are the point of the waking, asleep and error states; anything else is reported.
      const expected = /Failed to load resource/.test(m.text()) && /^(02|13|14|15)-/.test(name);
      if ((m.type() === 'error' || m.type() === 'warning') && !expected) report.console.push(`${v.name} ${name}: ${m.text()}`);
    });
    page.on('pageerror', (e) => report.console.push(`${v.name} ${name}: PAGEERROR ${e.message}`));
    let axed = false;
    const shot = async ({ suffix = '', full = false } = {}) => {
      const file = `${name}${suffix ? '-' + suffix : ''}-${v.name}.png`;
      await page.screenshot({ path: OUT + file, fullPage: full });
      if (!axed) {
        axed = true;
        const r = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'best-practice']).analyze();
        report.axe[`${name} ${v.name}`] = r.violations.map((x) => ({ id: x.id, impact: x.impact, n: x.nodes.length, help: x.help, targets: x.nodes.slice(0, 3).map((n) => n.target.join(' ')) }));
        const over = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
        if (over > 0) report.overflow.push(`${v.name} ${name}: page scrolls ${over}px sideways`);
      }
    };
    try {
      await fn(page, shot);
    } catch (e) {
      report.console.push(`${v.name} ${name}: SCENARIO FAILED ${e.message.split('\n')[0]}`);
      await page.screenshot({ path: OUT + `${name}-FAILED-${v.name}.png` }).catch(() => {});
    }
    await page.close();
  }
  await ctx.close();
}
await browser.close();
writeFileSync(OUT + 'report.json', JSON.stringify(report, null, 2));
const violations = Object.entries(report.axe).filter(([, v]) => v.length);
console.log(`axe: ${violations.length ? violations.map(([k, v]) => `${k}: ${v.map((x) => `${x.id}(${x.n})`).join(' ')}`).join('\n     ') : '0 violations'}`);
console.log(`console: ${report.console.length ? report.console.join('\n         ') : 'clean'}`);
console.log(`overflow: ${report.overflow.length ? report.overflow.join('\n          ') : 'none'}`);
cleanup();
process.exit(0);
