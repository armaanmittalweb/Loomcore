// `npm run e2e:live -- [--space <url>]`: drives the console against a real
// runtime, with no mocks: waits for "Live", runs one job, a load test and a
// hot-swap through the UI, and checks each result came back live with a
// trace drawn. Use it against `python space/server.py` (default
// http://127.0.0.1:7860) or, after deploying, https://loomcore-api.amittal.dev.
// Serves the console on port 5177, an origin the server's CORS allows.
import { spawn, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const args = process.argv.slice(2);
const i = args.indexOf('--space');
const SPACE = i >= 0 ? args[i + 1] : 'http://127.0.0.1:7860';
const WEB = fileURLToPath(new URL('..', import.meta.url));
const win = process.platform === 'win32';
const BASE = 'http://localhost:5177';

const dev = spawn(win ? 'npx.cmd' : 'npx', ['vite', '--port', '5177', '--strictPort'], { cwd: WEB, stdio: 'ignore', shell: win, env: { ...process.env, VITE_SPACE_URL: SPACE } });
const stop = () => {
  try {
    if (win) spawnSync('taskkill', ['/pid', String(dev.pid), '/T', '/F']);
    else dev.kill();
  } catch {
    /* gone */
  }
};
process.on('exit', stop);
for (let n = 0; n < 80; n++) {
  try {
    if ((await fetch(BASE)).ok) break;
  } catch {
    /* starting */
  }
  await new Promise((r) => setTimeout(r, 250));
}

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
const check = (ok, what) => {
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${what}`);
  if (!ok) process.exitCode = 1;
};

await page.goto(BASE);
await page.waitForSelector('.status.st-live', { timeout: 240_000 });
check(true, `live: ${await page.textContent('.status')}`);

await page.click('label.sample:has-text("Lighthouse")');
await page.click('.go .btn');
await page.waitForSelector('.p-timeline .src-live', { timeout: 60_000 });
check((await page.locator('.p-timeline .bar').count()) >= 1, `run: ${await page.textContent('.p-timeline .panel-title')}`);
// With no deadline (the Run tab's default) mobilenet runs FP32 and labels the photo correctly.
const precision = (await page.textContent('.rv-nodes tbody tr:first-child .pchip'))?.trim();
check(precision !== 'FP32' || (await page.textContent('.rv-name'))?.trim() === 'lighthouse', `run label: ${await page.textContent('.rv-name')} (mobilenet ${precision})`);

await page.click('#tab-load');
await page.click('.go .btn');
await page.waitForSelector('.p-result .stats', { timeout: 120_000 });
await page.waitForFunction(() => !document.querySelector('.tl-busy'), null, { timeout: 120_000 });
// Every job is accounted for: completed, rejected, cancelled or shed (which ones depends on load and budget).
const stat = async (label) => Number((await page.locator('.stat', { hasText: label }).locator('.stat-v').first().textContent().catch(() => '0'))?.replace(/,/g, '') || 0);
const [jobs, done, rej, can] = await Promise.all(['Jobs', 'Completed', 'Rejected', 'Cancelled'].map(stat));
const shed = await page.locator('.stat', { hasText: 'Shed' }).count() ? await stat('Shed') : 0;
check(jobs === 48 && done + rej + can + shed === jobs, `load: ${await page.textContent('.p-timeline .panel-title')} (${done} completed, ${rej} rejected, ${can} cancelled, ${shed} shed)`);

await page.click('#tab-swap');
await page.click('.go .btn');
await page.waitForFunction(() => document.querySelector('.p-result .swap-count') && !document.querySelector('.tl-busy'), null, { timeout: 120_000 });
const lost = (await page.textContent('.swap-count .is-ok dd, .swap-count .is-bad dd'))?.trim();
check(lost === '0', `hot-swap: ${await page.textContent('.p-timeline .panel-title')}, lost ${lost}`);
check((await page.locator('.p-timeline .swap').count()) >= 1, 'swap lines drawn');
check(errors.length === 0, `no page errors${errors.length ? ': ' + errors.join('; ') : ''}`);

await browser.close();
stop();
process.exit(process.exitCode ?? 0);
