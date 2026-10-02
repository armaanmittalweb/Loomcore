// After `vite build` and `vite build --ssr src/entry-server.tsx --outDir dist-ssr`:
// renders each page as it first appears (the recorded run, every panel's text)
// into dist/index.html (the landing page) and dist/console.html (the console),
// so crawlers and link previews read real content.
import { readFileSync, rmSync, writeFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

const dist = fileURLToPath(new URL('../dist/', import.meta.url));
const ssr = fileURLToPath(new URL('../dist-ssr/', import.meta.url));
const { render, renderLanding } = await import(pathToFileURL(ssr + 'entry-server.js').href);
for (const [name, fn] of [['index.html', renderLanding], ['console.html', render]]) {
  const html = fn();
  const file = dist + name;
  const page = readFileSync(file, 'utf8');
  if (!page.includes('<div id="app"></div>')) throw new Error(`no empty #app in dist/${name}`);
  writeFileSync(file, page.replace('<div id="app"></div>', `<div id="app">${html}</div>`));
  console.log(`prerendered ${(html.length / 1024).toFixed(0)} KB of HTML into dist/${name}`);
}
rmSync(ssr, { recursive: true, force: true });
