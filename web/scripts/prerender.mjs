// After `vite build` and `vite build --ssr src/entry-server.tsx --outDir dist-ssr`:
// renders the app as it first appears (the recorded run, every panel's text)
// into dist/index.html, so crawlers and link previews read real content.
import { readFileSync, rmSync, writeFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

const dist = fileURLToPath(new URL('../dist/', import.meta.url));
const ssr = fileURLToPath(new URL('../dist-ssr/', import.meta.url));
const { render } = await import(pathToFileURL(ssr + 'entry-server.js').href);
const html = render();
const file = dist + 'index.html';
const page = readFileSync(file, 'utf8');
if (!page.includes('<div id="app"></div>')) throw new Error('no empty #app in dist/index.html');
writeFileSync(file, page.replace('<div id="app"></div>', `<div id="app">${html}</div>`));
rmSync(ssr, { recursive: true, force: true });
console.log(`prerendered ${(html.length / 1024).toFixed(0)} KB of HTML into dist/index.html`);
