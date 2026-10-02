import { readFileSync } from 'node:fs';
import preact from '@preact/preset-vite';
import type { Plugin } from 'vite';
import { defineConfig } from 'vitest/config';

// Serve `vite preview` with the production headers from vercel.json (per path),
// so the screenshot run checks the app under the CSP it will ship with.
type Rule = { source: string; headers: { key: string; value: string }[] };
const vercel = JSON.parse(readFileSync(new URL('./vercel.json', import.meta.url), 'utf8')) as { headers: Rule[] };
const rules = vercel.headers.map((r) => ({ re: new RegExp(`^${r.source}$`), headers: r.headers }));

// vercel.json's cleanUrls serves console.html at /console; dev and preview do the same here.
const cleanConsole = (req: { url?: string }) => {
  const [path, query] = (req.url ?? '/').split('?');
  if (path === '/console' || path === '/console/') req.url = `/console.html${query ? `?${query}` : ''}`;
};

const vercelHeaders = (): Plugin => ({
  name: 'vercel-headers-in-preview',
  configureServer(server) {
    server.middlewares.use((req, _res, next) => {
      cleanConsole(req);
      next();
    });
  },
  configurePreviewServer(server) {
    server.middlewares.use((req, res, next) => {
      const path = (req.url ?? '/').split('?')[0];
      cleanConsole(req);
      for (const r of rules) if (r.re.test(path)) for (const h of r.headers) res.setHeader(h.key, h.value);
      next();
    });
  },
});

export default defineConfig({
  plugins: [preact(), vercelHeaders()],
  server: { port: 5177, strictPort: true },
  preview: { port: 5178, strictPort: true },
  // Two pages: the landing page at / and the console at /console (vercel.json's cleanUrls serves console.html there).
  build: { target: 'es2022', assetsInlineLimit: 0, rollupOptions: { input: { index: 'index.html', console: 'console.html' } } },
  test: { include: ['tests/**/*.test.ts'] },
});
