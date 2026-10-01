import { readFileSync } from 'node:fs';
import preact from '@preact/preset-vite';
import type { Plugin } from 'vite';
import { defineConfig } from 'vitest/config';

// Serve `vite preview` with the production headers from vercel.json (per path),
// so the screenshot run checks the app under the CSP it will ship with.
type Rule = { source: string; headers: { key: string; value: string }[] };
const vercel = JSON.parse(readFileSync(new URL('./vercel.json', import.meta.url), 'utf8')) as { headers: Rule[] };
const rules = vercel.headers.map((r) => ({ re: new RegExp(`^${r.source}$`), headers: r.headers }));

const vercelHeaders = (): Plugin => ({
  name: 'vercel-headers-in-preview',
  configurePreviewServer(server) {
    server.middlewares.use((req, res, next) => {
      const path = (req.url ?? '/').split('?')[0];
      for (const r of rules) if (r.re.test(path)) for (const h of r.headers) res.setHeader(h.key, h.value);
      next();
    });
  },
});

export default defineConfig({
  plugins: [preact(), vercelHeaders()],
  server: { port: 5177, strictPort: true },
  preview: { port: 5178, strictPort: true },
  build: { target: 'es2022', assetsInlineLimit: 0 },
  test: { include: ['tests/**/*.test.ts'] },
});
