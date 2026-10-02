// The console images on the landing page's "Try it" steps. Run
//   LANDING=1 DPR=2 ONLY=06-run-live,10-load,12-hotswap VARIANTS=desktop-light,desktop-dark npm run shots
// first: it saves the Controls panel after each step; this converts them to public/landing/<step>-<theme>.webp.
import { mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';

const SHOTS = fileURLToPath(new URL('../shots/', import.meta.url));
const OUT = fileURLToPath(new URL('../public/landing/', import.meta.url));
mkdirSync(OUT, { recursive: true });
for (const step of ['run', 'load', 'swap']) {
  for (const theme of ['light', 'dark']) {
    const info = await sharp(`${SHOTS}landing-${step}-${theme}.png`).webp({ quality: 82 }).toFile(`${OUT}${step}-${theme}.webp`);
    console.log(`${step}-${theme}.webp ${info.width}x${info.height} ${(info.size / 1024).toFixed(0)} KB`);
  }
}
