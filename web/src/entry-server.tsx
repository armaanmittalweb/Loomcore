// Built with `vite build --ssr` and run by scripts/prerender.mjs: each page as
// it first renders (the recorded run), as HTML.
import { renderToString } from 'preact-render-to-string';
import { App } from './App';
import { Landing } from './landing/Landing';

export function render(): string {
  return renderToString(<App />);
}

export function renderLanding(): string {
  return renderToString(<Landing />);
}
