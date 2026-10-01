// Built with `vite build --ssr` and run by scripts/prerender.mjs: the app as
// it first renders (the recorded run), as HTML.
import { renderToString } from 'preact-render-to-string';
import { App } from './App';

export function render(): string {
  return renderToString(<App />);
}
