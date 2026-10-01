import { hydrate, render } from 'preact';
import { App } from './App';
import { countViews } from './beacon';
import './styles.css';

const root = document.getElementById('app') as HTMLElement;
// The HTML is prerendered from the recorded run (scripts/prerender.mjs), so
// crawlers and no-JS readers get the real content; hydrate it in place.
if (root.firstElementChild) hydrate(<App />, root);
else render(<App />, root);
countViews('loomcore');
