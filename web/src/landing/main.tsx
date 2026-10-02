import { hydrate, render } from 'preact';
import { Landing } from './Landing';
import { countViews } from '../beacon';
import '../styles.css';
import './landing.css';

const root = document.getElementById('app') as HTMLElement;
// Prerendered like the console (scripts/prerender.mjs); hydrate it in place.
if (root.firstElementChild) hydrate(<Landing />, root);
else render(<Landing />, root);
countViews('loomcore');
