// Number and date formatting, one place, so every panel reads the same.

/** Milliseconds with precision that suits the size: 0.142, 3.99, 38.2, 1,541. */
export function ms(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return '-';
  const a = Math.abs(v);
  if (a >= 1000) return Math.round(v).toLocaleString('en-US');
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return v.toFixed(3);
}

export function pct(p: number | null | undefined, digits = 1): string {
  if (p == null || !Number.isFinite(p)) return '-';
  return `${(p * 100).toFixed(digits)}%`;
}

export function ratio(v: number): string {
  return `${v.toFixed(2)}x`;
}

export function count(n: number): string {
  return n.toLocaleString('en-US');
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

/** "1 Oct 2026", in UTC so the prerendered page and the browser agree. */
export function day(iso: string): string {
  const d = new Date(iso);
  return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
}

/** "14:02:11" local wall clock, for a live result. */
export function clock(t: number): string {
  const d = new Date(t);
  return [d.getHours(), d.getMinutes(), d.getSeconds()].map((n) => String(n).padStart(2, '0')).join(':');
}

/** "11th Gen Intel(R) Core(TM) i5-1145G7 @ 2.60GHz (8 logical CPUs)" -> "Intel Core i5-1145G7". */
export function shortCpu(cpu: string): string {
  const base = cpu
    .replace(/\(\d+ logical CPUs?\)/, '')
    .replace(/\((R|TM)\)/gi, '')
    .replace(/@.*$/, '')
    .replace(/\b\d+(st|nd|rd|th) Gen\b/i, '')
    .replace(/\bCPU\b/, '')
    .replace(/\s+/g, ' ')
    .trim();
  return base.replace(/^AMD Ryzen/, 'Ryzen');
}

export function logicalCpus(cpu: string): number | null {
  const m = cpu.match(/\((\d+) logical CPUs?\)/);
  return m ? Number(m[1]) : null;
}

/** Splits `code` spans out of a sentence: "a `b` c" -> ["a ", {code:"b"}, " c"]. */
export function codeSpans(text: string): (string | { code: string })[] {
  return text.split(/(`[^`]+`)/).filter(Boolean).map((p) => (p.startsWith('`') ? { code: p.slice(1, -1) } : p));
}

/** Strips C++ float noise from a router reason: "19.557800ms" -> "19.56 ms". */
export function tidyReason(reason: string): string {
  return reason
    .replace(/(\d+\.\d+)ms/g, (_, n: string) => `${ms(Number(n))} ms`)
    .replace(/(\d+\.\d{4,})/g, (_, n: string) => String(Number(Number(n).toFixed(3))));
}
