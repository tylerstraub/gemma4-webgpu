// Display-formatting helpers for data-dense panels.
//
// The site is monospace-heavy for numeric data. These helpers produce
// fixed-width, tightly-formatted strings that line up column-to-column when
// rendered in a monospace font.

/** Format nanoseconds adaptively — µs for small values, ms for big. */
export function formatNs(ns: number): string {
  if (ns < 1e3) return `${ns.toFixed(0)} ns`;
  if (ns < 1e6) return `${(ns / 1e3).toFixed(2)} µs`;
  if (ns < 1e9) return `${(ns / 1e6).toFixed(3)} ms`;
  return `${(ns / 1e9).toFixed(3)} s`;
}

/** Format milliseconds with three sig figs. */
export function formatMs(ms: number): string {
  if (ms < 1) return `${(ms * 1000).toFixed(0)} µs`;
  if (ms < 1000) return `${ms.toFixed(3)} ms`;
  return `${(ms / 1000).toFixed(3)} s`;
}

/** Format a byte count in human-readable units. */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KB', 'MB', 'GB', 'TB'];
  let v = bytes / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v >= 100 ? 0 : v >= 10 ? 1 : 2)} ${units[i]}`;
}

/** Format a tokens-per-second value with sensible precision. */
export function formatTps(tps: number): string {
  if (tps >= 100) return tps.toFixed(1);
  if (tps >= 10) return tps.toFixed(2);
  return tps.toFixed(3);
}

/** Format a cosine similarity: always four decimals to keep columns aligned. */
export function formatCos(cos: number): string {
  return cos.toFixed(4);
}

/** Format a percentage (input is 0–1). */
export function formatPct(frac: number, digits = 1): string {
  return `${(frac * 100).toFixed(digits)}%`;
}

/** Pad a string on the left to a fixed width with non-breaking spaces. */
export function padLeft(s: string, n: number): string {
  if (s.length >= n) return s;
  return '\u00a0'.repeat(n - s.length) + s;
}
