// Hand-rolled SVG chart helpers. The datasets here are all tiny (≤34 points,
// a handful of bars) and every chart needs project-specific annotation (cos
// thresholds, noise bands, short/long bench divide). A generic chart library
// would fight the annotation story; 200 lines of SVG buys full control.

export interface Axis {
  min: number;
  max: number;
  ticks?: number[];
}

export interface ChartTheme {
  /** Stroke / fill for the data series. */
  series: string;
  /** Background fill (matches page background; charts blend in). */
  bg: string;
  /** Faint grid lines. */
  grid: string;
  /** Text color for labels. */
  text: string;
  /** Muted text (ticks). */
  textMuted: string;
  /** Accent for threshold lines (pass-line, noise band). */
  accent: string;
  /** Warn color (crossed threshold). */
  warn: string;
  /** Good color (inside noise band). */
  good: string;
}

export const DEFAULT_THEME: ChartTheme = {
  series: 'var(--accent)',
  bg: 'transparent',
  grid: 'var(--border)',
  text: 'var(--fg)',
  textMuted: 'var(--muted)',
  accent: 'var(--accent)',
  warn: 'var(--warn)',
  good: 'var(--good)',
};

interface ScaleFn { (v: number): number; }

function makeScale(domMin: number, domMax: number, rangeMin: number, rangeMax: number): ScaleFn {
  const span = domMax - domMin || 1;
  const rspan = rangeMax - rangeMin;
  return (v: number) => rangeMin + ((v - domMin) / span) * rspan;
}

export interface LineChartSeries {
  label: string;
  points: { x: number; y: number }[];
  color?: string;
}

export interface LineChartOpts {
  width: number;
  height: number;
  xAxis: Axis;
  yAxis: Axis;
  series: LineChartSeries[];
  /** Horizontal guide lines, drawn dashed with a label. */
  yGuides?: { y: number; label: string; color?: string }[];
  /** X-axis label. */
  xLabel?: string;
  /** Y-axis label. */
  yLabel?: string;
  /** Marker-on-points. Default: true. */
  markers?: boolean;
  theme?: Partial<ChartTheme>;
}

/**
 * Build an SVG string for a line chart. Append via `el.innerHTML = lineChart(opts)`
 * — the output is a single `<svg>` tag.
 */
export function lineChart(opts: LineChartOpts): string {
  const theme = { ...DEFAULT_THEME, ...(opts.theme ?? {}) };
  const pad = { l: 48, r: 16, t: 12, b: 32 };
  const w = opts.width, h = opts.height;
  const plotW = w - pad.l - pad.r;
  const plotH = h - pad.t - pad.b;
  const sx = makeScale(opts.xAxis.min, opts.xAxis.max, pad.l, pad.l + plotW);
  const sy = makeScale(opts.yAxis.min, opts.yAxis.max, pad.t + plotH, pad.t);

  const yTicks = opts.yAxis.ticks ?? defaultTicks(opts.yAxis.min, opts.yAxis.max, 5);
  const xTicks = opts.xAxis.ticks ?? defaultTicks(opts.xAxis.min, opts.xAxis.max, 6);

  const gridY = yTicks.map((t) =>
    `<line x1="${pad.l}" y1="${sy(t).toFixed(1)}" x2="${pad.l + plotW}" y2="${sy(t).toFixed(1)}" stroke="${theme.grid}" stroke-width="1"/>`,
  ).join('');
  const labelsY = yTicks.map((t) =>
    `<text x="${pad.l - 6}" y="${sy(t).toFixed(1)}" fill="${theme.textMuted}" font-size="10" text-anchor="end" dominant-baseline="middle" font-family="var(--font-mono)">${formatTick(t)}</text>`,
  ).join('');
  const labelsX = xTicks.map((t) =>
    `<text x="${sx(t).toFixed(1)}" y="${pad.t + plotH + 14}" fill="${theme.textMuted}" font-size="10" text-anchor="middle" font-family="var(--font-mono)">${formatTick(t)}</text>`,
  ).join('');

  const guides = (opts.yGuides ?? []).map((g) => {
    const y = sy(g.y).toFixed(1);
    const c = g.color ?? theme.accent;
    return (
      `<line x1="${pad.l}" y1="${y}" x2="${pad.l + plotW}" y2="${y}" stroke="${c}" stroke-width="1" stroke-dasharray="3 3" opacity="0.7"/>` +
      `<text x="${pad.l + plotW - 4}" y="${Number(y) - 3}" fill="${c}" font-size="10" text-anchor="end" font-family="var(--font-mono)">${g.label}</text>`
    );
  }).join('');

  const seriesPaths = opts.series.map((s, idx) => {
    const color = s.color ?? (idx === 0 ? theme.series : theme.accent);
    const d = s.points.map((p, i) => `${i === 0 ? 'M' : 'L'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ');
    const markers = opts.markers !== false
      ? s.points.map((p) => `<circle cx="${sx(p.x).toFixed(1)}" cy="${sy(p.y).toFixed(1)}" r="2.5" fill="${color}"/>`).join('')
      : '';
    return `<path d="${d}" fill="none" stroke="${color}" stroke-width="1.5"/>${markers}`;
  }).join('');

  const xAxisLabel = opts.xLabel
    ? `<text x="${pad.l + plotW / 2}" y="${h - 4}" fill="${theme.textMuted}" font-size="10" text-anchor="middle" font-family="var(--font-sans)">${opts.xLabel}</text>`
    : '';
  const yAxisLabel = opts.yLabel
    ? `<text x="12" y="${pad.t + plotH / 2}" fill="${theme.textMuted}" font-size="10" text-anchor="middle" transform="rotate(-90, 12, ${pad.t + plotH / 2})" font-family="var(--font-sans)">${opts.yLabel}</text>`
    : '';

  return (
    `<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto;">` +
    gridY + guides + seriesPaths + labelsY + labelsX + xAxisLabel + yAxisLabel +
    `</svg>`
  );
}

export interface BarChartBar {
  label: string;
  value: number;
  /** Optional secondary value, drawn as a lighter bar behind (e.g. p95 vs p50). */
  secondary?: number;
  color?: string;
}

export interface BarChartOpts {
  width: number;
  height: number;
  bars: BarChartBar[];
  /** Y-axis domain. If omitted, inferred from bars. */
  yAxis?: Axis;
  xLabel?: string;
  yLabel?: string;
  /** Format function for value labels above each bar. */
  format?: (v: number) => string;
  theme?: Partial<ChartTheme>;
}

export function barChart(opts: BarChartOpts): string {
  const theme = { ...DEFAULT_THEME, ...(opts.theme ?? {}) };
  const pad = { l: 48, r: 16, t: 12, b: 60 };
  const w = opts.width, h = opts.height;
  const plotW = w - pad.l - pad.r;
  const plotH = h - pad.t - pad.b;
  const vals = opts.bars.flatMap((b) => [b.value, b.secondary ?? b.value]);
  const yMax = opts.yAxis?.max ?? (Math.max(0, ...vals) * 1.1 || 1);
  const yMin = opts.yAxis?.min ?? 0;
  const sy = makeScale(yMin, yMax, pad.t + plotH, pad.t);
  const yTicks = opts.yAxis?.ticks ?? defaultTicks(yMin, yMax, 4);
  const format = opts.format ?? ((v: number) => formatTick(v));

  const n = opts.bars.length;
  const slotW = plotW / Math.max(1, n);
  const barW = Math.min(slotW * 0.7, 40);

  const gridY = yTicks.map((t) =>
    `<line x1="${pad.l}" y1="${sy(t).toFixed(1)}" x2="${pad.l + plotW}" y2="${sy(t).toFixed(1)}" stroke="${theme.grid}" stroke-width="1"/>`,
  ).join('');
  const labelsY = yTicks.map((t) =>
    `<text x="${pad.l - 6}" y="${sy(t).toFixed(1)}" fill="${theme.textMuted}" font-size="10" text-anchor="end" dominant-baseline="middle" font-family="var(--font-mono)">${formatTick(t)}</text>`,
  ).join('');

  const bars = opts.bars.map((b, i) => {
    const cx = pad.l + slotW * (i + 0.5);
    const x = cx - barW / 2;
    const color = b.color ?? theme.series;
    const yV = sy(b.value).toFixed(1);
    const hV = (sy(yMin) - sy(b.value)).toFixed(1);
    const secondary = b.secondary != null && b.secondary !== b.value
      ? `<rect x="${x}" y="${sy(b.secondary).toFixed(1)}" width="${barW}" height="${(sy(yMin) - sy(b.secondary)).toFixed(1)}" fill="${color}" opacity="0.35"/>`
      : '';
    const valueLabel = `<text x="${cx}" y="${Number(yV) - 4}" fill="${theme.text}" font-size="10" text-anchor="middle" font-family="var(--font-mono)">${format(b.value)}</text>`;
    const xLabel = `<text x="${cx}" y="${pad.t + plotH + 14}" fill="${theme.textMuted}" font-size="10" text-anchor="middle" font-family="var(--font-mono)" transform="rotate(-30, ${cx}, ${pad.t + plotH + 14})">${escapeText(b.label)}</text>`;
    return `${secondary}<rect x="${x}" y="${yV}" width="${barW}" height="${hV}" fill="${color}"/>${valueLabel}${xLabel}`;
  }).join('');

  const yAxisLabel = opts.yLabel
    ? `<text x="12" y="${pad.t + plotH / 2}" fill="${theme.textMuted}" font-size="10" text-anchor="middle" transform="rotate(-90, 12, ${pad.t + plotH / 2})" font-family="var(--font-sans)">${opts.yLabel}</text>`
    : '';

  return (
    `<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto;">` +
    gridY + bars + labelsY + yAxisLabel +
    `</svg>`
  );
}

export interface ScatterPoint {
  x: number;
  y: number;
  label?: string;
  color?: string;
}

export interface ScatterChartOpts {
  width: number;
  height: number;
  xAxis: Axis;
  yAxis: Axis;
  points: ScatterPoint[];
  /** Horizontal span to shade as a band (e.g. ±3% noise around median). */
  yBand?: { from: number; to: number; label?: string };
  /** Horizontal guide line (e.g. median). */
  yGuides?: { y: number; label: string; color?: string }[];
  xLabel?: string;
  yLabel?: string;
  theme?: Partial<ChartTheme>;
}

export function scatterChart(opts: ScatterChartOpts): string {
  const theme = { ...DEFAULT_THEME, ...(opts.theme ?? {}) };
  const pad = { l: 48, r: 16, t: 12, b: 32 };
  const w = opts.width, h = opts.height;
  const plotW = w - pad.l - pad.r;
  const plotH = h - pad.t - pad.b;
  const sx = makeScale(opts.xAxis.min, opts.xAxis.max, pad.l, pad.l + plotW);
  const sy = makeScale(opts.yAxis.min, opts.yAxis.max, pad.t + plotH, pad.t);
  const yTicks = opts.yAxis.ticks ?? defaultTicks(opts.yAxis.min, opts.yAxis.max, 5);
  const xTicks = opts.xAxis.ticks ?? defaultTicks(opts.xAxis.min, opts.xAxis.max, 6);

  const gridY = yTicks.map((t) =>
    `<line x1="${pad.l}" y1="${sy(t).toFixed(1)}" x2="${pad.l + plotW}" y2="${sy(t).toFixed(1)}" stroke="${theme.grid}" stroke-width="1"/>`,
  ).join('');

  const band = opts.yBand
    ? (() => {
        const y1 = sy(opts.yBand.to).toFixed(1);
        const y2 = sy(opts.yBand.from).toFixed(1);
        const rh = (Number(y2) - Number(y1)).toFixed(1);
        const lbl = opts.yBand.label
          ? `<text x="${pad.l + plotW - 4}" y="${Number(y1) - 3}" fill="${theme.good}" font-size="10" text-anchor="end" font-family="var(--font-mono)">${opts.yBand.label}</text>`
          : '';
        return `<rect x="${pad.l}" y="${y1}" width="${plotW}" height="${rh}" fill="${theme.good}" opacity="0.08"/>${lbl}`;
      })()
    : '';

  const guides = (opts.yGuides ?? []).map((g) => {
    const y = sy(g.y).toFixed(1);
    const c = g.color ?? theme.accent;
    return (
      `<line x1="${pad.l}" y1="${y}" x2="${pad.l + plotW}" y2="${y}" stroke="${c}" stroke-width="1" stroke-dasharray="3 3" opacity="0.7"/>` +
      `<text x="${pad.l + 4}" y="${Number(y) - 3}" fill="${c}" font-size="10" font-family="var(--font-mono)">${g.label}</text>`
    );
  }).join('');

  const points = opts.points.map((p) => {
    const color = p.color ?? theme.series;
    return `<circle cx="${sx(p.x).toFixed(1)}" cy="${sy(p.y).toFixed(1)}" r="3" fill="${color}"/>`;
  }).join('');

  const labelsY = yTicks.map((t) =>
    `<text x="${pad.l - 6}" y="${sy(t).toFixed(1)}" fill="${theme.textMuted}" font-size="10" text-anchor="end" dominant-baseline="middle" font-family="var(--font-mono)">${formatTick(t)}</text>`,
  ).join('');
  const labelsX = xTicks.map((t) =>
    `<text x="${sx(t).toFixed(1)}" y="${pad.t + plotH + 14}" fill="${theme.textMuted}" font-size="10" text-anchor="middle" font-family="var(--font-mono)">${formatTick(t)}</text>`,
  ).join('');

  const xAxisLabel = opts.xLabel
    ? `<text x="${pad.l + plotW / 2}" y="${h - 4}" fill="${theme.textMuted}" font-size="10" text-anchor="middle" font-family="var(--font-sans)">${opts.xLabel}</text>`
    : '';
  const yAxisLabel = opts.yLabel
    ? `<text x="12" y="${pad.t + plotH / 2}" fill="${theme.textMuted}" font-size="10" text-anchor="middle" transform="rotate(-90, 12, ${pad.t + plotH / 2})" font-family="var(--font-sans)">${opts.yLabel}</text>`
    : '';

  return (
    `<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto;">` +
    band + gridY + guides + points + labelsY + labelsX + xAxisLabel + yAxisLabel +
    `</svg>`
  );
}

function defaultTicks(min: number, max: number, n: number): number[] {
  if (max === min) return [min];
  const step = (max - min) / n;
  const out: number[] = [];
  for (let i = 0; i <= n; i++) out.push(min + step * i);
  return out;
}

function formatTick(v: number): string {
  if (v === 0) return '0';
  const abs = Math.abs(v);
  if (abs >= 1000) return v.toFixed(0);
  if (abs >= 10) return v.toFixed(1);
  if (abs >= 1) return v.toFixed(2);
  return v.toFixed(3);
}

function escapeText(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
