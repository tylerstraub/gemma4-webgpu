// Bench panel — short/long runners + tps distribution scatter.
//
// Short config: "Hello, how are you?", maxTokens=64.
// Long  config: list-five-differences prompt, maxTokens=200.
// Both are greedy (temperature=0, repPenalty=1.0) so tps is the only
// independent variable. Each run does a warmup + 5 measured runs; median is
// the headline number. `CLAUDE.md` pins 3% as the noise threshold — the
// scatter shades ±3% as a reference band.

import {
  subscribe,
  getState,
  beginBusy,
  updateStatusLine,
  type EngineState,
} from '../lib/engine.js';
import { runBench, type BenchResult } from '../lib/diagnostics.js';
import { scatterChart } from '../lib/svg.js';
import { formatTps, formatMs, padLeft } from '../lib/format.js';
import { confirmDialog } from '../lib/dialog.js';

const SHORT_PROMPT = 'Hello, how are you?';
const LONG_PROMPT =
  'List five concrete differences between a raven and a crow. Number them.';

export function mountBenchPanel(root: HTMLElement): void {
  root.innerHTML = `
    <div class="card">
      <h3>benchmark</h3>
      <p class="card-sub">
        Measure decode throughput (tokens/second). Temperature is pinned to 0
        and repetition penalty to 1.0 so the only variable is hardware +
        tuning. Each run does one warmup and five measured runs; the median
        is the headline number. The ±3% noise band reflects this project's
        bench-noise threshold — anything inside the band is indistinguishable
        from measurement noise.
      </p>

      <div class="field-row">
        <label>
          prompt
          <select id="bench-prompt">
            <option value="short">short · "Hello, how are you?" · maxTokens=64</option>
            <option value="long" selected>long · five-differences · maxTokens=200</option>
          </select>
        </label>
        <label>
          runs
          <input type="number" id="bench-runs" min="1" max="20" step="1" value="5" />
        </label>
      </div>

      <div class="card-actions">
        <button type="button" class="primary" id="bench-run" disabled>run bench</button>
        <span class="muted mono-sm" id="bench-status"></span>
      </div>
    </div>

    <div id="bench-result" class="hidden"></div>
  `;

  const elRun = root.querySelector('#bench-run') as HTMLButtonElement;
  const elStatus = root.querySelector('#bench-status') as HTMLElement;
  const elResult = root.querySelector('#bench-result') as HTMLElement;
  const elPrompt = root.querySelector('#bench-prompt') as HTMLSelectElement;
  const elRuns = root.querySelector('#bench-runs') as HTMLInputElement;

  let latest: { label: string; result: BenchResult }[] = [];

  const renderResults = () => {
    if (latest.length === 0) { elResult.classList.add('hidden'); return; }
    elResult.classList.remove('hidden');
    elResult.innerHTML = latest.map(renderBenchCard).join('');
  };

  elRun.addEventListener('click', async () => {
    const s = getState();
    if (s.status !== 'ready') return;
    if (s.messages.length > 0) {
      const ok = await confirmDialog({
        title: 'Bench will reset your chat',
        message:
          'Running bench wipes the KV cache and the conversation transcript. The engine stays loaded. Continue?',
        confirmLabel: 'Run bench',
        danger: true,
      });
      if (!ok) return;
    }

    const promptKey = elPrompt.value;
    const prompt = promptKey === 'short' ? SHORT_PROMPT : LONG_PROMPT;
    const maxTokens = promptKey === 'short' ? 64 : 200;
    const runs = Math.max(1, Math.min(20, Number(elRuns.value) || 5));

    const engine = getState().engine!;
    const done = beginBusy(`benching (${promptKey}, ${runs} runs)…`);
    try {
      const result = await runBench(engine, prompt, maxTokens, runs, (p) => {
        const line = `${p.label}: ${p.tokens} tok in ${p.totalMs.toFixed(0)} ms · ${formatTps(p.tps)} tps  (${p.index + 1}/${p.total})`;
        updateStatusLine(line);
        elStatus.textContent = line;
      });
      const label = promptKey === 'short'
        ? `short · "${SHORT_PROMPT}" · maxTokens=64 · runs=${runs}`
        : `long · five-differences · maxTokens=200 · runs=${runs}`;
      latest = [{ label, result }, ...latest].slice(0, 4);
      renderResults();
      done({ ok: true });
      elStatus.textContent = `done · median ${formatTps(result.median)} tps`;
    } catch (e) {
      done({ ok: false, error: (e as Error).message });
      elStatus.textContent = `error: ${(e as Error).message}`;
    }
  });

  subscribe((s: EngineState) => {
    elRun.disabled = s.status !== 'ready';
  });
}

function renderBenchCard({ label, result }: { label: string; result: BenchResult }): string {
  const measured = result.runs.filter((r) => r.label.startsWith('m'));
  const warmup = result.runs.find((r) => r.label === 'warmup');

  const points = measured.map((r, i) => ({ x: i + 1, y: r.tps }));
  const noiseBand = {
    from: result.median * 0.97,
    to: result.median * 1.03,
    label: '±3% noise band',
  };
  const xMax = measured.length + 1;
  const yMin = Math.min(result.min, result.median * 0.95) * 0.97;
  const yMax = Math.max(result.max, result.median * 1.05) * 1.03;

  const chart = scatterChart({
    width: 640, height: 260,
    xAxis: { min: 0, max: xMax, ticks: measured.map((_, i) => i + 1) },
    yAxis: { min: yMin, max: yMax },
    points,
    yBand: noiseBand,
    yGuides: [{ y: result.median, label: `median ${formatTps(result.median)}` }],
    xLabel: 'run index',
    yLabel: 'tps',
  });

  const rows = result.runs.map((r) => `
    <tr>
      <td>${r.label}</td>
      <td class="num">${r.tokens}</td>
      <td class="num">${formatMs(r.totalMs)}</td>
      <td class="num">${r.firstMs != null ? formatMs(r.firstMs) : '—'}</td>
      <td class="num">${formatMs(r.decodeMs)}</td>
      <td class="num ${r.label.startsWith('m') ? 'accent' : 'muted'}">${formatTps(r.tps)}</td>
    </tr>
  `).join('');

  const statBlock = [
    `<b>median</b> ${formatTps(result.median)}`,
    `<b>mean</b> ${formatTps(result.mean)}`,
    `<b>min</b> ${formatTps(result.min)}`,
    `<b>max</b> ${formatTps(result.max)}`,
    `<b>spread</b> ${padLeft(((result.max - result.min) / result.median * 100).toFixed(2), 5)}%`,
  ].join(' · ');

  return `
    <div class="card">
      <h3>${escapeHtml(label)}</h3>
      <div class="mono-sm" style="margin-bottom: var(--s-3);">${statBlock}</div>
      <div class="chart-wrap">${chart}</div>
      <details style="margin-top: var(--s-3);">
        <summary class="mono-sm muted" style="cursor: pointer;">per-run detail</summary>
        <table class="data" style="margin-top: var(--s-2);">
          <thead><tr><th>label</th><th class="num">tokens</th><th class="num">total</th><th class="num">first</th><th class="num">decode</th><th class="num">tps</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
        ${warmup?.outHead ? `<div class="mono-sm muted" style="margin-top: var(--s-3);">warmup output head: ${escapeHtml(warmup.outHead)}…</div>` : ''}
      </details>
    </div>
  `;
}

function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
