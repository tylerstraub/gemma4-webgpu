// Profile panel — GPU timestamp-query breakdown + CPU phase breakdown.
//
// Two independent sub-panels. GPU path requires `timestamp-query`; if the
// device doesn't have it (common on consumer integrated GPUs and Safari),
// surface the reason in project vocabulary rather than throwing.

import {
  subscribe,
  getState,
  beginBusy,
  updateStatusLine,
  type EngineState,
} from '../lib/engine.js';
import { profile as runGpuProfile, cpuProfile as runCpuProfile } from '../../src/diagnostics/index.js';
import type { ProfileRunResult, CpuProfileRunResult } from '../../src/diagnostics/index.js';
import { barChart } from '../lib/svg.js';
import { formatNs, formatMs, formatTps } from '../lib/format.js';
import { confirmDialog } from '../lib/dialog.js';

const DEFAULT_PROMPT = 'List five concrete differences between a raven and a crow. Number them.';

export function mountProfilePanel(root: HTMLElement): void {
  root.innerHTML = `
    <div class="card">
      <h3>gpu profile · timestamp-query</h3>
      <p class="card-sub">
        Per-dispatch GPU timing via WebGPU's <code>timestamp-query</code>.
        Decodes <span class="mono">maxTokens=64</span> after a 10-token warmup.
        Bars are the p50 per dispatch label; the faint background bar is p95.
      </p>
      <div class="card-actions">
        <button type="button" class="primary" id="prof-gpu-run" disabled>run gpu profile</button>
        <span class="muted mono-sm" id="prof-gpu-status"></span>
      </div>
      <div id="prof-gpu-result" class="hidden" style="margin-top: var(--s-4);"></div>
    </div>

    <div class="card">
      <h3>cpu profile · phase breakdown</h3>
      <p class="card-sub">
        Host-side wall-clock per forward pass. The <code>mapAsync.wait</code>
        phase is dominated by GPU work completing; subtract it from
        <code>forward.total</code> to get the CPU-only floor. Secondary counts
        show median writeBuffer and pass-encode calls per forward.
      </p>
      <div class="card-actions">
        <button type="button" class="primary" id="prof-cpu-run" disabled>run cpu profile</button>
        <span class="muted mono-sm" id="prof-cpu-status"></span>
      </div>
      <div id="prof-cpu-result" class="hidden" style="margin-top: var(--s-4);"></div>
    </div>
  `;

  const elGpuRun = root.querySelector('#prof-gpu-run') as HTMLButtonElement;
  const elGpuStatus = root.querySelector('#prof-gpu-status') as HTMLElement;
  const elGpuResult = root.querySelector('#prof-gpu-result') as HTMLElement;
  const elCpuRun = root.querySelector('#prof-cpu-run') as HTMLButtonElement;
  const elCpuStatus = root.querySelector('#prof-cpu-status') as HTMLElement;
  const elCpuResult = root.querySelector('#prof-cpu-result') as HTMLElement;

  const confirmDestroy = async (): Promise<boolean> => {
    if (getState().messages.length === 0) return true;
    return confirmDialog({
      title: 'Profile will reset your chat',
      message: 'Profiling wipes the KV cache and the conversation transcript. Continue?',
      confirmLabel: 'Run profile',
      danger: true,
    });
  };

  elGpuRun.addEventListener('click', async () => {
    const s = getState();
    if (s.status !== 'ready') return;
    if (!await confirmDestroy()) return;
    const engine = s.engine!;
    const done = beginBusy('gpu profile (warmup + measure)…');
    try {
      updateStatusLine('gpu profile · warming up');
      const result = await runGpuProfile(engine, DEFAULT_PROMPT, { maxTokens: 64, warmup: 10 });
      elGpuResult.classList.remove('hidden');
      elGpuResult.innerHTML = renderGpuResult(result);
      done({ ok: true });
      if (result.ok) {
        elGpuStatus.textContent =
          `done · forward p50 ${formatNs(result.report.forwardMedian_ns)} · ${result.report.forwardSamples_n} samples`;
      } else {
        elGpuStatus.textContent = `unavailable: ${result.error}`;
      }
    } catch (e) {
      done({ ok: false, error: (e as Error).message });
      elGpuStatus.textContent = `error: ${(e as Error).message}`;
    }
  });

  elCpuRun.addEventListener('click', async () => {
    const s = getState();
    if (s.status !== 'ready') return;
    if (!await confirmDestroy()) return;
    const engine = s.engine!;
    const done = beginBusy('cpu profile (warmup + measure)…');
    try {
      updateStatusLine('cpu profile · warming up');
      const result = await runCpuProfile(engine, DEFAULT_PROMPT, { maxTokens: 64, warmup: 10 });
      elCpuResult.classList.remove('hidden');
      elCpuResult.innerHTML = renderCpuResult(result);
      done({ ok: true });
      elCpuStatus.textContent =
        `done · forward p50 ${formatMs(result.report.forwardMedian_ms)} · ${result.report.forwardSamples_n} samples`;
    } catch (e) {
      done({ ok: false, error: (e as Error).message });
      elCpuStatus.textContent = `error: ${(e as Error).message}`;
    }
  });

  subscribe((s: EngineState) => {
    const canRun = s.status === 'ready';
    elGpuRun.disabled = !canRun;
    elCpuRun.disabled = !canRun;
  });
}

function renderGpuResult(r: ProfileRunResult): string {
  if (!r.ok) {
    return `
      <div class="empty">
        <div>gpu profile unavailable</div>
        <div class="muted" style="margin-top: var(--s-2);">${escapeHtml(r.error)}</div>
        <div class="mono-sm muted" style="margin-top: var(--s-3);">
          timestampQuerySupported: <b>${r.capability.timestampQuerySupported}</b> ·
          onAdapterUnrequested: <b>${r.capability.timestampQueryOnAdapterButUnrequested}</b> ·
          querySlots: <b>${r.capability.querySlots}</b>
        </div>
      </div>
    `;
  }

  const rows = [...r.report.perLabel].sort((a, b) => b.p50_ns - a.p50_ns);
  const bars = rows.map((x) => ({
    label: x.label,
    value: x.p50_ns,
    secondary: x.p95_ns,
  }));

  const chart = barChart({
    width: 780, height: 340,
    bars,
    yLabel: 'nanoseconds (p50 solid, p95 faded)',
    format: formatNs,
  });

  const table = rows.map((x) => `
    <tr>
      <td>${escapeHtml(x.label)}</td>
      <td class="num">${x.samples_n}</td>
      <td class="num">${formatNs(x.p50_ns)}</td>
      <td class="num">${formatNs(x.p95_ns)}</td>
      <td class="num">${formatNs(x.mean_ns)}</td>
      <td class="num">${formatNs(x.total_ns)}</td>
    </tr>
  `).join('');

  const derivedTps = r.tokensMeasured > 1 && r.wallMs > 0
    ? (r.tokensMeasured - 1) / (r.wallMs / 1000)
    : 0;

  return `
    <div class="mono-sm" style="margin-bottom: var(--s-3);">
      <b>forward p50</b> ${formatNs(r.report.forwardMedian_ns)} ·
      <b>samples</b> ${r.report.forwardSamples_n} ·
      <b>wall</b> ${formatMs(r.wallMs)} (${r.tokensMeasured} tok, ~${formatTps(derivedTps)} tps)
      ${r.report.overflow ? ' · <span class="warn">overflow</span>' : ''}
    </div>
    <div class="chart-wrap">${chart}</div>
    <details style="margin-top: var(--s-3);">
      <summary class="mono-sm muted" style="cursor: pointer;">per-label detail</summary>
      <table class="data" style="margin-top: var(--s-2);">
        <thead><tr><th>label</th><th class="num">samples</th><th class="num">p50</th><th class="num">p95</th><th class="num">mean</th><th class="num">total</th></tr></thead>
        <tbody>${table}</tbody>
      </table>
    </details>
  `;
}

function renderCpuResult(r: CpuProfileRunResult): string {
  const rows = [...r.report.perPhase].sort((a, b) => b.p50_ms - a.p50_ms);
  const bars = rows.map((x) => ({
    label: x.phase,
    value: x.p50_ms,
    secondary: x.p95_ms,
  }));

  const chart = barChart({
    width: 780, height: 340,
    bars,
    yLabel: 'milliseconds (p50 solid, p95 faded)',
    format: formatMs,
  });

  const table = rows.map((x) => `
    <tr>
      <td>${escapeHtml(x.phase)}</td>
      <td class="num">${x.samples_n}</td>
      <td class="num">${formatMs(x.p50_ms)}</td>
      <td class="num">${formatMs(x.p95_ms)}</td>
      <td class="num">${formatMs(x.mean_ms)}</td>
      <td class="num">${formatMs(x.total_ms)}</td>
    </tr>
  `).join('');

  return `
    <div class="mono-sm" style="margin-bottom: var(--s-3);">
      <b>forward p50</b> ${formatMs(r.report.forwardMedian_ms)} ·
      <b>samples</b> ${r.report.forwardSamples_n} ·
      <b>wb/fwd</b> ${r.report.writeBufferCountMedian} ·
      <b>encode/fwd</b> ${r.report.passEncodeCountMedian} ·
      <b>wall</b> ${formatMs(r.wallMs)} (${r.tokensMeasured} tok)
    </div>
    <div class="chart-wrap">${chart}</div>
    <details style="margin-top: var(--s-3);">
      <summary class="mono-sm muted" style="cursor: pointer;">per-phase detail</summary>
      <table class="data" style="margin-top: var(--s-2);">
        <thead><tr><th>phase</th><th class="num">samples</th><th class="num">p50</th><th class="num">p95</th><th class="num">mean</th><th class="num">total</th></tr></thead>
        <tbody>${table}</tbody>
      </table>
    </details>
  `;
}

function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
