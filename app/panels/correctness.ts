// Correctness panel — the crossLabDiff sweep.
//
// This is the flagship artifact of the project's methodology (per
// docs/methodology.md and CLAUDE.md §Development loop). Load a reference
// npz, capture hidden states at every layer boundary, compare against the
// external backend, and plot cosine similarity per layer with the project's
// gates (0.94 all-layers, 0.9978 logits) annotated as horizontal guides.

import {
  subscribe,
  getState,
  beginBusy,
  updateStatusLine,
  type EngineState,
} from '../lib/engine.js';
import {
  loadReferenceTensors,
  crossLabDiff,
} from '../../src/diagnostics/index.js';
import type {
  CrossLabDiffResult,
  LoadedReference,
  ReferenceTensors,
} from '../../src/diagnostics/index.js';
import { lineChart } from '../lib/svg.js';
import { formatCos, formatBytes, formatPct } from '../lib/format.js';
import { confirmDialog } from '../lib/dialog.js';

const DEFAULT_REF_URL = '/ref/smoke.npz';

interface SweepPoint {
  label: string;
  layerIndex: number; // -1 for embed, num_layers for logits
  cos: number;
  maxAbs: number;
  detail: CrossLabDiffResult;
}

export function mountCorrectnessPanel(root: HTMLElement): void {
  root.innerHTML = `
    <div class="card">
      <h3>reference tensors</h3>
      <p class="card-sub">
        Cross-lab diff compares this engine's hidden states against reference
        activations from an external backend (HuggingFace PyTorch forward-
        hooks, packed as an npz). The canonical <code>/ref/smoke.npz</code>
        ships in <code>public/ref/</code> and covers every capture point at
        position 14 of the chat-templated "Hello, how are you?" prompt.
      </p>
      <div class="field-row">
        <label>
          reference url
          <input type="url" id="ref-url" value="${DEFAULT_REF_URL}" />
        </label>
      </div>
      <div class="card-actions">
        <button type="button" id="ref-load" class="primary" disabled>load reference</button>
        <span class="mono-sm muted" id="ref-status">not loaded</span>
      </div>
    </div>

    <div class="card">
      <h3>cross-lab sweep</h3>
      <p class="card-sub">
        Capture per-layer hidden states and diff each against the reference.
        The <span class="good">green band</span> marks this project's
        correctness gate — cosine similarity ≥ <span class="mono">0.94</span>
        at every boundary. The logits gate is stricter at
        <span class="mono">0.9978</span>. Points dipping below either line
        are regressions, not rounding noise.
      </p>
      <div class="card-actions">
        <button type="button" id="sweep-run" class="primary" disabled>run full sweep</button>
        <span class="mono-sm muted" id="sweep-status"></span>
      </div>
      <div id="sweep-result" class="hidden" style="margin-top: var(--s-4);"></div>
    </div>

    <div class="card">
      <h3>capture-point detail</h3>
      <p class="card-sub">
        Click a point on the chart above to see its full diff: cosine
        similarity, concentration metrics (uniform vs single-dim
        divergence), top-K worst dimensions, and first-over-threshold
        index.
      </p>
      <div id="detail" class="empty">no point selected</div>
    </div>
  `;

  const elUrl = root.querySelector('#ref-url') as HTMLInputElement;
  const elLoad = root.querySelector('#ref-load') as HTMLButtonElement;
  const elRefStatus = root.querySelector('#ref-status') as HTMLElement;
  const elSweep = root.querySelector('#sweep-run') as HTMLButtonElement;
  const elSweepStatus = root.querySelector('#sweep-status') as HTMLElement;
  const elSweepResult = root.querySelector('#sweep-result') as HTMLElement;
  const elDetail = root.querySelector('#detail') as HTMLElement;

  let refTensors: ReferenceTensors | null = null;
  let refInfo: LoadedReference | null = null;
  let sweepPoints: SweepPoint[] = [];

  const updateRefStatus = () => {
    if (refInfo) {
      elRefStatus.innerHTML = `<span class="good">loaded</span> · ${refInfo.keys.length} entries · ${formatBytes(refInfo.sizeBytes)}`;
    } else {
      elRefStatus.textContent = 'not loaded';
    }
    const canSweep = getState().status === 'ready' && refTensors !== null;
    elSweep.disabled = !canSweep;
  };

  elLoad.addEventListener('click', async () => {
    const url = elUrl.value.trim() || DEFAULT_REF_URL;
    elRefStatus.innerHTML = '<span class="spinner"></span> fetching…';
    try {
      const { info, tensors } = await loadReferenceTensors(url);
      refTensors = tensors;
      refInfo = info;
      updateRefStatus();
    } catch (e) {
      elRefStatus.innerHTML = `<span class="bad">error:</span> ${escapeHtml((e as Error).message)}`;
    }
  });

  elSweep.addEventListener('click', async () => {
    const s = getState();
    if (s.status !== 'ready' || !refTensors) return;
    if (s.messages.length > 0) {
      const ok = await confirmDialog({
        title: 'Sweep will reset your chat',
        message: 'The capture flow resets the KV cache between points. Your conversation transcript will be cleared. Continue?',
        confirmLabel: 'Run sweep',
        danger: true,
      });
      if (!ok) return;
    }
    const engine = s.engine!;
    const numLayers = s.config!.num_layers;

    const done = beginBusy(`crossLabDiff sweep · ${numLayers + 2} points`);
    sweepPoints = [];
    try {
      const run = async (label: string, layerIndex: number, fn: () => Promise<CrossLabDiffResult>) => {
        updateStatusLine(`sweep · ${label} (${sweepPoints.length + 1}/${numLayers + 2})`);
        elSweepStatus.textContent = `${label}…`;
        const detail = await fn();
        sweepPoints.push({ label, layerIndex, cos: detail.cosineSimilarity, maxAbs: detail.maxAbsDiff, detail });
        renderSweep();
      };

      await run('embed', -1, () => crossLabDiff(engine, refTensors!, 'embed'));
      for (let L = 0; L < numLayers; L++) {
        await run(`afterLayer:${L}`, L, () => crossLabDiff(engine, refTensors!, { afterLayer: L }));
      }
      await run('logits', numLayers, () => crossLabDiff(engine, refTensors!, 'logits'));

      const minCos = Math.min(...sweepPoints.filter((p) => p.label !== 'logits').map((p) => p.cos));
      const logitsCos = sweepPoints[sweepPoints.length - 1].cos;
      const layerGatePass = minCos >= 0.94;
      const logitsGatePass = logitsCos >= 0.9978;
      const verdict = layerGatePass && logitsGatePass ? 'PASS' : 'FAIL';
      elSweepStatus.innerHTML =
        `<b class="${verdict === 'PASS' ? 'good' : 'bad'}">${verdict}</b> · ` +
        `min layer cos ${formatCos(minCos)} (gate 0.94) · logits ${formatCos(logitsCos)} (gate 0.9978)`;
      done({ ok: true });
    } catch (e) {
      done({ ok: false, error: (e as Error).message });
      elSweepStatus.innerHTML = `<span class="bad">error:</span> ${escapeHtml((e as Error).message)}`;
    }
  });

  const renderSweep = () => {
    if (sweepPoints.length === 0) { elSweepResult.classList.add('hidden'); return; }
    elSweepResult.classList.remove('hidden');
    const numLayers = getState().config?.num_layers ?? 34;
    const xMax = numLayers + 1;
    const yMin = Math.min(0.88, ...sweepPoints.map((p) => p.cos)) - 0.01;
    const yMax = 1.001;

    const chart = lineChart({
      width: 780, height: 320,
      xAxis: { min: -1, max: xMax, ticks: [-1, 0, Math.floor(numLayers / 2), numLayers - 1, numLayers] },
      yAxis: {
        min: yMin,
        max: yMax,
        ticks: [yMin, 0.94, 0.9978, 1].filter((v) => v >= yMin && v <= yMax),
      },
      series: [{
        label: 'cosine similarity',
        points: sweepPoints.map((p) => ({ x: p.layerIndex, y: p.cos })),
      }],
      yGuides: [
        { y: 0.94,   label: 'layers gate 0.94',  color: 'var(--warn)' },
        { y: 0.9978, label: 'logits gate 0.9978', color: 'var(--good)' },
      ],
      xLabel: 'layer index (−1=embed, N=logits)',
      yLabel: 'cos',
      markers: true,
    });

    elSweepResult.innerHTML = `
      <div class="chart-wrap" id="sweep-chart">${chart}</div>
      <table class="data" style="margin-top: var(--s-3);">
        <thead>
          <tr><th>point</th><th class="num">cos</th><th class="num">maxAbs</th><th class="num">concentration</th><th class="num">firstOver</th><th></th></tr>
        </thead>
        <tbody>${sweepPoints.map(rowHtml).join('')}</tbody>
      </table>
    `;

    // Wire row clicks + svg point clicks to the detail drawer.
    elSweepResult.querySelectorAll<HTMLElement>('[data-point-idx]').forEach((el) => {
      el.addEventListener('click', () => {
        const idx = Number(el.dataset.pointIdx);
        renderDetail(sweepPoints[idx]);
      });
    });
  };

  const rowHtml = (p: SweepPoint, i: number): string => {
    const cosClass = p.cos >= 0.9978 ? 'good' : p.cos >= 0.94 ? '' : 'bad';
    return `
      <tr data-point-idx="${i}" style="cursor: pointer;">
        <td>${escapeHtml(p.label)}</td>
        <td class="num ${cosClass}">${formatCos(p.cos)}</td>
        <td class="num">${p.maxAbs.toExponential(2)}</td>
        <td class="num">${formatPct(p.detail.concentration.top1PctFraction)}</td>
        <td class="num">${p.detail.firstOverIndex < 0 ? '—' : p.detail.firstOverIndex}</td>
        <td><span class="muted">detail →</span></td>
      </tr>
    `;
  };

  const renderDetail = (p: SweepPoint) => {
    const d = p.detail;
    const c = d.concentration;
    elDetail.classList.remove('empty');
    elDetail.innerHTML = `
      <h4 class="mono-sm" style="margin: 0 0 var(--s-2);">${escapeHtml(p.label)}</h4>
      <table class="data">
        <tbody>
          <tr><th>cosine similarity</th><td class="num">${formatCos(d.cosineSimilarity)}</td></tr>
          <tr><th>max abs diff</th><td class="num">${d.maxAbsDiff.toExponential(4)}</td></tr>
          <tr><th>mean abs diff</th><td class="num">${d.meanAbsDiff.toExponential(4)}</td></tr>
          <tr><th>stddev diff</th><td class="num">${d.stddevDiff.toExponential(4)}</td></tr>
          <tr><th>ref L2 norm</th><td class="num">${d.refNorm.toFixed(4)}</td></tr>
          <tr><th>our L2 norm</th><td class="num">${d.ourNorm.toFixed(4)}</td></tr>
          <tr><th>first over ${d.firstOverThreshold}</th><td class="num">${d.firstOverIndex < 0 ? 'none' : `idx ${d.firstOverIndex}`}</td></tr>
          <tr><th>over-threshold count</th><td class="num">${d.overCount} / ${d.length}</td></tr>
          <tr><th>top-1% mass</th><td class="num">${formatPct(c.top1PctFraction)}</td></tr>
          <tr><th>top-10% mass</th><td class="num">${formatPct(c.top10PctFraction)}</td></tr>
          <tr><th>participation ratio</th><td class="num">${c.participationRatio.toFixed(4)}</td></tr>
          <tr><th>effective dim count</th><td class="num">${c.effectiveDimCount.toFixed(1)}</td></tr>
        </tbody>
      </table>
      <details style="margin-top: var(--s-3);">
        <summary class="mono-sm muted" style="cursor: pointer;">top worst dims</summary>
        <table class="data" style="margin-top: var(--s-2);">
          <thead><tr><th class="num">idx</th><th class="num">ref</th><th class="num">ours</th><th class="num">diff</th></tr></thead>
          <tbody>
            ${d.topDiffs.map((t) => `
              <tr>
                <td class="num">${t.index}</td>
                <td class="num">${t.ref.toFixed(5)}</td>
                <td class="num">${t.ours.toFixed(5)}</td>
                <td class="num bad">${t.diff.toExponential(2)}</td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </details>
    `;
  };

  subscribe((s: EngineState) => {
    elLoad.disabled = s.status !== 'ready';
    elSweep.disabled = s.status !== 'ready' || refTensors === null;
    updateRefStatus();
  });
}

function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
