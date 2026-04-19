// Weight probes panel — advanced, contributor-facing.
//
// The verbs here diagnose GGUF-upload layout bugs. Expected audience is
// someone porting the engine to another quant or debugging a suspect
// RMSNorm/matmul hookup. Surfaces are kept close to the raw diagnostic
// API because the data matters more than the chrome.

import {
  subscribe,
  getState,
  beginBusy,
  updateStatusLine,
  type EngineState,
} from '../lib/engine.js';
import {
  compareWeightSlice,
  compareWeightRow,
  probeLayerWeight,
  probePerLayerModelProjLayout,
  GGUF_TYPE_NAMES,
} from '../../src/diagnostics/index.js';
import type {
  WeightCompareResult,
  LayerWeightProbe,
  PerLayerModelProjProbe,
} from '../../src/diagnostics/index.js';

export function mountWeightProbesPanel(root: HTMLElement): void {
  root.innerHTML = `
    <div class="card">
      <h3>weight probes · contributor-facing</h3>
      <p class="card-sub">
        Read a slice from the GPU F16 upload and from the raw GGUF source,
        diff them element-wise. These verbs catch routing / transpose /
        dequant bugs that numerical sweeps at attention boundaries miss.
        Useful for porting to new quantizations and debugging upload-path
        changes.
      </p>
    </div>

    <details class="card">
      <summary>compareWeightSlice · arbitrary slice</summary>
      <p class="card-sub" style="margin-top: var(--s-3);">
        Compare an arbitrary element-range of any named tensor.
      </p>
      <div class="field-row">
        <label>tensor name<input type="text" id="cws-name" value="token_embd.weight" /></label>
        <label>elem offset<input type="number" id="cws-off" value="0" min="0" /></label>
        <label>elem count<input type="number" id="cws-count" value="128" min="1" /></label>
        <label>head<input type="number" id="cws-head" value="16" min="1" max="128" /></label>
      </div>
      <div class="card-actions">
        <button type="button" class="primary" id="cws-run" disabled>run</button>
        <span class="mono-sm muted" id="cws-status"></span>
      </div>
      <div id="cws-result" style="margin-top: var(--s-3);"></div>
    </details>

    <details class="card">
      <summary>compareWeightRow · single row</summary>
      <p class="card-sub" style="margin-top: var(--s-3);">
        Diff one named-tensor row. Useful for 2D layouts where you want to
        verify a specific row rather than a flat slice.
      </p>
      <div class="field-row">
        <label>tensor name<input type="text" id="cwr-name" value="token_embd.weight" /></label>
        <label>row<input type="number" id="cwr-row" value="0" min="0" /></label>
        <label>cols<input type="number" id="cwr-cols" value="2560" min="1" /></label>
        <label>head<input type="number" id="cwr-head" value="16" min="1" max="128" /></label>
      </div>
      <div class="card-actions">
        <button type="button" class="primary" id="cwr-run" disabled>run</button>
        <span class="mono-sm muted" id="cwr-status"></span>
      </div>
      <div id="cwr-result" style="margin-top: var(--s-3);"></div>
    </details>

    <details class="card">
      <summary>probeLayerWeight · layer × row sweep</summary>
      <p class="card-sub" style="margin-top: var(--s-3);">
        Sweep a named tensor across a set of layers and rows, flag any
        target whose maxAbsDiff exceeds F16 upload noise (≥ 0.02).
      </p>
      <div class="field-row">
        <label>tensor name<input type="text" id="plw-name" value="attn_q.weight" /></label>
        <label>layers (comma)<input type="text" id="plw-layers" value="0,8,16,24,33" /></label>
        <label>rows (comma)<input type="text" id="plw-rows" value="0,1,63,127" /></label>
        <label>head<input type="number" id="plw-head" value="16" min="1" max="128" /></label>
      </div>
      <div class="card-actions">
        <button type="button" class="primary" id="plw-run" disabled>run</button>
        <span class="mono-sm muted" id="plw-status"></span>
      </div>
      <div id="plw-result" style="margin-top: var(--s-3);"></div>
    </details>

    <details class="card">
      <summary>probePerLayerModelProjLayout · layout hypothesis test</summary>
      <p class="card-sub" style="margin-top: var(--s-3);">
        Run four flatten-hypothesis probes against <code>per_layer_model_proj</code>
        to identify which memory layout the GGUF encoder used. Prints a
        verdict for each hypothesis.
      </p>
      <div class="card-actions">
        <button type="button" class="primary" id="pmp-run" disabled>run</button>
        <span class="mono-sm muted" id="pmp-status"></span>
      </div>
      <div id="pmp-result" style="margin-top: var(--s-3);"></div>
    </details>
  `;

  // compareWeightSlice
  wireAction(root, 'cws-run', 'cws-status', 'cws-result', async () => {
    const engine = getState().engine!;
    const name = (root.querySelector('#cws-name') as HTMLInputElement).value.trim();
    const off = Number((root.querySelector('#cws-off') as HTMLInputElement).value);
    const count = Number((root.querySelector('#cws-count') as HTMLInputElement).value);
    const head = Number((root.querySelector('#cws-head') as HTMLInputElement).value);
    updateStatusLine(`compareWeightSlice ${name}[${off}..+${count}]`);
    const result = await compareWeightSlice(engine, name, off, count, head);
    return renderCompare(result);
  });

  // compareWeightRow
  wireAction(root, 'cwr-run', 'cwr-status', 'cwr-result', async () => {
    const engine = getState().engine!;
    const name = (root.querySelector('#cwr-name') as HTMLInputElement).value.trim();
    const row = Number((root.querySelector('#cwr-row') as HTMLInputElement).value);
    const cols = Number((root.querySelector('#cwr-cols') as HTMLInputElement).value);
    const head = Number((root.querySelector('#cwr-head') as HTMLInputElement).value);
    updateStatusLine(`compareWeightRow ${name} row=${row}`);
    const result = await compareWeightRow(engine, name, row, cols, head);
    return renderCompare(result);
  });

  // probeLayerWeight
  wireAction(root, 'plw-run', 'plw-status', 'plw-result', async () => {
    const state = getState();
    const engine = state.engine!;
    const config = state.config!;
    const name = (root.querySelector('#plw-name') as HTMLInputElement).value.trim();
    const layers = parseIntList((root.querySelector('#plw-layers') as HTMLInputElement).value);
    const rows = parseIntList((root.querySelector('#plw-rows') as HTMLInputElement).value);
    const head = Number((root.querySelector('#plw-head') as HTMLInputElement).value);
    updateStatusLine(`probeLayerWeight ${name}`);
    const result = await probeLayerWeight(engine, config, name, { layers, rows, head });
    return renderLayerProbe(result);
  });

  // probePerLayerModelProjLayout
  wireAction(root, 'pmp-run', 'pmp-status', 'pmp-result', async () => {
    const state = getState();
    const engine = state.engine!;
    const config = state.config!;
    updateStatusLine('probePerLayerModelProjLayout');
    const result = await probePerLayerModelProjLayout(engine, config);
    return renderPerLayerProbe(result);
  });

  subscribe((s: EngineState) => {
    const canRun = s.status === 'ready';
    root.querySelectorAll<HTMLButtonElement>('button.primary').forEach((b) => { b.disabled = !canRun; });
  });
}

function wireAction(
  root: HTMLElement,
  btnId: string,
  statusId: string,
  resultId: string,
  fn: () => Promise<string>,
): void {
  const btn = root.querySelector(`#${btnId}`) as HTMLButtonElement;
  const status = root.querySelector(`#${statusId}`) as HTMLElement;
  const result = root.querySelector(`#${resultId}`) as HTMLElement;
  btn.addEventListener('click', async () => {
    const s = getState();
    if (s.status !== 'ready') return;
    const done = beginBusy('weight probe…');
    status.textContent = 'running…';
    try {
      const html = await fn();
      result.innerHTML = html;
      status.innerHTML = '<span class="good">done</span>';
      done({ ok: true });
    } catch (e) {
      status.innerHTML = `<span class="bad">error:</span> ${escapeHtml((e as Error).message)}`;
      done({ ok: false, error: (e as Error).message });
    }
  });
}

function parseIntList(s: string): number[] {
  return s.split(',').map((x) => x.trim()).filter(Boolean).map(Number).filter((n) => Number.isFinite(n));
}

function renderCompare(r: WeightCompareResult): string {
  const dims = Array.isArray(r.dims) ? r.dims.map((d) => typeof d === 'bigint' ? d.toString() : String(d)).join('×') : String(r.dims);
  const verdictClass =
    r.stats.maxAbsDiff < 0.02 ? 'good' :
    r.stats.maxAbsDiff < 0.1 ? 'warn' : 'bad';
  const typeName = GGUF_TYPE_NAMES[r.sourceType] ?? `type${r.sourceType}`;
  const topRows = r.topDiffIndices.map((t) => `
    <tr>
      <td class="num">${t.index}</td>
      <td class="num">${t.gpu.toFixed(6)}</td>
      <td class="num">${t.src.toFixed(6)}</td>
      <td class="num ${t.diff > 0.02 ? 'bad' : 'muted'}">${t.diff.toExponential(2)}</td>
    </tr>
  `).join('');
  return `
    <div class="mono-sm" style="margin-bottom: var(--s-2);">
      <b>${escapeHtml(r.name)}</b> · type <b>${typeName}</b> · dims <b>${dims}</b> · elemOffset <b>${r.elemOffset}</b> · count <b>${r.elemCount}</b>
    </div>
    <table class="data">
      <tbody>
        <tr><th>max abs diff</th><td class="num ${verdictClass}">${r.stats.maxAbsDiff.toExponential(3)}</td></tr>
        <tr><th>mean abs diff</th><td class="num">${r.stats.meanAbsDiff.toExponential(3)}</td></tr>
        <tr><th>stddev abs diff</th><td class="num">${r.stats.stddevAbsDiff.toExponential(3)}</td></tr>
        <tr><th>exact match</th><td class="num">${r.stats.exactMatchCount} / ${r.elemCount}</td></tr>
        <tr><th>within F16 noise</th><td class="num">${r.stats.withinF16NoiseCount} / ${r.elemCount}</td></tr>
        <tr><th>gross mismatch (&gt;0.01)</th><td class="num ${r.stats.grossMismatchCount > 0 ? 'bad' : ''}">${r.stats.grossMismatchCount}</td></tr>
        <tr><th>NaN (gpu / src)</th><td class="num">${r.stats.nanCountGpu} / ${r.stats.nanCountSrc}</td></tr>
        <tr><th>Inf (gpu / src)</th><td class="num">${r.stats.infCountGpu} / ${r.stats.infCountSrc}</td></tr>
      </tbody>
    </table>
    ${topRows ? `
    <details style="margin-top: var(--s-3);">
      <summary class="mono-sm muted" style="cursor: pointer;">top diffs</summary>
      <table class="data" style="margin-top: var(--s-2);">
        <thead><tr><th class="num">idx</th><th class="num">gpu</th><th class="num">src</th><th class="num">diff</th></tr></thead>
        <tbody>${topRows}</tbody>
      </table>
    </details>
    ` : ''}
  `;
}

function renderLayerProbe(r: LayerWeightProbe): string {
  const dims = Array.isArray(r.dims) ? r.dims.map((d) => typeof d === 'bigint' ? d.toString() : String(d)).join('×') : String(r.dims);
  const typeName = GGUF_TYPE_NAMES[r.sourceType] ?? `type${r.sourceType}`;
  const verdictClass = r.verdict === 'OK' ? 'good' : 'bad';
  const rows = r.targets.map((t) => `
    <tr>
      <td class="num">${t.layer ?? 'tied'}</td>
      <td class="num">${t.row}</td>
      <td class="num">${t.elemOffset}</td>
      <td class="num">${t.elemCount}</td>
      <td class="num">${t.maxAbsDiff.toExponential(2)}</td>
      <td class="num ${t.pass ? 'good' : 'bad'}">${t.pass ? 'pass' : 'FAIL'}</td>
    </tr>
  `).join('');
  return `
    <div class="mono-sm" style="margin-bottom: var(--s-2);">
      <b>${escapeHtml(r.name)}</b> · type <b>${typeName}</b> · dims <b>${dims}</b> · verdict <b class="${verdictClass}">${r.verdict}</b>
    </div>
    <table class="data">
      <thead><tr><th class="num">layer</th><th class="num">row</th><th class="num">elemOffset</th><th class="num">count</th><th class="num">maxAbsDiff</th><th>pass</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function renderPerLayerProbe(r: PerLayerModelProjProbe): string {
  const dims = Array.isArray(r.dims) ? r.dims.map((d) => typeof d === 'bigint' ? d.toString() : String(d)).join('×') : String(r.dims);
  const probes = r.probes.map((pr) => {
    const hypRows = Object.entries(pr.hypotheses).map(([name, h]) => `
      <tr>
        <td>${escapeHtml(name)}</td>
        <td class="muted">${escapeHtml(h.description)}</td>
        <td class="num">${h.sampleDiff.maxAbsDiff.toExponential(2)}</td>
        <td class="num">${h.sampleDiff.exactMatchCount} / ${h.elemCount}</td>
      </tr>
    `).join('');
    return `
      <details style="margin-top: var(--s-3);">
        <summary class="mono-sm" style="cursor: pointer;">layer ${pr.layer}, pos ${pr.pos}</summary>
        <table class="data" style="margin-top: var(--s-2);">
          <thead><tr><th>hypothesis</th><th>desc</th><th class="num">maxAbsDiff</th><th class="num">exact</th></tr></thead>
          <tbody>${hypRows}</tbody>
        </table>
      </details>
    `;
  }).join('');
  const verdictClass = /OK|MATCH|PASS/i.test(r.verdict) ? 'good' : 'warn';
  return `
    <div class="mono-sm" style="margin-bottom: var(--s-2);">
      dims <b>${dims}</b> · numElems <b>${r.numElems}</b> · assumption <b>${r.assumption.M}×${r.assumption.N}</b> ·
      verdict <b class="${verdictClass}">${escapeHtml(r.verdict)}</b>
    </div>
    ${probes}
  `;
}

function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
