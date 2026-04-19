// Device report panel — the community-contribution flow.
//
// Wraps `collectDiagnosticReport` and its moving parts (device info, short
// + long bench, per-layer crossLabDiff sweep). The user clicks one button;
// we run the full suite and produce the markdown the device-report issue
// template expects. Copy-to-clipboard works from anywhere; the "open issue
// with this report" link is a prefill-via-URL convenience that degrades
// gracefully when the body is too large.

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
  formatDiagnosticReport,
} from '../../src/diagnostics/index.js';
import type {
  BenchMeasurement,
  DiagnosticReportInput,
  ReferenceTensors,
} from '../../src/diagnostics/index.js';
import {
  runBench,
  benchToMeasurement,
  collectDeviceInfo,
  collectBrowserInfo,
} from '../lib/diagnostics.js';
import { renderMarkdown } from '../lib/markdown.js';
import { buildDeviceReportIssueUrl, blankIssueUrl, MAX_PREFILL_BODY, REPO_SLUG } from '../lib/github.js';
import { confirmDialog } from '../lib/dialog.js';
import { formatBytes } from '../lib/format.js';

const SHORT_PROMPT = 'Hello, how are you?';
const LONG_PROMPT = 'List five concrete differences between a raven and a crow. Number them.';
const DEFAULT_REF_URL = '/ref/smoke.npz';

export function mountDeviceReportPanel(root: HTMLElement): void {
  root.innerHTML = `
    <div class="card">
      <h3>device report</h3>
      <p class="card-sub">
        Run the full diagnostic suite and produce a markdown payload you can
        paste into a GitHub <code>device-report</code> issue. This is how
        hardware data from real devices rolls into the project's calibration
        and tuning-profile registry. Takes ~90 s — warmups + 5 bench runs ×
        two prompt configs + a 34-layer cross-lab sweep.
      </p>
      <div class="field-row">
        <label>
          reference url
          <input type="url" id="rep-ref-url" value="${DEFAULT_REF_URL}" />
        </label>
      </div>
      <details style="margin: var(--s-3) 0;">
        <summary class="mono-sm muted" style="cursor: pointer;">optional notes</summary>
        <label style="margin-top: var(--s-2); display: block;">
          free-form notes (appended to the report)
          <textarea id="rep-notes" rows="3" placeholder="e.g. Chrome Canary 134, fans at 60%, …"></textarea>
        </label>
      </details>
      <div class="card-actions">
        <button type="button" id="rep-run" class="primary" disabled>run full diagnostic (~90s)</button>
        <span class="mono-sm muted" id="rep-progress"></span>
      </div>
    </div>

    <div id="rep-result" class="hidden">
      <div class="card">
        <h3>report preview · rendered</h3>
        <div class="msg-text" id="rep-rendered" style="background: var(--bg); padding: var(--s-4); border: 1px solid var(--border);"></div>
      </div>

      <div class="card">
        <h3>report markdown · raw</h3>
        <pre id="rep-raw" style="white-space: pre-wrap; max-height: 24rem; overflow: auto; background: var(--bg); padding: var(--s-3); border: 1px solid var(--border); font-size: 0.78rem;"></pre>
        <div class="mono-sm muted" id="rep-size"></div>
        <div class="card-actions">
          <button type="button" class="primary" id="rep-copy">copy markdown</button>
          <a class="btn" id="rep-open-issue" target="_blank" rel="noopener">open issue with this report ↗</a>
          <span class="mono-sm muted" id="rep-open-fallback"></span>
        </div>
      </div>
    </div>
  `;

  const elRefUrl = root.querySelector('#rep-ref-url') as HTMLInputElement;
  const elNotes = root.querySelector('#rep-notes') as HTMLTextAreaElement;
  const elRun = root.querySelector('#rep-run') as HTMLButtonElement;
  const elProgress = root.querySelector('#rep-progress') as HTMLElement;
  const elResult = root.querySelector('#rep-result') as HTMLElement;
  const elRendered = root.querySelector('#rep-rendered') as HTMLElement;
  const elRaw = root.querySelector('#rep-raw') as HTMLElement;
  const elSize = root.querySelector('#rep-size') as HTMLElement;
  const elCopy = root.querySelector('#rep-copy') as HTMLButtonElement;
  const elOpen = root.querySelector('#rep-open-issue') as HTMLAnchorElement;
  const elOpenFallback = root.querySelector('#rep-open-fallback') as HTMLElement;

  let latestMarkdown: string | null = null;

  elRun.addEventListener('click', async () => {
    const s = getState();
    if (s.status !== 'ready') return;
    if (s.messages.length > 0) {
      const ok = await confirmDialog({
        title: 'Diagnostic will reset your chat',
        message:
          'The diagnostic suite runs benches and captures hidden states — both reset the KV cache between runs. Your conversation transcript will be cleared. Continue?',
        confirmLabel: 'Run diagnostic',
        danger: true,
      });
      if (!ok) return;
    }
    await runFullDiagnostic();
  });

  async function runFullDiagnostic(): Promise<void> {
    const s = getState();
    const engine = s.engine!;
    const config = s.config!;
    const tuning = s.tuning!;
    const tuningReason = s.tuningReason ?? 'auto';
    const refUrl = elRefUrl.value.trim() || DEFAULT_REF_URL;
    const notes = elNotes.value.trim() || undefined;

    const done = beginBusy('collecting diagnostic report…');
    elResult.classList.add('hidden');
    elProgress.textContent = '';

    try {
      updateStatusLine('device + browser info');
      elProgress.textContent = 'step 1/5 · device + browser info';
      const [device, browser] = await Promise.all([collectDeviceInfo(), Promise.resolve(collectBrowserInfo())]);

      updateStatusLine('bench · short (maxTokens=64)');
      elProgress.textContent = 'step 2/5 · short bench';
      const shortRaw = await runBench(engine, SHORT_PROMPT, 64, 5, (p) => {
        elProgress.textContent = `step 2/5 · short bench · ${p.label} (${p.index + 1}/${p.total})`;
      });
      const shortMeasurement = benchToMeasurement('short (Hello, how are you?, maxTokens=64)', shortRaw);

      updateStatusLine('bench · long (maxTokens=200)');
      elProgress.textContent = 'step 3/5 · long bench';
      const longRaw = await runBench(engine, LONG_PROMPT, 200, 5, (p) => {
        elProgress.textContent = `step 3/5 · long bench · ${p.label} (${p.index + 1}/${p.total})`;
      });
      const longMeasurement = benchToMeasurement('long (default, maxTokens=200)', longRaw);

      const bench: BenchMeasurement[] = [shortMeasurement, longMeasurement];

      let crossLabSweep: { point: string; cos: number; maxAbs: number }[] | undefined;
      let crossLabDetail: DiagnosticReportInput['crossLabDetail'];
      try {
        updateStatusLine('fetching reference tensors');
        elProgress.textContent = `step 4/5 · loading reference tensors from ${refUrl}`;
        const { tensors } = await loadReferenceTensors(refUrl);
        const refTensors: ReferenceTensors = tensors;

        elProgress.textContent = `step 5/5 · cross-lab sweep (0/${config.num_layers + 2})`;
        const sweep: { point: string; cos: number; maxAbs: number }[] = [];
        const embedR = await crossLabDiff(engine, refTensors, 'embed');
        sweep.push({ point: embedR.capturePoint, cos: embedR.cosineSimilarity, maxAbs: embedR.maxAbsDiff });
        for (let L = 0; L < config.num_layers; L++) {
          elProgress.textContent = `step 5/5 · cross-lab sweep (${L + 1}/${config.num_layers + 2})`;
          const r = await crossLabDiff(engine, refTensors, { afterLayer: L });
          sweep.push({ point: r.capturePoint, cos: r.cosineSimilarity, maxAbs: r.maxAbsDiff });
        }
        const logitsR = await crossLabDiff(engine, refTensors, 'logits');
        sweep.push({ point: logitsR.capturePoint, cos: logitsR.cosineSimilarity, maxAbs: logitsR.maxAbsDiff });
        crossLabSweep = sweep;
        crossLabDetail = {
          capturePoint: logitsR.capturePoint,
          cosineSimilarity: logitsR.cosineSimilarity,
          maxAbsDiff: logitsR.maxAbsDiff,
          concentration: logitsR.concentration,
          refNorm: logitsR.refNorm,
          ourNorm: logitsR.ourNorm,
        };
      } catch (e) {
        elProgress.textContent = `sweep skipped: ${(e as Error).message}`;
      }

      const input: DiagnosticReportInput = {
        repoName: 'gemma4-webgpu',
        generatedAt: new Date().toISOString(),
        device,
        browser,
        engineConfig: {
          num_layers: config.num_layers,
          hidden_size: config.hidden_size,
          vocab_size: config.vocab_size,
          context_length: config.context_length,
          per_layer_input_dim: config.per_layer_input_dim,
        },
        tuning: {
          id: tuning.id,
          description: tuning.description,
          verified: tuning.verified,
          reason: tuningReason,
          rowsPerWorkgroup_ffnGateUp:
            tuning.matmul.rowsPerWorkgroupByKernel?.['ffn.linearGateUp']
            ?? tuning.matmul.defaultRowsPerWorkgroup,
          decodeDepth: tuning.pipeline.decodeDepth,
          greedyFastPath: tuning.pipeline.greedyFastPath,
        },
        bench,
        crossLabSweep,
        crossLabDetail,
        notes,
      };

      const markdown = formatDiagnosticReport(input);
      latestMarkdown = markdown;
      renderPreview(markdown);
      elResult.classList.remove('hidden');
      elProgress.textContent = 'report ready';
      updateStatusLine(`report ready · ${formatBytes(markdown.length)} markdown`);
      done({ ok: true });
    } catch (e) {
      done({ ok: false, error: (e as Error).message });
      elProgress.textContent = `error: ${(e as Error).message}`;
    }
  }

  function renderPreview(md: string): void {
    elRendered.innerHTML = renderMarkdown(md);
    elRaw.textContent = md;

    const { url, truncated, encodedLength } = buildDeviceReportIssueUrl(md);
    elSize.innerHTML =
      `markdown: <b>${formatBytes(md.length)}</b> ` +
      `· url-encoded: <b>${formatBytes(encodedLength)}</b> ` +
      `(prefill ceiling ${formatBytes(MAX_PREFILL_BODY)})`;

    if (truncated || !url) {
      elOpen.classList.add('hidden');
      elOpenFallback.classList.remove('hidden');
      elOpenFallback.innerHTML = `<span class="warn">report too large to prefill —</span> <a href="${blankIssueUrl()}" target="_blank" rel="noopener">open a blank device-report issue</a> and paste.`;
    } else {
      elOpen.classList.remove('hidden');
      elOpen.href = url;
      elOpenFallback.classList.add('hidden');
    }
  }

  elCopy.addEventListener('click', async () => {
    if (!latestMarkdown) return;
    try {
      await navigator.clipboard.writeText(latestMarkdown);
      elCopy.textContent = 'copied ✓';
      setTimeout(() => { elCopy.textContent = 'copy markdown'; }, 1500);
    } catch (e) {
      elCopy.textContent = `clipboard failed: ${(e as Error).message}`;
    }
  });

  subscribe((s: EngineState) => {
    elRun.disabled = s.status !== 'ready';
  });
}
