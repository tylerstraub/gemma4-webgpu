// Workbench entry point — agent-first diagnostic interface.
//
// All state and verbs live on `window.lab`. The DOM is a side effect for
// humans; Playwright MCP driving `browser_evaluate` is the primary user.
//
// Verbs (grouped):
//   Lifecycle:     init, dispose, state
//   Generation:    bench
//   Correctness:   loadReferenceTensors, capture, captureAll, crossLabDiff
//   Profiling:     profile, cpuProfile
//   Weight probe:  compareWeightSlice, compareWeightRow, probeLayerWeight,
//                  probePerLayerModelProjLayout
//   Contribution:  collectDiagnosticReport, formatDiagnosticReport,
//                  copyToClipboard

import { createGemmaEngine, PROFILES } from '../src/index.js';
import type {
  GemmaEngine,
  GemmaEngineOptions,
  GemmaConfig,
  ProgressInfo,
  TuningProfile,
  TuningOverride,
  LoadTimings,
} from '../src/index.js';
import {
  loadReferenceTensors as loadRefTensors,
  capture as captureOne,
  captureAll as captureMany,
  crossLabDiff as runCrossLabDiff,
  profile as runProfile,
  cpuProfile as runCpuProfile,
  compareWeightSlice as runCompareWeightSlice,
  compareWeightRow as runCompareWeightRow,
  probeLayerWeight as runProbeLayerWeight,
  probePerLayerModelProjLayout as runProbePerLayerModelProjLayout,
  formatDiagnosticReport,
} from '../src/diagnostics/index.js';
import type {
  CapturePointShorthand,
  CaptureSummary,
  CrossLabDiffResult,
  LoadedReference,
  ReferenceTensors,
  ProfileRunResult,
  CpuProfileRunResult,
  WeightCompareResult,
  LayerWeightProbe,
  PerLayerModelProjProbe,
  BenchMeasurement,
  DiagnosticReportInput,
  DeviceInfo,
  BrowserInfo,
} from '../src/diagnostics/index.js';

const DEFAULT_PROMPT =
  'List five concrete differences between a raven and a crow. Number them.';
const SHORT_PROMPT = 'Hello, how are you?';

interface BenchRun {
  label: string;
  tokens: number;
  firstMs: number | null;
  totalMs: number;
  decodeMs: number;
  tps: number;
  outHead: string;
}

interface BenchConfig {
  prompt: string;
  promptCharCount: number;
  maxTokens: number;
  runs: number;
  temperature: number;
  repPenalty: number;
}

interface BenchResult {
  config: BenchConfig;
  runs: BenchRun[];
  median: number;
  mean: number;
  min: number;
  max: number;
}

interface CaptureAllResult {
  prefillMs: number;
  capturesMs: number;
  tokens: number[];
  position: number;
  capturedTokenId: number;
  pointCount: number;
  captures: Record<string, CaptureSummary>;
}

interface CollectedDiagnostic {
  markdown: string;
  raw: DiagnosticReportInput;
}

interface Lab {
  engine: GemmaEngine | null;
  config: Readonly<GemmaConfig> | null;
  modelKey: string | null;
  loadProgress: ProgressInfo | null;
  reference: LoadedReference | null;

  init: (modelKey?: string, opts?: { contextLength?: number; tuning?: TuningOverride }) => Promise<Readonly<GemmaConfig>>;
  bench: (prompt?: string, opts?: { maxTokens?: number; runs?: number }) => Promise<BenchResult>;
  dispose: () => void;
  state: () => {
    loaded: boolean;
    modelKey: string | null;
    loadProgress: ProgressInfo | null;
    config: Readonly<GemmaConfig> | null;
    referenceLoaded: boolean;
    tuning: { profile: Readonly<TuningProfile>; reason: string } | null;
  };
  /** Enumerate available tuning profiles. Pure — does not require init. */
  tuningProfiles: () => Record<string, Readonly<TuningProfile>>;

  /**
   * End-to-end load phase breakdown from the last `init()`. Returns null
   * if the engine hasn't been initialized. Used to diagnose where load
   * wall-clock actually goes (network vs CPU dequant vs GPU upload vs
   * shader compile) rather than guessing from code inspection.
   */
  loadTimings: () => LoadTimings | null;

  loadReferenceTensors: (url: string) => Promise<LoadedReference>;
  capture: (
    prompt: string,
    stopAt: CapturePointShorthand,
    opts?: { applyChatTemplate?: boolean; includeValues?: boolean },
  ) => Promise<CaptureSummary>;
  captureAll: (
    prompt: string,
    points?: CapturePointShorthand[],
    opts?: { applyChatTemplate?: boolean; includeValues?: boolean },
  ) => Promise<CaptureAllResult>;
  crossLabDiff: (
    capturePoint: CapturePointShorthand,
    opts?: {
      prompt?: string;
      applyChatTemplate?: boolean;
      thresholdOver?: number;
      headN?: number;
      topK?: number;
    },
  ) => Promise<CrossLabDiffResult>;

  profile: (prompt?: string, opts?: { maxTokens?: number; warmup?: number }) => Promise<ProfileRunResult>;
  cpuProfile: (prompt?: string, opts?: { maxTokens?: number; warmup?: number }) => Promise<CpuProfileRunResult>;

  compareWeightSlice: (name: string, elemOffset: number, elemCount: number, opts?: { head?: number }) => Promise<WeightCompareResult>;
  compareWeightRow: (name: string, row: number, cols: number, opts?: { head?: number }) => Promise<WeightCompareResult>;
  probeLayerWeight: (name: string, opts?: { layers?: number[]; rows?: number[]; head?: number }) => Promise<LayerWeightProbe>;
  probePerLayerModelProjLayout: () => Promise<PerLayerModelProjProbe>;

  /**
   * One-shot: run the canonical diagnostic suite (adapter info + short bench
   * + long bench + crossLabDiff sweep) and return a structured report plus
   * the markdown rendering ready to paste into a device-report issue.
   */
  collectDiagnosticReport: (opts?: {
    includeShortBench?: boolean;
    includeLongBench?: boolean;
    includeCrossLabSweep?: boolean;
    referenceUrl?: string;
    notes?: string;
  }) => Promise<CollectedDiagnostic>;
  /** Format a pre-assembled report input as markdown. Pure, no side effects. */
  formatDiagnosticReport: (input: DiagnosticReportInput) => string;
  /** Write text to the system clipboard via `navigator.clipboard`. */
  copyToClipboard: (text: string) => Promise<{ ok: boolean; error?: string }>;
}

declare global {
  interface Window {
    lab: Lab;
  }
}

function setStatus(text: string): void {
  const el = document.getElementById('status');
  if (el) el.textContent = text;
}

async function runBench(
  engine: GemmaEngine,
  prompt: string,
  maxTokens: number,
  runs: number,
): Promise<BenchResult> {
  const results: BenchRun[] = [];
  const labels = ['warmup', ...Array.from({ length: runs }, (_, i) => `m${i + 1}`)];

  for (const label of labels) {
    engine.resetConversation();
    engine.addUserMessage(prompt);
    const gs = performance.now();
    let tokens = 0;
    let firstMs: number | null = null;
    let out = '';
    for await (const t of engine.generate({
      temperature: 0,
      maxTokens,
      repPenalty: 1.0,
    })) {
      tokens++;
      if (tokens === 1) firstMs = performance.now() - gs;
      out += t;
    }
    const totalMs = performance.now() - gs;
    const decodeMs = totalMs - (firstMs ?? 0);
    const tps = tokens > 1 ? (tokens - 1) / (decodeMs / 1000) : 0;
    results.push({ label, tokens, firstMs, totalMs, decodeMs, tps, outHead: out.slice(0, 180) });
    setStatus(`${label}: ${tokens} tok in ${totalMs.toFixed(0)}ms (${tps.toFixed(2)} tps)`);
  }

  const measured = results.filter((r) => r.label.startsWith('m')).map((r) => r.tps);
  measured.sort((a, b) => a - b);
  const median = measured[Math.floor(measured.length / 2)];
  const mean = measured.reduce((a, b) => a + b, 0) / measured.length;
  const min = measured[0];
  const max = measured[measured.length - 1];

  return {
    config: {
      prompt,
      promptCharCount: prompt.length,
      maxTokens,
      runs,
      temperature: 0,
      repPenalty: 1.0,
    },
    runs: results,
    median,
    mean,
    min,
    max,
  };
}

/** Convert a workbench bench result into the report-module BenchMeasurement shape. */
function benchToMeasurement(label: string, r: BenchResult): BenchMeasurement {
  return {
    label,
    config: r.config,
    median: r.median,
    mean: r.mean,
    min: r.min,
    max: r.max,
    tpsRuns: r.runs.filter((run) => run.label.startsWith('m')).map((run) => run.tps),
  };
}

async function collectDeviceInfo(): Promise<DeviceInfo | undefined> {
  if (!('gpu' in navigator)) return undefined;
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return undefined;
    const info = adapter.info;
    const features = Array.from(adapter.features);
    const limits: Partial<Record<string, number>> = {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
    };
    return {
      vendor: info.vendor ?? 'unknown',
      architecture: info.architecture ?? 'unknown',
      description: info.description,
      features,
      limits,
    };
  } catch {
    return undefined;
  }
}

function collectBrowserInfo(): BrowserInfo {
  const nav = navigator as Navigator & { deviceMemory?: number };
  return {
    userAgent: nav.userAgent,
    platform: nav.platform,
    hardwareConcurrency: nav.hardwareConcurrency,
    deviceMemoryGB: nav.deviceMemory,
  };
}

let refTensors: ReferenceTensors | null = null;

const lab: Lab = {
  engine: null,
  config: null,
  modelKey: null,
  loadProgress: null,
  reference: null,

  async init(modelKey = 'e2b', opts = {}) {
    if (lab.engine) throw new Error('engine already loaded — call dispose() first');
    lab.modelKey = modelKey;
    const engineOpts: GemmaEngineOptions = {
      model: modelKey,
      contextLength: opts.contextLength,
      tuning: opts.tuning ?? undefined,
      onProgress: (p) => {
        lab.loadProgress = p;
        const pct = p.total ? `${((p.loaded / p.total) * 100).toFixed(1)}%` : '';
        setStatus(`${p.status} ${pct}`);
      },
    };
    lab.engine = await createGemmaEngine(engineOpts);
    lab.config = lab.engine.config;
    setStatus(
      `ready: ${modelKey} · ${lab.config.num_layers}L hidden=${lab.config.hidden_size} · tuning=${lab.engine.tuning.id}`,
    );
    return lab.config;
  },

  async bench(prompt = DEFAULT_PROMPT, opts = {}) {
    if (!lab.engine) throw new Error('call init() first');
    return runBench(lab.engine, prompt, opts.maxTokens ?? 200, opts.runs ?? 5);
  },

  dispose() {
    lab.engine?.dispose();
    lab.engine = null;
    lab.config = null;
    lab.modelKey = null;
    lab.loadProgress = null;
    refTensors = null;
    lab.reference = null;
    setStatus('disposed');
  },

  state() {
    return {
      loaded: lab.engine !== null,
      modelKey: lab.modelKey,
      loadProgress: lab.loadProgress,
      config: lab.config,
      referenceLoaded: refTensors !== null,
      tuning: lab.engine
        ? { profile: lab.engine.tuning, reason: lab.engine.tuningReason }
        : null,
    };
  },

  tuningProfiles() {
    return PROFILES;
  },

  loadTimings() {
    return lab.engine?.loadTimings ?? null;
  },

  async loadReferenceTensors(url) {
    setStatus(`fetching reference npz ${url}…`);
    const { info, tensors } = await loadRefTensors(url);
    refTensors = tensors;
    lab.reference = info;
    setStatus(`reference loaded · ${info.keys.length} entries · ${(info.sizeBytes / 1024 / 1024).toFixed(2)} MB`);
    return info;
  },

  async capture(prompt, stopAt, opts = {}) {
    if (!lab.engine) throw new Error('call init() first');
    const summary = await captureOne(lab.engine, prompt, stopAt, opts);
    setStatus(`capture ${summary.stopAt} · len=${summary.length} · ‖x‖=${summary.norm.toFixed(3)}`);
    return summary;
  },

  async captureAll(prompt, points, opts = {}) {
    if (!lab.engine) throw new Error('call init() first');
    if (!lab.config) throw new Error('config missing');
    const result = await captureMany(lab.engine, lab.config, prompt, points, opts);
    setStatus(`captureAll done · ${result.pointCount} points · prefill=${result.prefillMs}ms · captures=${result.capturesMs}ms`);
    return result;
  },

  async crossLabDiff(capturePoint, opts = {}) {
    if (!lab.engine) throw new Error('call init() first');
    if (!refTensors) {
      throw new Error(
        'no reference loaded — call window.lab.loadReferenceTensors("/ref/smoke.npz") first',
      );
    }
    const result = await runCrossLabDiff(lab.engine, refTensors, capturePoint, opts);
    setStatus(
      `crossLabDiff ${result.capturePoint} · cos=${result.cosineSimilarity.toFixed(4)} · maxAbsDiff=${result.maxAbsDiff.toFixed(4)}`,
    );
    return result;
  },

  async profile(prompt = DEFAULT_PROMPT, opts = {}) {
    if (!lab.engine) throw new Error('call init() first');
    setStatus('profiling (GPU timestamp-query)…');
    const result = await runProfile(lab.engine, prompt, opts);
    if (result.ok) {
      setStatus(
        `profile · forward p50=${(result.report.forwardMedian_ns / 1e6).toFixed(3)}ms · samples=${result.report.forwardSamples_n}`,
      );
    } else {
      setStatus(`profile unavailable: ${result.error}`);
    }
    return result;
  },

  async cpuProfile(prompt = DEFAULT_PROMPT, opts = {}) {
    if (!lab.engine) throw new Error('call init() first');
    setStatus('profiling (CPU phases)…');
    const result = await runCpuProfile(lab.engine, prompt, opts);
    setStatus(
      `cpuProfile · forward p50=${result.report.forwardMedian_ms.toFixed(3)}ms · wb/fwd=${result.report.writeBufferCountMedian}`,
    );
    return result;
  },

  async compareWeightSlice(name, elemOffset, elemCount, opts = {}) {
    if (!lab.engine) throw new Error('call init() first');
    return runCompareWeightSlice(lab.engine, name, elemOffset, elemCount, opts.head ?? 16);
  },

  async compareWeightRow(name, row, cols, opts = {}) {
    if (!lab.engine) throw new Error('call init() first');
    return runCompareWeightRow(lab.engine, name, row, cols, opts.head ?? 16);
  },

  async probeLayerWeight(name, opts = {}) {
    if (!lab.engine) throw new Error('call init() first');
    if (!lab.config) throw new Error('config missing');
    const result = await runProbeLayerWeight(lab.engine, lab.config, name, opts);
    setStatus(`probeLayerWeight ${name} · ${result.verdict}`);
    return result;
  },

  async probePerLayerModelProjLayout() {
    if (!lab.engine) throw new Error('call init() first');
    if (!lab.config) throw new Error('config missing');
    const result = await runProbePerLayerModelProjLayout(lab.engine, lab.config);
    setStatus(`probePerLayerModelProjLayout · ${result.verdict}`);
    return result;
  },

  async collectDiagnosticReport(opts = {}) {
    if (!lab.engine || !lab.config) throw new Error('call init() first');
    const includeShort = opts.includeShortBench ?? true;
    const includeLong = opts.includeLongBench ?? true;
    const includeSweep = opts.includeCrossLabSweep ?? true;
    const referenceUrl = opts.referenceUrl ?? '/ref/smoke.npz';

    setStatus('collecting device + browser info…');
    const [device, browser] = await Promise.all([collectDeviceInfo(), Promise.resolve(collectBrowserInfo())]);

    const bench: BenchMeasurement[] = [];
    if (includeShort) {
      setStatus('benchmarking (short config)…');
      const r = await runBench(lab.engine, SHORT_PROMPT, 64, 5);
      bench.push(benchToMeasurement('short (Hello, how are you?, maxTokens=64)', r));
    }
    if (includeLong) {
      setStatus('benchmarking (long config)…');
      const r = await runBench(lab.engine, DEFAULT_PROMPT, 200, 5);
      bench.push(benchToMeasurement('long (default, maxTokens=200)', r));
    }

    let crossLabSweep: { point: string; cos: number; maxAbs: number }[] | undefined;
    let crossLabDetail: Parameters<typeof formatDiagnosticReport>[0]['crossLabDetail'];
    if (includeSweep) {
      try {
        if (!refTensors) {
          setStatus(`loading reference tensors from ${referenceUrl}…`);
          const { info, tensors } = await loadRefTensors(referenceUrl);
          refTensors = tensors;
          lab.reference = info;
        }
        setStatus('running crossLabDiff sweep…');
        const sweep: { point: string; cos: number; maxAbs: number }[] = [];
        const embedR = await runCrossLabDiff(lab.engine, refTensors, 'embed');
        sweep.push({ point: embedR.capturePoint, cos: embedR.cosineSimilarity, maxAbs: embedR.maxAbsDiff });
        for (let L = 0; L < lab.config.num_layers; L++) {
          const r = await runCrossLabDiff(lab.engine, refTensors, { afterLayer: L });
          sweep.push({ point: r.capturePoint, cos: r.cosineSimilarity, maxAbs: r.maxAbsDiff });
        }
        const logitsR = await runCrossLabDiff(lab.engine, refTensors, 'logits');
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
        setStatus(`crossLabDiff sweep skipped: ${(e as Error).message}`);
      }
    }

    const profile = lab.engine.tuning;
    const input: DiagnosticReportInput = {
      repoName: 'gemma4-webgpu',
      generatedAt: new Date().toISOString(),
      device,
      browser,
      engineConfig: {
        num_layers: lab.config.num_layers,
        hidden_size: lab.config.hidden_size,
        vocab_size: lab.config.vocab_size,
        context_length: lab.config.context_length,
        per_layer_input_dim: lab.config.per_layer_input_dim,
      },
      tuning: {
        id: profile.id,
        description: profile.description,
        verified: profile.verified,
        reason: lab.engine.tuningReason,
        rowsPerWorkgroup_ffnGateUp:
          profile.matmul.rowsPerWorkgroupByKernel?.['ffn.linearGateUp']
          ?? profile.matmul.defaultRowsPerWorkgroup,
        decodeDepth: profile.pipeline.decodeDepth,
        greedyFastPath: profile.pipeline.greedyFastPath,
      },
      bench,
      crossLabSweep,
      crossLabDetail,
      notes: opts.notes,
    };

    const markdown = formatDiagnosticReport(input);
    setStatus('diagnostic report ready — copy + paste into a new GitHub issue');
    return { markdown, raw: input };
  },

  formatDiagnosticReport(input) {
    return formatDiagnosticReport(input);
  },

  async copyToClipboard(text) {
    try {
      await navigator.clipboard.writeText(text);
      setStatus(`copied ${text.length} chars to clipboard`);
      return { ok: true };
    } catch (e) {
      return { ok: false, error: (e as Error).message };
    }
  },
};

window.lab = lab;

// Tiny UI wiring — the two buttons in workbench.html. No framework; the
// DOM is a side effect for humans. The primary consumer is Playwright
// MCP via `browser_evaluate(() => window.lab.*)`.
function wireButtons(): void {
  const initBtn = document.getElementById('btn-init');
  if (initBtn) {
    initBtn.addEventListener('click', async () => {
      try {
        await lab.init('e2b');
      } catch (e) {
        setStatus(`init failed: ${(e as Error).message}`);
      }
    });
  }
  const reportBtn = document.getElementById('btn-collect-report');
  if (reportBtn) {
    reportBtn.addEventListener('click', async () => {
      try {
        const { markdown } = await lab.collectDiagnosticReport();
        const pre = document.getElementById('report');
        if (pre) pre.textContent = markdown;
        const copyResult = await lab.copyToClipboard(markdown);
        setStatus(
          copyResult.ok
            ? `report ready — copied ${markdown.length} chars to clipboard`
            : `report ready but clipboard write failed: ${copyResult.error} — copy the <pre> below manually`,
        );
      } catch (e) {
        setStatus(`collectDiagnosticReport failed: ${(e as Error).message}`);
      }
    });
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', wireButtons);
} else {
  wireButtons();
}

setStatus('ready — call lab.init("e2b") to load the model');
