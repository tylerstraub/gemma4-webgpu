// Shared diagnostics helpers lifted from workbench/main.ts so the human-
// panel versions don't duplicate the bench/device-info logic. These are
// stateless utilities that operate against a `GemmaEngine` instance —
// nothing subscribed to the app store lives here.

import type { GemmaEngine } from '../../src/index.js';
import type {
  BenchMeasurement,
  DeviceInfo,
  BrowserInfo,
} from '../../src/diagnostics/index.js';

export interface BenchRun {
  label: string;
  tokens: number;
  firstMs: number | null;
  totalMs: number;
  decodeMs: number;
  tps: number;
  outHead: string;
}

export interface BenchConfig {
  prompt: string;
  promptCharCount: number;
  maxTokens: number;
  runs: number;
  temperature: number;
  repPenalty: number;
}

export interface BenchResult {
  config: BenchConfig;
  runs: BenchRun[];
  median: number;
  mean: number;
  min: number;
  max: number;
}

export type BenchProgress = (
  e: { label: string; index: number; total: number; tokens: number; totalMs: number; tps: number },
) => void;

export async function runBench(
  engine: GemmaEngine,
  prompt: string,
  maxTokens: number,
  runs: number,
  onProgress?: BenchProgress,
): Promise<BenchResult> {
  const results: BenchRun[] = [];
  const labels = ['warmup', ...Array.from({ length: runs }, (_, i) => `m${i + 1}`)];

  for (let i = 0; i < labels.length; i++) {
    const label = labels[i];
    engine.resetConversation();
    engine.addUserMessage(prompt);
    const gs = performance.now();
    let tokens = 0;
    let firstMs: number | null = null;
    let out = '';
    for await (const t of engine.generate({ temperature: 0, maxTokens, repPenalty: 1.0 })) {
      tokens++;
      if (tokens === 1) firstMs = performance.now() - gs;
      out += t;
    }
    const totalMs = performance.now() - gs;
    const decodeMs = totalMs - (firstMs ?? 0);
    const tps = tokens > 1 && decodeMs > 0 ? (tokens - 1) / (decodeMs / 1000) : 0;
    results.push({ label, tokens, firstMs, totalMs, decodeMs, tps, outHead: out.slice(0, 180) });
    onProgress?.({ label, index: i, total: labels.length, tokens, totalMs, tps });
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

/** Convert a bench result into the diagnostics-report BenchMeasurement shape. */
export function benchToMeasurement(label: string, r: BenchResult): BenchMeasurement {
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

export async function collectDeviceInfo(): Promise<DeviceInfo | undefined> {
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

export function collectBrowserInfo(): BrowserInfo {
  const nav = navigator as Navigator & { deviceMemory?: number };
  return {
    userAgent: nav.userAgent,
    platform: nav.platform,
    hardwareConcurrency: nav.hardwareConcurrency,
    deviceMemoryGB: nav.deviceMemory,
  };
}
