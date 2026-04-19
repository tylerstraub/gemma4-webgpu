/**
 * gemma4-webgpu
 *
 * Run Gemma 4 E2B (text-only) locally in the browser via WebGPU. The GGUF
 * weights are streamed from Hugging Face at init time and CPU-dequantized
 * to F16 once at upload; all subsequent compute runs on the GPU via raw
 * WebGPU and purpose-built WGSL.
 *
 * @example
 * ```typescript
 * import { createGemmaEngine } from 'gemma4-webgpu';
 *
 * const engine = await createGemmaEngine({
 *   model: 'e2b',
 *   onProgress: (p) => console.log(p.status),
 * });
 *
 * engine.addUserMessage('What is the capital of France?');
 * for await (const token of engine.generate({ temperature: 0.7 })) {
 *   process.stdout.write(token);
 * }
 *
 * // Follow-up reuses KV cache.
 * engine.addUserMessage('And what about Germany?');
 * for await (const token of engine.generate()) {
 *   process.stdout.write(token);
 * }
 *
 * engine.dispose();
 * ```
 */

import { GemmaEngineImpl } from './engine.js';
import type { GemmaEngine, GemmaEngineOptions } from './types.js';

export async function createGemmaEngine(options: GemmaEngineOptions = {}): Promise<GemmaEngine> {
  const engine = new GemmaEngineImpl(options);
  await engine.init(options);
  return engine;
}

export type {
  GemmaEngine,
  GemmaEngineOptions,
  GemmaConfig,
  GenerateOptions,
  ProgressInfo,
  ConversationTurn,
  CapturePoint,
  LoadTimings,
} from './types.js';

// Tuning — device profiles and per-kernel knobs. Consumers can pass a
// profile id, a full `TuningProfile`, or a `TuningProfileOverrides`
// deep-partial via `GemmaEngineOptions.tuning`.
export type {
  TuningProfile,
  TuningProfileOverrides,
  KernelName,
  FeatureIntent,
  SelectedProfile,
  TuningOverride,
} from './tuning/index.js';
export {
  PROFILES,
  nvidiaBlackwell,
  appleMSeries,
  generic,
  selectDeviceProfile,
  overrideProfile,
  rowsPerWorkgroupFor,
} from './tuning/index.js';
