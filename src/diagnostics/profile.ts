/**
 * GPU-side (timestamp-query) and CPU-side (performance.now phase breakdown)
 * profiling — high-level verbs that wrap the engine's per-pass
 * profiling primitives.
 *
 * Both profilers arm, run a measured decode, and disarm — there's no
 * runtime cost when they're off. Keeping them separate keeps CPU-side
 * timing clean: enabling the GPU profiler adds a `resolveQuerySet` +
 * `copyBufferToBuffer` to every forward, and the extra CPU wall cost
 * would show up in `cpuProfile`.
 */

import type {
  GemmaEngine,
  ProfileCapability,
  ProfileReport,
  CpuProfileReport,
} from '../types.js';

/** GPU profile run result. */
export type ProfileRunResult =
  | {
      ok: true;
      capability: ProfileCapability;
      report: ProfileReport;
      wallMs: number;
      tokensMeasured: number;
    }
  | {
      ok: false;
      error: string;
      capability: ProfileCapability;
    };

/** CPU profile run result. */
export interface CpuProfileRunResult {
  ok: true;
  report: CpuProfileReport;
  wallMs: number;
  tokensMeasured: number;
}

/**
 * GPU per-dispatch profiler. Requires `timestamp-query` on the adapter.
 * Runs a warmup (untimed) decode, then an armed decode, returns the
 * per-label aggregate report.
 */
export async function profile(
  engine: GemmaEngine,
  prompt: string,
  opts: { maxTokens?: number; warmup?: number } = {},
): Promise<ProfileRunResult> {
  const capability = engine.getProfileCapability();
  if (!capability.timestampQuerySupported) {
    return {
      ok: false,
      error: capability.timestampQueryOnAdapterButUnrequested
        ? 'timestamp-query present on adapter but not enabled on device — reload the page'
        : 'timestamp-query feature unavailable on this device',
      capability,
    };
  }

  const maxTokens = opts.maxTokens ?? 64;
  const warmup = opts.warmup ?? 10;
  const genOpts = { temperature: 0, maxTokens, repPenalty: 1.0 } as const;

  // Warmup — untimed decode to warm pipeline cache + GPU clock state.
  engine.resetConversation();
  engine.addUserMessage(prompt);
  let warmupTokens = 0;
  for await (const _ of engine.generate(genOpts)) {
    warmupTokens++;
    if (warmupTokens >= warmup + 1) break; // +1 for the first-token-after-prefill
  }

  // Measured pass. Profile is armed after prefill completes (prefill
  // internally toggles capture off via prefillBatched).
  engine.resetConversation();
  engine.addUserMessage(prompt);
  engine.enableProfile();
  engine.resetProfileSamples();
  const t0 = performance.now();
  let tokensMeasured = 0;
  for await (const _ of engine.generate(genOpts)) {
    tokensMeasured++;
  }
  const wallMs = performance.now() - t0;
  engine.disableProfile();

  const report = engine.getProfileReport();
  return { ok: true, capability, report, wallMs, tokensMeasured };
}

/**
 * CPU-side phase profiler — wall-clock breakdown of the host-side work
 * per forward pass (writeBuffer, encode, submit, mapAsync.wait, etc.).
 * Does NOT enable GPU profiling; the two are separate knobs so CPU
 * numbers aren't contaminated by the resolve/copy path.
 */
export async function cpuProfile(
  engine: GemmaEngine,
  prompt: string,
  opts: { maxTokens?: number; warmup?: number } = {},
): Promise<CpuProfileRunResult> {
  const maxTokens = opts.maxTokens ?? 64;
  const warmup = opts.warmup ?? 10;
  const genOpts = { temperature: 0, maxTokens, repPenalty: 1.0 } as const;

  engine.resetConversation();
  engine.addUserMessage(prompt);
  let warmupTokens = 0;
  for await (const _ of engine.generate(genOpts)) {
    warmupTokens++;
    if (warmupTokens >= warmup + 1) break;
  }

  engine.resetConversation();
  engine.addUserMessage(prompt);
  engine.enableCpuProfile();
  engine.resetCpuProfileSamples();
  const t0 = performance.now();
  let tokensMeasured = 0;
  for await (const _ of engine.generate(genOpts)) {
    tokensMeasured++;
  }
  const wallMs = performance.now() - t0;
  engine.disableCpuProfile();

  const report = engine.getCpuProfileReport();
  return { ok: true, report, wallMs, tokensMeasured };
}
