/**
 * Diagnostic modules — correctness verification, hidden-state capture,
 * and cross-lab numerical comparison.
 *
 * The core loop:
 *   1. `loadReferenceTensors(url)` reads an npz file of reference
 *      activations produced by the external backend (HF PyTorch forward-
 *      hooks, etc.), returns the parsed tensors.
 *   2. `crossLabDiff(engine, ref, capturePoint, opts)` runs the engine up
 *      to the named boundary, diffs against the reference entry, and
 *      reports cosine similarity + concentration + first-divergence index.
 *   3. Any shader change runs this sweep BEFORE bench. Reverse order
 *      (bench first, verify later) is a trap.
 *
 * See `docs/methodology.md` (in an upcoming release) for the full
 * framing of why this is the durable contribution.
 */

export { loadReferenceTensors, parseNpz } from './npz.js';
export {
  capture,
  captureAll,
  normalizeCapturePoint,
  stringifyStopAt,
  defaultCapturePoints,
  buildSummary,
} from './capture.js';
export { crossLabDiff, capturePointToRefKey } from './cross-lab-diff.js';
export { profile, cpuProfile } from './profile.js';
export {
  compareWeightSlice,
  compareWeightRow,
  probeLayerWeight,
  probePerLayerModelProjLayout,
  GGUF_TYPE_NAMES,
} from './weight-probe.js';

export type {
  CapturePoint,
  CapturePointShorthand,
  CaptureSummary,
  CrossLabDiffResult,
  DiffConcentration,
  LoadedReference,
  ReferenceTensors,
  WeightCompareStats,
  WeightCompareResult,
  LayerWeightProbeTarget,
  LayerWeightProbe,
  PerLayerModelProjProbe,
  ProfileCapability,
  ProfileReport,
  CpuProfileReport,
} from './types.js';
export type { ProfileRunResult, CpuProfileRunResult } from './profile.js';
export { formatDiagnosticReport } from './report.js';
export type {
  BenchMeasurement,
  DeviceInfo,
  BrowserInfo,
  DiagnosticReportInput,
} from './report.js';
