/**
 * Type definitions for the diagnostic modules: capture summaries,
 * cross-lab diff results, reference tensor manifests.
 */

import type { CapturePoint } from '../types.js';

/**
 * Compact form for addressing a capture point. String shortcuts for
 * layer-invariant points; `{ afterLayer: N }` etc. for per-layer.
 */
export type CapturePointShorthand =
  | 'embed'
  | 'plePmProjected'
  | 'pleStage1'
  | 'final'
  | 'logits'
  | { afterLayer: number }
  | { preRopeQ: number }
  | { preRopeK: number }
  | { preRopeV: number }
  | { postRopeQ: number }
  | { postRopeK: number }
  | { attnOut: number };

export interface CaptureSummary {
  /** Serialized CapturePoint, e.g. "embed", "afterLayer:0", "pleStage1". */
  stopAt: string;
  /** Chat-templated prompt (or raw if applyChatTemplate=false), truncated to 200 chars. */
  promptHead: string;
  /** Token IDs after tokenization, for exact cross-lab comparison. */
  tokens: number[];
  /** Position index of the token whose forward pass was captured. */
  position: number;
  /** Token ID captured (= tokens[tokens.length - 1] for first-token probe). */
  capturedTokenId: number;
  /** Element count of the returned vector. */
  length: number;
  /** L2 norm. */
  norm: number;
  /** Mean and stddev (useful for scale comparison). */
  mean: number;
  stddev: number;
  /** Min / max / abs-max. */
  min: number;
  max: number;
  absMax: number;
  /** NaN / Inf counts — should be 0 on a healthy pass. */
  nanCount: number;
  infCount: number;
  /** First 16 and last 16 values for sanity-checking shape + magnitude. */
  first16: number[];
  last16: number[];
  /** Optional: full F32 vector when opts.includeValues=true. */
  values?: Float32Array;
}

/**
 * Direction-specific vs uniform-scale discriminator. Sort |diff| descending,
 * read off cumulative mass at the top 1% / 10% / 50% cutoffs. A uniform bug
 * (every dim drifts by the same factor → RMSNorm weight off by 2×) gives
 * values near the cutoff itself. A concentrated bug (a handful of dims
 * carry most of the divergence → one attention head transposed) gives top
 * fractions much larger than the cutoff.
 */
export interface DiffConcentration {
  top1PctFraction: number;
  top10PctFraction: number;
  top50PctFraction: number;
  /**
   * Participation ratio: (Σ|d|)² / (n · Σd²). 1.0 = perfectly uniform;
   * near 0 = one dim dominates. Think "effective fraction of dimensions
   * contributing meaningfully to the divergence."
   */
  participationRatio: number;
  /** Raw effective-dim count: (Σ|d|)² / Σd². */
  effectiveDimCount: number;
}

export interface CrossLabDiffResult {
  capturePoint: string;
  refKey: string;
  length: number;
  maxAbsDiff: number;
  meanAbsDiff: number;
  stddevDiff: number;
  /** First index where |diff| exceeds `thresholdOver` (default 0.05). -1 if none. */
  firstOverIndex: number;
  firstOverThreshold: number;
  overCount: number;
  refNorm: number;
  ourNorm: number;
  refMean: number;
  ourMean: number;
  refAbsMax: number;
  ourAbsMax: number;
  /** Cosine similarity of the two vectors (-1..1). This is the headline metric. */
  cosineSimilarity: number;
  refHead: number[];
  ourHead: number[];
  diffHead: number[];
  topDiffs: { index: number; ref: number; ours: number; diff: number }[];
  concentration: DiffConcentration;
}

export interface LoadedReference {
  url: string;
  sizeBytes: number;
  entries: { name: string; dtype: string; shape: number[]; length: number }[];
  keys: string[];
}

/** Parsed reference tensors keyed by npz entry name (without `.npy` suffix). */
export type ReferenceTensors = Record<string, Float32Array | Int32Array>;

export type { CapturePoint };

// ─── Weight-layout verification ────────────────────────────────────

export interface WeightCompareStats {
  /** Max absolute diff between GPU F16→F32 values and GGUF source→F32 values. */
  maxAbsDiff: number;
  /** Mean absolute diff (ignoring NaN/Inf on either side). */
  meanAbsDiff: number;
  /** Stddev of absolute diffs. */
  stddevAbsDiff: number;
  /** Count of elements where GPU == source bitwise. */
  exactMatchCount: number;
  /** Count within 2⁻¹⁰ absolute — F16 quantization noise. */
  withinF16NoiseCount: number;
  /** Count of diffs > 0.01 — clear layout / dequant bug indicator. */
  grossMismatchCount: number;
  nanCountGpu: number;
  nanCountSrc: number;
  infCountGpu: number;
  infCountSrc: number;
}

export interface WeightCompareResult {
  name: string;
  sourceType: number;
  sourceTypeName: string;
  dims: number[] | bigint[];
  elemOffset: number;
  elemCount: number;
  stats: WeightCompareStats;
  headGpu: number[];
  headSrc: number[];
  headDiff: number[];
  topDiffIndices: { index: number; gpu: number; src: number; diff: number }[];
}

export interface LayerWeightProbeTarget {
  /** Layer index, or `null` for the tied LM head (`token_embd` — no per-layer buffer). */
  layer: number | null;
  row: number;
  elemOffset: number;
  elemCount: number;
  maxAbsDiff: number;
  meanAbsDiff: number;
  exactMatchCount: number;
  headGpu: number[];
  headSrc: number[];
  pass: boolean;
}

export interface LayerWeightProbe {
  name: string;
  sourceType: number;
  sourceTypeName: string;
  dims: number[] | bigint[];
  targets: LayerWeightProbeTarget[];
  /** OK if every target's maxAbsDiff < 0.02 (F16 upload noise). LIKELY_BUG otherwise. */
  verdict: string;
}

export interface PerLayerModelProjProbe {
  dims: number[] | bigint[];
  numElems: number;
  assumption: { M: number; N: number };
  probes: {
    layer: number;
    pos: number;
    hypotheses: Record<string, {
      description: string;
      gpuElemOffset: number;
      srcElemOffset: number;
      elemCount: number;
      sampleDiff: { maxAbsDiff: number; meanAbsDiff: number; exactMatchCount: number };
      headGpu: number[];
      headSrc: number[];
    }>;
  }[];
  verdict: string;
}

// ─── Profiling — re-exported from engine types for convenience ────

export type { ProfileCapability, ProfileReport, CpuProfileReport } from '../types.js';
