/**
 * Cross-lab diff — the core of the correctness-verification methodology.
 * Capture a hidden-state boundary on this engine, compare it against the
 * matching reference tensor from an external backend (HF PyTorch
 * forward-hooks packed into an `np.savez` file), and report cosine
 * similarity + concentration metrics + first-divergence index.
 *
 * This is the durable contribution of the research — the kernels are a
 * point-in-time optimization, but the methodology for verifying numerical
 * equivalence between two independent implementations generalizes to any
 * transformer architecture and any backend.
 */

import type { GemmaEngine, CapturePoint } from '../types.js';
import type { CapturePointShorthand, CrossLabDiffResult, ReferenceTensors } from './types.js';
import { capture, normalizeCapturePoint, stringifyStopAt } from './capture.js';

/**
 * Map a capture point to its reference-npz entry name. The reference
 * generator (reference/pytorch/smoke_test.py) dumps entries keyed by
 * `<point>_pos14` for position 14 of the canonical prompt.
 */
export function capturePointToRefKey(p: CapturePoint): string {
  if (p.kind === 'embed') return 'embed_pos14';
  if (p.kind === 'logits') return 'logits_pos14';
  if (p.kind === 'final') return 'final_pos14';
  if (p.kind === 'pleStage1') return 'pleStage1_pos14';
  if (p.kind === 'afterLayer') return `afterLayer_${String(p.layer).padStart(2, '0')}_pos14`;
  if (p.kind === 'preRopeQ') return `preRopeQ_${String(p.layer).padStart(2, '0')}_pos14`;
  if (p.kind === 'preRopeK') return `preRopeK_${String(p.layer).padStart(2, '0')}_pos14`;
  if (p.kind === 'preRopeV') return `preRopeV_${String(p.layer).padStart(2, '0')}_pos14`;
  if (p.kind === 'postRopeQ') return `postRopeQ_${String(p.layer).padStart(2, '0')}_pos14`;
  if (p.kind === 'postRopeK') return `postRopeK_${String(p.layer).padStart(2, '0')}_pos14`;
  if (p.kind === 'attnOut') return `attnOut_${String(p.layer).padStart(2, '0')}_pos14`;
  throw new Error(`capturePointToRefKey: unhandled kind ${(p as { kind: string }).kind}`);
}

/**
 * Capture at `capturePoint`, diff against the reference entry, compute
 * cosine similarity + concentration metrics + first-divergence index +
 * top-K worst dims.
 */
export async function crossLabDiff(
  engine: GemmaEngine,
  ref: ReferenceTensors,
  capturePoint: CapturePointShorthand,
  opts: {
    prompt?: string;
    applyChatTemplate?: boolean;
    thresholdOver?: number;
    headN?: number;
    topK?: number;
  } = {},
): Promise<CrossLabDiffResult> {
  const point = normalizeCapturePoint(capturePoint);
  const refKey = capturePointToRefKey(point);
  const refRaw = ref[refKey];
  if (!refRaw) {
    throw new Error(
      `crossLabDiff: no reference entry '${refKey}'. Known keys: ${Object.keys(ref).sort().join(', ')}`,
    );
  }
  if (!(refRaw instanceof Float32Array)) {
    throw new Error(`crossLabDiff: reference '${refKey}' is not F32`);
  }

  const prompt = opts.prompt ?? 'Hello, how are you?';
  const applyChatTemplate = opts.applyChatTemplate ?? true;
  const threshold = opts.thresholdOver ?? 0.05;
  const headN = opts.headN ?? 16;
  const topK = opts.topK ?? 20;

  const summary = await capture(engine, prompt, capturePoint, {
    applyChatTemplate,
    includeValues: true,
  });
  const ours = summary.values;
  if (!ours) throw new Error('crossLabDiff: capture did not return values');
  if (ours.length !== refRaw.length) {
    throw new Error(
      `crossLabDiff: length mismatch — ours=${ours.length} ref=${refRaw.length} for ${refKey}`,
    );
  }

  const n = ours.length;
  const absDiffs = new Float32Array(n);
  let maxAbs = 0;
  let sumAbs = 0;
  let sumSqDiff = 0;
  let sumAbsSq = 0; // Σ|d|² — same value as sumSqDiff, kept explicit for clarity
  let firstOver = -1;
  let overCount = 0;
  let sumRefSq = 0, sumOurSq = 0, sumDot = 0;
  let sumRef = 0, sumOur = 0;
  let absMaxRef = 0, absMaxOur = 0;

  for (let i = 0; i < n; i++) {
    const r = refRaw[i];
    const o = ours[i];
    const d = o - r;
    const ad = Math.abs(d);
    absDiffs[i] = ad;
    sumAbs += ad;
    sumSqDiff += d * d;
    sumAbsSq += ad * ad;
    if (ad > maxAbs) maxAbs = ad;
    if (ad > threshold) {
      if (firstOver === -1) firstOver = i;
      overCount++;
    }
    sumRef += r; sumOur += o;
    sumRefSq += r * r; sumOurSq += o * o;
    sumDot += r * o;
    const ar = Math.abs(r); if (ar > absMaxRef) absMaxRef = ar;
    const ao = Math.abs(o); if (ao > absMaxOur) absMaxOur = ao;
  }

  // Concentration analysis: sort |diff| descending, read off cumulative
  // mass at top 1% / 10% / 50%. Discriminates direction-specific bugs
  // (handful of dims carry most divergence — e.g. one attention head
  // transposed) from uniform-scale bugs (every dim drifts by same factor —
  // e.g. an RMSNorm weight off by 2×).
  const sortIdx = new Int32Array(n);
  for (let i = 0; i < n; i++) sortIdx[i] = i;
  const sortedIdx = Array.from(sortIdx).sort((a, b) => absDiffs[b] - absDiffs[a]);

  const cutoff = (pct: number): number => {
    const k = Math.max(1, Math.ceil((n * pct) / 100));
    let cum = 0;
    for (let i = 0; i < k; i++) cum += absDiffs[sortedIdx[i]];
    return sumAbs > 0 ? cum / sumAbs : 0;
  };
  const top1 = cutoff(1);
  const top10 = cutoff(10);
  const top50 = cutoff(50);
  const effectiveDim = sumAbsSq > 0 ? (sumAbs * sumAbs) / sumAbsSq : 0;
  const participation = effectiveDim / n;

  const topDiffs: { index: number; ref: number; ours: number; diff: number }[] = [];
  const kActual = Math.min(topK, n);
  for (let i = 0; i < kActual; i++) {
    const idx = sortedIdx[i];
    topDiffs.push({
      index: idx,
      ref: +refRaw[idx].toFixed(6),
      ours: +ours[idx].toFixed(6),
      diff: +(ours[idx] - refRaw[idx]).toFixed(6),
    });
  }

  const mean = sumAbs / n;
  const variance = sumSqDiff / n - mean * mean;
  const stddev = Math.sqrt(Math.max(0, variance));
  const cos = sumDot / (Math.sqrt(sumRefSq) * Math.sqrt(sumOurSq) + 1e-30);
  const head = Math.min(headN, n);
  const refHead = new Array<number>(head);
  const ourHead = new Array<number>(head);
  const diffHead = new Array<number>(head);
  for (let i = 0; i < head; i++) {
    refHead[i] = +refRaw[i].toFixed(6);
    ourHead[i] = +ours[i].toFixed(6);
    diffHead[i] = +(ours[i] - refRaw[i]).toFixed(6);
  }

  return {
    capturePoint: stringifyStopAt(point),
    refKey,
    length: n,
    maxAbsDiff: +maxAbs.toFixed(6),
    meanAbsDiff: +mean.toFixed(6),
    stddevDiff: +stddev.toFixed(6),
    firstOverIndex: firstOver,
    firstOverThreshold: threshold,
    overCount,
    refNorm: +Math.sqrt(sumRefSq).toFixed(4),
    ourNorm: +Math.sqrt(sumOurSq).toFixed(4),
    refMean: +(sumRef / n).toFixed(6),
    ourMean: +(sumOur / n).toFixed(6),
    refAbsMax: +absMaxRef.toFixed(4),
    ourAbsMax: +absMaxOur.toFixed(4),
    cosineSimilarity: +cos.toFixed(6),
    refHead,
    ourHead,
    diffHead,
    topDiffs,
    concentration: {
      top1PctFraction: +top1.toFixed(4),
      top10PctFraction: +top10.toFixed(4),
      top50PctFraction: +top50.toFixed(4),
      participationRatio: +participation.toFixed(4),
      effectiveDimCount: +effectiveDim.toFixed(2),
    },
  };
}
