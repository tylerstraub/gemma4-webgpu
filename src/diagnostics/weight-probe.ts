/**
 * Weight-layout verification — read arbitrary slices from the uploaded
 * GPU F16 buffer AND from raw GGUF source bytes, diff them element-wise.
 * Catches routing / transpose / dequant bugs that numerical sweeps at
 * attention boundaries might not isolate.
 *
 * Three main verbs:
 *   - `compareWeightSlice` — arbitrary slice from any named tensor.
 *   - `probeLayerWeight` — sweep a named tensor across a set of layers
 *     and rows, flag any that exceed F16 upload noise.
 *   - `probePerLayerModelProjLayout` — specifically for the
 *     per-layer-model-proj tensor, probes four flatten hypotheses to
 *     identify which memory layout the GGUF encoder used.
 */

import type { GemmaEngine, GemmaConfig } from '../types.js';
import type {
  WeightCompareResult,
  LayerWeightProbe,
  LayerWeightProbeTarget,
  PerLayerModelProjProbe,
} from './types.js';

export const GGUF_TYPE_NAMES: Record<number, string> = {
  0: 'F32',
  1: 'F16',
  2: 'Q4_0',
  3: 'Q4_1',
  6: 'Q5_0',
  7: 'Q5_1',
  8: 'Q8_0',
  9: 'Q8_1',
  10: 'Q2_K',
  11: 'Q3_K',
  12: 'Q4_K',
  13: 'Q5_K',
  14: 'Q6_K',
  15: 'Q8_K',
  16: 'IQ2_XXS',
  17: 'IQ2_XS',
  18: 'IQ3_XXS',
  19: 'IQ1_S',
  20: 'IQ4_NL',
  21: 'IQ3_S',
  22: 'IQ2_S',
  23: 'IQ4_XS',
  24: 'I8',
  25: 'I16',
  26: 'I32',
  27: 'I64',
  28: 'F64',
  30: 'BF16',
};

/**
 * Compare an arbitrary slice of a named tensor between the GPU upload
 * (F16 → F32) and the raw GGUF source (dequantized via Range refetch).
 * Returns full stats plus head preview plus the 5 largest diffs.
 */
export async function compareWeightSlice(
  engine: GemmaEngine,
  name: string,
  elemOffset: number,
  elemCount: number,
  headN: number = 16,
): Promise<WeightCompareResult> {
  const info = engine.getGgufTensorInfo(name);
  if (!info) throw new Error(`compareWeightSlice: no tensor '${name}'`);

  const [gpu, src] = await Promise.all([
    engine.readGlobalTensor(name, elemOffset, elemCount),
    engine.fetchRawTensorSlice(name, elemOffset, elemCount),
  ]);

  let sumAbs = 0;
  let sumSqAbs = 0;
  let maxAbs = 0;
  let exact = 0;
  let within = 0;
  let gross = 0;
  let nanGpu = 0, nanSrc = 0, infGpu = 0, infSrc = 0;
  const topDiffs: { index: number; gpu: number; src: number; diff: number }[] = [];
  const pushTop = (i: number, g: number, s: number, d: number) => {
    if (topDiffs.length < 5 || Math.abs(d) > Math.abs(topDiffs[topDiffs.length - 1].diff)) {
      topDiffs.push({ index: i, gpu: g, src: s, diff: d });
      topDiffs.sort((a, b) => Math.abs(b.diff) - Math.abs(a.diff));
      if (topDiffs.length > 5) topDiffs.pop();
    }
  };
  let finite = 0;
  for (let i = 0; i < elemCount; i++) {
    const g = gpu[i], s = src[i];
    if (Number.isNaN(g)) nanGpu++;
    if (Number.isNaN(s)) nanSrc++;
    if (g === Infinity || g === -Infinity) infGpu++;
    if (s === Infinity || s === -Infinity) infSrc++;
    if (!Number.isFinite(g) || !Number.isFinite(s)) continue;
    const d = g - s;
    const ad = Math.abs(d);
    sumAbs += ad;
    sumSqAbs += ad * ad;
    if (ad > maxAbs) maxAbs = ad;
    if (g === s) exact++;
    if (ad < 1 / 1024) within++; // ~F16 LSB
    if (ad > 0.01) gross++;
    finite++;
    pushTop(i, g, s, d);
  }
  const mean = finite > 0 ? sumAbs / finite : 0;
  const variance = finite > 0 ? sumSqAbs / finite - mean * mean : 0;

  const head = Math.min(headN, elemCount);
  const headGpu: number[] = new Array(head);
  const headSrc: number[] = new Array(head);
  const headDiff: number[] = new Array(head);
  for (let i = 0; i < head; i++) {
    headGpu[i] = +gpu[i].toFixed(6);
    headSrc[i] = +src[i].toFixed(6);
    headDiff[i] = +(gpu[i] - src[i]).toFixed(6);
  }

  return {
    name,
    sourceType: info.type,
    sourceTypeName: GGUF_TYPE_NAMES[info.type] ?? `T${info.type}`,
    dims: info.dims,
    elemOffset,
    elemCount,
    stats: {
      maxAbsDiff: +maxAbs.toFixed(6),
      meanAbsDiff: +mean.toFixed(6),
      stddevAbsDiff: +Math.sqrt(Math.max(0, variance)).toFixed(6),
      exactMatchCount: exact,
      withinF16NoiseCount: within,
      grossMismatchCount: gross,
      nanCountGpu: nanGpu,
      nanCountSrc: nanSrc,
      infCountGpu: infGpu,
      infCountSrc: infSrc,
    },
    headGpu,
    headSrc,
    headDiff,
    topDiffIndices: topDiffs.map((d) => ({
      index: d.index,
      gpu: +d.gpu.toFixed(6),
      src: +d.src.toFixed(6),
      diff: +d.diff.toFixed(6),
    })),
  };
}

/** Convenience wrapper: treat a tensor as `[M, N]` row-major, pull row `r`. */
export async function compareWeightRow(
  engine: GemmaEngine,
  name: string,
  row: number,
  cols: number,
  headN: number = 16,
): Promise<WeightCompareResult> {
  return compareWeightSlice(engine, name, row * cols, cols, headN);
}

function diffStats(g: Float32Array, s: Float32Array, n: number) {
  let sumAbs = 0, maxAbs = 0, exact = 0;
  for (let i = 0; i < n; i++) {
    const d = Math.abs(g[i] - s[i]);
    sumAbs += d;
    if (d > maxAbs) maxAbs = d;
    if (g[i] === s[i]) exact++;
  }
  return { maxAbsDiff: +maxAbs.toFixed(6), meanAbsDiff: +(sumAbs / n).toFixed(6), exact };
}

function verdictFor(targets: LayerWeightProbeTarget[]): string {
  const fails = targets.filter((t) => !t.pass);
  if (fails.length === 0) return `OK — all ${targets.length} targets within F16 noise`;
  const where = fails
    .map((t) => (t.layer === null ? `lmHead row ${t.row}` : `L${t.layer} row ${t.row}`))
    .join(', ');
  return `LIKELY_BUG — ${fails.length}/${targets.length} targets diverge at: ${where}`;
}

/**
 * Per-layer weight probe — for each (layer, row) target, read `head`
 * F16 elements from the GPU buffer (`layers[L][name]` or `tokenEmbed`
 * for `token_embd`) and from GGUF source at the same logical offset,
 * diff. Flag any target whose `maxAbsDiff` exceeds F16 upload noise
 * (0.02).
 *
 * Default targets: all full-attention layers (global) + layer 3
 * (sliding) — mixes both head-dim regimes so layer-confusion bugs
 * can't hide behind uniform shape.
 */
export async function probeLayerWeight(
  engine: GemmaEngine,
  config: Readonly<GemmaConfig>,
  name: string,
  opts: { layers?: number[]; rows?: number[]; head?: number } = {},
): Promise<LayerWeightProbe> {
  const head = opts.head ?? 16;
  const isTiedLmHead = name === 'token_embd';
  const NOISE = 0.02;

  if (isTiedLmHead) {
    const info = engine.getGgufTensorInfo('token_embd');
    if (!info) throw new Error('probeLayerWeight: token_embd not loaded');
    const N = info.dims[0];
    const rows = opts.rows ?? [0, 1, 100, 236777 % Number(info.dims[1])];
    const targets: LayerWeightProbeTarget[] = [];
    for (const row of rows) {
      const elemOffset = row * Number(N);
      const [g, s] = await Promise.all([
        engine.readEmbeddingTensor(elemOffset, head),
        engine.fetchRawTensorSlice('token_embd', elemOffset, head),
      ]);
      const stats = diffStats(g, s, head);
      targets.push({
        layer: null,
        row,
        elemOffset,
        elemCount: head,
        maxAbsDiff: stats.maxAbsDiff,
        meanAbsDiff: stats.meanAbsDiff,
        exactMatchCount: stats.exact,
        headGpu: Array.from(g).map((v) => +v.toFixed(6)),
        headSrc: Array.from(s).map((v) => +v.toFixed(6)),
        pass: stats.maxAbsDiff < NOISE,
      });
    }
    return {
      name,
      sourceType: info.type,
      sourceTypeName: GGUF_TYPE_NAMES[info.type] ?? `T${info.type}`,
      dims: info.dims,
      targets,
      verdict: verdictFor(targets),
    };
  }

  // Per-layer case. Default: all full-attention layers + layer 3.
  const faLayers = [4, 9, 14, 19, 24, 29, 34].filter((L) => L < config.num_layers);
  const layers = opts.layers ?? [3, ...faLayers];
  const rows = opts.rows ?? [0];
  const targets: LayerWeightProbeTarget[] = [];

  let firstInfo: ReturnType<typeof engine.getGgufTensorInfo> | null = null;
  for (const L of layers) {
    const info = engine.getGgufTensorInfo(`blk.${L}.${name}`);
    if (!info) {
      targets.push({
        layer: L,
        row: -1,
        elemOffset: -1,
        elemCount: 0,
        maxAbsDiff: Infinity,
        meanAbsDiff: Infinity,
        exactMatchCount: 0,
        headGpu: [],
        headSrc: [],
        pass: false,
      });
      continue;
    }
    if (!firstInfo) firstInfo = info;
    const N = Number(info.dims[0]); // inner dim (fast axis) per GGUF convention
    for (const row of rows) {
      const elemOffset = row * N;
      const [g, s] = await Promise.all([
        engine.readLayerTensor(L, name, elemOffset, head),
        engine.fetchRawTensorSlice(`blk.${L}.${name}`, elemOffset, head),
      ]);
      const stats = diffStats(g, s, head);
      targets.push({
        layer: L,
        row,
        elemOffset,
        elemCount: head,
        maxAbsDiff: stats.maxAbsDiff,
        meanAbsDiff: stats.meanAbsDiff,
        exactMatchCount: stats.exact,
        headGpu: Array.from(g).map((v) => +v.toFixed(6)),
        headSrc: Array.from(s).map((v) => +v.toFixed(6)),
        pass: stats.maxAbsDiff < NOISE,
      });
    }
  }

  const info = firstInfo ?? { type: -1, dims: [] as number[], offset: 0, byteSize: 0 };
  return {
    name,
    sourceType: info.type,
    sourceTypeName: GGUF_TYPE_NAMES[info.type] ?? `T${info.type}`,
    dims: info.dims,
    targets,
    verdict: verdictFor(targets),
  };
}

/**
 * Probe four flattening hypotheses for `per_layer_model_proj`. The
 * tensor's JAX shape is `(d=hidden, n=num_layers, p=per_layer_dim)`;
 * GGUF reports dims `[hidden, num_layers * per_layer_input_dim]`. Our
 * matmul treats the GPU buffer as `[M=num_layers*p, N=hidden]`
 * row-major — M is the slow axis — so GPU elem offset for `(L, P)` is
 * `(L * P + Pidx) * hidden`. Question: what does that correspond to in
 * the GGUF bytes?
 */
export async function probePerLayerModelProjLayout(
  engine: GemmaEngine,
  config: Readonly<GemmaConfig>,
): Promise<PerLayerModelProjProbe> {
  const info = engine.getGgufTensorInfo('per_layer_model_proj');
  if (!info) throw new Error('per_layer_model_proj not loaded');
  const N = config.hidden_size;
  const NL = config.num_layers;
  const P = config.per_layer_input_dim;
  const M = NL * P;
  const numElems = M * N;
  const head = 16;

  const targets: [number, number][] = [[0, 0], [1, 0], [17, 128], [34, 255]];
  const probes: PerLayerModelProjProbe['probes'] = [];

  for (const [L, Pidx] of targets) {
    const m_n_p_fastest = L * P + Pidx;
    const m_p_n_fastest = Pidx * NL + L;

    const gpuOff_A = m_n_p_fastest * N;
    const srcOff_A = m_n_p_fastest * N;
    const gpuOff_B = m_n_p_fastest * N;
    const srcOff_B = m_p_n_fastest * N;
    const gpuOff_C = m_p_n_fastest * N;
    const srcOff_C = m_n_p_fastest * N;

    const runOne = async (gpuOff: number, srcOff: number) => {
      const [gArr, sArr] = await Promise.all([
        engine.readGlobalTensor('per_layer_model_proj', gpuOff, head),
        engine.fetchRawTensorSlice('per_layer_model_proj', srcOff, head),
      ]);
      let sumAbs = 0, maxAbs = 0, exact = 0;
      for (let i = 0; i < head; i++) {
        const d = Math.abs(gArr[i] - sArr[i]);
        sumAbs += d;
        if (d > maxAbs) maxAbs = d;
        if (gArr[i] === sArr[i]) exact++;
      }
      return {
        gpuElemOffset: gpuOff,
        srcElemOffset: srcOff,
        elemCount: head,
        sampleDiff: {
          maxAbsDiff: +maxAbs.toFixed(6),
          meanAbsDiff: +(sumAbs / head).toFixed(6),
          exactMatchCount: exact,
        },
        headGpu: Array.from(gArr).map((x) => +x.toFixed(5)),
        headSrc: Array.from(sArr).map((x) => +x.toFixed(5)),
      };
    };

    const [A, B, C] = await Promise.all([
      runOne(gpuOff_A, srcOff_A),
      runOne(gpuOff_B, srcOff_B),
      runOne(gpuOff_C, srcOff_C),
    ]);

    probes.push({
      layer: L,
      pos: Pidx,
      hypotheses: {
        A_m_np_both: { description: 'GPU m=L*P+Pidx, src m=L*P+Pidx (p-fastest, JAX default flatten)', ...A },
        B_m_np_vs_pn: { description: 'GPU m=L*P+Pidx, src m=Pidx*NL+L (n-fastest in src)', ...B },
        C_m_pn_vs_np: { description: 'GPU m=Pidx*NL+L, src m=L*P+Pidx (inverse interp)', ...C },
      },
    });
  }

  const pickVerdict = (): string => {
    const scores: Record<string, number> = { A_m_np_both: 0, B_m_np_vs_pn: 0, C_m_pn_vs_np: 0 };
    for (const p of probes) {
      for (const [k, h] of Object.entries(p.hypotheses)) {
        if (h.sampleDiff.maxAbsDiff < 0.02) scores[k]++;
      }
    }
    const best = Object.entries(scores).sort((a, b) => b[1] - a[1])[0];
    return best[1] === probes.length
      ? `${best[0]} matches all ${probes.length} targets`
      : best[1] === 0
        ? 'none — all hypotheses diverge (check transposed layout or quant decode)'
        : `${best[0]} matches ${best[1]}/${probes.length} targets — inconclusive, probe more targets`;
  };

  return {
    dims: info.dims,
    numElems,
    assumption: { M, N },
    probes,
    verdict: pickVerdict(),
  };
}
