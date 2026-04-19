/**
 * Hidden-state capture — run a forward pass up to a named boundary and
 * return the F32 contents of the underlying work buffer. Used by the
 * cross-lab diff methodology to compare against an external reference
 * (HF PyTorch forward-hooks).
 */

import type { GemmaEngine, GemmaConfig, CapturePoint } from '../types.js';
import type { CapturePointShorthand, CaptureSummary } from './types.js';

/** Convert a shorthand form into a full `CapturePoint`. */
export function normalizeCapturePoint(p: CapturePointShorthand): CapturePoint {
  if (typeof p === 'string') return { kind: p };
  if ('afterLayer' in p) return { kind: 'afterLayer', layer: p.afterLayer };
  if ('preRopeQ' in p) return { kind: 'preRopeQ', layer: p.preRopeQ };
  if ('preRopeK' in p) return { kind: 'preRopeK', layer: p.preRopeK };
  if ('preRopeV' in p) return { kind: 'preRopeV', layer: p.preRopeV };
  if ('postRopeQ' in p) return { kind: 'postRopeQ', layer: p.postRopeQ };
  if ('postRopeK' in p) return { kind: 'postRopeK', layer: p.postRopeK };
  if ('attnOut' in p) return { kind: 'attnOut', layer: p.attnOut };
  throw new Error(`unrecognized CapturePointShorthand: ${JSON.stringify(p)}`);
}

/** Serialized form, e.g. `"afterLayer:7"`, `"embed"`, `"logits"`. */
export function stringifyStopAt(p: CapturePoint): string {
  if ('layer' in p) return `${p.kind}:${p.layer}`;
  return p.kind;
}

/** Full sweep across every layer plus the surrounding boundaries. */
export function defaultCapturePoints(numLayers: number, perLayerInputDim: number): CapturePoint[] {
  const pts: CapturePoint[] = [{ kind: 'embed' }];
  if (perLayerInputDim > 0) pts.push({ kind: 'pleStage1' });
  for (let i = 0; i < numLayers; i++) pts.push({ kind: 'afterLayer', layer: i });
  pts.push({ kind: 'final' });
  return pts;
}

/** Summarize an F32 capture vector — stats + head/tail for eyeballing. */
export function buildSummary(
  values: Float32Array,
  tokens: number[],
  position: number,
  capturedTokenId: number,
  point: CapturePoint,
  prompt: string,
  includeValues: boolean,
): CaptureSummary {
  let sum = 0;
  let sumSq = 0;
  let min = Infinity;
  let max = -Infinity;
  let nanCount = 0;
  let infCount = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (Number.isNaN(v)) { nanCount++; continue; }
    if (!Number.isFinite(v)) { infCount++; continue; }
    sum += v;
    sumSq += v * v;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const finiteCount = values.length - nanCount - infCount;
  const mean = finiteCount > 0 ? sum / finiteCount : 0;
  const variance = finiteCount > 0 ? sumSq / finiteCount - mean * mean : 0;
  const stddev = Math.sqrt(Math.max(0, variance));
  const norm = Math.sqrt(sumSq);
  const head = Array.from(values.slice(0, Math.min(16, values.length)));
  const tail = Array.from(values.slice(Math.max(0, values.length - 16)));

  const summary: CaptureSummary = {
    stopAt: stringifyStopAt(point),
    promptHead: prompt.slice(0, 200),
    tokens,
    position,
    capturedTokenId,
    length: values.length,
    norm,
    mean,
    stddev,
    min: min === Infinity ? 0 : min,
    max: max === -Infinity ? 0 : max,
    absMax: Math.max(
      Math.abs(min === Infinity ? 0 : min),
      Math.abs(max === -Infinity ? 0 : max),
    ),
    nanCount,
    infCount,
    first16: head,
    last16: tail,
  };
  if (includeValues) summary.values = values;
  return summary;
}

/** Single-point capture. Resets KV, prefills all-but-last, captures on last. */
export async function capture(
  engine: GemmaEngine,
  prompt: string,
  stopAt: CapturePointShorthand,
  opts: { applyChatTemplate?: boolean; includeValues?: boolean } = {},
): Promise<CaptureSummary> {
  const point = normalizeCapturePoint(stopAt);
  const applyChatTemplate = opts.applyChatTemplate ?? true;

  const tokens = engine.encodePromptTokens(prompt, applyChatTemplate);
  if (tokens.length === 0) throw new Error('tokenizer returned empty token stream');

  engine.resetKVForCapture();
  const prefix = tokens.slice(0, tokens.length - 1);
  const last = tokens[tokens.length - 1];
  const position = prefix.length;
  if (prefix.length > 0) await engine.prefillForCapture(prefix, 0);

  const values = await engine.captureHidden(last, position, point);
  return buildSummary(values, tokens, position, last, point, prompt, opts.includeValues === true);
}

/**
 * Multi-point capture — prefill once, capture at each point. ~25× faster
 * than a loop of single-point captures because prefill dominates.
 *
 * After prefill, KV cache holds positions 0..n-2. Each `captureHidden`
 * re-runs the forward pass for `token[n-1]` at position n-1. That pass
 * overwrites KV-cache entry at position n-1 each time with the same
 * deterministic values, so sequential captures on the same prompt are
 * safe without re-prefilling.
 */
export async function captureAll(
  engine: GemmaEngine,
  config: Readonly<GemmaConfig>,
  prompt: string,
  points: CapturePointShorthand[] | undefined,
  opts: { applyChatTemplate?: boolean; includeValues?: boolean } = {},
): Promise<{
  prefillMs: number;
  capturesMs: number;
  tokens: number[];
  position: number;
  capturedTokenId: number;
  pointCount: number;
  captures: Record<string, CaptureSummary>;
}> {
  const applyChatTemplate = opts.applyChatTemplate ?? true;
  const resolved: CapturePoint[] = points
    ? points.map(normalizeCapturePoint)
    : defaultCapturePoints(config.num_layers, config.per_layer_input_dim);

  const tokens = engine.encodePromptTokens(prompt, applyChatTemplate);
  if (tokens.length === 0) throw new Error('tokenizer returned empty token stream');

  const prefix = tokens.slice(0, tokens.length - 1);
  const last = tokens[tokens.length - 1];
  const position = prefix.length;

  engine.resetKVForCapture();
  const tPrefillStart = performance.now();
  if (prefix.length > 0) await engine.prefillForCapture(prefix, 0);
  const prefillMs = performance.now() - tPrefillStart;

  const captures: Record<string, CaptureSummary> = {};
  const tCapStart = performance.now();
  for (const p of resolved) {
    const values = await engine.captureHidden(last, position, p);
    captures[stringifyStopAt(p)] = buildSummary(
      values, tokens, position, last, p, prompt, opts.includeValues === true,
    );
  }
  const capturesMs = performance.now() - tCapStart;

  return {
    prefillMs: Math.round(prefillMs),
    capturesMs: Math.round(capturesMs),
    tokens,
    position,
    capturedTokenId: last,
    pointCount: resolved.length,
    captures,
  };
}
