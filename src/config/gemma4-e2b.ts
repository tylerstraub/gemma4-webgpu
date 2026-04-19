/**
 * Gemma 4 E2B configuration: verified baseline + GGUF metadata hydration.
 *
 * The baseline values are drawn from Google's Gemma 4 model card
 * (https://ai.google.dev/gemma/docs/core/model_card_4) and verified against
 * the Unsloth Q4_K_M GGUF that this engine loads by default. `configFromGGUF`
 * reads the actual GGUF metadata and overrides any baseline value that the
 * file specifies — so in practice the engine runs on the file's exact
 * architecture, not the hardcoded defaults.
 */

import type { GemmaConfig, GGUFParsed } from '../types.js';
import { kvArray, kvNumberOrNull } from '../gguf.js';

/** Baseline Gemma 4 E2B config. Overridden from GGUF metadata by `configFromGGUF`. */
export function defaultConfig(): GemmaConfig {
  const num_layers = 35;

  // Full-attention layers (every fifth, final always global).
  const attention_is_sliding = new Array<boolean>(num_layers).fill(true);
  for (const i of [4, 9, 14, 19, 24, 29, 34]) attention_is_sliding[i] = false;

  const num_unshared_layers = 15;

  // Producer layers (0..14) use I=6144. Consumer layers (15..34) use I=12288.
  const intermediate_sizes = new Array<number>(num_layers);
  for (let i = 0; i < num_layers; i++) {
    intermediate_sizes[i] = i < num_unshared_layers ? 6144 : 12288;
  }

  const num_q_heads = 8;
  const num_kv_heads = 1;
  const head_dim_local = 256;
  const head_dim_global = 512;
  const maxHeadDim = Math.max(head_dim_local, head_dim_global);

  const config: GemmaConfig = {
    hidden_size: 1536,
    q_dim: num_q_heads * maxHeadDim,
    kv_dim: num_kv_heads * maxHeadDim,
    num_q_heads,
    num_kv_heads,
    head_dim: head_dim_local,
    intermediate_size: 12288,
    vocab_size: 262144,
    num_layers,
    context_length: 2048,
    rms_norm_eps: 1e-6,
    rope_theta_global: 1_000_000.0,
    rope_theta_swa: 10_000.0,
    head_dim_local,
    head_dim_global,
    sliding_window: 512,
    attention_is_sliding,
    intermediate_sizes,
    num_unshared_layers,
    kv_producer_for_layer: [], // computed below
    per_layer_input_dim: 256,
    final_logit_softcapping: 30.0,
  };
  config.kv_producer_for_layer = computeKvProducerMap(config);
  return config;
}

/**
 * Hydrate a `GemmaConfig` from GGUF metadata. Throws if the file isn't a
 * Gemma 4 GGUF (no `gemma4.*` keys). Returns a fresh config where every
 * value is either from the GGUF or from the Gemma 4 E2B baseline.
 */
export function configFromGGUF(gguf: GGUFParsed, maxContextLength?: number): GemmaConfig {
  const isGemma4 = gguf.kv.has('gemma4.block_count') || gguf.kv.has('gemma4.embedding_length');
  if (!isGemma4) {
    throw new Error('Not a Gemma 4 GGUF — no `gemma4.*` metadata keys found.');
  }

  const config = defaultConfig();

  // Core scalars.
  const hidden = kvNumberOrNull(gguf, 'gemma4.embedding_length');
  if (hidden !== null) config.hidden_size = hidden;

  const layers = kvNumberOrNull(gguf, 'gemma4.block_count');
  if (layers !== null) config.num_layers = layers;

  const qHeads = kvNumberOrNull(gguf, 'gemma4.attention.head_count');
  if (qHeads !== null) config.num_q_heads = qHeads;

  const kvHeads = kvNumberOrNull(gguf, 'gemma4.attention.head_count_kv');
  if (kvHeads !== null) config.num_kv_heads = kvHeads;

  const maxCtx = maxContextLength ?? 2048;
  const ctxLen = kvNumberOrNull(gguf, 'gemma4.context_length');
  if (ctxLen !== null) config.context_length = Math.min(ctxLen, maxCtx);

  // Per-layer FFN intermediate size (array in Gemma 4).
  const intermediateArr = kvArray(gguf, 'gemma4.feed_forward_length');
  if (intermediateArr) {
    const asNums = intermediateArr.map((v) => Number(v));
    config.intermediate_sizes = asNums;
    config.intermediate_size = Math.max(...asNums);
  } else {
    const scalar = kvNumberOrNull(gguf, 'gemma4.feed_forward_length');
    if (scalar !== null) {
      config.intermediate_size = scalar;
      config.intermediate_sizes = new Array<number>(config.num_layers).fill(scalar);
    }
  }

  // Split head dims: local (sliding) vs global (full-attention).
  const hdGlobal = kvNumberOrNull(gguf, 'gemma4.attention.key_length');
  const hdLocal = kvNumberOrNull(gguf, 'gemma4.attention.key_length_swa');
  if (hdGlobal !== null) config.head_dim_global = hdGlobal;
  if (hdLocal !== null) config.head_dim_local = hdLocal;
  config.head_dim = config.head_dim_local;

  // Dual RoPE.
  const freqBase = kvNumberOrNull(gguf, 'gemma4.rope.freq_base');
  const freqBaseSwa = kvNumberOrNull(gguf, 'gemma4.rope.freq_base_swa');
  if (freqBase !== null) config.rope_theta_global = freqBase;
  if (freqBaseSwa !== null) config.rope_theta_swa = freqBaseSwa;

  // Sliding-window + pattern.
  const sw = kvNumberOrNull(gguf, 'gemma4.attention.sliding_window');
  if (sw !== null) config.sliding_window = sw;

  const pattern = kvArray(gguf, 'gemma4.attention.sliding_window_pattern');
  if (pattern) {
    config.attention_is_sliding = pattern.map((v) => Boolean(v));
  }

  // Per-layer-embedding dim + softcap.
  const pleDim = kvNumberOrNull(gguf, 'gemma4.embedding_length_per_layer_input');
  if (pleDim !== null) config.per_layer_input_dim = pleDim;

  const softcap = kvNumberOrNull(gguf, 'gemma4.final_logit_softcapping');
  if (softcap !== null) config.final_logit_softcapping = softcap;

  // Shared-KV layer count → consumer/producer split.
  const sharedKv = kvNumberOrNull(gguf, 'gemma4.attention.shared_kv_layers');
  if (sharedKv !== null) config.num_unshared_layers = config.num_layers - sharedKv;

  // Recompute derived aggregates now that the primary dims are settled.
  const maxHeadDim = Math.max(config.head_dim_local, config.head_dim_global);
  config.q_dim = config.num_q_heads * maxHeadDim;
  config.kv_dim = config.num_kv_heads * maxHeadDim;
  config.kv_producer_for_layer = computeKvProducerMap(config);

  return config;
}

/**
 * Compute per-layer KV producer index. Each consumer layer shares its KV
 * cache with the last unshared (producer) layer of the same attention type
 * (sliding or full).
 *
 * This mirrors the rule in DeepMind's Gemma 4 JAX reference
 * (`_config.py` `shared_kv_layers` handling).
 */
function computeKvProducerMap(config: GemmaConfig): number[] {
  const num = config.num_layers;
  const unshared = config.num_unshared_layers;
  const result: number[] = new Array<number>(num);
  // Scan for the last unshared layer of each type.
  let lastUnsharedLocal = -1;
  let lastUnsharedGlobal = -1;
  for (let i = 0; i < unshared; i++) {
    if (config.attention_is_sliding[i]) lastUnsharedLocal = i;
    else lastUnsharedGlobal = i;
  }
  for (let i = 0; i < num; i++) {
    if (i < unshared) {
      result[i] = i;
    } else {
      result[i] = config.attention_is_sliding[i] ? lastUnsharedLocal : lastUnsharedGlobal;
    }
  }
  return result;
}
