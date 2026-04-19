# Architecture

This page catalogs the ways Gemma 4 E2B differs from Gemma 3 inside the
hot path, with pointers to where each delta is implemented. If you're
porting Gemma 4 to another backend, or reading this repo after spending
time in a Gemma-3 codebase, start here.

Everything below is verified against the HuggingFace Transformers
reference via the per-kernel crossLabDiff methodology documented in
[`methodology.md`](./methodology.md).

## At a glance

Gemma 4 E2B is a 35-layer decoder with hidden=1536, vocab=262144, and a
mixed attention pattern (every third layer is full-attention, the rest
are local sliding-window). Text-only weights land at 2.89 GiB as Q4_K_M;
with runtime work buffers the browser holds ~3.2 GiB resident. First
load streams from HuggingFace and CPU-dequantizes to F16 once, so all
GPU compute runs on the single F16 matmul path regardless of source
quantization.

## The nine deltas

### 1. Per-Layer Embeddings (PLE)

Every layer reads a small (`per_layer_input_dim = 256`) token-indexed
vector in addition to the main hidden state. The PLE table is
`[vocab × (num_layers × per_layer_input_dim)]` — 35 × 256 floats per
token, looked up once per forward and folded into the layer's residual
stream via a two-stage pipeline:

- **Stage 1** (once per forward): project hidden state through
  `per_layer_model_proj` to produce a 35×256 working vector, RMS-norm,
  and fuse with the token's PLE slice.
- **Stage 2** (per layer, after the FFN block): gate-project, GELU-mul
  with the PLE slice, up-project back to hidden, post-norm, skip-add
  into the residual with a per-layer scale.

Shaders: `ple_stage1_fuse.wgsl`, `ple_gelu_mul.wgsl`,
`ple_skip_scale_add.wgsl`. The PLE table is split across
`num_layers` GPU buffers to stay under WebGPU's 2 GiB buffer ceiling.

### 2. Shared KV caches

Only 15 of 35 layers (the "producer" layers) compute their own K/V and
write to KV cache. The remaining 20 "consumer" layers reuse the cache
of the nearest earlier producer of the same attention type. Shaders
`kv_cache_store.wgsl`, `attn_score.wgsl`, and `attn_output.wgsl` are
invariant under the sharing pattern — the engine simply maps each
consumer's reads to the producer's cache. See
`config.kv_producer_for_layer[]` for the mapping and
`isKvProducerLayer()` for the predicate.

### 3. Dual RoPE frequencies

Full-attention layers use `rope_theta_global = 1e6`; sliding-window
layers use `rope_theta_swa = 1e4`. Both frequencies are precomputed at
init from `rope_freqs` in the GGUF and uniform-injected per dispatch.
Shader `fused_per_head_norm_rope.wgsl` reads `theta` per invocation.
See `getRopeTheta(layerIdx, config)`.

### 4. Mixed head dimensions

Sliding-window (LOCAL) layers use `head_dim_local = 256`; full-attention
(GLOBAL) layers use `head_dim_global = 512`. The attention shaders take
`head_dim` as a uniform; the Q/K/V linear widths are computed per layer.
See `getHeadDim(layerIdx, config)`.

### 5. Sliding window attention

The LOCAL layers attend to the most recent `sliding_window = 512`
positions only. Enforced via a position-delta check inside
`attn_score.wgsl` (masked to `-inf` before softmax).

### 6. Final logit softcap

`logits := tanh(logits / cap) * cap` with `cap = 30.0`. Stabilizes
greedy sampling against outlier logits. Shader
`logit_softcap.wgsl`. Skip gracefully when `final_logit_softcapping = 0`.

### 7. `v_norm` per KV head (weight ≡ 1, not in GGUF)

A fused RMS-norm without learnable weights, applied to V per KV head
between `linearV` and `kvCacheStore`. The GGUF quantizer drops this
tensor because a weight-1 tensor is architecturally a no-op — but
dropping it silently inflates V entering attention by ~67×. Shader
`per_head_rms_norm_no_weight.wgsl` is the fix. See
[`findings/gemma4-attention-quirks.md`](./findings/gemma4-attention-quirks.md)
for the full writeup.

### 8. Attention scale is `1.0`, not `1/sqrt(head_dim)`

Gemma 4's `q_norm` and `k_norm` normalize each head to unit RMS
pre-attention, so the standard Transformer `1/sqrt(d)` softmax
compensation is dropped. The scale is hardcoded at `1.0` inside
`attn_score.wgsl`. Using the wrong scale flattens softmax by 16× on
HD=256 and smears attention uniformly across positions. See
[`findings/gemma4-attention-quirks.md`](./findings/gemma4-attention-quirks.md).

### 9. Mixed FFN widths

Producer layers (indices 0–14) have intermediate size 6144; consumer
layers (15–34) double it to 12288. Both are divisible by 4, so the
MR4 matmul variant (`matmul_quant_mr4.wgsl`) dispatches cleanly on
either width. Per-layer `I` is read from `config.intermediate_sizes[]`
via `getIntermediateSize(layerIdx, config)`.

## Forward pass at a glance

One forward pass encodes roughly the following, per layer:

```
attn.rmsNorm → linearQ → linearK* → linearV*
  → perHeadNorm(Q, K, V*) + RoPE(Q, K*)
  → kvCacheStore* → attnScore → softmax → attnOutput
  → linearOut → postNormAdd
ffn.rmsNorm → linearGateUp → geluMul → linearDown → postNormAdd
ple.linearInpGate → geluMul → linearPostProj → rmsNorm → skipScaleAdd
```

Starred kernels are producer-only; consumer layers skip linearK/V and
kvCacheStore and read from the mapped producer's cache.

Final pipeline:
```
final.rmsNorm → lmHead (matmul over vocab) → logitSoftcap
             → sample (argmax | topk256) → ...
```

## Load-bearing architectural decisions

**F16-everywhere on GPU.** Every weight is CPU-dequantized to F16 once
at upload, regardless of source type (Q4_K / Q5_K / Q6_K / Q8_0 / BF16 /
F16 / F32). One `matmul_quant.wgsl` covers all matmuls. The cost is
slow first load (~90 s for Q4_K_M on a fast link); the benefit is a
dramatically simpler correctness surface — one kernel, not eight.

**Gemma 4 E2B only.** This repo is not a model-agnostic inference
framework. Init asserts `per_layer_input_dim > 0` (PLE is
architectural); shaders encode the deltas above. Gemma 5, Gemma 4 4B,
and multimodal variants would require meaningful engine changes.

**Q4_K_M default, bring-your-own-URL escape.** The verified default is
`unsloth/gemma-4-E2B-it-GGUF`. Pass `model: '<full URL>'` to load any
other GGUF (quant tested: Q4_K / Q5_K / Q6_K / Q8_0 / BF16 / F16 / F32).
See [`findings/why-q4km.md`](./findings/why-q4km.md).

## Where to read next

- **Methodology** — [`methodology.md`](./methodology.md): how numerical
  equivalence is verified per-kernel against HuggingFace.
- **Findings** — [`findings/`](./findings/): discovered Gemma 4
  quirks worth knowing about.
- **Tuning** — [`tuning.md`](./tuning.md): device profile system and
  how to calibrate on new hardware.
- **Future work** — [`future-work.md`](./future-work.md): known
  deferred optimizations, framed as contribution targets.
