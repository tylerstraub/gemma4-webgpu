# Gemma 4 attention quirks

Two details about Gemma 4's attention block are silent-corruption
hazards for anyone porting the model to a new backend. Neither
appears in the Gemma 4 model card or in DeepMind's JAX docstrings;
both were found by comparing per-kernel hidden states against the
HuggingFace Transformers reference (see
[`../methodology.md`](../methodology.md)).

If you're porting Gemma 4 to MLX, ONNX, llama.cpp (for a new quant
kernel), custom CUDA, or any WebGPU variant — check both of these
before trusting generation output.

## Finding 1: `v_norm` is applied per-KV-head, weight ≡ 1

**The behavior**: between `linear_v` (projecting hidden to V space)
and `kv_cache_store`, V is RMS-normalized per KV head with a
learnable weight of all ones.

**Why you can miss it**:
- The Q4_K_M GGUF quantizer drops the `v_norm` tensor entirely —
  llama.cpp's quantizer treats a weight-1 RMSNorm as a no-op
  because it's algebraically equivalent to a plain rescale. But it
  isn't a no-op: a norm with weight=1 still normalizes magnitude,
  and you still need to apply it.
- The Gemma 4 model card describes `q_norm` and `k_norm` explicitly
  (both are applied per-head, with learnable weights); `v_norm` is
  omitted from the spec.
- Inspecting the JAX reference implementation also doesn't surface
  it directly; it's a single line inside a helper function.

**What goes wrong**:
V enters attention at ~67× the correct magnitude. Softmax(QK) is
fine (the scale is self-normalizing), but `attn_output = softmax @ V`
inherits V's inflation directly. The residual stream drifts
catastrophically across layers; generation produces ungrammatical
output within ~20 tokens.

**The fix**:
Apply RMSNorm per KV head (weights = 1) between `linear_v` and
`kv_cache_store`. In this repo, shader
[`shaders/per_head_rms_norm_no_weight.wgsl`](../../shaders/per_head_rms_norm_no_weight.wgsl)
is a fused weightless RMSNorm. It runs on producer layers only
(consumer layers read from the producer's already-normed cache).

**Code pointer**: in `src/engine.ts`, search for
`perHeadRmsNormNoWeight` — it's dispatched in the producer-layer
branch between the V linear projection and the KV cache store.

**How to verify in your port**:
1. Add a capture hook at `v` after the linear projection.
2. Add a capture hook at `k_v_cache` after the store.
3. Run a forward pass on a known prompt; compare the V magnitude
   before and after.
4. Reference V should be ~17% of pre-store magnitude; if yours is
   100%, `v_norm` is missing.

## Finding 2: Attention scaling is `1.0`, not `1/sqrt(head_dim)`

**The behavior**: in the attention scoring step,
`scores = Q @ K^T` (no scale factor). The standard Transformer
`1/sqrt(d)` compensation is dropped.

**Why**: Gemma 4 applies `q_norm` and `k_norm` per head with
learnable weights right before attention, and those norms are
calibrated to produce unit-RMS heads. Since Q and K already arrive
at unit scale, the `sqrt(d)` compensation isn't needed — it would
over-correct.

**Why you can miss it**:
- Most Transformer implementations treat `1/sqrt(d)` as
  architecturally universal. It's baked into `F.scaled_dot_product_attention`,
  Flash Attention variants, and most hand-written attention kernels.
- The compensation is often spelled `attention_scale = head_dim ** -0.5`
  in config objects and silently threaded into the kernel.
- Gemma 4's config doesn't explicitly say `attention_scale = 1.0` —
  it says `q_norm` and `k_norm` are applied, and the scale is
  implicit in the combined attention computation.

**What goes wrong**:
On `head_dim = 256` (Gemma 4 E2B's sliding-window layers), applying
`1/sqrt(256) = 1/16` flattens softmax by a factor of 16.
Attention becomes nearly uniform across all positions, so the
output is approximately `mean(V)` instead of attended V. The
residual stream still carries information (V itself is
informative), but attention no longer selects — generation produces
context-unaware output.

On `head_dim = 512` (the full-attention layers), the flattening is
~23×.

**The fix**:
Leave the scale at `1.0` inside `attn_score`. In this repo, see
[`shaders/attn_score.wgsl`](../../shaders/attn_score.wgsl) —
there's no scale multiply in the dot-product accumulator.

**How to verify in your port**:
1. Run a forward pass on a prompt where you know the right answer
   should attend heavily to a specific earlier token (e.g.
   "The capital of France is" — softmax should peak hard at
   "France").
2. Capture the softmax output of the final full-attention layer.
3. Compute entropy or look at the max-probability position.
4. If softmax is nearly uniform (max ~1/seq_len), scale is
   wrong. If it peaks sharply on a semantically-meaningful position,
   scale is right.

## Putting it together

Both findings were caught by running the engine's per-kernel
output through cosine similarity against the HuggingFace reference
at the `afterLayer:0` capture point. Cosine at layer 0 dropped to
~0.4, which is far below the F16-accumulation floor and therefore
pointed at a real algorithmic bug rather than precision drift.

Bisection:
1. `afterLayer:0` cos ≈ 0.41 → bug somewhere in layer 0
2. `attnOut:0` cos ≈ 0.35 → bug in the attention block
3. `postRopeQ:0`, `postRopeK:0` both cos ≈ 0.9997 → Q, K are fine
4. This pointed at V, the attention scores, or the output
   projection. The V path turned out to have two compounding bugs:
   `v_norm` missing (finding 1), and the attention scale wrong
   (finding 2). Fixing either alone left cosine at ~0.6; fixing
   both recovered >0.99.

See [`../journey/crosslab-diff-debugging.md`](../journey/crosslab-diff-debugging.md)
for the narrative version of the bug hunt.

## Acknowledgments

The `v_norm` discovery came from instrumenting the HuggingFace
reference with forward hooks on every module boundary — the hook on
`GemmaV2RMSNorm` inside the attention block surfaced the call that
didn't appear in the model card or the initial port. If you're
doing this kind of port in the future, forward hooks on every
`nn.Module` in the reference are a cheap and thorough way to make
sure you haven't missed one.
