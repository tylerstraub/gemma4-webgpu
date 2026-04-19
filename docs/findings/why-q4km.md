# Why Q4_K_M

The default GGUF this engine loads is
[`unsloth/gemma-4-E2B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF)
at the Q4_K_M quantization — 2.89 GiB on disk, mixed Q4_K + Q5_K +
Q6_K + F32 + BF16 internally. This page explains why, what the
alternatives look like, and how to swap.

## The architectural floor

Gemma 4 E2B's Per-Layer Embedding (PLE) table is
`[vocab × (num_layers × per_layer_input_dim)]` =
`[262144 × (35 × 256)]` ≈ **2.35 GB at BF16**.

The PLE table is hot — every token access reads one row of 256
floats for every layer. It cannot be quantized aggressively without
perceptible quality loss (downstream this shows up as degraded
grammar and factual recall). The GGUF variants we looked at all
keep the PLE as Q8_0 or Q5_K at best; below that, generation quality
drops visibly.

So the minimum on-disk size for a well-behaved Gemma 4 E2B GGUF is
roughly **2 GB**. There's no sub-2GB variant that's usable; don't
chase it.

## The choices in the Q4 band

The practical candidates, within the ~2.3–3.2 GB band:

| Variant | Size | PLE | Matmuls | Notes |
|---|---:|---|---|---|
| `Q4_K_S` | ~2.5 GB | Q4_K | Q4_K | Smallest usable; slight quality cost on long outputs |
| **`Q4_K_M`** | **2.89 GB** | **Q5_K** | **Q4_K + Q6_K (mixed)** | **Default** — PLE upgraded to Q5_K; some matmuls Q6_K |
| `Q5_K_M` | ~3.3 GB | Q6_K | Q5_K | A step up, ~15% bigger |
| `Q6_K` | ~3.8 GB | F32 | Q6_K | PLE uncompressed; marginal quality gain |
| `Q8_0` | ~5.3 GB | F32 | Q8_0 | Reference quality; 2× our default size |

## Why `Q4_K_M` won

Five reasons, decreasingly important:

1. **It fits.** 2.89 GB downloads acceptably on consumer home
   connections (~60–120 s on 200–400 Mbps cable, the typical user
   connection). `Q5_K_M` at 3.3 GB pushes into "people close the
   tab" territory on slower connections. `Q4_K_S` saves 12% for a
   detectable quality regression on longer rollouts.

2. **PLE stays at Q5_K.** Unsloth's Q4_K_M variant specifically
   upgrades the PLE from Q4_K to Q5_K. Given how hot the PLE is and
   how sensitive generation quality is to it, this is the kind of
   mixed-precision decision that pays off out-of-proportion to the
   size cost.

3. **Correctness is measurable.** `crossLabDiff` against the HF
   `google/gemma-4-E2B-it` reference (at BF16) shows Q4_K_M layer
   cosines land within F16-accumulation-noise of the reference.
   That means the quant error is within the arithmetic error
   already accepted by running the inference in F16 — we're paying
   no visible extra cost for the 4-bit weights.

4. **Llama.cpp's mixed-quant strategy is mature.** The K-quant
   family in llama.cpp has seen years of evaluation; the S/M/L
   tiers reflect real tradeoffs measured across many models.
   Importing that curation saves us the eval infrastructure we
   don't yet have.

5. **Unsloth's imatrix calibration.** The `unsloth/` GGUFs are
   importance-matrix-calibrated, which shifts quantization error
   away from the tensors that matter most. Free quality.

## The escape hatch: bring your own URL

You're not locked into the default. Pass a full URL and the
engine's GGUF parser handles Q4_K / Q5_K / Q6_K / Q8_0 / BF16 / F16
/ F32:

```ts
const engine = await createGemmaEngine({
  model: 'https://huggingface.co/<author>/<repo>/resolve/main/<file>.gguf',
});
```

Tested ✓ — the dequant path is shared across variants; all quantize
down to F16 at upload.

Things to watch when swapping:

- **SHA256 and size**: run `scripts/audit_gguf.mjs <URL>` to verify
  the file is what you think it is before the engine spends 90 s
  downloading it.
- **Architectural invariants**: the engine asserts
  `per_layer_input_dim > 0` at init. Any non-Gemma-4 variant fails
  this check — by design, since the shaders encode Gemma 4's
  specifics.
- **crossLabDiff**: run the sweep after swapping. Reference tensors
  in `public/ref/smoke.npz` were captured against the BF16
  HuggingFace model; any quant variant's cosine should land at
  ≥ 0.99 on `logits` against that reference. If it doesn't, the
  quant is damaging output — try a higher tier.

## What's worth evaluating next

Good candidates for a future calibration / quality-comparison pass,
if you have eval infrastructure:

- **`unsloth/gemma-4-E2B-it-GGUF:Q5_K_M`** — 15% larger, should show
  small cosine improvement at deep layers.
- **`unsloth/gemma-4-E2B-it-GGUF:UD-Q4_K_XL`** (dynamic quant if
  Unsloth publishes it for Gemma 4) — same size band, potentially
  better quality via dynamic per-tensor quant.
- **`bartowski/gemma-4-E2B-it-GGUF:IQ3_M`** — sub-Q4 i-matrix
  variants for low-bandwidth connections. Likely crosses the
  correctness boundary at deeper layers; worth measuring to know.

## Multimodal extension

Gemma 4 E2B's `general.tags = ["any-to-any"]` refers to the base
model — it has vision and audio projectors documented in the model
card. This engine is text-only: the vision/audio tensors (the
`mmproj.*` family) are not loaded and the input path doesn't accept
image or audio.

The extension path for multimodal is additive:
1. An `mmproj.*` sidecar GGUF with the projector weights
2. New WGSL shaders for the projectors (vision uses a patchified
   ViT-style head; audio is a CNN)
3. A projection of the image/audio latents into the token embedding
   space, interleaved with text token embeddings before the main
   forward pass

This is on [`../future-work.md`](../future-work.md) as an extension
point, not an in-progress feature. If you're interested,
`research-finding` issue first so scope is clear before code.
