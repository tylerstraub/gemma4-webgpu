# crossLabDiff debugging: the v_norm hunt

A narrative walk-through of how the two Gemma 4 attention findings
were localized. The findings themselves are documented in
[`../findings/gemma4-attention-quirks.md`](../findings/gemma4-attention-quirks.md);
this page is the story of how they got found.

If you're reading this to learn a methodology for localizing
correctness bugs in a ported LLM inference stack, the lesson is:
get a per-kernel cosine-similarity trace against a trusted
reference *first*, before trying to optimize anything.

## The symptom

First working Gemma 4 forward pass produced output like:

> "I'm I'm the I the I'm you you to I you I I to you..."

Tokens generated, logits had some signal, but the output was
nonsense. Classic "something is approximately working but
seriously wrong" shape.

## First reflex that didn't help

Staring at the generation output. The tokens aren't obvious bugs —
they're real vocabulary items. The repetition pattern is consistent
with attention being broken (heads attending uniformly = output
≈ `mean(V)` = similar logits every step), but it's also consistent
with a dozen other problems. Inspection doesn't bisect.

## The setup that did help

A Python-side HuggingFace Transformers reference with forward hooks
on every `nn.Module` inside the attention block, a text prompt, and
a specific token position to capture at. The hooks dumped hidden
state tensors to a `.npz` file:

```python
# Inside the forward hook:
captures[f"afterLayer:{layer_idx}"] = hidden_state.cpu().numpy()
captures[f"attnOut:{layer_idx}"] = attn_output_before_linearOut.cpu().numpy()
captures[f"preRopeQ:{layer_idx}"] = q_before_rope.cpu().numpy()
# ... and so on
```

On the engine side, `captureHidden()` ran the forward pass on the
same prompt to the same position with a `stopAt` capture point, read
back the relevant GPU work buffer as `Float32Array`, and compared
against the Python-dumped tensor via cosine similarity.

Building this harness took about a day. It paid for itself in the
first hour of using it.

## The sweep

With the harness in place, running the correctness sweep was a
single for-loop:

```
embed        cos = 0.997232
afterLayer:0 cos = 0.414
```

That's a catastrophic drop at layer 0. The embedding is right (cos
0.997 is pure F16 rounding noise); the output of layer 0 is wrong.

F16 accumulation across one layer loses you maybe 0.001 in cosine.
A cos of 0.414 at layer 0 is 400× more error than float precision
explains. That's an algorithmic bug, not a precision issue.

## Bisecting within layer 0

```
attnOut:0    cos = 0.35
```

Even worse — so the bug is before the attention-output stage
(`attnOut` is captured just after `attn_output` writes the attended
V values). Narrow further:

```
preRopeQ:0   cos = 0.9997
preRopeK:0   cos = 0.9997
preRopeV:0   cos = 0.9997
```

Q, K, V linear projections are correct (cosines indistinguishable
from the embedding). So the bug is somewhere between the linear
projections and the attention-output kernel.

```
postRopeQ:0  cos = 0.9997
postRopeK:0  cos = 0.9997
```

RoPE is correct on Q and K. (No `postRopeV` — V doesn't get RoPE.)

So the bug is in one of: attention scoring (Q @ K), softmax, or
attention output (softmax @ V). Or something happening to V between
the linear projection and the scoring kernel.

## Reading the reference more carefully

At this point the working theory was "something's wrong with the
softmax scale or the V path." Staring at the HuggingFace reference
code:

```python
# Gemma4Attention.forward
...
query_states = self.q_norm(query_states)
key_states = self.k_norm(key_states)
value_states = self.v_norm(value_states)   # <--- wait, what?
...
attn_output = scaled_dot_product_attention(
    query_states, key_states, value_states,
    scale=1.0,                              # <--- and what?
)
```

Two things jumped out:

1. **`self.v_norm`** — a V-side RMSNorm. The Gemma 4 model card
   doesn't mention one, and our initial port didn't include one.
   But `v_norm` is a real module in the reference, with a weight
   tensor called `v_norm.weight`.

   Checking the GGUF: no `v_norm.weight` in the tensor list. Not
   under any naming variant. Huh?

   The llama.cpp GGUF quantizer treats RMSNorm with weight ≡ 1 as
   a no-op and drops the tensor from the export. `v_norm.weight`
   is all ones (we verified by dumping `model.layers[0].self_attn
   .v_norm.weight` from the HuggingFace model — all 1.0). So the
   quantizer dropped the tensor; the port skipped the module; and
   V entered attention unnormed, at ~67× its correct magnitude.

2. **`scale=1.0`** — the attention scale. Standard Transformer
   attention uses `1/sqrt(head_dim)` as the scale. The reference
   says `scale=1.0`. On `head_dim=256`, that's a 16× difference;
   applying the wrong scale would flatten softmax by that factor.

   Why `scale=1.0`? Because `q_norm` and `k_norm` already normalize
   Q and K to unit RMS, so the compensation is redundant. Gemma 4
   bakes the "scaling" into the norms.

## Fixing both

- Added a weightless fused per-head RMSNorm on V between `linearV`
  and `kvCacheStore`:
  [`shaders/per_head_rms_norm_no_weight.wgsl`](../../shaders/per_head_rms_norm_no_weight.wgsl)
- Removed the `/sqrt(head_dim)` scale factor from
  [`shaders/attn_score.wgsl`](../../shaders/attn_score.wgsl).

After the fix, the sweep:

```
embed        cos = 0.997232
afterLayer:0 cos = 0.999841
afterLayer:33 cos = 0.941056
logits       cos = 0.997847
```

Generation output:

> "I'm doing well, thank you for asking! I'm ready to assist you.
>  How are you today? And what can I help you with? 😊"

Not a revolution in LLM output, but — coherent, grammatical,
context-aware. Exactly what Gemma 4 E2B should produce.

## Lessons

- **Per-kernel cosine similarity is a superpower.** Knowing the
  bug is "before attnOut, after linearV" eliminates 90% of the
  codebase in one observation. Generation inspection would never
  get you there.

- **Forward hooks on every nn.Module in the reference.** Five
  minutes of Python, a day of engine-side harness, and you get a
  bisection tree for every future bug you'll hit. Worth the
  investment.

- **A weight-1 RMSNorm is not a no-op.** The learnable weight
  being all-ones doesn't make the norm itself a no-op; the norm
  still divides by the RMS. Quantizer dropping the tensor is
  correct (per llama.cpp convention) but the engine has to know
  to apply the norm anyway. Check every `nn.Module` in the
  reference, including the ones that look trivial.

- **Attention scale isn't universal.** The `1/sqrt(d)` compensation
  is a convention, not a law. Architectures that normalize Q/K
  pre-attention (Gemma 4, some OLMo variants, various Mistral
  forks) don't want it. Always verify by reading the reference,
  not by assuming.

## Where to read the finding

[`../findings/gemma4-attention-quirks.md`](../findings/gemma4-attention-quirks.md)
is the evergreen writeup — what the bug is, why it happens, how to
fix it in any backend. This page is the narrative; that page is the
reference.
