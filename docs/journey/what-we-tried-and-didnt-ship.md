# What we tried and didn't ship

Dead-end paths and deferred optimizations. Each entry includes what
was tried, why it didn't ship, and — where applicable — the
conditions under which revisiting it would make sense.

Publishing this is deliberate: future contributors save the time
it would cost to re-discover these dead ends, and "we considered
this and here's why it isn't live" is more useful to a porter than
silence.

## Subgroup-aware matmul — regressed ~5%, reverted

**What it was**: a variant of the main matmul shader that used
WebGPU subgroup intrinsics (`subgroupAdd`, `subgroupBroadcast`) to
amortize the tree-reduce across warps instead of through shared
memory.

**What happened**: On the reference RTX 5090 / Chrome stable, the
subgroup variant was ~5% *slower* than the plain shared-memory
tree-reduce. The regression was reproducible across multiple
benches.

**Why we think**: Chrome's subgroup codegen as of the test window
compiled `subgroupAdd` into a sequence of DPP-style shuffles that
aren't as well-optimized as the compiler's hand-rolled
shared-memory reduce. The shared-memory path hits a well-trodden
fast path in the NVIDIA driver; the subgroup path doesn't.

**When to revisit**: when Chrome's subgroup codegen matures, or
when calibrating a profile for hardware where the tradeoff
reverses. The tuning profile shape already has
`features.subgroups: 'use' | 'avoid' | 'auto'` so flipping it in
one profile without rewriting the default is a one-line change
once the alternate shader variant is committed.

## Option C — sampled/blocked attention — deferred

**What it is**: at longer contexts, most attention weight
concentrates on a small subset of positions (the sliding-window
layers have this concentration architecturally; the full-attention
layers develop it empirically). Option C would skip the scoring
work on low-importance positions — either by bucketing positions
into blocks and attending at block-level, or by sampling from a
score-sketched distribution.

**Why deferred**: on the reference workload (ctx=2048, batch=1),
the predicted win is ~3–5% — real, but below the threshold that
made sense to ship alongside the release. At longer contexts
(8k+) the win scales meaningfully; at the current default it
doesn't pay for the correctness-surface complexity.

**The correctness wrinkle**: sampled attention isn't numerically
equivalent to dense attention, by design. crossLabDiff against a
dense reference would show divergence that *isn't* a bug — so
either the reference has to grow a "sampled mode" or the
correctness check has to shift to a generation-quality comparison.

**When to revisit**: when either (a) someone's running this engine
on longer contexts where the win is real, or (b) the
generation-quality evaluation infrastructure lands (see
[`../future-work.md`](../future-work.md) for that infrastructure's
scope).

## Fusion sweep on single-workgroup ops — deferred

**What it is**: the forward pass has ~100 single-workgroup
dispatches (small norms, adds, per-head ops). Each costs ~1–2 µs
of CPU pass-encode overhead and ~30 ns of GPU launch. Fusing pairs
or triples of chained single-workgroup ops over the same tensor
into one kernel would reduce both.

**What we did**: the `fused_norm_add.wgsl` and
`fused_per_head_norm_rope.wgsl` shaders are shipped. They together
eliminated ~10 dispatches per forward.

**Why more wasn't done**: diminishing returns. The top-2 fusion
targets (norm+add, norm+rope) are done; the next few candidates
would each save ~1% and the shader count grows linearly.
Correctness-surface cost per fusion is fixed; value shrinks.

**When to revisit**: if someone's profiling on a different vendor
where the CPU-overhead ratio is different, more fusions may pay
off. Also if pass encoding becomes a measured bottleneck (it isn't
on the reference machine — the pipelined decode path hides most
of the CPU-side overhead anyway).

## Quant-in-shader — deferred

**What it is**: instead of CPU-dequantizing weights to F16 at
upload, store them on GPU in their native Q4/Q5/Q6 format and
decode on-the-fly in the matmul shader.

**The win**: ~2× faster first load (skip CPU dequant, ~90 s → ~45
s). No tps change (the matmul bandwidth is the same once the
weights are on GPU).

**Why deferred**: load time is annoying but it only hits once per
session. tps matters every session. The F16-everywhere decision
was load-bearing for the correctness story — one matmul shader,
one dequant path, one set of bit-patterns to verify against the
reference. Adding shader-side quant decode multiplies the
correctness surface by the number of quant formats supported (7
today). Not worth it for load-time only.

**When to revisit**: when a generation-quality evaluation suite
is in place. Shader-side decode can subtly change numerics (F32
accumulator vs on-the-fly dequant's lower precision); knowing
whether that costs 0.1% MMLU or 2% requires eval infrastructure
we don't have yet.

## Multimodal (vision + audio projectors) — deferred

**What it is**: Gemma 4 E2B's `general.tags = ["any-to-any"]` — the
base model supports image and audio inputs via projector networks
that live in an `mmproj.*` sidecar GGUF. Loading both the projector
weights and the main model, running the projector forward pass,
interleaving its output with text token embeddings, and extending
the context window for image/audio tokens.

**Why deferred**: substantial scope. New shader pipelines for
vision (ViT-style patch projector) and audio (CNN-style
convolutional projector). New input API surface (image / audio
data). Correctness verification against the HF multimodal pipeline
(non-trivial — the HF image preprocessing chain alone has ~5
configurable steps).

**Why text-only first**: text generation is the most widely
useful mode and the core engine can be verified, shipped, and
used standalone. Multimodal is additive and can land later
without restructuring the text path.

**When to revisit**: when there's community demand (open a
`research-finding` issue) or when someone wants to take it on.
The extension points are documented in
[`../findings/why-q4km.md`](../findings/why-q4km.md) and
[`../future-work.md`](../future-work.md).

## Lessons

- **Revert numbers that don't survive a second measurement.** The
  subgroups variant looked like a win on the first bench, then
  regressed consistently on subsequent benches. One measurement
  isn't a measurement; take five and look at the median.

- **Complexity budget is real.** Every shipped optimization grows
  the correctness surface. Deferring a 3% win that doubles the
  number of quant decoders you have to verify is the right call
  on a reference implementation.

- **"Deferred" is informative.** Publishing what was considered
  and *why it wasn't done* is as useful to future readers as
  publishing what shipped. This page is the record of that
  decision process.
