# Future work

Deliberately-deferred optimizations, framed as contribution targets.
Each entry has a predicted win band, a complexity note, and a pointer
to where in the code to start. Numbers are base-rate priors for the
reference hardware; expect drift across vendors.

This is the closest thing the repo has to a roadmap, but it's a menu
rather than a plan — pick what excites you. The correctness surface
is shared (crossLabDiff must stay green), so parallel contributions
don't collide.

## Category A: Kernel fusion

Low complexity, small-to-medium wins. Good first-time contributions.

### lmHead + argmax fusion

**Predicted win**: ~1–2% on greedy short generations.
**Complexity**: Medium. Single new shader, two dispatch sites to wire.
**Where to start**: `shaders/argmax.wgsl` + `shaders/matmul_quant.wgsl`
— fuse the vocab-sized matmul output with the argmax reduction so the
intermediate 262144-wide logits buffer doesn't round-trip.

The greedy path already uses on-GPU argmax and a ping-pong readback;
this just eliminates one intermediate. Non-greedy paths keep the
current split (argmax fusion doesn't help topk256).

### lmHead + topk fusion

**Predicted win**: ~0.5–1% on sampled generations.
**Complexity**: Medium-high. Topk is already shared-memory heavy;
fusing with the matmul's tree-reduce requires careful bank-conflict
management.
**Where to start**: `shaders/topk256.wgsl`.

### Fusion sweep on single-workgroup ops

**Predicted win**: ~2–4% short config on RTX 5090.
**Complexity**: Medium-high. Many small shader changes.
**Where to start**: look at the list of single-workgroup dispatches
in the profile output — norms (`rms_norm`, `per_head_rms_norm*`),
adds (`add`, `fused_norm_add`), and the small PLE ops. Each pair of
chained single-workgroup ops over the same tensor can fuse into one
kernel. The `fused_norm_add.wgsl` and `fused_per_head_norm_rope.wgsl`
are the template.

The win is from ~20 fewer pass encodes per forward (each ~1–2 µs of
CPU overhead + ~30 ns GPU barrier). Diminishing returns after the
top few fusions; profile first.

## Category B: Algorithmic

Medium-to-large complexity, larger wins at longer sequences.

### Sampled / blocked attention (Option C)

**Predicted win**: ~3–5% at ctx=2048; scales with sequence length.
Meaningful at 8k+.
**Complexity**: High. New attention-score shader variant; new softmax
variant; selector logic at dispatch time.
**Where to start**: `shaders/attn_score.wgsl` + `shaders/softmax.wgsl`.

The idea: instead of computing attention scores for every position up
to the current token, subsample positions past some threshold (e.g.
sliding-window layers past a certain distance) and reconstruct. Works
because attention distributions are heavily concentrated at Gemma 4's
softcap-normalized scale.

Correctness constraint: crossLabDiff may diverge at longer contexts
because the reference doesn't sample. A separate "sampled mode"
reference may need adding.

### Long-context speculation

**Predicted win**: varies wildly by workload (2×–0.5×).
**Complexity**: Very high. Draft model + verification harness.
**Where to start**: not today — the architectural plumbing for a
draft model is missing and would be substantial. Noted for
completeness.

## Category C: WebGPU feature-gated

Low complexity when the feature lands; large predicted wins. Blocked
on browser feature delivery.

### Cooperative-matrix matmul (`chromium-experimental-subgroup-matrix`)

**Predicted win**: ~1.2× ceiling on the matmul family.
**Complexity**: Medium when available. New shader variants using the
subgroup-matrix intrinsics; profile bit-equivalence against the
existing path.
**Blocked on**: Chrome stable exposing the feature. Re-check on
each Chrome version bump — the flag name is
`chromium-experimental-subgroup-matrix`. Adapter feature check:
`adapter.features.has('chromium-experimental-subgroup-matrix')`.

The tuning profile shape already has `features.subgroupMatrix` as a
`'use' | 'avoid' | 'auto'` flag; when the feature lands, flip
profile defaults and register the variant shader.

### Subgroup-aware matmul variants

**Predicted win**: ~3–5% on some vendors; regression on others.
**Complexity**: Medium. New shader using `subgroupBroadcast` / warp
intrinsics to amortize the tree-reduce.
**Status**: tried during development, regressed ~5% on Blackwell with
Chrome's current subgroup codegen — reverted. Revisit when Chrome's
subgroup path matures, or when calibrating a profile for hardware
where the tradeoff reverses.

## Category D: Weight / memory

### Quant-in-shader

**Predicted win**: Reduces first-load time by ~2× (skip CPU dequant).
**Complexity**: High. Requires shader-side Q4_K_M decode; requires
a generation-quality eval loop (MMLU, perplexity) since F16-accum
vs Q4-accum differs numerically.
**Status**: Out of scope for this repo's current "numerical
equivalence first" correctness story. The F16-everywhere decision is
load-bearing; don't undo it without eval infrastructure landing
first.

### Streaming load during first tokens

**Predicted win**: Perceived latency improvement, not tps.
**Complexity**: Medium. The uploader already streams from HF
Range-fetched; the engine could start running forward passes after
early layers land while later layers still upload in the background.

### Alternative quant support

**Predicted win**: Enables more community models.
**Complexity**: Low (each decoder is ~50 lines). Tested today:
Q4_K / Q5_K / Q6_K / Q8_0 / BF16 / F16 / F32. Missing:
Q3_K / Q2_K / IQ variants.
**Where to start**: `src/gguf.ts` has the decoder registry.

## Category E: Evaluation infrastructure

### Generation-quality suite

**Not a perf optimization** — but essential before shipping any perf
optimization that crosses a numerical-equivalence boundary
(quant-in-shader, sampled attention, speculative decoding).

A harness that runs MMLU, a small perplexity set, and a handful of
coding/reasoning prompts with reproducible seeds, producing a scored
report. Could live as a separate repo that depends on this engine via
commit pin.

### Long-context correctness

The reference `smoke.npz` is at position 14. A longer-context
reference (~1k tokens of prefill + capture at the end) would
validate the sliding-window path and the KV-cache sharing pattern
more thoroughly.

## Contribution shape

For any of the above:

1. Open a `research-finding` issue with the plan + predicted win band
2. Implement on a branch
3. crossLabDiff sweep green (no regression beyond archival baseline)
4. Bench before/after, ideally on multiple hardware (or flag which
   hardware you measured on)
5. PR with the predicted-vs-measured delta in the description
6. Durable findings land in [`calibrations.md`](./calibrations.md)
   and (if worth it) in tuning-profile updates

Prediction accuracy is itself a calibration signal — if you predict
5% and measure 8%, that's a calibration update worth writing down.
If you predict 5% and measure 0%, that's a more interesting
calibration update.
