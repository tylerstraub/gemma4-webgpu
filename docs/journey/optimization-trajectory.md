# Optimization trajectory

From 13 tps to 127 tps on RTX 5090 / Chrome stable, Gemma 4 E2B at
Q4_K_M. Four landmarks, roughly in order of magnitude contribution.

## Baseline: 13 tps

The first correctness-green Gemma 4 forward pass — crossLabDiff
sweep all-green against the PyTorch reference — sat at ~13 tps on
short generations.

At 13 tps, most of the forward-pass time sat in matmul kernels with
non-coalesced weight reads. Every warp of 32 threads was issuing 32
scattered F16 loads instead of one coalesced 64-byte transaction.

## Landmark 1: Weight-coalescing matmul — 13 → 58 tps (4.5×)

The first matmul kernel kept the outer-M loop threaded across
workgroups (one output row per workgroup of 256 threads) and had
threads within a workgroup cooperate on the N-long dot product.
That's the right skeleton but the memory-access pattern in the
baseline had each thread reading `weight[m, k]` for its own `k` —
which for `k` strided by thread-index *across a row* gives you
coalesced reads if and only if weights are laid out row-major with
N as the fast dimension. They were not.

The fix: transpose the hot matmul weights to put N fast (the
accumulator dimension the threads cooperate on), so thread `tid`
reads `weight[m, tid..N stride WG]`. After the transpose, 256
threads issue 16 coalesced 1024-byte transactions per iteration
instead of 256 scattered 64-byte reads.

4.5× speedup on short config. The win was "free" — pure memory
bandwidth, no algorithmic change, no correctness risk.
[Shader](../../shaders/matmul_quant.wgsl) is the current `matmulQuant`.

## Landmark 2: Option B — pipelined decode — 58 → 92 tps (1.59×)

With matmuls coalesced, the next bottleneck was CPU overhead. At
~11 ms/token wall-clock, the WebGPU submit→callback scheduling
sat at ~2.7 ms — about 25% of each token. The GPU was idle for
that slice while JavaScript waited for the argmax readback.

"Option B" is a depth-2 pipeline for the greedy-sampling case
(`temperature === 0` + `repPenalty ≤ 1`):

1. The argmax is computed on GPU and its result copied directly
   into the `embeddingLookup` and `pleStage1` uniform buffers that
   the NEXT forward will read. No CPU round-trip needed for the
   sampled token.
2. The engine submits forward N+1 *before* awaiting forward N's
   readback. The GPU compute of N+1 hides inside the CPU-side wait
   on N.
3. A ping-pong pool of two argmax readback buffers lets both
   forwards have a distinct buffer to map (WebGPU forbids
   concurrent maps on one buffer).

This path only works for pure greedy (no topk, no repetition
penalty) because the speculative submit has to know the next
token's embedding input without consulting CPU state. Topk and
repPenalty paths stay on the serial route.

1.59× speedup on top of landmark 1. The win is almost pure
scheduling-overhead reclamation; per-token GPU compute is
unchanged. See [`src/engine.ts`](../../src/engine.ts) — search for
`argmaxReadbackPool` and `decodeDepth`.

## Landmark 3: Option A — MR4 FFN matmul — 92 → 117 tps (1.27×)

At 92 tps, `ffn.linearGateUp` was the single largest kernel in the
forward pass — it runs twice per layer at I=6144 (producer) or
I=12288 (consumer), which is the widest matmul outside of lmHead.

"Option A" is a 4-row-per-workgroup variant of the matmul:

- Each workgroup produces 4 consecutive output rows instead of 1.
- The input vector is read once, multiplied against 4 weight rows
  (one coalesced fetch per row).
- The tree-reduce is 4× wider in shared memory but takes the same
  number of barriers.

The key constraint: MR4 needs `I` divisible by 4. Gemma 4 E2B's
intermediate sizes (6144 and 12288) both are.

1.27× short-config speedup ("Option A" landed alongside a couple
of minor Option-B refinements; the combined 92→117 is mostly MR4).
[Shader](../../shaders/matmul_quant_mr4.wgsl), dispatched from
`ffn.linearGateUp` when the active tuning profile selects R=4.

## Landmark 4: Option A extension to long config — 117 → 127 / 108 → 116

Long-context runs (maxTokens=200) initially sat ~10 tps behind
short runs due to the growing KV cache inflating the attention
kernel times. Option A's FFN win applies uniformly across prefill
and decode, so both short and long benefited. Post-Option-A:

- short (maxTokens=64, "Hello, how are you?"): **127 tps**
- long (maxTokens=200, raven/crow default): **116 tps**

## Cumulative ratios

- M0 → current: 13 → 127 = **9.8×**
- A different non-WebGPU in-browser approach peaks at ~8.5 tps on
  comparable hardware; the current engine beats it by **14.9×**

## Why we stopped optimizing

After landmark 4, the short-term ceiling estimate for remaining
category-A and -B optimizations combined was ~160 tps short / ~148
tps long. Getting there would have required ~4–5 more landmarks of
diminishing size — fusion sweep (~3%), Option C sampled attention
(~4% at this context length), lmHead+argmax fusion (~1%), and so
on.

The decision was: publish the reference engine + methodology now
at 127 tps, let the curated docs and the contribution path
(device reports, community findings, the profile registry) extend
the work. Shipping the methodology has more leverage than
squeezing another 25% out on one hardware config.

See [`../future-work.md`](../future-work.md) for the deferred
optimizations, framed as contribution targets with predicted win
bands.

## Lessons

- **Memory bandwidth first.** Before writing any clever kernel,
  make sure the boring kernels are coalesced. 4.5× for an
  afternoon's weight-layout transpose is the best speedup-to-risk
  ratio in the ladder.
- **CPU-side overhead is real on WebGPU.** The ~2.7 ms of
  submit→callback slack per token was hiding in plain sight until
  the CPU profiler exposed it. If you're tps-bound and the GPU
  profiler shows headroom, check CPU scheduling.
- **Match kernel shape to model shape.** MR4 works because Gemma
  4 E2B's intermediate sizes are divisible by 4. It wouldn't on a
  model with I=6145. Before porting an optimization, check the
  hypotheses it encodes about shape.
- **Stop when the curve flattens.** The 10% landmarks turn into 3%
  landmarks turn into 1% landmarks. At some point the hours spent
  on the next landmark exceed the value of the win. Calibrate that
  tradeoff deliberately.
