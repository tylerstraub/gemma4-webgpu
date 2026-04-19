# Perf calibration process

How the base-rate priors in [`../calibrations.md`](../calibrations.md)
were built, and why calibration discipline matters more than
individual measurements.

If you take one thing from this page: **write down what you expect
before you measure.** Divergence between the prediction and the
measurement is itself the signal worth writing up.

## Why calibration discipline matters

LLM inference optimization is a long tail of small wins. Any
individual bench result is noisy by ±3%. Without priors, it's hard
to tell whether a 5% measured win is real or variance.

Once you have priors — "I expect this kernel to run at ~54% of
peak bandwidth because that's what unfused F16 matmul on this
silicon hits" — measurements become diagnostic:

- If the measurement matches the prior: no new information, but
  confidence the prior is still valid.
- If the measurement is 30% below prior: something regressed.
  Start looking.
- If the measurement is 30% above prior: either the prior was
  wrong or you just did something clever. Figure out which.

All three cases are more useful than "bench returned 91 tps, is
that good?"

## The base-rate shape

The calibrations are anchored to one hardware / browser / model
config and report:

- **Per-kernel p50 times** (from timestamp-query profiler)
- **Per-kernel BW/launch classification** (informed by the time
  divided by the theoretical minimum from weight bytes × peak BW)
- **Forward-pass wall-clock budget** (sum of the above, plus CPU
  overhead)
- **Cumulative ratios at each optimization landmark** — because
  "Option A won 8%" only means something alongside "what was the
  kernel mix at the time"

The discipline is: every optimization result carries a
before/after number *and* the config it was measured under. Tps
without config is noise.

## The predict-before-measure cadence

For any proposed optimization, the cadence was:

1. **Classify the target kernel.** Is it BW-bound or launch-bound?
   Estimate the current extract percentage.
2. **Predict the win band.** If the kernel is at 54% of peak and
   the optimization moves toward 75%, that's a ~39% improvement
   on that kernel. The kernel takes X% of the forward pass, so
   the whole-forward win is roughly X × 39% (minus whatever
   overhead the change adds).
3. **Commit the prediction.** Write it in the PR description before
   measuring: "expect 2.5–3.5% whole-bench win."
4. **Implement.**
5. **Measure.** 5 runs at each config.
6. **Compare predicted to measured.** If they match within
   noise, the prior is still valid. If they diverge, figure out
   why before landing.

This cadence is slower than "just try stuff" but catches a
surprising class of bugs — the ones where your change is wrong
but your kernel still happens to run fast.

## A worked example

**Option A (MR4 FFN matmul) prediction**:
- Target kernel: `ffn.linearGateUp`. Measured p50 before Option A:
  ~3.1 ms. That's ~37% of the forward-pass budget of 8.3 ms.
- Kernel is BW-bound. MR4 coalesces input-vector reads 4× (one
  input read per 4 weight rows instead of per 1). The input
  vector is tiny vs the weight matrix, so input-side BW savings
  are small. But MR4 also amortizes the shared-memory
  tree-reduce barrier count across 4× more FMAs, which hides
  about half of each barrier's ~30 ns.
- Predicted win on this kernel: ~15% (not huge, because the
  kernel was already well-coalesced).
- Predicted whole-bench win: ~6% (15% × 37%).

**Measured**:
- Kernel p50 after MR4: ~2.4 ms. That's a 23% kernel-time
  reduction — larger than predicted.
- Whole-bench tps: 92 → 117 for a 27% tps win.

**Divergence diagnosis**:
- The 27% tps win is 4× larger than the 6% predicted. Something
  else moved.
- Looking at the full profile: after the MR4 change, we also
  un-stalled the `ffn.geluMul` and `ffn.linearDown` kernels that
  were waiting behind `ffn.linearGateUp` — the GPU pipeline was
  filling more completely.
- The MR4 kernel's time reduction alone wasn't the whole story;
  it was that reducing this kernel's time also improved the
  overlap of the downstream kernels.

The diagnosis informed the base-rate update: "large kernels near
the critical path win more than their fraction of the forward when
their reduction also enables better overlap." That's now a prior
for future optimizations.

## When calibration divergence is a bug

Sometimes the prediction and measurement diverge because your
change is wrong, not because the prior is stale. A specific
example:

- Prediction: fusing norm + rope into one kernel should save the
  ~1 µs pass-encode cost, measured at ~0.3% of forward-pass time.
- Measurement: fused version was *slower* by ~5%.
- Diagnosis: the fused kernel had higher register pressure than
  either of the two unfused kernels, spilling to local memory on
  this hardware. The 1 µs save was real; the 5% regression was
  real; the register spill wasn't predicted.
- Fix: careful kernel layout to keep register pressure low, or
  accept that some fusions aren't worth it on this hardware.

The lesson: calibration priors are about the *units of work* (BW,
launches, barriers). Register pressure and instruction scheduling
are second-order effects that the priors don't cover. When
divergence appears, check both.

## Building your own priors

If you're porting this engine to new hardware, the workflow that
built the 5090 priors:

1. Run the profile: `await window.lab.profile(...)`.
2. For each kernel, compute the theoretical minimum time:
   `bytes_moved / hardware_peak_bandwidth`.
3. Divide measured p50 by theoretical min → extract percentage.
4. Classify: >40% extract = BW-bound. <10% extract = launch-bound.
   In between = either mixed or un-tuned.
5. Write it down.

Then for each hypothesized optimization:

1. Predict which classification the kernel will move to.
2. Estimate the new extract percentage.
3. Work backward to the predicted kernel time.
4. Multiply by the fraction of forward the kernel consumes.
5. Expect noise of ±20% on the whole-bench prediction.

The prediction discipline is what makes the 5090 priors useful.
The measurements themselves would be noise if not anchored to
priors that explain why they are what they are.

## Where the priors get written

[`../calibrations.md`](../calibrations.md) is the canonical home
for priors on the reference hardware. If a community member files
a device report with good profile data, the durable priors land
there too — labeled by hardware so readers can choose which
priors apply.

New priors become part of the data that future
"predict-before-measure" cycles consult. The discipline scales
with the community; every device report is a calibration
opportunity.
