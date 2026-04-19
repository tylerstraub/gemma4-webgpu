# Calibrations

Base-rate priors for performance work. These numbers are measured
against the `nvidia-blackwell` tuning profile on one reference machine;
they are useful as priors, not promises. If you see numbers that
diverge by more than ~5% on similar hardware, that's interesting —
file a device report and we'll fold the data in.

## Reference machine

- **GPU**: NVIDIA GeForce RTX 5090 (Blackwell)
- **CPU**: AMD Ryzen 9 9950X3D
- **OS**: Windows 11
- **Browser**: Chrome stable (no flags required)
- **WebGPU features active**: `shader-f16`, `timestamp-query`

## Headline numbers (maxTokens, prompt)

| Config | Median tps | Notes |
|---|---:|---|
| short (`"Hello, how are you?"`, 64) | **125–127** | Fully warmed, throwaway run first |
| long (raven/crow default, 200)      | **115–117** | First bench after init sits ~4% low |

Short and long configs both use `temperature: 0`, `repPenalty: 1.0`,
`runs: 5` (reported as the median across the five measurements).
Bench numbers always need the config to be meaningful; citing a tps
number without the config is noise.

## Per-kernel timing floor (RTX 5090)

Per-forward-pass p50 timings from the timestamp-query profiler, short
config warm state. These are forward-pass kernel totals — one token
worth of work across 35 layers.

| Kernel | p50 (ms) | BW/launch class | Notes |
|---|---:|---|---|
| `ffn.linearGateUp` | 2.4 | BW-bound | MR4 variant |
| `ffn.linearDown`   | 1.7 | BW-bound (54% peak) | Aggregate across producer + consumer |
| `attn.linearOut`   | 0.5 | BW-bound | |
| `attn.linearQ`     | 0.4 | BW-bound | |
| `attn.linearK*`    | 0.1 | BW-bound | Producer layers only |
| `attn.linearV*`    | 0.1 | BW-bound | Producer layers only |
| `lmHead`           | 0.9 | BW-bound | Not launch-bound — that was a mis-read |
| `ple2.linearInpGate` | 0.6 | BW-bound | Per-layer × 35 |
| `ple2.linearPostProj` | 0.6 | BW-bound | Per-layer × 35 |
| Other (norms/adds/softmax/RoPE) | ~1 | launch-bound | Aggregated |
| **Forward total** | **~7.8** | | |

Wall-clock per token is ~8.5 ms in the serial path, ~7.8 ms in the
Option-B pipelined path (the submit→callback scheduling slack, ~2.7 ms
per token, hides behind the GPU compute of the next forward).

## Bandwidth classifications

- **BW-bound**: kernel time scales with weight bytes moved. The matmul
  family is all BW-bound at this sequence length. The effective
  bandwidth on this hardware for F16 weights is ~54% of peak HBM3 —
  which is the "good for a non-cooperative-matrix kernel" number.
- **Launch-bound**: time dominated by kernel-launch overhead
  (~30 ns barrier on this hardware; individual pass encode ~1–2 µs).
  Single-workgroup norms and adds are launch-bound; they're cheap
  individually but there are ~100 of them per forward.
- **Compute-bound**: none of this engine's kernels currently sit here.
  F16 matmul at HD=256 / HD=512 has enough BW pressure that compute
  isn't the ceiling on this hardware.

## Ceiling estimates for deferred optimizations

Predictions, not measurements. If you're considering one of these,
measure first and compare to the predicted band.

- **Option C (sampled/blocked attention)** — ~3–5% win at the current
  context length (2048). Scales with sequence length; becomes
  meaningful at 8k+.
- **lmHead + argmax fusion** — ~1–2% win on greedy short generations.
  One round-trip eliminated.
- **lmHead + topk fusion** — ~0.5–1% on sampled generations.
- **Single-workgroup fusion sweep** (norms + adds) — ~2–4% short-config
  win. Reduces per-forward pass count by ~20.
- **Cooperative-matrix kernels** — ~1.2× ceiling on the matmul family
  if Chrome stable exposes `chromium-experimental-subgroup-matrix`.
  Not available today.

Combined short-term ceiling on this hardware: roughly 160 tps short /
148 tps long, assuming all the above land without correctness
regressions. See [`future-work.md`](./future-work.md) for the framing
of these as contribution targets.

## Things that moved and their ratios

| Change | Before | After | Ratio |
|---|---:|---:|---:|
| M0 baseline | 13 tps | — | 1.0× |
| Weight-coalescing matmul | 13 | 58 | 4.5× |
| Option B (pipelined decode) | 92 | 117 | 1.27× |
| Option A (MR4 FFN matmul) | 117 | 127 | 1.08× |

Each optimization's win ratio depends on what was in the kernel mix at
the time — Option A's 8% win is smaller than it would have been at the
baseline because the FFN was no longer the dominant kernel. Order
matters; predicting without the order is easy to get wrong.

## How to contribute calibration data

File a device report from the workbench:

```js
await window.lab.init('e2b');
const { markdown } = await window.lab.collectDiagnosticReport();
await window.lab.copyToClipboard(markdown);
// Paste into a new "device-report" issue on GitHub.
```

The report includes adapter info, feature flags, selected tuning
profile, short + long bench, crossLabDiff sweep, and (if available)
GPU timestamp-query profile. Durable learnings roll into this page
and into new tuning profiles.
