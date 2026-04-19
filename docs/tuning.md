# Tuning: device profiles

WebGPU runs on a wildly heterogeneous hardware base — a kernel tuned
for NVIDIA Blackwell runs great on RTX 50-series and often badly on
Apple M-series, and vice versa. This repo encodes tuning decisions as
immutable profile objects selected at engine init, so porting to new
hardware is a matter of finding the right profile (or adding one)
rather than rewriting shaders.

## Quick usage

The repo is designed to be forked or consumed via a git-commit pin,
not installed from a registry — so imports resolve to the engine
source directly (`src/index.ts`). The human-facing demo and the
dev-only workbench both show the pattern:

```ts
// app/lib/engine.ts and workbench/main.ts both import via:
import { createGemmaEngine } from '../src/index.js';

// Pure auto-selection — vendor match on adapter.info.
const engine1 = await createGemmaEngine({ model: 'e2b' });

// Explicit profile id.
const engine2 = await createGemmaEngine({
  model: 'e2b',
  tuning: 'apple-m-series',
});

// Partial override merged onto the auto-selected base.
const engine3 = await createGemmaEngine({
  model: 'e2b',
  tuning: { pipeline: { decodeDepth: 1, greedyFastPath: false } },
});

console.log(engine1.tuning.id, engine1.tuningReason);
// "nvidia-blackwell" "vendor match: nvidia → nvidia-blackwell"
```

The selected profile is visible on `engine.tuning` (the profile object)
and `engine.tuningReason` (a human-readable explanation). The workbench
surfaces both in `window.lab.state().tuning`.

## Shipped profiles

### `nvidia-blackwell` (verified)

Tuned against RTX 5090 / Chrome stable. Short bench 125 tps, long
bench 116 tps. Uses the 4-row MR4 matmul variant for `ffn.linearGateUp`
and the depth-2 pipelined (Option B) decode path for the greedy-fast
sampling route. Subgroups deliberately disabled — a prior variant
regressed ~5% on this hardware with Chrome's current subgroup codegen.

Likely also a good fit for RTX 4090 / Ada, RTX 3090 / Ampere, and H100
class hardware, though only Blackwell is verified. File a device report
if you run it elsewhere — results fold back into
[`calibrations.md`](./calibrations.md).

### `apple-m-series` (unverified)

Conservative starting defaults for M1/M2/M3. No MR4 (Apple GPUs prefer
single-row matmul for this kind of workload — prior tuning data
suggests the tile memory layout doesn't reward 4-row coalescing here).
Depth-2 pipelined decode kept on; greedy-fast path available.

This profile is **unverified** — file a device report with bench and
calibration sweep to upgrade it. The `verified: false` flag is
surfaced in diagnostics so consumers know the numbers are priors, not
measurements.

### `generic` (fallback)

Portable defaults. Scalar matmul everywhere, serial decode only, no
greedy-fast path. Expect ~20–30% lower tps than a tuned profile on
the same hardware; the tradeoff is that this profile works anywhere
WebGPU + `shader-f16` works. Use as the starting point when
calibrating new hardware.

## How a profile shapes engine init

Two knobs are wired end-to-end in v1:

### `matmul.rowsPerWorkgroupByKernel['ffn.linearGateUp']`

Controls the FFN gate/up matmul path. `4` selects the MR4 variant
(`matmul_quant_mr4.wgsl`, 4 output rows per workgroup — ~8% faster on
Blackwell). `1` (or unset with `defaultRowsPerWorkgroup: 1`) falls back
to the scalar `matmulQuant` kernel with a regular
`dispatchMatmul(I)` dispatch.

The WGSL side declares `override R: u32 = 4;`, and the engine injects
the profile's value via `createComputePipeline({ constants })`. Only
`R = 4` is supported by the shader's unrolled body today; future MR2 or
MR8 variants would add new shaders and register them here.

### `pipeline.decodeDepth` + `pipeline.greedyFastPath`

Controls the decode loop's critical path.

- `decodeDepth: 2, greedyFastPath: true` enables the Option-B
  pipelined path for greedy sampling (`temperature === 0` +
  `repPenalty <= 1`). The engine submits forward N+1 before awaiting
  forward N's token readback, hiding ~2.7 ms of submit→callback
  scheduling slack per token on the reference hardware.
- `decodeDepth: 1` (or `greedyFastPath: false`) routes every sample
  through the serial loop. Same correctness; slower.

Topk / temperature-sampled / repPenalty > 1 paths always go serial —
the JS-side sort and penalty update can't pipeline cleanly. The
`decodeDepth` knob only affects the greedy route.

## Adding a new profile

Three steps:

1. **Add the profile object** to `src/tuning/devices.ts`:

   ```ts
   export const intelArc: TuningProfile = {
     id: 'intel-arc',
     description: 'Intel Arc (Alchemist/Battlemage).',
     verified: false,
     notes: 'Starting defaults; calibration sweep pending.',
     matmul: {
       workgroupSize: 256,
       defaultRowsPerWorkgroup: 1,
       // rowsPerWorkgroupByKernel: {...} — add once measured
     },
     pipeline: { decodeDepth: 2, greedyFastPath: true },
     features: {
       shaderF16Required: true,
       subgroups: 'avoid',
       subgroupMatrix: 'avoid',
     },
   };
   ```

2. **Register it** in the `PROFILES` registry in the same file.

3. **Wire vendor matching** in `src/tuning/detect.ts`:

   ```ts
   if (combined.includes('intel')) {
     return { profile: PROFILES['intel-arc'], reason: `vendor match: intel → intel-arc` };
   }
   ```

Then calibrate — the bench + profile + crossLabDiff sweep give you
everything to validate and tune the new profile's knobs.

## Calibrating a profile

On the target hardware:

```js
// 1. Load the engine with the new profile.
await window.lab.init('e2b', { tuning: 'intel-arc' });

// 2. Run the correctness sweep. Must match archival cosines.
await window.lab.loadReferenceTensors('/ref/smoke.npz');
await window.lab.crossLabDiff('logits');

// 3. Baseline bench.
const baseline = await window.lab.bench('Hello, how are you?', {
  maxTokens: 64, runs: 5,
});

// 4. Try each knob in isolation.
window.lab.dispose();
await window.lab.init('e2b', {
  tuning: {
    matmul: { rowsPerWorkgroupByKernel: { 'ffn.linearGateUp': 4 } },
  },
});
const mr4 = await window.lab.bench(undefined, { maxTokens: 64, runs: 5 });
// ... etc

// 5. File a device report with the winning combination.
await window.lab.collectDiagnosticReport();
```

## What's architecturally ready but not yet wired

The `TuningProfile` shape includes fields that exist for v0.2+ work:

- `matmul.workgroupSize` — 256 is baked in today; knob is there for
  Apple M-series retuning.
- `features.subgroups` / `features.subgroupMatrix` — placeholders for
  when subgroup-aware matmul variants or cooperative-matrix kernels
  land. Chrome stable does not currently expose
  `chromium-experimental-subgroup-matrix`.
- `matmul.rowsPerWorkgroupByKernel` — only `ffn.linearGateUp` is
  wired; other kernel names are listed so future work on
  `attn.linearOut`, `linearQ.consumer`, and similar can register
  without a type change.

Adding a knob to the wire-up is localized to two places: the
`createPipelines` path in `src/engine.ts` (inject the WGSL override)
and the dispatch site (branch on the selected shader variant). See
the `ffn.linearGateUp` path as a template.
