# Contributing

Thanks for considering a contribution. This repo is positioned as an
open-source research home for browser-local Gemma 4 inference —
device reports, research findings, kernel optimizations, and
documentation contributions are all welcome.

The highest-value contributions, in rough order of impact per hour:

1. **Device reports** from hardware we don't have — one-click from
   the demo's device-report tab, folds into the tuning profile
   registry and calibrations doc.
2. **Research findings** — architectural details, perf calibrations,
   unexpected behavior worth documenting for future porters.
3. **Kernel optimizations** — from the deferred list in
   [`docs/future-work.md`](./docs/future-work.md) or your own ideas.
4. **Documentation** — corrections, clarifications, new device
   profiles, porting writeups.

If what you have in mind doesn't fit one of those categories,
please open an issue first — the scope / framing conversation is
usually the hardest part.

## The correctness-first loop

Any change that touches a shader, a matmul dispatch, a weight
upload path, or a normalization runs the
[`crossLabDiff`](./docs/methodology.md) sweep **before** bench.
Numerical equivalence against the HuggingFace reference is a
blocker; tps without correctness is theater.

The check that matters:

```js
// After any engine-side change:
await window.lab.loadReferenceTensors('/ref/smoke.npz');
for (const L of [0, 5, 10, 20, 33]) {
  const r = await window.lab.crossLabDiff({ afterLayer: L });
  console.log(`afterLayer:${L}`, r.cosineSimilarity);
}
const logits = await window.lab.crossLabDiff('logits');
console.log('logits', logits.cosineSimilarity);
```

Archival floor (on the reference hardware):

| Point | cosine ≥ |
|---|---:|
| `embed` | 0.99720 |
| `afterLayer:0` | 0.99980 |
| `afterLayer:33` | 0.94100 |
| `logits` | 0.99780 |

A drop of > 1e-3 below any of these is a regression, not float
noise. Investigate; don't paper over.

## The calibration discipline

Before proposing an optimization, classify the target kernel
(BW-bound / launch-bound / compute-bound) using
[`docs/calibrations.md`](./docs/calibrations.md) and bound the
expected win. Write the prediction in the PR description. Compare
predicted vs measured — divergence is diagnostic, not cause for
embarrassment. See
[`docs/journey/perf-calibration-process.md`](./docs/journey/perf-calibration-process.md)
for the full framing.

## Submitting a device report

The human demo at `/` has a one-click diagnostic collector:

1. Open the deployed demo (or `npm run dev` locally and hit `/`).
2. Click **"Load Gemma 4 E2B"** in the hero, wait for init
   (~70–90 s first time).
3. Switch to the **device report** tab, click **"run full
   diagnostic"**. This runs short bench + long bench + crossLabDiff
   sweep across every layer.
4. Click **"copy markdown"** — or use the **"open issue with this
   report"** link for a prefilled issue form. Paste into a new
   **device-report** issue either way.

The report format matches the issue template schema, so reports
land as comparable data. Durable findings roll into
[`docs/calibrations.md`](./docs/calibrations.md) and new or updated
tuning profiles.

Prefer the JSON surface? `npm run dev`, then hit
`/workbench.html` — `window.lab.collectDiagnosticReport()` returns
the same markdown payload.

## Submitting a research finding

Research findings — architectural quirks, unexpected behavior,
calibration observations worth writing up — go through a slightly
more structured path:

1. Open an issue using the **research-finding** template. Describe
   what you found, why it's worth documenting, and how to
   reproduce.
2. Wait for a maintainer response. Most findings get a "yes, write
   it up" within a day; some need scope conversation first.
3. PR a new `.md` file under `docs/findings/` (or, if the writeup
   is more narrative than evergreen, `docs/journey/`).
4. Update the relevant `README.md` index.

Findings are reviewed more for clarity and reproducibility than
style. What matters is that a porter reading cold can follow the
argument and run the verification.

## Submitting a kernel change

Kernel changes (WGSL shader edits, matmul dispatch changes,
optimization passes) follow this pattern:

1. **Open an issue or comment on an existing one** — describe the
   change, predicted win band, and which kernels are affected.
2. **Implement on a branch.**
3. **Correctness**: crossLabDiff sweep green (no regression beyond
   archival baseline, see floor table above).
4. **Performance**: bench before/after, 5 runs at each of short
   and long configs. Include the bench output in the PR
   description.
5. **Device coverage**: ideally run on multiple hardware. If you
   only have one, say so and flag it as a calibration gap worth
   closing.
6. **PR** with predicted-vs-measured delta, correctness deltas,
   and the commit style below.

## Submitting a tuning profile

Adding or updating a device profile needs a calibration sweep on
matching hardware:

1. Run `window.lab.collectDiagnosticReport()` — saves the bench,
   profile, crossLabDiff sweep into the copyable report.
2. Iterate on profile knobs; re-run the sweep per configuration.
3. Profile lands in `src/tuning/devices.ts` with
   `verified: true` and a `notes` field citing the short/long tps
   on the target hardware.
4. Vendor-match wiring in `src/tuning/detect.ts`.
5. Brief note in [`docs/tuning.md`](./docs/tuning.md) describing
   what was tuned.

A profile without hardware verification is fine to ship as
`verified: false` with educated-guess defaults. That's useful too —
the diagnostic surface makes it clear to users that the numbers are
priors, not measurements.

## Commit style

Lightweight type-prefix convention, not strict conventional
commits:

- `feat: ...` — new functionality
- `fix: ...` — bug fix
- `perf: ...` — performance improvement (include before/after
  numbers in the body)
- `refactor: ...` — structural change, no behavior delta
- `docs: ...` — documentation only
- `chore: ...` — build, CI, dependency bumps
- `test: ...` — correctness/regression harness changes

Subject under 70 chars. Body wrapped at 72. Perf and fix commits
include measured deltas with the bench config they were measured
under.

Example:

```
perf: MR4 variant for ffn.linearGateUp

Four-row-per-workgroup matmul amortizes tree-reduce barriers and
input-vector reads. Requires I divisible by 4 (Gemma 4 E2B's
6144 and 12288 both qualify).

Bench short (maxTokens=64, "Hello, how are you?", 5 runs):
  92 → 117 tps median  (+27%)
Bench long (maxTokens=200, raven/crow, 5 runs):
  84 → 108 tps median  (+29%)
crossLabDiff sweep: all layers green, unchanged from archival.
```

## What's out of scope

Some directions are deliberately non-goals for this repo:

- **Multimodal support** — documented as an extension point in
  [`docs/findings/why-q4km.md`](./docs/findings/why-q4km.md); no
  implementation yet.
- **Non-Gemma-4 models** — shaders encode Gemma 4's specifics. Other
  models want a different repo.
- **Framework integration** (React / Vue / Svelte components) — plain
  TypeScript + raw WebGPU is the scope.
- **Sub-2 GB builds** — the PLE table puts the architectural floor
  at ~2.3 GB. Don't chase it.
- **Generation-quality benchmark suite** — noted as future-work
  scope, not in-progress.

See [`docs/future-work.md`](./docs/future-work.md) for the longer
list of "considered and deferred" items.

## Reporting a bug

If you hit a correctness bug (output makes no sense,
crossLabDiff regressed beyond the floor table, engine crashes at
init) or a security issue (WebGPU driver crashes, memory leaks
that don't release on `dispose()`, anything that affects users
beyond this tab):

Open a **bug-report** issue. Include:
- Browser + version, OS + version, GPU model
- Reproduction steps
- The markdown from `await window.lab.collectDiagnosticReport()`
  (or the workbench's "Run diagnostic & copy report" button) — this
  bundles the adapter info, feature flags, engine config, benches,
  and crossLabDiff sweep needed to isolate environmental issues
  from real regressions. If the engine failed before init completed,
  `await window.lab.state()` plus the init error is the useful payload.
- Any console errors (Chrome DevTools → Console)

For security-sensitive issues, flag in the issue that you'd
prefer coordinated disclosure and a maintainer will reach out.

## The repo isn't frozen

This is living research scope. The engine, the shader set, the
tuning profile registry, the docs — all of them iterate as
community data arrives. If something you contribute doesn't land
on the first PR, it's almost certainly a scope / framing
conversation rather than a "no." Ask; we'll figure it out.

Thanks again for contributing.
