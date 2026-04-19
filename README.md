# gemma4-webgpu

> Browser-local Gemma 4 E2B inference via purpose-built WGSL and raw
> WebGPU. An open-source AI community research project.

**Status: v0.1 — reference implementation, iterating on community
data.** The engine achieves 125+ tps on a short workload
(maxTokens=64) and 116 tps on a longer decode (maxTokens=200) for
Gemma 4 E2B on an RTX 5090 in Chrome stable, with no browser flags
required. Engine, shader set, diagnostic harness, human-facing demo
with chat + bench + profile + crossLabDiff + device-report, curated
documentation, and device-profile tuning system are all live.

## What this is

A reference implementation and a living research home for running
Gemma 4 E2B (text-only, Q4_K_M) locally in a browser via WebGPU. The
audience is:

- **Porters** adapting browser-local LLM inference to other hardware
- **Kernel tinkerers** interested in WGSL optimizations that work
  across vendors
- **Researchers** studying the edge of what WebGPU can do for LLMs
- **Anyone** who wants to learn from the process — including the
  bug hunts, the dead ends, and the base-rate calibrations

The repo is **not** a framework or a drop-in library. It's a
reference you fork, adapt, and learn from. Pin to commits, not
version ranges — the surface is living, not frozen.

## Two Gemma 4 findings worth knowing

Caught via the project's `crossLabDiff` methodology (HF PyTorch
reference forward-hooks, compared per-kernel with cosine similarity).
Neither is in the Gemma 4 model card or DeepMind's JAX docstrings;
both silently corrupt output by >10× magnitude.

1. **`v_norm` is applied per-KV-head before attention**, weight ≡ 1,
   and the Q4_K_M GGUF export drops the tensor entirely (the
   quantizer treats a weight-1 tensor as a no-op). Skip this norm
   and V enters attention at ~67× magnitude. The fix is a fused
   weightless RMSNorm between `linearV` and `kvCacheStore` on
   producer layers.

2. **Attention scaling is `1.0`, not `1/sqrt(head_dim)`** — `q_norm`
   and `k_norm` already normalize each head to unit RMS, so the
   standard Transformer `sqrt(d)` compensation is dropped. Use the
   wrong scaling and softmax flattens by 16× on HD=256, smearing
   attention across all positions.

Full writeup with code pointers:
**[`docs/findings/gemma4-attention-quirks.md`](./docs/findings/gemma4-attention-quirks.md)**.
If you're porting Gemma 4 to any backend (ONNX, MLX, custom CUDA,
WebGPU variants), these two are the first things to check.

## Quickstart

```sh
# Clone this repo, then:
cd gemma4-webgpu
npm install
npm run dev
```

Opens at `http://127.0.0.1:5175`:

- **`/`** — the human-facing demo. One page, seven panels: chat,
  bench, profile, correctness (per-layer crossLabDiff sweep), tuning,
  device report, weight probes. Load the model once (~70–90 s first
  time), then explore.
- **`/workbench.html`** — agent-first diagnostic interface on
  `window.lab`. **Localdev only** — stripped from the production
  `vite build`. Used by Playwright-MCP-driven sessions and anyone
  who prefers the JSON surface over UI chrome. Every verb has a
  human equivalent on `/`.

Requires a WebGPU-capable browser with the `shader-f16` feature
(Chrome stable on recent hardware works out of the box; the engine
throws at init if the feature is unavailable).

The human page exposes a minimal JSON API on `window.chat` for
programmatic chat:

```js
await window.chat.init('e2b');                 // ~70–90s first load
const reply = await window.chat.send('Hello! Tell me about ravens.');
await window.chat.send('And what about crows?');  // reuses KV cache
window.chat.reset();                               // clear conversation
```

In localdev (`npm run dev`), the workbench at `/workbench.html`
exposes the fuller diagnostic surface on `window.lab`:

```js
await window.lab.init('e2b');
await window.lab.bench('Hello, how are you?', { maxTokens: 64, runs: 5 });
await window.lab.loadReferenceTensors('/ref/smoke.npz');
await window.lab.crossLabDiff({ afterLayer: 33 });
const { markdown } = await window.lab.collectDiagnosticReport();
await window.lab.copyToClipboard(markdown);      // paste into a new issue
```

Or use the equivalent panels on `/` — same underlying diagnostics
module, human-shaped UI.

## Structure

```
gemma4-webgpu/
├── src/               engine, shaders registry, diagnostics, tuning
├── shaders/           one .wgsl per kernel
├── app/               the human-facing demo at `/` (shipped to Pages)
├── workbench/         window.lab API (dev-only, not in prod build)
├── reference/         HF PyTorch reference generators
├── public/ref/        npz reference tensors, checked in
├── docs/              architecture / methodology / findings / journey
├── scripts/           GGUF audit helper
└── .github/           Pages deploy workflow + issue templates
```

## Documentation map

- **[`docs/architecture.md`](./docs/architecture.md)** — the nine
  Gemma 3 → Gemma 4 deltas with code pointers. Start here if you're
  porting to another backend.
- **[`docs/methodology.md`](./docs/methodology.md)** — the
  crossLabDiff-first dev loop. How correctness is verified
  per-kernel against a HuggingFace reference.
- **[`docs/tuning.md`](./docs/tuning.md)** — the device-profile
  system, the shipped profiles, and how to add one.
- **[`docs/calibrations.md`](./docs/calibrations.md)** — base-rate
  priors for performance work on RTX 5090 / Chrome. Invites
  corrections from other hardware.
- **[`docs/future-work.md`](./docs/future-work.md)** — deferred
  optimizations framed as contribution targets.
- **[`docs/findings/`](./docs/findings/)** — curated discoveries
  worth writing down for porters. Currently: Gemma 4 attention
  quirks and the Q4_K_M reasoning.
- **[`docs/journey/`](./docs/journey/)** — the development
  narrative: the 13 → 127 tps trajectory, the v_norm bug hunt,
  what we tried and didn't ship, how the base-rate priors were
  built.

## Non-goals

Deliberate scope limits — not things we might add later, things we
are saying no to on purpose.

- **No multimodal.** Vision and audio projector paths are
  documented as extension points but not implemented. Text only.
- **No tensor-core / cooperative-matrix path.** Chrome stable does
  not expose `chromium-experimental-subgroup-matrix` as of this
  writing. Re-evaluate when it ships.
- **No quant-in-shader.** F16-everywhere on GPU is a correctness-
  surface decision; CPU-dequants once at upload.
- **No API stability commitments.** Living research scope; fork
  and adapt, or pin to a commit.
- **No framework integration** (Svelte / React / Vue components).
  Plain TypeScript + raw WebGPU.
- **No sub-2GB build.** Gemma 4 E2B's PLE table puts the
  architectural floor at ~2.3 GB. Don't chase it.

## Contributing

Device reports and research findings are the highest-value
contributions. The human demo's **"device report"** panel runs the
full diagnostic suite (device + browser info, short + long bench,
34-layer crossLabDiff sweep) and produces a paste-ready markdown
report matching the `device-report.md` issue template — reports land
as structured, parseable issues, and durable findings roll into the
tuning-profile registry and
[`docs/calibrations.md`](./docs/calibrations.md).

See **[`CONTRIBUTING.md`](./CONTRIBUTING.md)** for the full loop —
the correctness-first cadence, the calibration discipline, and
how to propose kernel changes, tuning profiles, and findings.

## Acknowledgments

Model: **[google/gemma-4-E2B-it][gemma4]** by Google DeepMind,
governed by the [Gemma Terms of Use][gemma-terms]. Default GGUF:
**[unsloth/gemma-4-E2B-it-GGUF][unsloth]** (Q4_K_M, imatrix-
calibrated).

[gemma4]: https://huggingface.co/google/gemma-4-E2B-it
[gemma-terms]: https://ai.google.dev/gemma/terms
[unsloth]: https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF

## License

Apache License 2.0. See [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE).
