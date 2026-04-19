# CLAUDE.md — gemma4-webgpu

Project framing and collaboration conventions. Auto-loaded by Claude
Code on session start.

## What this project is

A browser-local inference engine for Gemma 4 E2B, purpose-built WGSL
over raw WebGPU. Reference implementation with two surfaces: a
human-facing demo at `/` (shipped to GitHub Pages) and a dev-only
diagnostic workbench at `/workbench.html` (stripped from the
production build; exists only under `npm run dev` for Playwright-MCP-
driven sessions and `window.lab` console work).

## Architectural commitments

Load-bearing decisions that shape the codebase. Don't introduce
conflicting changes without opening a design discussion.

- **F16 everywhere on GPU.** All weights are CPU-dequantized to F16
  once at upload, regardless of source quant. One matmul shader
  path. Trades slow first load for a radically simpler correctness
  surface.
- **Gemma 4 E2B, text-only.** The shaders encode Gemma 4's specific
  architecture: per-layer embeddings (PLE), shared KV across layer
  groups, dual RoPE with proportional factor on full-attention
  heads, mixed head dimensions (256 / 512), sliding-window attention,
  and final-logit softcapping.
- **Q4_K_M GGUF by default.** Users override the model URL at
  runtime to load other quants. Supported source quantizations:
  F32, F16, Q8_0, Q4_K, Q5_K, Q6_K, BF16.
- **No build-time ML framework.** Plain TypeScript + raw WebGPU API.
- **Reference-implementation scope.** No API stability commitments.
  Consumers fork or pin to commits, not version ranges.

## Development loop — correctness first

1. Make a change
2. Run the `crossLabDiff` sweep against `public/ref/smoke.npz`
3. Gate on all-layers cosine similarity ≥ 0.94, logits cos ≥ 0.9978
4. **Only after green**, run `lab.bench` and compare tps
5. The commit message carries both correctness and bench deltas

Reverse order (bench first, verify correctness later) is a trap.
See `docs/methodology.md` for the full reasoning and the
`crossLabDiff` design.

## Performance work discipline

Before proposing an optimization:

1. **Classify the target kernel** — bandwidth-bound, launch-overhead-
   bound, or compute-bound. Use `docs/calibrations.md` for base rates
2. **Bound the expected win** using the right mechanism for the
   kernel class. Matmul BW ceiling is ~70–75% of peak on Blackwell-
   class NVIDIA; don't overpredict
3. **Record the prediction before measuring.** Divergence between
   predicted and measured is itself diagnostic
4. **~3% bench noise threshold.** Changes inside noise aren't worth
   landing even if technically correct

## Bench config discipline

Every tps number cited — in code, in docs, in issues — must include
its config. `lab.bench()` returns
`{config, runs, median, mean, ...}` with `config` first for exactly
this reason. Short-config (maxTokens=64) and long-config
(maxTokens=200) are distinct workloads and produce distinct tps
because `attn.attnScore` is O(seqLen) per token.

A tps citation without config is ambiguous until verified.

## Workbench conventions — agent-first

The workbench is a JSON API on `window.lab`. The DOM is a side effect
for humans; Playwright MCP driving `browser_evaluate` is the primary
user. When adding verbs:

- Return structured JSON, not formatted text (the sole exception is
  `formatDiagnosticReport()`, which is explicitly a text output for
  the community-contribution flow)
- Errors as `{ ok: false, error }` discriminated unions, not thrown
- Include a `config` field on verbs that take options, so
  cross-session comparisons are self-documenting
- Long-running ops: fire-and-forget via a `window.__pending` stash
  pattern when the browser tool call would otherwise time out

## Vite HMR

After editing engine modules or workbench code, do a full browser
reload rather than relying on HMR. HMR re-executes the module and
leaves orphaned GPU buffers behind a stale reference; the engine
object on `window.lab` becomes unusable. Accept the re-init cost.

## Commit message style

Lightweight type-prefix (inspired by conventional commits, not
strict). Subject ≤ 70 characters; body wrapped at 72.

- `feat:` new functionality
- `fix:` bug fix
- `perf:` performance change (include before/after numbers in body)
- `refactor:` structural change with no behavior delta
- `docs:` documentation only
- `chore:` build, CI, dependency bumps
- `test:` correctness or regression harness

Perf and fix commits include measured deltas in the body (the
`crossLabDiff` result, the bench before/after). The commit log is
the public-facing history — write it for a reader skimming in 18
months.

## Knowledge capture

The repo is a living research home. Capturing lessons is part of the
work, not an afterthought.

- **Curate, don't dump.** Write-ups land as `docs/journey/`
  entries (narrative — trajectory, bug hunts, deferred work) or
  `docs/findings/` entries (evergreen porting lessons). Commit
  messages carry the rest
- Device reports arriving via the issue-template workflow get
  reviewed, and durable findings roll into `docs/calibrations.md`
  and the tuning-profile registry
- Predictions get recorded before runs, not after

## Where things live

- `src/` — engine, shaders registry, diagnostics, tuning (the library)
- `shaders/` — one .wgsl per kernel
- `app/` — the human-facing demo at `/`. Ships to GitHub Pages
- `workbench/` — agent-first `window.lab` surface at `/workbench.html`.
  Dev-only; stripped from production `vite build`
- `docs/architecture.md` — the Gemma 4 delta spec
- `docs/methodology.md` — the `crossLabDiff`-first correctness loop
- `docs/calibrations.md` — perf base rates and prediction priors
- `docs/future-work.md` — deferred opportunities, contribution targets
- `docs/findings/` — published discoveries
- `docs/journey/` — curated development narrative

## When in doubt

Surface trade-offs honestly. Ask rather than assume. Prefer consult-
first discussion over barreling ahead on an ambiguous path. The
project values considered choices over fast ones.
