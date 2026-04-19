# Load-path reimplementation: 100s → 50s, gremlin killed

Until v0.1 the load path had been left alone while the forward-pass
engine and the tuning infrastructure got most of the attention.
Users reported two problems: cold loads on the reference workload
took just over 100 seconds on a good connection, and every so often
the progress bar would silently stall at "8.2%" and never recover.

This entry documents the reimplementation: what shipped (−50.5% wall
clock, gremlin fixed), what didn't (Phase 5 overlap, Phase 6 worker
decode), and — most usefully for the next contributor — which priors
held up under measurement and which didn't.

## What the bottleneck actually was

Before reimplementing anything, we instrumented `init()` with
`engine.loadTimings` (see commit `9e72539`) and drove one load via
Playwright-MCP against `window.lab`. On the reference workload
(Q4_K_M, Range-supported CDN, 100 Mbps residential link) the
baseline was:

| Phase | ms | % of total |
|---|---:|---:|
| pleDecode | 31,365 | 31% |
| pleFetch | 28,715 | 29% |
| layersFetch | 24,230 | 24% |
| layersProcess | 8,576 | 9% |
| headerFetch | 978 | 1% |
| ggufParse + tokenizer | 434 | 0.4% |
| initWebGPU | 100 | 0.1% |
| createPipelines | **0.3** | **~0%** |
| createBindGroups | 3.8 | ~0% |
| createWorkBuffers | 0.4 | ~0% |
| createUniformBuffers | 2.3 | ~0% |

~90% of wall-clock is network I/O plus PLE decode. Shader compile,
bind-group creation, uniform-buffer setup, and work-buffer allocation
are collectively noise.

This matters because a pre-measurement audit had named shader compile
as a likely bottleneck ("21 shaders created serially, parallelize!").
Measurement put it at 0.3 ms — four orders of magnitude off. One of
the two big lessons of the project's "predict before measuring"
discipline paying off directly.

## What shipped

### Streaming PLE fetch+decode pipeline (commit `07d058a`)

The old PLE path: `fetchRange` pulled all 1.6 GB into a JS `Uint8Array`
via `Response.arrayBuffer()`, then `streamDecodePleQ5_K` walked that
buffer super-block by super-block. 28.7s of fetch strictly before
31.4s of decode — sequential, with both phases running at roughly
the same rate.

The replacement: `RangedReader.stream()` yields 16–64 KB network chunks
as they arrive; the decoder accumulates until it has one processing
chunk worth (512 vocab entries × 35 layers × 176 bytes ≈ 3.15 MB),
decodes in a burst, writes to GPU, clears, repeats. Fetch and decode
overlap: while JS runs a decode burst, the browser's network stack
continues filling its internal buffer; the next `await reader.read()`
drains it in one gulp.

Peak JS memory for PLE dropped from ~1.6 GB to ~12 MB (3 MB accumulator
+ 9 MB per-layer f32 staging). The old `await setTimeout(0)` yields
(~500 ms of clamp overhead across 512 chunks) disappeared — the
`for await` on the stream yields naturally between network chunks.

**pleFetch + pleDecode: 60.1s → plePipeline: 26.4s.** Prediction was
~25s saved; actual ~34s. The delta over prediction was GC pressure
from the 1.6 GB JS buffer and the `setTimeout` clamp; both vanished
in the streaming design.

### Concurrent layer fetches (commit `a673cdb`)

Old: a serial `for` loop over 35 layers. 35 sequential Range requests
totaling 24.2s at ~50 MB/s per request.

New: a bounded-concurrency scheduler with `LAYER_CONCURRENCY = 4`.
Order is preserved via indexed assignment into
`modelBuffers.layers[i]` — finish order doesn't matter since each
worker writes to its own slot. Accumulator `+=` ops are safe under JS
event-loop semantics.

**layers wall-clock: ~32s → ~26s.** Prediction was −15s on total;
actual −16.4s. Individual fetches got *slower* (600ms → 2100ms each)
because 4 connections share the pipe, but aggregate effective
concurrency was 3.1× — the per-connection slowdown is sublinear in
stream count, so 4× concurrency still wins.

### RangedReader with stall detection (commit `07d058a`)

The "stuck at 8.2%" gremlin was a silent TCP zombie on the 1.6 GB PLE
fetch. `fetch()` + `arrayBuffer()` exposes no observable progress;
when the underlying connection goes dead without a reset, the promise
never resolves and never rejects.

`RangedReader` uses `fetch()` + `body.getReader()` so bytes are
observable as they arrive. A no-progress timer (default 30 s) arms
before each `reader.read()` and disarms on chunk arrival; if it fires,
`AbortController.abort()` kills the active request and a new one
issues with `Range: bytes=<last-received+1>-<end>`. Retries use
exponential backoff.

Forward compatibility: the retry events surface through the engine's
`onProgress` callback with `kind: 'retrying'` so UI surfaces can
render "connection stalled, retrying…" instead of an eerily quiet
progress bar.

## What we tried and didn't ship

### Phase 5: overlap PLE pipeline with layer fetches

**The idea**: don't `await` PLE before kicking off layer fetches.
Run both under `Promise.all`. Theoretical floor is `max(pleWall,
layersWall)` ≈ 26s, saving the ~22s PLE phase entirely.

**What happened**: regressed from 49.7s → 52.0s. Per-phase:
plePipeline went from 22s solo → 44s concurrent. Layer wall went from
26s solo → 51s concurrent. Both phases ran ~2× slower when overlapped.

**Why**: the HF CDN's per-connection throughput *falls* when more
connections open. Phase 4's 4-way layer concurrency already hit the
aggregate ceiling (~47 MB/s across 4 streams); adding a 5th stream
(PLE) pushed every stream into slow-start-thrashing territory.

Tried dropping `LAYER_CONCURRENCY` to 3 during overlap (keeping total
streams at 4). Still regressed (56s) — the PLE stream and the layer
streams still contended for the same per-connection-allocated
bandwidth.

**When to revisit**: on a CDN that doesn't throttle per-connection,
or after a `document.visibilityState === 'hidden'` background prefetch
that decouples from the user-visible load time.

### Phase 6: worker-offload Q5_K decode

**The idea**: move Q5_K super-block decoding to a Web Worker. In
theory this unblocks the main thread for fetch chunk processing.

**Why skipped**: the Phase 3 measurements pinned the PLE pipeline at
73 MB/s — that's the *single-connection* network ceiling. Decode is
already overlapped with fetch, and the pipeline is fetch-bound, not
decode-bound. Worker offload would smooth out the ~60 ms main-thread
bursts during `drainChunk` but wouldn't shrink wall clock.

Cost was non-trivial: Vite worker entry, `SharedArrayBuffer` /
transferable handling, message protocol, error propagation. Didn't
justify the ~0 perf win.

**When to revisit**: if the model moves to a faster CDN where the
decode becomes the binding constraint, or if progressive-loading
(decode layers in the order they're used during forward pass) lands
and needs off-thread concurrency.

### Splitting PLE into concurrent Range sub-requests

**The idea**: the PLE is contiguous in the GGUF file and has vocab as
the outer dim, so 4 sub-requests (each covering ~400 MB / one quarter
of vocab) could theoretically parallelize the fetch.

**Why not tried**: extrapolating from Phase 4's empirical data. With
1 PLE stream we get 73 MB/s; with 4 layer streams we get 47 MB/s
aggregate (slower per byte). The evidence says splitting PLE into
more streams would likely reduce aggregate throughput, not increase
it. Not worth the refactor effort for a predicted regression.

**When to revisit**: same as Phase 5 — a CDN without per-connection
throttling.

## Calibrations this run captured

- **HF CDN per-connection ceiling**: ~55–73 MB/s. 4 concurrent streams
  is the sweet spot; 5+ oversaturates.
- **21 WGSL shader compile + pipeline creation**: 0.3 ms. For shaders
  of this complexity, compile is effectively free.
- **350+ `createBindGroup` calls**: ~4 ms. Not a hot spot.
- **Native `Float16Array` (Chrome 135+)** makes f32→f16 conversion
  essentially free — decode-side bottleneck is the Q5_K block walk,
  not the f16 packing.

These move into `docs/calibrations.md` as base rates for future load-
path work.

## Pre-measurement priors: a scorecard

| Prior | Measured | Verdict |
|---|---|---|
| Shader compile 500–2000 ms, parallelize | 0.3 ms | **Wrong by ~4 OoM** |
| Network dominates wall clock | 53.9s of 100.5s | Confirmed |
| Per-layer fetch > process | 24s vs 9s | Confirmed |
| Bind groups / uniforms < 100 ms | < 5 ms each | Confirmed, stronger |
| 5-stream overlap wins 15–20s | Regressed 2s | **Wrong** |
| Worker decode would save ~15s | Can't save past network floor | Moot given measured ceiling |

Two priors were meaningfully wrong. Both were about assumed
parallelism: "more concurrency = more throughput" turns out to require
a qualifier ("up to 4 streams on this CDN"). Worth keeping that
qualifier in mind the next time something in this project wants to
go concurrent.

## End state

**100.5s → 49.7s (−50.5%).** Gremlin architecturally killed. Peak JS
memory for the PLE phase dropped from ~1.6 GB to ~12 MB. All API
changes are additive, so both the human demo at `/` and the workbench
at `/workbench.html` pick up the gains without code changes on their
end.

`crossLabDiff` sweep after each phase landed bit-identical to the
archival baseline in `docs/methodology.md` — zero correctness drift
across the entire reimplementation.
