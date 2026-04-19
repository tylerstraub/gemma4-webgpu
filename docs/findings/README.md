# Findings

Curated discoveries from building this engine. Each page stands alone —
framed for a reader who landed here cold, in search of a specific
answer. If you're porting Gemma 4 to any backend (MLX, ONNX, custom
CUDA, a different WebGPU variant), start with
[`gemma4-attention-quirks.md`](./gemma4-attention-quirks.md) — both
findings there silently corrupt output by >10× and neither is in the
model card.

## Current findings

- **[Gemma 4 attention quirks](./gemma4-attention-quirks.md)**
  — `v_norm` applied per-KV-head weight≡1 (dropped by the Q4_K_M
  quantizer), and attention scaling is `1.0` not `1/sqrt(d)`. Two
  architectural details not in the Gemma 4 model card.

- **[Why Q4_K_M](./why-q4km.md)**
  — The reasoning behind the default GGUF selection, the 2.3 GB
  architectural floor (PLE table), the bring-your-own-URL escape
  hatch, and pointers to variants worth evaluating next.

## Findings vs journey

The [`findings/`](.) directory is the "what we discovered" pages —
things worth knowing that generalize beyond this repo. Each finding
is evergreen: the writeup should still be useful in two years.

The [`journey/`](../journey/) directory is the "how we got there"
pages — the optimization trajectory, the bug hunts, the dead ends.
More narrative, more time-stamped, tied to specific commits.

Some community-submitted content might fit either category; when in
doubt, if you'd file it under "porting lesson" (not "performance
tuning"), it probably belongs here.

## Contributing a finding

Device reports go in GitHub issues via the `device-report` template.
Findings worth a writeup go through a PR:

1. Open a `research-finding` issue describing what you found and why
   it's worth a dedicated page
2. A maintainer comments with drafting notes or "yes, go ahead"
3. PR the new `.md` with code pointers, a reproducible test (ideally
   a crossLabDiff capture or numeric verification), and links to any
   upstream context (HuggingFace, JAX, the model card)
4. Update [`README.md`](./README.md) (this file) to index the new page

Findings are reviewed more for clarity and reproducibility than
style — what matters is that a porter reading cold can follow the
argument and run the verification.

## What doesn't belong as a finding

- **Generic Transformer explanations** — there are better resources
  elsewhere
- **Tuning numbers for one hardware setup** — those belong in
  [`calibrations.md`](../calibrations.md)
- **Optimization trajectory** — those belong in
  [`journey/`](../journey/)
- **Bug fixes without a generalizable lesson** — those belong in git
  commit messages
