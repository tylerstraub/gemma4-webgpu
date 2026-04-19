# Journey

The development trajectory from 13 tps at M0 to the current 125+
tps short / 116+ tps long. Curated for readers, not a
raw session log — what got tried, what worked, what got reverted,
what was deferred.

Why publish this? Because "here is a clean reference implementation"
is less useful than "here is how the reference implementation came to
be" for anyone planning similar work. The optimization ladder isn't
obvious in retrospect; the bugs we hit aren't in the model card; the
calibration discipline wasn't free to learn. We saved the journey
here so future contributors don't pay the same learning cost.

## What to read first

- **If you're interested in the big picture** — start with
  [`optimization-trajectory.md`](./optimization-trajectory.md). Four
  landmarks from 13 → 127 tps with the per-landmark ratios.

- **If you're interested in a correctness bug hunt** — start with
  [`crosslab-diff-debugging.md`](./crosslab-diff-debugging.md).
  Narrative version of how the two Gemma 4 attention findings were
  localized.

- **If you're interested in what we tried that didn't ship** —
  [`what-we-tried-and-didnt-ship.md`](./what-we-tried-and-didnt-ship.md).
  Subgroups variant (reverted), fusion sweep (deferred), quant-in-shader
  (deferred).

- **If you want to understand the base rates** —
  [`perf-calibration-process.md`](./perf-calibration-process.md).
  How the priors in [`../calibrations.md`](../calibrations.md) were
  built and why they're useful.

- **If you're interested in the load path** —
  [`load-path-reimplementation.md`](./load-path-reimplementation.md).
  100s → 50s cold load, the "stuck at 8.2%" gremlin fix, and a
  scorecard of which pre-measurement priors held up (and which
  missed by four orders of magnitude).

## Reading order

These pages stand alone. Read in any order. The optimization
trajectory references the bug-hunt page where the v_norm /
attention-scale issues were fixed, and the "what we tried and
didn't ship" page references the trajectory for the kernels that
were in flight at revert time — but each page has enough context to
make sense cold.

## Curation principle

These pages are *curated* narrative. What lands here:

- the decision, the reasoning, and the outcome
- pointers to the commit(s) or files where the change lives
- the lesson, when there is one that generalizes

If you're writing a new journey entry, that's the shape to aim for.
In-progress debugging, tried-and-failed patches, and session-level
state stay in commit messages and PR descriptions rather than
graduating to a page here.
