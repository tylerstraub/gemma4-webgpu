---
name: Bug report
about: Something in the engine or workbench isn't working as documented.
title: 'bug: <brief summary>'
labels: bug
assignees: ''
---

## What went wrong

<!-- A clear statement of the expected vs actual behavior. -->

## How to reproduce

<!--
Exact steps. Ideally:
1. Browser + version + OS
2. Commit hash
3. Sequence of workbench verbs (or console snippet)
4. Observed output / error message
-->

## Device + engine context

<!--
Paste the Device and Engine-config sections from the workbench's
`formatDiagnosticReport` output. These isolate environmental issues
from real regressions.
-->

## Impact

<!--
- Is the engine's forward pass still numerically correct? (Run the
  crossLabDiff sweep and report the minimum cosine.)
- Is this a perf regression, a correctness issue, or a workbench UX
  issue?
-->
