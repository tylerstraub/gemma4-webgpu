---
name: Device report
about: Share how gemma4-webgpu performs on your hardware.
title: 'device: <GPU> / <browser>'
labels: device-report
assignees: ''
---

<!--
Thanks for running gemma4-webgpu on your hardware. The quickest way to
fill this out: open the deployed demo (or `npm run dev` locally), load
the model, switch to the "device report" tab, click "run full
diagnostic", then "copy markdown" — paste the result here. The
generator produces every section this template expects.

If you're filling this out manually, the sections below are the schema
the project uses to compare reports across hardware — keep the
structure so everything stays grepable.
-->

## Summary

<!-- One line: GPU / browser / key numbers. -->

## Device

- **GPU vendor**:
- **GPU architecture**:
- **GPU description**:
- **WebGPU features**:
- **User agent**:
- **Platform**:
- **Hardware concurrency**:
- **Device memory (GB)**:

## Engine config

<!-- From the workbench formatter. Usually just the Gemma 4 E2B defaults. -->

- **num_layers**:
- **hidden_size**:
- **vocab_size**:
- **context_length**:
- **per_layer_input_dim**:
- **init time (s)**:

## Benchmark

<!--
Canonical configs:
  short = "Hello, how are you?" / maxTokens=64 / runs=5
  long  = default (raven/crow) / maxTokens=200 / runs=5
Always cite both. Different prompts + maxTokens produce
systematically different tps — numbers without config are ambiguous.
-->

## Correctness (crossLabDiff sweep)

<!--
Optional but strongly appreciated. The workbench sweep runs in ~5s
after the engine is initialized and the reference tensors are loaded.
Include the per-capture-point cosine table and the minimum cosine
across all layer boundaries.
-->

## GPU profile (timestamp-query)

<!--
Only available on adapters that expose `timestamp-query`. Omit this
section if your adapter doesn't — note it in the Device section above.
-->

## CPU profile (phase breakdown)

<!-- Wall-clock breakdown of the host side. -->

## Notes

<!--
Anything specific to this run — thermal state, other GPU workloads,
whether this was a first-run or warm-run, unusual browser flags, etc.
-->
