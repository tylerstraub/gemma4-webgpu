---
name: Research finding
about: Share a Gemma 4 architectural detail, shader optimization, or methodological discovery.
title: 'finding: <brief summary>'
labels: research-finding
assignees: ''
---

<!--
This repo is a living research home. Findings that would help someone
else porting Gemma 4 to a different backend, or tuning the WebGPU path
on different hardware, are some of the most valuable contributions.
-->

## What you found

<!-- One paragraph. The claim, in plain language. -->

## How you found it

<!--
- Which verb / sweep / measurement surfaced it?
- What was the observed signal (cosine delta, tps regression, etc.)?
- What was the repro — prompt, capture point, layer range?
-->

## Why it matters

<!--
- What does it change about how we port, tune, or consume the engine?
- Is this Gemma-4-specific, WebGPU-specific, hardware-specific?
- Is it publishable as a standalone finding, or does it fold into an
  existing section of `docs/`?
-->

## Evidence

<!--
Code snippets, numbers, capture tables, reference-impl pointers —
whatever you can share. If this is about a specific capture point,
include the relevant crossLabDiff result.
-->

## Proposed action

<!--
- Suggested code change (link to a PR if you have one).
- Docs update that should land alongside the code.
- Further investigation that'd strengthen the claim.
-->
