/**
 * Device-tuning registry.
 *
 * Three shipped profiles in v1:
 *   - `nvidia-blackwell` — verified on RTX 5090 / Chrome stable.
 *   - `apple-m-series`   — educated-guess defaults; awaits calibration.
 *   - `generic`          — conservative, portable; ~20–30% slower than tuned.
 *
 * Add a profile by adding an entry here and wiring the vendor / feature
 * match in `selectDeviceProfile` (see `./detect.ts`). Calibration sweeps
 * should set `verified: true` and update `notes` with the measured
 * short/long tps on the hardware.
 */

import type { TuningProfile } from './profile.js';

/** NVIDIA Blackwell (RTX 50-series). Verified on RTX 5090 / Chrome stable. */
export const nvidiaBlackwell: TuningProfile = {
  id: 'nvidia-blackwell',
  description: 'NVIDIA Blackwell (RTX 50-series). F16 matmul, MR4 FFN, depth-2 pipelined decode.',
  verified: true,
  notes:
    'Measured ~127 short / ~116 long tps on RTX 5090 / Chrome stable ' +
    '(prompt "Hello, how are you?" maxTokens=64; raven/crow maxTokens=200). ' +
    'MR4 wins ~8% on ffn.linearGateUp; pipelined decode hides ~2.7 ms/token ' +
    'of submit→callback slack.',
  matmul: {
    workgroupSize: 256,
    defaultRowsPerWorkgroup: 1,
    rowsPerWorkgroupByKernel: {
      'ffn.linearGateUp': 4,
    },
  },
  pipeline: {
    decodeDepth: 2,
    greedyFastPath: true,
  },
  features: {
    shaderF16Required: true,
    // Tried and reverted — ~5% regression on RTX 5090 with Chrome's current
    // subgroup codegen. Revisit when the subgroup-matrix feature lands.
    subgroups: 'avoid',
    // Chromium experimental, not exposed in Chrome stable as of this writing.
    subgroupMatrix: 'avoid',
  },
};

/** Apple M-series (M1/M2/M3). Educated-guess defaults; not yet calibrated. */
export const appleMSeries: TuningProfile = {
  id: 'apple-m-series',
  description: 'Apple M-series (M1/M2/M3). Conservative starting defaults; awaits calibration.',
  verified: false,
  notes:
    'Starting defaults based on Apple GPU architectural priors: prefers ' +
    'single-row matmul, serial decode is robust. Run a calibration sweep ' +
    'and update this profile — file a device-report issue with results.',
  matmul: {
    workgroupSize: 256,
    defaultRowsPerWorkgroup: 1,
  },
  pipeline: {
    decodeDepth: 2,
    greedyFastPath: true,
  },
  features: {
    shaderF16Required: true,
    subgroups: 'avoid',
    subgroupMatrix: 'avoid',
  },
};

/** Generic fallback. Works on any WebGPU + shader-f16 device. */
export const generic: TuningProfile = {
  id: 'generic',
  description: 'Conservative portable defaults. Works on any WebGPU + shader-f16 device.',
  verified: false,
  notes:
    'Baseline portable defaults. Expect ~20–30% lower tps vs a tuned ' +
    'vendor profile (measured ~27% short on RTX 5090 with both MR4 + ' +
    'depth-2 pipelined decode off). Use as the starting point when ' +
    'calibrating new hardware.',
  matmul: {
    workgroupSize: 256,
    defaultRowsPerWorkgroup: 1,
  },
  pipeline: {
    decodeDepth: 1,
    greedyFastPath: false,
  },
  features: {
    shaderF16Required: true,
    subgroups: 'avoid',
    subgroupMatrix: 'avoid',
  },
};

/** Profile registry keyed by `id`. Order here controls listing order. */
export const PROFILES: Record<string, TuningProfile> = {
  'nvidia-blackwell': nvidiaBlackwell,
  'apple-m-series': appleMSeries,
  'generic': generic,
};
