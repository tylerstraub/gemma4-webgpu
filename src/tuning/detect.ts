/**
 * Auto-detect the best-matching tuning profile for a WebGPU adapter.
 *
 * The match is intentionally coarse (vendor substring on adapter.info).
 * WebGPU's `adapter.info` surface has been stable since the feature
 * shipped, and Chrome reports useful vendor strings for discrete GPUs;
 * on integrated hardware and mobile the match often falls through to
 * `generic`, which is the safe default.
 *
 * Override precedence (highest first):
 *   1. full `TuningProfile` object passed by the caller
 *   2. string id (e.g. `'apple-m-series'`) — looked up in `PROFILES`
 *   3. `TuningProfileOverrides` — deep-merged onto the auto-selected base
 *   4. vendor auto-match
 */

import type { TuningProfile, TuningProfileOverrides } from './profile.js';
import { overrideProfile } from './profile.js';
import { PROFILES } from './devices.js';

export interface SelectedProfile {
  profile: TuningProfile;
  /** Human-readable explanation of why this profile was picked. */
  reason: string;
}

/** Is the value a full `TuningProfile`? Used to distinguish from a partial. */
function isFullProfile(x: unknown): x is TuningProfile {
  if (typeof x !== 'object' || x === null) return false;
  const o = x as Record<string, unknown>;
  return (
    typeof o.id === 'string' &&
    typeof o.description === 'string' &&
    typeof o.verified === 'boolean' &&
    typeof o.matmul === 'object' &&
    typeof o.pipeline === 'object' &&
    typeof o.features === 'object'
  );
}

export type TuningOverride =
  | string
  | TuningProfile
  | TuningProfileOverrides
  | undefined
  | null;

/** Pick a profile for this adapter, applying any caller override. */
export function selectDeviceProfile(adapter: GPUAdapter, override?: TuningOverride): SelectedProfile {
  // 1) Full profile object — use verbatim.
  if (isFullProfile(override)) {
    return { profile: override, reason: `explicit profile object: ${override.id}` };
  }
  // 2) String id — look up in the registry.
  if (typeof override === 'string') {
    const p = PROFILES[override];
    if (!p) {
      const available = Object.keys(PROFILES).join(', ');
      throw new Error(`unknown tuning profile id "${override}" (available: ${available})`);
    }
    return { profile: p, reason: `explicit override: "${override}"` };
  }
  // 3) Partial override — auto-select then deep-merge.
  if (override !== undefined && override !== null && typeof override === 'object') {
    const base = autoSelect(adapter);
    return {
      profile: overrideProfile(base.profile, override),
      reason: `${base.reason}; then applied caller overrides`,
    };
  }
  // 4) Pure auto.
  return autoSelect(adapter);
}

function autoSelect(adapter: GPUAdapter): SelectedProfile {
  // `adapter.info` is the W3C-standard API. The vendor / architecture /
  // description strings are browser-reported and may be empty on some
  // platforms (Linux Mesa, some mobile GPUs); fall through to `generic`
  // when we can't make a confident match.
  const info = adapter.info;
  const vendor = (info.vendor ?? '').toLowerCase();
  const architecture = (info.architecture ?? '').toLowerCase();
  const description = (info.description ?? '').toLowerCase();
  const combined = `${vendor} ${architecture} ${description}`;

  if (combined.includes('nvidia')) {
    return {
      profile: PROFILES['nvidia-blackwell'],
      reason: `vendor match: nvidia → ${PROFILES['nvidia-blackwell'].id}`,
    };
  }
  if (combined.includes('apple')) {
    return {
      profile: PROFILES['apple-m-series'],
      reason: `vendor match: apple → ${PROFILES['apple-m-series'].id}`,
    };
  }
  return {
    profile: PROFILES.generic,
    reason: `no vendor match (vendor="${info.vendor ?? ''}", architecture="${info.architecture ?? ''}") → generic`,
  };
}
