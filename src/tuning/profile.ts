/**
 * Tuning-profile types. A `TuningProfile` captures the two classes of knobs
 * that matter for WebGPU matmul-driven LLM inference: per-kernel matmul
 * shape (rows per workgroup, workgroup size) and decode-loop shape
 * (pipeline depth, greedy fast path). Feature flags describe what the
 * profile wants to *do* when a WebGPU feature is (or isn't) available —
 * the engine does its own hard requirement check on `shader-f16`.
 *
 * v1 ships two exercised knobs:
 *   - `matmul.rowsPerWorkgroupByKernel['ffn.linearGateUp']`  (1 or 4)
 *   - `pipeline.decodeDepth`                                  (1 or 2)
 *
 * Other fields are shape-ready for v0.2+ without breaking existing profiles.
 */

/** Kernels the tuning layer can target. Only a subset is wired in v1 —
 * the rest are shape-ready for profile growth. */
export type KernelName =
  | 'attn.linearQ'
  | 'attn.linearK'
  | 'attn.linearV'
  | 'attn.linearOut'
  | 'ffn.linearGateUp'
  | 'ffn.linearDown'
  | 'ple.linearInpGate'
  | 'ple.linearPostProj'
  | 'ple.linearPmProj'
  | 'lmHead';

/** Feature-intent value. `'auto'` means defer to the profile's normal
 * vendor-appropriate default; `'use'` and `'avoid'` are explicit. */
export type FeatureIntent = 'use' | 'avoid' | 'auto';

export interface TuningProfile {
  /** Stable identifier. Matches the `PROFILES` registry key. */
  id: string;
  /** One-line description shown in diagnostics. */
  description: string;
  /** True if a real calibration sweep has been run on matching hardware. */
  verified: boolean;
  /** Free-form profile notes (what was tuned, what wasn't). */
  notes?: string;

  matmul: {
    /** Threads per workgroup for matmul kernels. v1 ships 256. */
    workgroupSize: number;
    /** Default rows-per-workgroup for matmul kernels not listed in the
     * per-kernel override map. 1 = scalar matmul; 4 = MR4 variant. */
    defaultRowsPerWorkgroup: number;
    /** Per-kernel override. Values here win over `defaultRowsPerWorkgroup`. */
    rowsPerWorkgroupByKernel?: Partial<Record<KernelName, number>>;
  };

  pipeline: {
    /** Decode-loop depth: 1 (serial) or 2 (Option-B pipelined). */
    decodeDepth: 1 | 2;
    /** Whether the greedy-fast (temp=0, repPenalty≤1) on-GPU argmax→embed
     * fan-out path is available. `decodeDepth: 2` requires this. */
    greedyFastPath: boolean;
  };

  features: {
    /** Engine init throws without `shader-f16`; the field is informational. */
    shaderF16Required: boolean;
    /** Intent for WebGPU `subgroups` feature. `'avoid'` = do not request
     * subgroup-specific kernels even if the feature is available. */
    subgroups: FeatureIntent;
    /** Intent for Chromium's experimental cooperative-matrix feature. */
    subgroupMatrix: FeatureIntent;
  };
}

/** Resolve the effective rows-per-workgroup for one kernel. */
export function rowsPerWorkgroupFor(profile: TuningProfile, kernel: KernelName): number {
  return profile.matmul.rowsPerWorkgroupByKernel?.[kernel] ?? profile.matmul.defaultRowsPerWorkgroup;
}

/** Deep-partial of a TuningProfile for `overrideProfile` input. */
export type TuningProfileOverrides = {
  id?: string;
  description?: string;
  verified?: boolean;
  notes?: string;
  matmul?: Partial<TuningProfile['matmul']> & {
    rowsPerWorkgroupByKernel?: Partial<Record<KernelName, number>>;
  };
  pipeline?: Partial<TuningProfile['pipeline']>;
  features?: Partial<TuningProfile['features']>;
};

/** Deep-merge `partial` onto `base`. `rowsPerWorkgroupByKernel` merges
 * per-key; all other nested objects merge one level deep; scalars replace. */
export function overrideProfile(base: TuningProfile, partial: TuningProfileOverrides): TuningProfile {
  return {
    id: partial.id ?? base.id,
    description: partial.description ?? base.description,
    verified: partial.verified ?? base.verified,
    notes: partial.notes ?? base.notes,
    matmul: {
      workgroupSize: partial.matmul?.workgroupSize ?? base.matmul.workgroupSize,
      defaultRowsPerWorkgroup:
        partial.matmul?.defaultRowsPerWorkgroup ?? base.matmul.defaultRowsPerWorkgroup,
      rowsPerWorkgroupByKernel: {
        ...(base.matmul.rowsPerWorkgroupByKernel ?? {}),
        ...(partial.matmul?.rowsPerWorkgroupByKernel ?? {}),
      },
    },
    pipeline: {
      decodeDepth: partial.pipeline?.decodeDepth ?? base.pipeline.decodeDepth,
      greedyFastPath: partial.pipeline?.greedyFastPath ?? base.pipeline.greedyFastPath,
    },
    features: {
      shaderF16Required: partial.features?.shaderF16Required ?? base.features.shaderF16Required,
      subgroups: partial.features?.subgroups ?? base.features.subgroups,
      subgroupMatrix: partial.features?.subgroupMatrix ?? base.features.subgroupMatrix,
    },
  };
}
