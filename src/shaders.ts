// Shader registry. Each kernel lives in its own `shaders/*.wgsl` file and
// is imported here as a raw string via Vite's `?raw` suffix. The engine's
// pipeline-creation step iterates over `SHADERS` and builds one compute
// pipeline per entry.
//
// Weight storage is F16 on GPU: shaders that read weights declare
// `array<f16>` and promote to `f32` for arithmetic. The `shader-f16`
// feature is required on the active adapter — engine init throws if it's
// missing.

import addSrc from '../shaders/add.wgsl?raw';
import argmaxSrc from '../shaders/argmax.wgsl?raw';
import attnOutputSrc from '../shaders/attn_output.wgsl?raw';
import attnScoreSrc from '../shaders/attn_score.wgsl?raw';
import embeddingLookupSrc from '../shaders/embedding_lookup.wgsl?raw';
import fusedNormAddSrc from '../shaders/fused_norm_add.wgsl?raw';
import fusedPerHeadNormRopeSrc from '../shaders/fused_per_head_norm_rope.wgsl?raw';
import geluMulSrc from '../shaders/gelu_mul.wgsl?raw';
import kvCacheStoreSrc from '../shaders/kv_cache_store.wgsl?raw';
import logitSoftcapSrc from '../shaders/logit_softcap.wgsl?raw';
import matmulQuantSrc from '../shaders/matmul_quant.wgsl?raw';
import matmulQuantMR4Src from '../shaders/matmul_quant_mr4.wgsl?raw';
import perHeadRmsNormSrc from '../shaders/per_head_rms_norm.wgsl?raw';
import perHeadRmsNormNoWeightSrc from '../shaders/per_head_rms_norm_no_weight.wgsl?raw';
import pleGeluMulSrc from '../shaders/ple_gelu_mul.wgsl?raw';
import pleSkipScaleAddSrc from '../shaders/ple_skip_scale_add.wgsl?raw';
import pleStage1FuseSrc from '../shaders/ple_stage1_fuse.wgsl?raw';
import rmsNormSrc from '../shaders/rms_norm.wgsl?raw';
import ropeSrc from '../shaders/rope.wgsl?raw';
import softmaxSrc from '../shaders/softmax.wgsl?raw';
import topk256Src from '../shaders/topk256.wgsl?raw';

/** Kernel name → WGSL source. The engine creates one compute pipeline per entry. */
export const SHADERS: Record<string, string> = {
  add: addSrc,
  argmax: argmaxSrc,
  attnOutput: attnOutputSrc,
  attnScore: attnScoreSrc,
  embeddingLookup: embeddingLookupSrc,
  fusedNormAdd: fusedNormAddSrc,
  fusedPerHeadNormRope: fusedPerHeadNormRopeSrc,
  geluMul: geluMulSrc,
  kvCacheStore: kvCacheStoreSrc,
  logitSoftcap: logitSoftcapSrc,
  matmulQuant: matmulQuantSrc,
  matmulQuantMR4: matmulQuantMR4Src,
  perHeadRmsNorm: perHeadRmsNormSrc,
  perHeadRmsNormNoWeight: perHeadRmsNormNoWeightSrc,
  pleGeluMul: pleGeluMulSrc,
  pleSkipScaleAdd: pleSkipScaleAddSrc,
  pleStage1Fuse: pleStage1FuseSrc,
  rmsNorm: rmsNormSrc,
  rope: ropeSrc,
  softmax: softmaxSrc,
  topk256: topk256Src,
};

/** Named type for kernel names — useful when wiring per-kernel tuning overrides. */
export type KernelName = keyof typeof SHADERS;
