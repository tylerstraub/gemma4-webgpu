/**
 * Model configuration for Gemma 4 E2B.
 *
 * The baseline values are set by `defaultConfig()` in
 * `./config/gemma4-e2b.ts` and overridden from GGUF metadata by
 * `configFromGGUF()` in the same module.
 */
export interface GemmaConfig {
  hidden_size: number;
  /** Scalar max across layers. Use `getHeadDim(il)` for per-layer. */
  q_dim: number;
  /** Scalar max across layers. */
  kv_dim: number;
  num_q_heads: number;
  num_kv_heads: number;
  /** Scalar (= head_dim_local). Use `getHeadDim(il)` for per-layer. */
  head_dim: number;
  /** Scalar max across layers. Use `getIntermediateSize(il)` for per-layer. */
  intermediate_size: number;
  vocab_size: number;
  num_layers: number;
  context_length: number;
  rms_norm_eps: number;

  /** Full-attention RoPE theta (Gemma 4 E2B default: 1e6). */
  rope_theta_global: number;
  /** Sliding-window-attention RoPE theta (Gemma 4 E2B default: 1e4). */
  rope_theta_swa: number;

  /** Sliding-layer head_dim (Gemma 4 E2B default: 256). */
  head_dim_local: number;
  /** Full-attention-layer head_dim (Gemma 4 E2B default: 512). */
  head_dim_global: number;
  /** Sliding-window attention window size (Gemma 4 E2B default: 512). */
  sliding_window: number;
  /**
   * Per-layer sliding/full marker. `true` at index `i` means layer `i` uses
   * sliding-window (LOCAL) attention; `false` means full (GLOBAL) attention.
   * Length = num_layers.
   */
  attention_is_sliding: boolean[];
  /**
   * Per-layer FFN intermediate size. Gemma 4 E2B uses a mixed pattern:
   * producer layers (0..14) have I=6144, consumer layers (15..34) have I=12288.
   * Length = num_layers.
   */
  intermediate_sizes: number[];
  /**
   * Number of layers that compute their own K/V. The remaining
   * `num_layers - num_unshared_layers` layers are "consumers" that share the
   * KV cache of an earlier producer layer of the same attention type.
   * Gemma 4 E2B: 15.
   */
  num_unshared_layers: number;
  /**
   * For each layer `i`, the index of the layer whose K/V cache layer `i`
   * reads from. For producer layers this is `i` itself; for consumer layers
   * it's the last unshared layer of the same attention type (sliding or full).
   * Length = num_layers.
   */
  kv_producer_for_layer: number[];
  /** Per-layer input embedding dim (Gemma 4 E2B PLE: 256). */
  per_layer_input_dim: number;
  /** tanh softcap applied to final logits (Gemma 4 E2B: 30.0). */
  final_logit_softcapping: number;
}

/** True if layer `il` uses sliding-window (LOCAL) attention. */
export function isSwaLayer(il: number, config: GemmaConfig): boolean {
  return config.attention_is_sliding[il];
}

/** RoPE theta for layer `il`: sliding vs full-attention. */
export function getRopeTheta(il: number, config: GemmaConfig): number {
  return isSwaLayer(il, config) ? config.rope_theta_swa : config.rope_theta_global;
}

/** Per-layer head_dim: LOCAL (sliding) or GLOBAL (full-attention). */
export function getHeadDim(il: number, config: GemmaConfig): number {
  return isSwaLayer(il, config) ? config.head_dim_local : config.head_dim_global;
}

/** Per-layer FFN intermediate size. */
export function getIntermediateSize(il: number, config: GemmaConfig): number {
  return config.intermediate_sizes[il] ?? config.intermediate_size;
}

/** True if layer produces its own K/V (and writes to its own cache). */
export function isKvProducerLayer(il: number, config: GemmaConfig): boolean {
  return config.kv_producer_for_layer[il] === il;
}

/** Options for creating a GemmaEngine. */
export interface GemmaEngineOptions {
  /** Model to load: `'e2b'` (the verified default) or a full URL to a `.gguf` file. */
  model?: string;
  /** Progress callback during weight loading. */
  onProgress?: (progress: ProgressInfo) => void;
  /** Maximum context length (tokens). Defaults to 2048. */
  contextLength?: number;
  /**
   * Tuning profile override. Accepts a profile id (string), a full
   * `TuningProfile` object, or a deep-partial `TuningProfileOverrides`
   * merged onto the auto-selected base. Leave unset for pure vendor
   * auto-selection. See `src/tuning/` for the registry.
   */
  tuning?: string | import('./tuning/index.js').TuningProfile | import('./tuning/index.js').TuningProfileOverrides;
  /**
   * Optional cancellation signal for the load. If aborted, the active
   * HTTP Range fetch is aborted and `init()` rejects with an AbortError.
   * Note: no partial-load GPU cleanup is attempted — callers that plan
   * to abort should throw away any engine reference they hold, since
   * some buffers may have been created before the abort landed.
   */
  signal?: AbortSignal;
}

export interface ProgressInfo {
  /** Bytes downloaded so far. */
  loaded: number;
  /** Total bytes to download. */
  total: number;
  /** Human-readable status message. */
  status: string;
  /**
   * Discrete state classification. Surfaces that want to render different
   * UX per state (e.g. show a "connection stalled, retrying…" banner)
   * branch on this rather than parsing `status`. Optional — most call
   * sites omit it and the default intent is `'downloading'`.
   */
  kind?: 'downloading' | 'decoding' | 'retrying';
}

/**
 * One-shot phase breakdown of `createGemmaEngine()` / `engine.init()`.
 * Populated by the engine as init runs; exposed as
 * `engine.loadTimings` for the agent-first workbench
 * (`window.lab.loadTimings()`). Intended for answering
 * "where does the load wall-clock actually go" with an
 * empirical, deterministic measurement rather than a guess.
 *
 * All durations are wall-clock milliseconds. Phase fields are
 * undefined when that phase didn't run on the active load path
 * (e.g. the full-buffer path doesn't populate `layersFetch_ms`).
 */
export interface LoadTimings {
  /** True if the server honored the Range request and the streaming path ran. */
  supportsRange: boolean;
  /** Resolved model URL. */
  modelFile: string;
  /** `performance.now()` at the instant `init()` was entered. */
  startedAt_ms: number;
  /** End-to-end wall clock from `init()` entry to return. */
  totalMs: number;
  phases: {
    /** `requestAdapter` + `requestDevice` + feature negotiation. */
    initWebGPU_ms?: number;
    /** Time spent in the header `fetch(...)` + `arrayBuffer()` (streaming path: 20 MB; fallback path: full file). */
    headerFetch_ms?: number;
    /** GGUFParser.parse() + config hydration + tokenizer extraction. */
    ggufParse_ms?: number;
    /** 21 `createShaderModule` + 21 `createComputePipeline` calls, serial. */
    createPipelines_ms?: number;
    /** Per-layer and global uniform buffer creation. */
    createUniformBuffers_ms?: number;
    /** One monolithic Range request for `per_layer_token_embd` (~1.6 GB Q5_K). Legacy; populated only by the full-buffer fallback path. */
    pleFetch_ms?: number;
    /** CPU Q5_K decode + writeBuffer into 35 per-layer GPU buffers. Legacy; populated only by the full-buffer fallback path. */
    pleDecode_ms?: number;
    /**
     * Streaming path only. Combined wall-clock of the PLE pipeline, where
     * network fetch and CPU decode run concurrently over the same byte
     * stream (no monolithic 1.6 GB buffer ever held). Replaces
     * `pleFetch_ms` + `pleDecode_ms` on the streaming path.
     */
    plePipeline_ms?: number;
    /** Sum of `await fetchRange(...)` time across all 35 layers. Network-bound on the streaming path. */
    layersFetch_ms?: number;
    /** Sum of CPU dequant + GPU writeBuffer per layer across all 35 layers. */
    layersProcess_ms?: number;
    /** `layersFetch_ms + layersProcess_ms`. */
    layersTotal_ms?: number;
    /** Full-buffer (non-Range) path: total time downloading the entire GGUF. */
    fullBufferDownload_ms?: number;
    /** Full-buffer path: CPU dequant + GPU upload for every tensor. */
    fullBufferUpload_ms?: number;
    /** `createWorkBuffers` — scratch + KV cache allocation. */
    createWorkBuffers_ms?: number;
    /** `createBindGroups` — ~350 bind groups built serially. */
    createBindGroups_ms?: number;
  };
  /** Streaming path only. One entry per decoder layer, in load order. */
  perLayer?: Array<{
    layerIdx: number;
    /** `await fetchRange(layerStart, layerSize)` time. */
    fetchMs: number;
    /** CPU dequant + GPU writeBuffer for every tensor in this layer. */
    processMs: number;
    /** Bytes in the Range response for this layer's tensors. */
    bytes: number;
  }>;
  /** Disk bytes for the Q5_K `per_layer_token_embd` tensor. */
  pleBytes?: number;
  /** Sum of bytes across all per-layer Range responses. */
  layerBytesTotal?: number;
}

/**
 * Timing stats from the most recent `generate()` call. See
 * `GemmaEngine.lastGenerateStats` for the full contract.
 */
export interface LastGenerateStats {
  /** Number of tokens yielded. */
  tokens: number;
  /** Wall-clock ms from `generate()` entry to the first yield. */
  firstTokenMs: number;
  /**
   * Sum of time spent inside the generator from the first yield to the
   * last, excluding time paused waiting for the caller's body. Divide
   * `(tokens - 1)` by `(coreDecodeMs / 1000)` to get pure-engine tps.
   */
  coreDecodeMs: number;
  /**
   * Wall-clock ms from `generate()` entry to the last yield. Includes
   * time the generator was paused waiting for the caller's body.
   */
  totalWallMs: number;
  /** Whether the generator exited via an abort signal. */
  aborted: boolean;
}

export interface GenerateOptions {
  /** Sampling temperature. 0 = greedy. Default: 0.7 */
  temperature?: number;
  /** Top-P nucleus sampling threshold. Default: 0.9 */
  topP?: number;
  /** Repetition penalty. 1.0 = no penalty. Default: 1.2 */
  repPenalty?: number;
  /** Maximum tokens to generate. Default: 32768 */
  maxTokens?: number;
  /** Tools JSON string for function calling. Default: '[]' */
  toolsJson?: string;
  /** AbortSignal to cancel generation mid-stream. */
  signal?: AbortSignal;
}

/** A single turn in conversation history. */
export interface ConversationTurn {
  role: 'user' | 'model';
  text: string;
}

/**
 * Hidden-state capture point for numerical-equivalence debugging.
 * Halts the forward pass at the named boundary and exposes the underlying
 * GPU work buffer for readback.
 */
export type CapturePoint =
  | { kind: 'embed' }                        // after embeddingLookup writes `hidden`
  | { kind: 'plePmProjected' }               // after per_layer_model_proj matmul, before rmsnorm+fuse
  | { kind: 'pleStage1' }                    // after PLE Stage 1 writes `pleInputs` [num_layers * per_layer_input_dim]
  | { kind: 'preRopeQ'; layer: number }      // after linearQ matmul, before fusedPerHeadNormRope
  | { kind: 'preRopeK'; layer: number }      // after linearK matmul (producer layers only)
  | { kind: 'preRopeV'; layer: number }      // after linearV matmul (producer layers only)
  | { kind: 'postRopeQ'; layer: number }     // after fusedPerHeadNormRope on Q
  | { kind: 'postRopeK'; layer: number }     // after fusedPerHeadNormRope on K (producer only)
  | { kind: 'attnOut'; layer: number }       // after attnOutput kernel, before linearAttnOut
  | { kind: 'afterLayer'; layer: number }    // after layer L's final block write to `hidden`
  | { kind: 'final' }                        // after finalNorm writes `normed`
  | { kind: 'logits' };                      // after LM head + softcap writes `logits` [vocab_size]

export interface GemmaEngine {
  /** Add a user message to the conversation. */
  addUserMessage(text: string): void;
  /** Generate a response as an async iterator of token strings. */
  generate(options?: GenerateOptions): AsyncGenerator<string, void, undefined>;
  /**
   * Timing stats from the most recent `generate()` call. Populated at each
   * yield and finalized when the loop exits (naturally or via abort).
   * `null` until the first `generate()` completes at least one yield.
   *
   * `coreDecodeMs` measures time spent INSIDE the generator between the
   * first and last yield — explicitly excluding the time the generator
   * is paused waiting for the consumer's `for await` body to run. Use
   * `(tokens - 1) / (coreDecodeMs / 1000)` for a pure-engine tps that's
   * methodology-matched to `lab.bench` / `runBench`, independent of any
   * DOM work the caller does per token.
   *
   * `totalWallMs` is the full wall-clock including body time — useful
   * for surfacing the end-to-end user-perceived rate.
   */
  readonly lastGenerateStats: LastGenerateStats | null;
  /** Reset conversation history and KV cache. */
  resetConversation(): void;
  /**
   * Currently-applied system instruction, emitted as a `developer`-role
   * preamble in the chat template. `null` when none is set.
   */
  readonly systemPrompt: string | null;
  /**
   * Set the system instruction. Pass `null` or `''` to clear. Does NOT
   * reset conversation state — callers should call `resetConversation()`
   * before the next `generate()` for the new preamble to take effect, or
   * the stale preamble in the existing KV cache will be used.
   */
  setSystemPrompt(text: string | null): void;
  /** Release all GPU resources. */
  dispose(): void;
  /** Current config (read-only). */
  readonly config: Readonly<GemmaConfig>;
  /**
   * Active tuning profile + the reason it was selected. Frozen at init
   * time; dispose and re-create the engine to switch profiles.
   */
  readonly tuning: Readonly<import('./tuning/index.js').TuningProfile>;
  readonly tuningReason: string;
  /**
   * End-to-end phase breakdown of `init()`. Populated as init runs and
   * finalized by the time the engine is returned from
   * `createGemmaEngine`. `null` until init() starts. Consumers use this
   * to answer "where did load wall-clock actually go" without guessing.
   */
  readonly loadTimings: LoadTimings | null;

  // ─── Diagnostic: hidden-state capture ──────────────────────────
  /**
   * Run forward pass up to `stopAt` and return the F32 contents of the
   * relevant work buffer. Does NOT advance KV state beyond what the pass
   * writes to producer-layer KV caches. Caller is responsible for prefill
   * state and for calling `resetConversation` before / between captures if
   * the KV cache must start empty.
   */
  captureHidden(tokenId: number, position: number, stopAt: CapturePoint): Promise<Float32Array>;
  /** Tokenize a prompt for capture-path prefill. Exposes the engine's tokenizer. */
  encodePromptTokens(text: string, applyChatTemplate: boolean): number[];
  /** Prefill a contiguous token range starting at `startPos` without sampling. */
  prefillForCapture(tokens: number[], startPos: number): Promise<void>;
  /** Reset KV caches without clearing conversation history. */
  resetKVForCapture(): void;
  /**
   * Read the per-layer embedding slice for a single token from each
   * `perLayerEmbeddings[L]` GPU buffer. Returns `num_layers × per_layer_dim`
   * F32 arrays — useful for verifying that the Q5_K streaming decode packed
   * the per-layer rows correctly.
   */
  captureEmbedSliceAll(tokenId: number): Promise<Float32Array[]>;

  // ─── Diagnostic: weight readback ───────────────────────────────
  /**
   * Per-named global tensor (e.g. `per_layer_model_proj`), read `elemCount`
   * F16 elements starting at `elemOffset`, converted to F32. Used to verify
   * weight-layout assumptions by comparing against raw GGUF source bytes.
   */
  readGlobalTensor(name: string, elemOffset: number, elemCount: number): Promise<Float32Array>;
  /**
   * Read a slice of a per-layer weight buffer (`layers[L][name]`). Uploaded
   * with COPY_SRC when diagnostics are enabled.
   */
  readLayerTensor(layer: number, name: string, elemOffset: number, elemCount: number): Promise<Float32Array>;
  /** Read a slice of the tied embedding / LM head buffer. */
  readEmbeddingTensor(elemOffset: number, elemCount: number): Promise<Float32Array>;
  /**
   * Dequantize a slice of a named GGUF source tensor directly from the
   * on-disk bytes (refetched via Range request). Used as ground-truth oracle
   * against `readGlobalTensor`.
   */
  fetchRawTensorSlice(name: string, elemOffset: number, elemCount: number): Promise<Float32Array>;
  /** Look up name / type / dims / offset for a loaded tensor. */
  getGgufTensorInfo(name: string): { dims: number[]; type: number; offset: number; byteSize: number } | null;

  // ─── Diagnostic: per-dispatch GPU timing ───────────────────────
  /**
   * Capability flags that the profiler needs to decide if it can run at all.
   * Captured once at init from `adapter.features` and the active `GPUDevice`.
   */
  getProfileCapability(): ProfileCapability;
  /** Arm profiling so the next forward passes emit timestampWrites. */
  enableProfile(): void;
  /** Disarm. Subsequent forward passes run with zero timing overhead. */
  disableProfile(): void;
  /** Forget previously-accumulated samples. Call between warmup and measurement. */
  resetProfileSamples(): void;
  /** Aggregate current samples into a JSON report. */
  getProfileReport(): ProfileReport;

  // ─── Diagnostic: CPU-side timing ───────────────────────────────
  /** Arm CPU-side phase timing. Zero cost when off; O(~200ns) per phase when on. */
  enableCpuProfile(): void;
  disableCpuProfile(): void;
  resetCpuProfileSamples(): void;
  getCpuProfileReport(): CpuProfileReport;
}

export interface ProfileCapability {
  /** `timestamp-query` feature was requested and is active on this device. */
  timestampQuerySupported: boolean;
  /** `timestamp-query` feature was present on the adapter but not enabled (should never happen in practice; diagnostic). */
  timestampQueryOnAdapterButUnrequested: boolean;
  /** Max query slots in the profiler's pre-allocated `GPUQuerySet`. Forward passes exceeding this drop overflow silently — we log once. */
  querySlots: number;
}

export interface ProfileReport {
  /** Aggregates per label. `samples_n` is how many forward passes contributed. */
  perLabel: Array<{
    label: string;
    samples_n: number;
    p50_ns: number;
    p95_ns: number;
    mean_ns: number;
    total_ns: number;
  }>;
  /** Sum across all labels per forward pass (median over samples). Rough upper bound on forward GPU time. */
  forwardMedian_ns: number;
  /** Number of forward passes contributing to the report. */
  forwardSamples_n: number;
  /** True if at least one forward pass tried to dispatch more passes than `querySlots` could hold. */
  overflow: boolean;
}

/**
 * CPU-side timing report. Durations are in milliseconds. The
 * `mapAsync.wait` phase is dominated by GPU work completing, not CPU work;
 * subtract it from forward.total to get the CPU-only floor.
 */
export interface CpuProfileReport {
  perPhase: Array<{
    phase: string;
    samples_n: number;
    p50_ms: number;
    p95_ms: number;
    mean_ms: number;
    total_ms: number;
  }>;
  /** p50 of full forward-pass wall-clock per token. */
  forwardMedian_ms: number;
  forwardSamples_n: number;
  /** Median writeBuffer calls per forward. */
  writeBufferCountMedian: number;
  /** Median pass-encode count per forward (beginPass + end). */
  passEncodeCountMedian: number;
}

/** Parsed GGUF file structure. */
export interface GGUFParsed {
  version: number;
  tensor_count: bigint;
  kv_count: bigint;
  kv: Map<string, GGUFValue>;
  tensors: GGUFTensor[];
  dataOffset: number;
}

export interface GGUFValue {
  type: string;
  value: unknown;
}

export interface GGUFTensor {
  name: string;
  dims: bigint[];
  type: number;
  offset: bigint;
}

export interface ModelBuffers {
  /**
   * Main token-embedding weights. F16 on GPU (u16-packed `array<f16>`).
   * Also used as the tied LM head weight.
   */
  tokenEmbed: GPUBuffer | null;
  /** Per-layer weights. Each is F16 on GPU. Keyed by tensor name (`attn_q`, `ffn_gate`, …). */
  layers: Record<string, GPUBuffer>[];
  finalNorm: GPUBuffer | null;
  /**
   * Global tensors: `per_layer_model_proj`, `per_layer_proj_norm`,
   * `rope_freqs`. F16 on GPU.
   */
  globals: Record<string, GPUBuffer>;
  /**
   * Per-layer embedding tables, split across `num_layers` buffers to stay
   * under WebGPU's 2 GB buffer ceiling. Each slice is
   * `[vocab, per_layer_input_dim]` F16. `perLayerEmbeddings[l]` holds the
   * lookup table for decoder layer `l`.
   */
  perLayerEmbeddings: GPUBuffer[];
}

export interface WorkBuffers {
  hidden: GPUBuffer;
  hiddenReadback: GPUBuffer;
  residual: GPUBuffer;
  normed: GPUBuffer;
  q: GPUBuffer;
  k: GPUBuffer;
  v: GPUBuffer;
  attnOut: GPUBuffer;
  attnProj: GPUBuffer;
  postAttnNormed: GPUBuffer;
  attnScores: GPUBuffer;
  ffnGate: GPUBuffer;
  ffnUp: GPUBuffer;
  ffnMul: GPUBuffer;
  ffnDown: GPUBuffer;
  postFfnNormed: GPUBuffer;
  logits: GPUBuffer;
  logitsReadback: GPUBuffer;
  argmaxResult: GPUBuffer;
  /**
   * Ping-pong readback pool for depth-2 pipelined decode. Two concurrent
   * greedy-fast forwards would otherwise `mapAsync` the same buffer, which
   * WebGPU forbids ("Buffer already has an outstanding map pending"). The
   * sampler alternates between `pool[0]` and `pool[1]`.
   */
  argmaxReadbackPool: [GPUBuffer, GPUBuffer];
  topk256Result: GPUBuffer;
  topk256Readback: GPUBuffer;
  // Per-layer embedding (PLE) work buffers.
  /** Stage 1 intermediate: `per_layer_model_proj @ hidden` → `[num_layers * per_layer_input_dim]` F32. */
  plePmProjected: GPUBuffer;
  /** Stage 1 output (= Stage 2 input): `[num_layers * per_layer_input_dim]` F32. Indexed by layer. */
  pleInputs: GPUBuffer;
  /** Stage 2 step a: `inp_gate @ hidden` result. `[per_layer_input_dim]` F32. Reused across layers. */
  pleGate: GPUBuffer;
  /** Stage 2 step b: `gelu(gate) * pleInputs[L]`. `[per_layer_input_dim]` F32. Reused across layers. */
  pleGated: GPUBuffer;
  /** Stage 2 step c: `proj @ pleGated`. `[hidden_size]` F32. Reused. */
  pleProjOut: GPUBuffer;
  /** Stage 2 step d: `post_norm(pleProjOut)`. `[hidden_size]` F32. Reused. */
  plePostNormed: GPUBuffer;
}

export interface KVCache {
  k: GPUBuffer;
  v: GPUBuffer;
}

export interface UniformBuffers {
  // Shape-invariant uniforms (same across all layers).
  rmsNorm: GPUBuffer;
  sizeH: GPUBuffer;
  embeddingLookup: GPUBuffer;
  softmax: GPUBuffer;
  linearQ8_V_H: GPUBuffer;
  argmaxSize: GPUBuffer;
  logitSoftcap: GPUBuffer;
  // PLE uniforms.
  /** matmulQuant params for `per_layer_model_proj`: M = num_layers * per_layer_input_dim, N = hidden_size. */
  plePmProjMM: GPUBuffer;
  /** matmulQuant params for `inp_gate`: M = per_layer_input_dim, N = hidden_size. Shared across layers. */
  pleInpGateMM: GPUBuffer;
  /** matmulQuant params for `per_layer_projection`: M = hidden_size, N = per_layer_input_dim. Shared across layers. */
  plePostProjMM: GPUBuffer;
  /** Stage 1 params, per layer: `{ layer_idx, token_id, per_layer_dim, eps }`. `token_id` updated per forward pass. */
  pleStage1: GPUBuffer[];
  /** Stage 2 GELU-mul params, per layer: `{ layer_offset = il * per_layer_input_dim, size = per_layer_input_dim }`. */
  pleGeluMulParams: GPUBuffer[];
  /** Stage 2 skip_scale_add params (size = hidden_size). Shared across layers. */
  pleSkipScaleAdd: GPUBuffer;

  // Per-layer uniforms. Populated per layer using `getHeadDim(il)` and
  // `getIntermediateSize(il)`, since LOCAL/GLOBAL layers have head_dim
  // 256/512 and consumer layers have double-wide FFN (6144 → 12288).
  perHeadRmsNormQ: GPUBuffer[];
  perHeadRmsNormK: GPUBuffer[];
  perHeadRmsNormV: GPUBuffer[];
  linearQ8_Q_H: GPUBuffer[];
  linearQ8_KV_H: GPUBuffer[];
  linearQ8_H_Q: GPUBuffer[];
  linearQ8_I_H: GPUBuffer[];
  linearQ8_H_I: GPUBuffer[];
  sizeI: GPUBuffer[];
  kvCacheStore: GPUBuffer[];
  attnScore: GPUBuffer[];
  attnOutput: GPUBuffer[];
  ropeQ: GPUBuffer[];
  ropeK: GPUBuffer[];
  fusedNormRopeQ: GPUBuffer[];
  fusedNormRopeK: GPUBuffer[];
}

export interface BindGroupCache {
  embeddingLookup: GPUBindGroup;
  finalNorm: GPUBindGroup;
  lmHead: GPUBindGroup;
  logitSoftcap: GPUBindGroup;
  argmax: GPUBindGroup;
  topk256: GPUBindGroup;
  layers: LayerBindGroups[];
  // Per-layer-embedding bind groups. Always populated.
  plePmProjMatmul: GPUBindGroup;
  pleStage1Fuse: GPUBindGroup[];
}

export interface LayerBindGroups {
  attnNorm: GPUBindGroup;
  linearQ: GPUBindGroup;
  linearK: GPUBindGroup;
  linearV: GPUBindGroup;
  ropeQ: GPUBindGroup;
  ropeK: GPUBindGroup;
  qNorm: GPUBindGroup;
  kNorm: GPUBindGroup;
  vNorm: GPUBindGroup;
  fusedNormRopeQ: GPUBindGroup;
  fusedNormRopeK: GPUBindGroup;
  kvStore: GPUBindGroup;
  attnScore: GPUBindGroup;
  softmax: GPUBindGroup;
  attnOutput: GPUBindGroup;
  linearAttnOut: GPUBindGroup;
  postAttnNorm: GPUBindGroup;
  residualAdd1: GPUBindGroup;
  ffnNorm: GPUBindGroup;
  ffnGate: GPUBindGroup;
  ffnUp: GPUBindGroup;
  geluMul: GPUBindGroup;
  ffnDown: GPUBindGroup;
  postFfnNorm: GPUBindGroup;
  residualAdd2: GPUBindGroup;
  fusedPostAttnNormAdd: GPUBindGroup;
  fusedPostFfnNormAdd: GPUBindGroup;
  // PLE Stage 2 bind groups. Populated for every layer.
  pleInpGateMatmul: GPUBindGroup;
  pleGeluMul: GPUBindGroup;
  plePostProjMatmul: GPUBindGroup;
  plePostNorm: GPUBindGroup;
  pleSkipScaleAdd: GPUBindGroup;
}
