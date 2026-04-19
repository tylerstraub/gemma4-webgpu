// Shared engine singleton + pub-sub store for the human-first demo.
//
// Panels subscribe for state changes and call lifecycle verbs. This module
// is the sole owner of the `GemmaEngine` instance across the app — only one
// engine is ever live (2.9 GB of weights is enough to load once).
//
// `window.chat` is mounted for back-compat with the pre-rewrite chat API
// (kept so console-driven debugging keeps working), but the human page does
// not introduce a broader `window.lab`-equivalent surface. Power users who
// want JSON go to the dev-only workbench.

import { createGemmaEngine } from '../../src/index.js';
import type {
  GemmaEngine,
  GemmaEngineOptions,
  GemmaConfig,
  GenerateOptions,
  ProgressInfo,
  TuningProfile,
  TuningOverride,
} from '../../src/index.js';

export type EngineStatus =
  | 'uninit'
  | 'loading'
  | 'ready'
  | 'generating'
  | 'busy'
  | 'reiniting'
  | 'error';

export interface ChatMessage {
  id: number;
  role: 'user' | 'model';
  text: string;
  /** Set while the model is still streaming into this message. */
  streaming?: boolean;
  /** ms from `generate()` start to first token for this reply. */
  firstTokenMs?: number;
  /**
   * Wall-clock tps measured by the app — `(tokens - 1) / (wall decode ms)`.
   * Includes time spent in the panel subscriber's DOM work between
   * tokens. This is the user-perceived rate.
   */
  tps?: number;
  /**
   * Pure engine-side tps — `(tokens - 1) / (coreDecodeMs / 1000)` from
   * the engine's internal timing. Excludes caller body time, so it's
   * methodology-matched to `lab.bench` / `runBench`.
   */
  engineTps?: number;
  /** Token count for this reply. */
  tokens?: number;
}

export interface EngineState {
  status: EngineStatus;
  engine: GemmaEngine | null;
  config: Readonly<GemmaConfig> | null;
  tuning: Readonly<TuningProfile> | null;
  tuningReason: string | null;
  modelKey: string | null;
  loadProgress: ProgressInfo | null;
  messages: ChatMessage[];
  /**
   * System instruction currently applied to the engine (the preamble
   * baked into the KV cache). Reflects `engine.systemPrompt`. Changes only
   * when `applySystemPrompt` is called — edits made in panel-local state
   * aren't applied until the user commits them.
   */
  systemPrompt: string | null;
  /** Most recent error, if any. Cleared on the next successful lifecycle op. */
  error: string | null;
  /** Free-form status line — panels can read and render this verbatim. */
  statusLine: string;
}

type Listener = (state: EngineState) => void;

const DEFAULT_GEN_OPTS: Required<Pick<GenerateOptions, 'temperature' | 'topP' | 'repPenalty' | 'maxTokens'>> = {
  temperature: 0.7,
  topP: 0.9,
  repPenalty: 1.2,
  maxTokens: 1024,
};

const state: EngineState = {
  status: 'uninit',
  engine: null,
  config: null,
  tuning: null,
  tuningReason: null,
  modelKey: null,
  loadProgress: null,
  messages: [],
  systemPrompt: null,
  error: null,
  statusLine: 'Idle · load the model to begin.',
};

const listeners = new Set<Listener>();
let messageIdSeq = 0;
let abortController: AbortController | null = null;

export function getState(): EngineState {
  return state;
}

export function subscribe(fn: Listener): () => void {
  listeners.add(fn);
  // Emit current state so panels render immediately on mount.
  fn(state);
  return () => listeners.delete(fn);
}

function emit(): void {
  for (const fn of listeners) fn(state);
}

function setStatusLine(line: string): void {
  state.statusLine = line;
  emit();
}

function progressLine(p: ProgressInfo): string {
  const pct = p.total ? `${((p.loaded / p.total) * 100).toFixed(1)}%` : '';
  return `${p.status}${pct ? ` · ${pct}` : ''}`.trim();
}

// ─── Lifecycle ─────────────────────────────────────────────────────

export interface InitOptions {
  modelKey?: string;
  contextLength?: number;
  tuning?: TuningOverride;
}

export async function init(opts: InitOptions = {}): Promise<void> {
  if (state.engine) throw new Error('engine already loaded — call dispose() first');
  state.status = 'loading';
  state.error = null;
  state.modelKey = opts.modelKey ?? 'e2b';
  setStatusLine('Loading model · this takes 70–90s on a fast link.');
  const engineOpts: GemmaEngineOptions = {
    model: state.modelKey,
    contextLength: opts.contextLength,
    tuning: opts.tuning,
    onProgress: (p) => {
      state.loadProgress = p;
      setStatusLine(progressLine(p));
    },
  };
  try {
    state.engine = await createGemmaEngine(engineOpts);
    state.config = state.engine.config;
    state.tuning = state.engine.tuning;
    state.tuningReason = state.engine.tuningReason;
    // Re-apply any previously-configured system prompt onto the fresh
    // engine instance. Tuning re-inits preserve the user's instruction.
    if (state.systemPrompt) {
      state.engine.setSystemPrompt(state.systemPrompt);
    }
    state.loadProgress = null;
    state.status = 'ready';
    setStatusLine(
      `Ready · ${state.config.num_layers}L · ctx=${state.config.context_length} · tuning=${state.tuning.id}`,
    );
  } catch (e) {
    state.status = 'error';
    state.error = (e as Error).message;
    setStatusLine(`Error: ${state.error}`);
    throw e;
  }
}

export async function reinitWithTuning(tuning: TuningOverride): Promise<void> {
  if (!state.engine) {
    await init({ tuning });
    return;
  }
  state.status = 'reiniting';
  setStatusLine('Disposing engine for tuning change…');
  state.engine.dispose();
  state.engine = null;
  state.config = null;
  state.tuning = null;
  state.tuningReason = null;
  // Note: messages are preserved visually. The KV cache is gone; future turns
  // re-prefill from scratch. Panels should annotate that re-init happened.
  await init({ modelKey: state.modelKey ?? 'e2b', tuning });
}

export function dispose(): void {
  abort();
  state.engine?.dispose();
  state.engine = null;
  state.config = null;
  state.tuning = null;
  state.tuningReason = null;
  state.loadProgress = null;
  state.messages = [];
  state.status = 'uninit';
  state.error = null;
  setStatusLine('Disposed · reload the page to start again.');
}

// ─── Chat ──────────────────────────────────────────────────────────

export async function send(text: string, opts: GenerateOptions = {}): Promise<string> {
  if (!state.engine) throw new Error('engine not loaded — call init() first');
  if (state.status === 'generating') throw new Error('already generating — call abort() first');
  if (state.status === 'busy' || state.status === 'reiniting')
    throw new Error(`engine is ${state.status}; wait for it to finish`);

  const userMsg: ChatMessage = { id: ++messageIdSeq, role: 'user', text };
  const modelMsg: ChatMessage = { id: ++messageIdSeq, role: 'model', text: '', streaming: true };
  state.messages.push(userMsg, modelMsg);
  state.status = 'generating';
  setStatusLine('Generating…');

  abortController = new AbortController();
  const gen: GenerateOptions = { ...DEFAULT_GEN_OPTS, ...opts, signal: abortController.signal };

  state.engine.addUserMessage(text);
  const startMs = performance.now();
  let firstMs: number | null = null;
  let tokens = 0;
  try {
    for await (const token of state.engine.generate(gen)) {
      tokens++;
      if (tokens === 1) {
        firstMs = performance.now() - startMs;
        modelMsg.firstTokenMs = firstMs;
      }
      modelMsg.text += token;
      modelMsg.tokens = tokens;
      emit();
    }
  } catch (e) {
    state.status = 'error';
    state.error = (e as Error).message;
    modelMsg.streaming = false;
    setStatusLine(`Error: ${state.error}`);
    throw e;
  }

  const totalMs = performance.now() - startMs;
  const decodeMs = totalMs - (firstMs ?? totalMs);
  modelMsg.tps = tokens > 1 && decodeMs > 0 ? (tokens - 1) / (decodeMs / 1000) : 0;

  // Pull the engine-side timing — this is the pure-decode tps that
  // matches bench methodology (excludes panel DOM work between yields).
  const engineStats = state.engine.lastGenerateStats;
  if (engineStats && engineStats.tokens > 1 && engineStats.coreDecodeMs > 0) {
    modelMsg.engineTps = (engineStats.tokens - 1) / (engineStats.coreDecodeMs / 1000);
  }

  modelMsg.streaming = false;
  modelMsg.tokens = tokens;
  state.status = 'ready';
  abortController = null;
  const engineLine = modelMsg.engineTps != null
    ? ` (engine ${modelMsg.engineTps.toFixed(1)})`
    : '';
  setStatusLine(
    `Ready · last reply ${tokens} tok, ${modelMsg.tps!.toFixed(1)} tps${engineLine}, first token ${(firstMs ?? 0).toFixed(0)}ms`,
  );
  return modelMsg.text;
}

export function abort(): void {
  abortController?.abort();
  abortController = null;
}

export function reset(): void {
  state.engine?.resetConversation();
  state.messages = [];
  if (state.status !== 'uninit' && state.status !== 'error') {
    state.status = 'ready';
    setStatusLine('Conversation reset.');
  } else {
    emit();
  }
}

/**
 * Set the system instruction on the engine and reset the conversation
 * so the new preamble takes effect on the next generation. Stores the
 * instruction in app state so it survives tuning re-inits.
 */
export function applySystemPrompt(text: string | null): void {
  const trimmed = text?.trim() ?? '';
  const normalized = trimmed === '' ? null : trimmed;
  state.systemPrompt = normalized;
  if (state.engine) {
    state.engine.setSystemPrompt(normalized);
    // Reset so the new preamble is rebuilt into the next KV prefill.
    state.engine.resetConversation();
    state.messages = [];
    if (state.status === 'ready' || state.status === 'error') {
      state.status = 'ready';
      state.error = null;
    }
  }
  setStatusLine(normalized
    ? `System instruction applied · ${normalized.length} chars · conversation reset.`
    : 'System instruction cleared · conversation reset.');
}

// ─── Long-running ops — panel-driven bench/profile/report ──────────
//
// These mutate engine state (resetConversation between runs). Callers must
// gate on `status === 'ready'` and warn the user — they will clobber the KV
// cache and in-flight chat streaming.

export function assertReady(): GemmaEngine {
  if (!state.engine || state.status !== 'ready') {
    throw new Error(`engine not ready (status=${state.status})`);
  }
  return state.engine;
}

/**
 * Put the engine into a `busy` state so chat/other panels refuse to run in
 * parallel. Returns a disposer that restores `ready` (or `error` if you
 * pass one). Use inside bench/profile/report async flows.
 */
export function beginBusy(message: string): (finalStatus?: { ok: true } | { ok: false; error: string }) => void {
  if (state.status !== 'ready') throw new Error(`cannot begin busy from status=${state.status}`);
  state.status = 'busy';
  setStatusLine(message);
  return (result = { ok: true }) => {
    if (result.ok) {
      state.status = 'ready';
      state.error = null;
    } else {
      state.status = 'error';
      state.error = result.error;
    }
    emit();
  };
}

/** Update the status-line message while in a busy span. */
export function updateStatusLine(line: string): void {
  setStatusLine(line);
}

// ─── window.chat back-compat surface ───────────────────────────────
//
// Thin shim that matches the API surface the pre-rewrite chat/main.ts
// exposed. Kept so console-driven debugging and any existing snippet that
// references `window.chat.send('...')` continues to work.

interface ChatCompat {
  readonly engine: GemmaEngine | null;
  readonly status: EngineStatus;
  readonly error: string | null;
  readonly messages: ReadonlyArray<ChatMessage>;
  readonly loadProgress: ProgressInfo | null;
  init: (modelKey?: string, opts?: { contextLength?: number; tuning?: TuningOverride }) => Promise<void>;
  send: (text: string, opts?: GenerateOptions) => Promise<string>;
  abort: () => void;
  reset: () => void;
  dispose: () => void;
}

declare global {
  interface Window {
    chat: ChatCompat;
  }
}

export function installWindowChat(): void {
  const compat: ChatCompat = {
    get engine() { return state.engine; },
    get status() { return state.status; },
    get error() { return state.error; },
    get messages() { return state.messages; },
    get loadProgress() { return state.loadProgress; },
    init: (modelKey, opts) => init({ modelKey, contextLength: opts?.contextLength, tuning: opts?.tuning }),
    send,
    abort,
    reset,
    dispose,
  };
  window.chat = compat;
}
