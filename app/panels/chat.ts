// Chat panel — streaming markdown, generation controls, live stats.
//
// The default landing view. Primary concerns: render markdown safely and
// responsively as tokens stream in, expose the useful generation knobs
// without cluttering, and surface per-reply tps/first-token latency so the
// engine's performance story is visible while you chat.

import {
  subscribe,
  getState,
  send,
  abort,
  reset,
  applySystemPrompt,
  type ChatMessage,
  type EngineState,
} from '../lib/engine.js';
import type { GenerateOptions } from '../../src/index.js';
import { renderMarkdown } from '../lib/markdown.js';
import { formatMs, formatTps } from '../lib/format.js';
import { confirmDialog } from '../lib/dialog.js';

const GEN_DEFAULTS = {
  temperature: 0.7,
  topP: 0.9,
  repPenalty: 1.2,
  maxTokens: 1024,
};

export function mountChatPanel(root: HTMLElement): void {
  root.innerHTML = `
    <div class="chat-grid">
      <div class="chat-surface card">
        <div class="chat-messages-wrap">
          <div class="chat-messages" id="chat-messages" aria-live="polite"></div>
          <button type="button" class="jump-to-bottom" id="chat-jump" hidden aria-label="Jump to latest">
            ↓ jump to latest
          </button>
        </div>
        <form class="chat-composer" id="chat-composer">
          <textarea
            id="chat-input"
            rows="2"
            placeholder="Ask the model anything. Enter sends, Shift+Enter adds a newline."
            disabled
            aria-label="Chat message"
          ></textarea>
          <div class="composer-actions">
            <button type="submit" class="primary" id="chat-send" disabled>send</button>
            <button type="button" class="ghost"   id="chat-stop" hidden>stop</button>
          </div>
        </form>
      </div>

      <aside class="chat-side">
        <div class="card">
          <h3>session</h3>
          <div class="stat"><span class="stat-label">turns</span>   <span class="stat-value" id="stat-turns">0</span></div>
          <div class="stat" title="Wall-clock tps — includes panel DOM work between tokens."><span class="stat-label">wall tps</span><span class="stat-value" id="stat-tps">—</span></div>
          <div class="stat" title="Pure engine-side tps — methodology-matched to bench, excludes caller body time."><span class="stat-label">engine tps</span><span class="stat-value" id="stat-engine-tps">—</span></div>
          <div class="stat"><span class="stat-label">first tok</span><span class="stat-value" id="stat-first">—</span></div>
          <div class="stat"><span class="stat-label">last tokens</span><span class="stat-value" id="stat-tokens">—</span></div>
          <div class="stat"><span class="stat-label">system</span>  <span class="stat-value" id="stat-system">none</span></div>
          <div class="card-actions" style="margin-top: var(--s-3);">
            <button type="button" class="ghost" id="chat-reset" disabled>reset conversation</button>
          </div>
        </div>

        <details class="card" id="system-card">
          <summary>system instruction</summary>
          <div class="card-sub" style="margin-top: var(--s-3);">
            Gemma 4 accepts a <code>developer</code>-role preamble. Leave
            blank for no instruction. Applying commits the text and resets
            the conversation — the preamble is baked into the KV cache on
            first prefill, so mid-conversation changes require a reset.
          </div>
          <textarea
            id="system-input"
            rows="5"
            placeholder="e.g. You are a terse, technically-precise assistant. Cite sources when relevant."
          ></textarea>
          <div class="card-actions">
            <button type="button" class="primary" id="system-apply" disabled>apply</button>
            <button type="button" class="ghost"   id="system-clear" disabled>clear</button>
            <span class="mono-sm muted" id="system-status"></span>
          </div>
        </details>

        <details class="card">
          <summary>generation knobs</summary>
          <div class="card-sub" style="margin-top: var(--s-3);">
            Per-request. Defaults are safe for conversation; drop temperature to 0 for
            deterministic output.
          </div>
          <div class="col">
            <label>
              temperature <span class="muted" id="lbl-temp">${GEN_DEFAULTS.temperature}</span>
              <input type="range" id="gen-temp" min="0" max="1.5" step="0.05" value="${GEN_DEFAULTS.temperature}" />
            </label>
            <label>
              top-p <span class="muted" id="lbl-topp">${GEN_DEFAULTS.topP}</span>
              <input type="range" id="gen-topp" min="0.1" max="1" step="0.05" value="${GEN_DEFAULTS.topP}" />
            </label>
            <label>
              rep penalty <span class="muted" id="lbl-rep">${GEN_DEFAULTS.repPenalty}</span>
              <input type="range" id="gen-rep" min="1" max="1.6" step="0.05" value="${GEN_DEFAULTS.repPenalty}" />
            </label>
            <label>
              max tokens <span class="muted" id="lbl-max">${GEN_DEFAULTS.maxTokens}</span>
              <input type="number" id="gen-max" min="16" max="8192" step="16" value="${GEN_DEFAULTS.maxTokens}" />
            </label>
          </div>
        </details>
      </aside>
    </div>
  `;

  const elMessages = root.querySelector('#chat-messages') as HTMLElement;
  const elForm = root.querySelector('#chat-composer') as HTMLFormElement;
  const elInput = root.querySelector('#chat-input') as HTMLTextAreaElement;
  const elSend = root.querySelector('#chat-send') as HTMLButtonElement;
  const elStop = root.querySelector('#chat-stop') as HTMLButtonElement;
  const elReset = root.querySelector('#chat-reset') as HTMLButtonElement;
  const elJump = root.querySelector('#chat-jump') as HTMLButtonElement;

  // ── Scroll latch ─────────────────────────────────────────────────
  //
  // Standard chat UX: while attached to the bottom, new content scrolls
  // into view; when the user scrolls up, auto-scroll pauses and a
  // "jump to latest" pill appears. Clicking the pill re-attaches.
  //
  // The key design choice: `userDetached` is toggled by EXPLICIT user
  // gestures (wheel-up, touch-drag-up, PageUp/ArrowUp/Home) — NOT by
  // observing scrollTop changes. Gesture events don't fire from our
  // programmatic scrollTop writes, so there's no race between "the rAF
  // flush scrolled us to the bottom" and "the scroll handler inferred
  // the user is at the bottom." Inferring intent from position was the
  // source of the earlier fight-with-the-user behavior.
  //
  // The scroll event is still useful for two cases:
  //   1. Scrollbar drag — the only user-initiated scroll that doesn't
  //      fire wheel/touch/key events. We detect it via scrollTop-delta.
  //   2. Re-attachment — when the user manually scrolls back to the
  //      bottom (within a tight threshold), auto-scroll resumes.
  //
  // `scrollLock` gates the scroll handler during our own programmatic
  // scrolls so we don't misclassify them as user gestures. It's released
  // on the first subsequent scroll event (the one our write produces),
  // with a defensive setTimeout fallback in case the write didn't
  // actually change scrollTop.
  const REATTACH_THRESHOLD = 20;
  let userDetached = false;
  let scrollLock = false;
  let lastScrollTop = elMessages.scrollTop;

  const isAtBottom = (): boolean =>
    elMessages.scrollHeight - elMessages.scrollTop - elMessages.clientHeight < REATTACH_THRESHOLD;

  const setDetached = (value: boolean): void => {
    if (userDetached === value) return;
    userDetached = value;
    updateJumpButton();
  };

  const updateJumpButton = (): void => {
    const hasContent = elMessages.children.length > 0;
    elJump.hidden = !userDetached || !hasContent;
  };

  const snapToBottom = (): void => {
    const target = elMessages.scrollHeight - elMessages.clientHeight;
    if (Math.abs(elMessages.scrollTop - target) < 0.5) {
      lastScrollTop = elMessages.scrollTop;
      return;  // already there; no scroll event will fire
    }
    scrollLock = true;
    const release = () => { scrollLock = false; };
    elMessages.addEventListener('scroll', release, { once: true });
    // Defensive fallback in case the scroll event somehow doesn't fire.
    setTimeout(() => {
      elMessages.removeEventListener('scroll', release);
      scrollLock = false;
    }, 200);
    elMessages.scrollTop = elMessages.scrollHeight;
    lastScrollTop = elMessages.scrollTop;
  };

  // ── User-intent detach signals ──────────────────────────────────
  // All three fire only from real user input; no race with programmatic
  // scrolls.
  elMessages.addEventListener('wheel', (e) => {
    if (e.deltaY < 0) setDetached(true);
  }, { passive: true });

  let touchStartTop = 0;
  elMessages.addEventListener('touchstart', () => {
    touchStartTop = elMessages.scrollTop;
  }, { passive: true });
  elMessages.addEventListener('touchmove', () => {
    if (elMessages.scrollTop < touchStartTop - 2) setDetached(true);
  }, { passive: true });

  elMessages.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowUp' || e.key === 'PageUp' || e.key === 'Home') {
      setDetached(true);
    }
  });

  // ── Scroll event: scrollbar-drag detection + re-attach ─────────
  elMessages.addEventListener('scroll', () => {
    const cur = elMessages.scrollTop;
    const delta = cur - lastScrollTop;
    lastScrollTop = cur;

    if (scrollLock) return;  // our own programmatic scroll

    // Scrollbar drag up (the only user scroll not caught above).
    if (delta < -1) setDetached(true);

    // Re-attach when user manually reaches the bottom.
    if (userDetached && isAtBottom()) setDetached(false);
  }, { passive: true });

  elJump.addEventListener('click', () => {
    setDetached(false);
    snapToBottom();
  });

  const gen = { ...GEN_DEFAULTS };
  const wireSlider = (id: string, labelId: string, key: keyof typeof gen, format = (v: number) => v.toFixed(2)) => {
    const el = root.querySelector(`#${id}`) as HTMLInputElement;
    const lbl = root.querySelector(`#${labelId}`) as HTMLElement;
    el.addEventListener('input', () => {
      const v = Number(el.value);
      gen[key] = v;
      lbl.textContent = format(v);
    });
  };
  wireSlider('gen-temp', 'lbl-temp', 'temperature');
  wireSlider('gen-topp', 'lbl-topp', 'topP');
  wireSlider('gen-rep',  'lbl-rep',  'repPenalty');
  const maxEl = root.querySelector('#gen-max') as HTMLInputElement;
  const maxLbl = root.querySelector('#lbl-max') as HTMLElement;
  maxEl.addEventListener('input', () => {
    const v = Math.max(16, Math.floor(Number(maxEl.value)));
    gen.maxTokens = v;
    maxLbl.textContent = String(v);
  });

  elForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = elInput.value.trim();
    if (!text) return;
    const s = getState();
    if (s.status !== 'ready') return;
    elInput.value = '';
    // Re-latch on send — users expect their own message + the incoming
    // reply to scroll into view.
    setDetached(false);
    const opts: GenerateOptions = { ...gen };
    try {
      await send(text, opts);
    } catch {
      // error rendered via status bar
    }
    elInput.focus();
  });

  elInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      elForm.requestSubmit();
    }
  });

  elStop.addEventListener('click', () => abort());

  elReset.addEventListener('click', async () => {
    if (getState().messages.length === 0) return;
    const ok = await confirmDialog({
      title: 'Reset conversation?',
      message: 'This clears all messages and wipes the KV cache. The engine stays loaded.',
      confirmLabel: 'Reset',
      danger: true,
    });
    if (ok) reset();
  });

  // System-instruction controls.
  const elSystemInput = root.querySelector('#system-input') as HTMLTextAreaElement;
  const elSystemApply = root.querySelector('#system-apply') as HTMLButtonElement;
  const elSystemClear = root.querySelector('#system-clear') as HTMLButtonElement;
  const elSystemStatus = root.querySelector('#system-status') as HTMLElement;

  const refreshSystemControls = () => {
    const s = getState();
    const applied = s.systemPrompt ?? '';
    const draft = elSystemInput.value;
    const isDirty = draft.trim() !== applied.trim();
    const canApply = s.status === 'ready' || s.status === 'uninit';
    elSystemApply.disabled = !canApply || !isDirty;
    elSystemClear.disabled = !canApply || (applied === '' && draft === '');
    if (!isDirty && applied) elSystemStatus.innerHTML = `<span class="good">applied · ${applied.length} chars</span>`;
    else if (isDirty && applied) elSystemStatus.innerHTML = '<span class="warn">modified — apply to commit</span>';
    else if (isDirty && !applied) elSystemStatus.innerHTML = '<span class="warn">draft — apply to commit</span>';
    else elSystemStatus.textContent = 'no instruction set';
  };

  elSystemInput.addEventListener('input', refreshSystemControls);

  elSystemApply.addEventListener('click', async () => {
    const s = getState();
    const text = elSystemInput.value;
    if (s.messages.length > 0) {
      const ok = await confirmDialog({
        title: 'Apply system instruction?',
        message:
          'The current conversation will be cleared so the new preamble takes effect on the next generation. Continue?',
        confirmLabel: 'Apply & reset',
        danger: true,
      });
      if (!ok) return;
    }
    applySystemPrompt(text);
    refreshSystemControls();
  });

  elSystemClear.addEventListener('click', async () => {
    const s = getState();
    if (s.messages.length > 0 || s.systemPrompt) {
      const ok = await confirmDialog({
        title: 'Clear system instruction?',
        message: 'This removes the system preamble and resets the conversation.',
        confirmLabel: 'Clear & reset',
        danger: true,
      });
      if (!ok) return;
    }
    elSystemInput.value = '';
    applySystemPrompt(null);
    refreshSystemControls();
  });

  // ── Per-message DOM tracking ─────────────────────────────────────
  //
  // The engine's `emit()` fires once per streamed token. At 120 tps that's
  // 120 subscriber calls per second, each executed synchronously inside
  // the generator's `for await` body — so any CPU work here directly
  // inflates wall-clock time per token and drags measured tps.
  //
  // The split that keeps streaming cheap:
  //   - syncStructure (per-emit, cheap): iterate messages, create/remove
  //     DOM rows as needed. During a streaming turn no rows are added or
  //     removed, so this is just Map bookkeeping.
  //   - scheduleFlush (per-emit, cheap): set a flag and request one rAF.
  //     Idempotent — only the first call per frame actually schedules.
  //   - flushRender (rAF, expensive work allowed): re-render markdown
  //     on the streaming message at most once per display frame (~60 Hz).
  //     All scroll adjustments happen at the tail, inside the same frame.

  interface RenderEntry {
    row: HTMLElement;
    textEl: HTMLElement;
    metaEl: HTMLElement;
    renderedLen: number;
    renderedStreaming: boolean;
    renderedMetaKey: string;
  }
  const rendered = new Map<number, RenderEntry>();

  let flushPending = false;

  const flushRender = (): void => {
    flushPending = false;
    const s = getState();

    for (const msg of s.messages) {
      const entry = rendered.get(msg.id);
      if (!entry) continue;

      entry.row.classList.toggle('streaming', !!msg.streaming);

      const contentChanged =
        entry.renderedLen !== msg.text.length ||
        entry.renderedStreaming !== !!msg.streaming;
      if (contentChanged) {
        if (msg.role === 'user') {
          entry.textEl.textContent = msg.text;
        } else {
          // The only place `renderMarkdown` is called. Runs at most once
          // per frame regardless of token rate.
          entry.textEl.innerHTML = renderMarkdown(
            msg.text || (msg.streaming ? '' : '_(empty reply)_'),
          );
        }
        entry.renderedLen = msg.text.length;
        entry.renderedStreaming = !!msg.streaming;
      }

      // Meta line — changes on every streamed token ("N tok streaming…"),
      // so cheap-diff it against the rendered key to avoid redundant
      // textContent writes.
      let metaKey = '';
      let metaText = '';
      if (msg.role === 'model' && !msg.streaming && msg.tps != null && msg.tokens) {
        metaKey = `done|${msg.tokens}|${msg.tps.toFixed(3)}|${msg.firstTokenMs ?? 0}`;
        metaText = `${msg.tokens} tok · ${formatTps(msg.tps)} tps · first ${formatMs(msg.firstTokenMs ?? 0)}`;
      } else if (msg.role === 'model' && msg.streaming) {
        metaKey = `stream|${msg.tokens ?? 0}`;
        metaText = msg.tokens ? `${msg.tokens} tok streaming…` : 'generating…';
      }
      if (metaKey !== entry.renderedMetaKey) {
        entry.metaEl.textContent = metaText;
        entry.renderedMetaKey = metaKey;
      }
    }

    if (!userDetached) snapToBottom();
    updateJumpButton();
  };

  const scheduleFlush = (): void => {
    if (flushPending) return;
    flushPending = true;
    requestAnimationFrame(flushRender);
  };

  const syncStructure = (s: EngineState): void => {
    // Remove DOM rows for messages no longer in state (reset flow).
    const currentIds = new Set(s.messages.map((m) => m.id));
    for (const id of Array.from(rendered.keys())) {
      if (!currentIds.has(id)) {
        rendered.get(id)?.row.remove();
        rendered.delete(id);
      }
    }
    // Create DOM rows for new messages. The row starts empty; the rAF
    // flush paints its content on the next frame.
    for (const msg of s.messages) {
      if (rendered.has(msg.id)) continue;
      const row = buildMessageRow(msg);
      elMessages.append(row);
      rendered.set(msg.id, {
        row,
        textEl: row.querySelector<HTMLElement>('[data-role="text"]')!,
        metaEl: row.querySelector<HTMLElement>('[data-role="meta"]')!,
        renderedLen: -1,
        renderedStreaming: false,
        renderedMetaKey: '',
      });
    }
  };

  const renderSide = (s: EngineState) => {
    const lastModel = [...s.messages].reverse().find((m) => m.role === 'model' && !m.streaming);
    (root.querySelector('#stat-turns') as HTMLElement).textContent =
      String(s.messages.filter((m) => m.role === 'user').length);
    (root.querySelector('#stat-tps') as HTMLElement).textContent =
      lastModel?.tps != null ? formatTps(lastModel.tps) : '—';
    (root.querySelector('#stat-engine-tps') as HTMLElement).textContent =
      lastModel?.engineTps != null ? formatTps(lastModel.engineTps) : '—';
    (root.querySelector('#stat-first') as HTMLElement).textContent =
      lastModel?.firstTokenMs != null ? formatMs(lastModel.firstTokenMs) : '—';
    (root.querySelector('#stat-tokens') as HTMLElement).textContent =
      lastModel?.tokens != null ? String(lastModel.tokens) : '—';
    (root.querySelector('#stat-system') as HTMLElement).innerHTML =
      s.systemPrompt ? `<span class="accent">${s.systemPrompt.length} chars</span>` : 'none';
  };

  const renderComposer = (s: EngineState) => {
    const canChat = s.status === 'ready';
    elInput.disabled = !canChat;
    elSend.disabled = !canChat;
    elSend.textContent = s.status === 'generating' ? 'generating…' : 'send';
    elStop.hidden = s.status !== 'generating';
    elReset.disabled = s.status !== 'ready' || s.messages.length === 0;
    if (canChat && !elInput.value && document.activeElement === document.body) {
      // Leave focus up to the user; we don't steal it.
    }
  };

  // Track last-seen inputs for the cold-path updates. The engine emits
  // on every streamed token — at 120 tps that's 120 subscriber calls per
  // second. Only the streaming-message-text render is hot; system-
  // controls, stats, and composer button state only need updating when
  // their inputs change. Gating on these diffs keeps the per-token work
  // at roughly O(markdown re-parse) instead of O(markdown + DOM thrash).
  const panelState = {
    lastSeenApplied: '',
    lastStatus: null as EngineState['status'] | null,
    lastMsgCount: -1,
    lastLastModelSig: '',  // "tokens|tps|firstMs" of the last completed model msg
    lastStreaming: false,
  };

  const lastModelSig = (s: EngineState): string => {
    for (let i = s.messages.length - 1; i >= 0; i--) {
      const m = s.messages[i];
      if (m.role === 'model' && !m.streaming) {
        return `${m.tokens ?? 0}|${m.tps ?? 0}|${m.engineTps ?? 0}|${m.firstTokenMs ?? 0}`;
      }
    }
    return '';
  };

  subscribe((s) => {
    // Hot path — runs on every streamed-token emit. Does only cheap
    // work: reconcile DOM structure (no-op during a streaming turn
    // since no messages are added/removed) and schedule a flush.
    syncStructure(s);
    scheduleFlush();

    // Cold path — only when inputs actually changed. During a 120 tps
    // stream, `s.systemPrompt` / `s.status` / completed-model stats
    // don't change, so this whole block is skipped token-to-token.
    const applied = s.systemPrompt ?? '';
    const currentStreaming = s.messages.some((m) => m.streaming);
    const sig = lastModelSig(s);
    const statusChanged = panelState.lastStatus !== s.status;
    const systemChanged = panelState.lastSeenApplied !== applied;
    const turnsChanged = panelState.lastMsgCount !== s.messages.length;
    const statsChanged = panelState.lastLastModelSig !== sig;
    const streamingChanged = panelState.lastStreaming !== currentStreaming;

    if (statusChanged || systemChanged || turnsChanged || statsChanged || streamingChanged) {
      renderSide(s);
      renderComposer(s);
      if (elSystemInput.value.trim() === panelState.lastSeenApplied.trim()) {
        elSystemInput.value = applied;
      }
      refreshSystemControls();

      panelState.lastStatus = s.status;
      panelState.lastSeenApplied = applied;
      panelState.lastMsgCount = s.messages.length;
      panelState.lastLastModelSig = sig;
      panelState.lastStreaming = currentStreaming;
    }
  });

  refreshSystemControls();
}

function buildMessageRow(msg: ChatMessage): HTMLElement {
  const row = document.createElement('div');
  row.className = `msg ${msg.role}`;
  row.dataset.msgId = String(msg.id);
  row.innerHTML = `
    <div class="msg-who">
      <span>${msg.role === 'user' ? 'you' : 'gemma'}</span>
      <span class="meta" data-role="meta"></span>
    </div>
    <div class="msg-text" data-role="text"></div>
  `;
  return row;
}

