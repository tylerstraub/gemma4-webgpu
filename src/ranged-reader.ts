/**
 * Streaming HTTP Range reader with stall detection and
 * resume-from-offset retry.
 *
 * Replaces the ad-hoc `fetchRange` closure in the engine that
 * accumulated the entire response via `Response.arrayBuffer()`
 * and had no way to observe, let alone recover from, a silent
 * connection stall — the "stuck at 8.2%" gremlin, which occurs
 * when the underlying TCP connection goes zombie (bytes neither
 * arriving nor erroring) mid-fetch on a large monolithic Range
 * request (notably the ~1.6 GB `per_layer_token_embd` tensor).
 *
 * Semantics:
 *  - Uses `fetch()` + `body.getReader()` so bytes are observable
 *    as they arrive.
 *  - If no bytes arrive within `noProgressTimeoutMs`, the current
 *    request is aborted and a new Range request is issued
 *    starting from the next-unread byte offset.
 *  - Retries use exponential backoff (500ms, 1s, 2s, 4s...).
 *  - External `AbortSignal` is honored: an external abort
 *    propagates to the active fetch and re-throws without retry.
 *  - A premature EOF (server closes early) is treated like a
 *    stall — retry with resumption.
 *
 * Caller ergonomics:
 *  - `readAll(start, size)` mimics the old `fetchRange` API for
 *    drop-in replacement. Returns a single `Uint8Array`.
 *  - `stream(start, size)` yields chunks as they arrive, for
 *    consumers (like the streaming PLE decoder) that process
 *    data incrementally.
 */

export interface RangedReaderOptions {
  /**
   * Maximum ms the active request may go without receiving any
   * bytes before it is aborted and retried from the last-received
   * byte offset. Default: 30_000 (30s). Pick longer for flaky
   * connections; shorter if you'd rather fail fast.
   */
  noProgressTimeoutMs?: number;
  /** Max total attempts across one read. Default: 4 (≤ 7.5s of backoff). */
  maxAttempts?: number;
  /** External cancellation. If this aborts, the active fetch aborts and the read rejects with no retry. */
  signal?: AbortSignal;
  /** Fine-grained event stream — useful for surfacing "retrying…" progress to users. */
  onEvent?: (ev: RangedReaderEvent) => void;
}

/**
 * Events emitted during a read.
 *  - `chunk`: a new bytes-arrived checkpoint. Fires on every
 *    `reader.read()` that returns data.
 *  - `retry`: the current attempt failed (stall / error /
 *    premature EOF) and we're about to back off and try again
 *    from byte offset `bytesReadBeforeRetry`.
 */
export type RangedReaderEvent =
  | { type: 'chunk'; bytesRead: number; totalSize: number }
  | {
      type: 'retry';
      attempt: number;
      reason: string;
      bytesReadBeforeRetry: number;
    };

export class RangedReader {
  constructor(
    private readonly url: string,
    private readonly options: RangedReaderOptions = {},
  ) {}

  /** Read a byte range fully into memory. Drop-in for a plain `fetchRange(start, size)` call. */
  async readAll(start: number, size: number): Promise<Uint8Array> {
    const buf = new Uint8Array(size);
    let offset = 0;
    for await (const chunk of this.stream(start, size)) {
      buf.set(chunk, offset);
      offset += chunk.byteLength;
    }
    return buf;
  }

  /** Yield `Uint8Array` chunks in arrival order. Handles stall detection and resumption transparently. */
  async *stream(start: number, size: number): AsyncGenerator<Uint8Array, void, void> {
    const noProgressTimeoutMs = this.options.noProgressTimeoutMs ?? 30_000;
    const maxAttempts = this.options.maxAttempts ?? 4;
    const externalSignal = this.options.signal;
    const onEvent = this.options.onEvent;

    // `bytesRead` is the cross-retry cursor. Each retry issues a
    // Range request starting at `start + bytesRead`, so a stall
    // 80% of the way through a 1.6 GB fetch doesn't restart from 0.
    let bytesRead = 0;

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      const inner = new AbortController();
      // Cascade external cancellation into the inner controller
      // (plain addEventListener with `once: true` — no AbortSignal.any for broader browser support).
      const onExternalAbort = (): void => {
        inner.abort(externalSignal?.reason ?? new DOMException('Aborted', 'AbortError'));
      };
      if (externalSignal) {
        if (externalSignal.aborted) {
          onExternalAbort();
        } else {
          externalSignal.addEventListener('abort', onExternalAbort, { once: true });
        }
      }

      let stallTimer: ReturnType<typeof setTimeout> | null = null;
      const armStallTimer = (): void => {
        if (stallTimer) clearTimeout(stallTimer);
        stallTimer = setTimeout(() => {
          inner.abort(new Error(`no progress for ${noProgressTimeoutMs}ms`));
        }, noProgressTimeoutMs);
      };
      const disarmStallTimer = (): void => {
        if (stallTimer) {
          clearTimeout(stallTimer);
          stallTimer = null;
        }
      };

      try {
        const resumeStart = start + bytesRead;
        const resumeEnd = start + size - 1;
        const resp = await fetch(this.url, {
          headers: { Range: `bytes=${resumeStart}-${resumeEnd}` },
          signal: inner.signal,
        });

        // We insist on 206 so we know the server honored the Range. A 200 would
        // mean the server sent the full file; we'd need to seek into it, and that
        // case didn't appear in our CDN probes. Fail loud if it ever does.
        if (resp.status !== 206) {
          throw new Error(`expected HTTP 206 for Range ${resumeStart}-${resumeEnd}, got ${resp.status}`);
        }

        const reader = resp.body!.getReader();
        while (true) {
          // Timer covers only the network read, not consumer work
          // between yields — a slow consumer doesn't spuriously trip
          // the stall detector.
          armStallTimer();
          const { done, value } = await reader.read();
          disarmStallTimer();
          if (done) break;
          bytesRead += value.byteLength;
          onEvent?.({ type: 'chunk', bytesRead, totalSize: size });
          yield value;
        }

        // Server closed the stream before delivering all expected bytes —
        // treat as a stall and resume from the current offset on retry.
        if (bytesRead < size) {
          throw new Error(`premature EOF: ${bytesRead}/${size} bytes`);
        }

        externalSignal?.removeEventListener('abort', onExternalAbort);
        return;
      } catch (err) {
        disarmStallTimer();
        externalSignal?.removeEventListener('abort', onExternalAbort);

        // External cancellation — rethrow without retry.
        if (externalSignal?.aborted) {
          throw err;
        }

        // Out of attempts — surface the last error to the caller.
        if (attempt >= maxAttempts - 1) {
          throw err;
        }

        onEvent?.({
          type: 'retry',
          attempt: attempt + 1,
          reason: err instanceof Error ? err.message : String(err),
          bytesReadBeforeRetry: bytesRead,
        });

        const backoffMs = 500 * Math.pow(2, attempt);
        await new Promise((r) => setTimeout(r, backoffMs));
      }
    }
  }
}
