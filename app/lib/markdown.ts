// Streaming-safe markdown renderer.
//
// The model streams token-by-token — we re-parse the growing buffer on every
// append. At 30–120 tps the O(n) cost is trivial; trading perf for
// correctness (no half-rendered fences, no dangling inline code) is the
// right call for a reference-implementation demo.
//
// Sanitization is mandatory: the model can and will emit <script>-shaped
// text. `DOMPurify` strips HTML-hazard nodes before the string ever reaches
// innerHTML.

import { marked } from 'marked';
import DOMPurify from 'dompurify';

marked.setOptions({
  gfm: true,
  breaks: false,
});

/**
 * Render a markdown string as sanitized HTML, safe to assign to innerHTML.
 * Synchronous; `marked.parse` with default options returns a string.
 */
export function renderMarkdown(src: string): string {
  const rawHtml = marked.parse(src, { async: false }) as string;
  return DOMPurify.sanitize(rawHtml, {
    ADD_ATTR: ['target', 'rel'],
  });
}

