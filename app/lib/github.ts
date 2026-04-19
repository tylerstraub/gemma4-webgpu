// Prefilled-issue URL builder for the device-report contribution flow.
//
// GitHub's URL limit is ~8KB in practice. A full device report with a 34-
// layer sweep can hit 6–10KB, so the "open issue with this report" link
// is a best-effort convenience. When the encoded body would exceed the
// threshold, callers should fall back to clipboard copy + a blank-issue
// link.

/** Slug of the published repo. Drives all `github.com/...` links in the app. */
export const REPO_SLUG = 'tylerstraub/gemma4-webgpu';

/** ~7000 encoded chars is a conservative ceiling under GitHub's ~8KB URL limit. */
export const MAX_PREFILL_BODY = 7000;

export interface IssueUrlResult {
  /** URL to GitHub's new-issue form; null if the body was too large to prefill. */
  url: string | null;
  /** True if the body exceeded the prefill ceiling. */
  truncated: boolean;
  /** Length of the encoded body. */
  encodedLength: number;
}

/**
 * Build a prefilled new-issue URL for the device-report template. If the
 * encoded body exceeds `MAX_PREFILL_BODY`, returns `{ url: null, truncated: true }`
 * — the caller should surface the copy-markdown path instead.
 */
export function buildDeviceReportIssueUrl(
  markdown: string,
  slug: string = REPO_SLUG,
): IssueUrlResult {
  const encodedBody = encodeURIComponent(markdown);
  if (encodedBody.length > MAX_PREFILL_BODY) {
    return { url: null, truncated: true, encodedLength: encodedBody.length };
  }
  const url =
    `https://github.com/${slug}/issues/new?template=device-report.md&body=${encodedBody}`;
  return { url, truncated: false, encodedLength: encodedBody.length };
}

/** Fallback: a plain new-issue link (no body prefill), same repo/template. */
export function blankIssueUrl(slug: string = REPO_SLUG): string {
  return `https://github.com/${slug}/issues/new?template=device-report.md`;
}
