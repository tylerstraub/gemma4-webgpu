/**
 * Diagnostic report formatter — produces a markdown payload ready to
 * paste into the `device-report.md` GitHub issue template. All inputs
 * are optional; the formatter emits whatever sections the caller has
 * collected.
 *
 * The payload is structured for easy downstream parsing. Headings and
 * code-fence layouts match the issue template's schema so device-report
 * issues land as consistent, comparable data.
 */

import type { GemmaConfig } from '../types.js';
import type { CrossLabDiffResult, ProfileRunResult, CpuProfileRunResult } from './index.js';

export interface BenchMeasurement {
  label: string;
  config: {
    prompt: string;
    promptCharCount: number;
    maxTokens: number;
    runs: number;
    temperature: number;
    repPenalty: number;
  };
  median: number;
  mean: number;
  min: number;
  max: number;
  tpsRuns: number[];
}

export interface DeviceInfo {
  vendor: string;
  architecture: string;
  description?: string;
  features: string[];
  limits?: Partial<Record<string, number>>;
}

export interface BrowserInfo {
  userAgent: string;
  platform?: string;
  hardwareConcurrency?: number;
  deviceMemoryGB?: number;
}

export interface DiagnosticReportInput {
  /** Repo identity — used for the opening line and any header framing. */
  repoName?: string;
  /** Git commit hash the engine was built from, if known. */
  commit?: string;
  /** When this report was generated (ISO 8601). Defaults to `new Date().toISOString()`. */
  generatedAt?: string;

  device?: DeviceInfo;
  browser?: BrowserInfo;
  engineConfig?: {
    num_layers: number;
    hidden_size: number;
    vocab_size: number;
    context_length: number;
    per_layer_input_dim: number;
  };
  /** Active tuning profile summary. Captured at init. */
  tuning?: {
    id: string;
    description: string;
    verified: boolean;
    reason: string;
    rowsPerWorkgroup_ffnGateUp: number;
    decodeDepth: 1 | 2;
    greedyFastPath: boolean;
  };
  /** Time from `lab.init()` start to readiness, in seconds. */
  initTimeSec?: number;

  /** Bench measurements. Short + long are the canonical two, but the
   *  formatter accepts any number of named measurements. */
  bench?: BenchMeasurement[];
  /** Cross-lab diff sweep results at multiple capture points. */
  crossLabSweep?: { point: string; cos: number; maxAbs: number }[];
  /** Optional full crossLabDiff result for a specific point (e.g. logits). */
  crossLabDetail?: Pick<CrossLabDiffResult, 'capturePoint' | 'cosineSimilarity' | 'maxAbsDiff' | 'concentration' | 'refNorm' | 'ourNorm'>;

  profile?: ProfileRunResult;
  cpuProfile?: CpuProfileRunResult;

  /** Free-form notes the contributor wants to include. */
  notes?: string;
}

const DIVIDER = '\n---\n\n';

function fmtTps(v: number): string {
  return v.toFixed(2);
}

function fmtMs(v: number): string {
  return v.toFixed(2);
}

function fmtPct(v: number): string {
  return (v * 100).toFixed(1) + '%';
}

function section(heading: string, body: string): string {
  return `## ${heading}\n\n${body.trim()}\n`;
}

function kvTable(entries: [string, string][]): string {
  const width = Math.max(...entries.map(([k]) => k.length), 1);
  return entries.map(([k, v]) => `- **${k.padEnd(width)}**  ${v}`).join('\n');
}

function benchBlock(m: BenchMeasurement): string {
  const lines = [
    `**${m.label}** — ${fmtTps(m.median)} tps median, ${fmtTps(m.mean)} tps mean (min ${fmtTps(m.min)}, max ${fmtTps(m.max)}).`,
    '',
    '```',
    `prompt:          ${JSON.stringify(m.config.prompt)}`,
    `promptCharCount: ${m.config.promptCharCount}`,
    `maxTokens:       ${m.config.maxTokens}`,
    `runs:            ${m.config.runs}`,
    `temperature:     ${m.config.temperature}`,
    `repPenalty:      ${m.config.repPenalty}`,
    `tps runs:        ${m.tpsRuns.map((v) => fmtTps(v)).join(', ')}`,
    '```',
  ];
  return lines.join('\n');
}

function sweepTable(sweep: { point: string; cos: number; maxAbs: number }[]): string {
  const lines = [
    '| Capture point | cosine | maxAbsDiff |',
    '|---|---:|---:|',
  ];
  for (const row of sweep) {
    lines.push(`| \`${row.point}\` | ${row.cos.toFixed(6)} | ${row.maxAbs.toFixed(4)} |`);
  }
  return lines.join('\n');
}

function profileTable(rp: ProfileRunResult): string {
  if (!rp.ok) {
    return [
      `**profile unavailable**`,
      '',
      `> ${rp.error}`,
      '',
      '```',
      `timestampQuerySupported:            ${rp.capability.timestampQuerySupported}`,
      `timestampQueryOnAdapterButUnreq:    ${rp.capability.timestampQueryOnAdapterButUnrequested}`,
      `querySlots:                         ${rp.capability.querySlots}`,
      '```',
    ].join('\n');
  }
  const r = rp.report;
  const topN = Math.min(12, r.perLabel.length);
  const sorted = [...r.perLabel].sort((a, b) => b.total_ns - a.total_ns).slice(0, topN);
  const lines = [
    `**forward p50**: ${(r.forwardMedian_ns / 1e6).toFixed(3)} ms · **samples**: ${r.forwardSamples_n}${r.overflow ? ' · **OVERFLOW**' : ''}`,
    `**tokens measured**: ${rp.tokensMeasured} · **wall**: ${fmtMs(rp.wallMs)} ms`,
    '',
    `Top ${topN} kernels by total time:`,
    '',
    '| Kernel | p50 (µs) | p95 (µs) | samples | total (ms) |',
    '|---|---:|---:|---:|---:|',
  ];
  for (const k of sorted) {
    lines.push(
      `| \`${k.label}\` | ${(k.p50_ns / 1000).toFixed(1)} | ${(k.p95_ns / 1000).toFixed(1)} | ${k.samples_n} | ${(k.total_ns / 1e6).toFixed(2)} |`,
    );
  }
  return lines.join('\n');
}

function cpuProfileTable(rp: CpuProfileRunResult): string {
  const r = rp.report;
  const topN = Math.min(10, r.perPhase.length);
  const sorted = [...r.perPhase].sort((a, b) => b.total_ms - a.total_ms).slice(0, topN);
  const lines = [
    `**forward p50**: ${fmtMs(r.forwardMedian_ms)} ms · **samples**: ${r.forwardSamples_n}`,
    `**writeBuffer/forward p50**: ${r.writeBufferCountMedian} · **passEncode/forward p50**: ${r.passEncodeCountMedian}`,
    `**tokens measured**: ${rp.tokensMeasured} · **wall**: ${fmtMs(rp.wallMs)} ms`,
    '',
    `Top ${topN} CPU phases by total time:`,
    '',
    '| Phase | p50 (ms) | p95 (ms) | samples | total (ms) |',
    '|---|---:|---:|---:|---:|',
  ];
  for (const p of sorted) {
    lines.push(
      `| \`${p.phase}\` | ${fmtMs(p.p50_ms)} | ${fmtMs(p.p95_ms)} | ${p.samples_n} | ${fmtMs(p.total_ms)} |`,
    );
  }
  return lines.join('\n');
}

/**
 * Format a diagnostic report as GitHub-issue-ready markdown. All sections
 * are optional — pass only what's been collected.
 */
export function formatDiagnosticReport(input: DiagnosticReportInput): string {
  const repo = input.repoName ?? 'gemma4-webgpu';
  const ts = input.generatedAt ?? new Date().toISOString();
  const parts: string[] = [];

  parts.push(`# ${repo} — device report`);
  parts.push('');
  parts.push(
    `_Generated ${ts}${input.commit ? ` · commit \`${input.commit}\`` : ''}._`,
  );

  // ── Device + browser
  if (input.device || input.browser) {
    const d = input.device;
    const b = input.browser;
    const rows: [string, string][] = [];
    if (d) {
      rows.push(['GPU vendor', d.vendor || '(unknown)']);
      rows.push(['GPU architecture', d.architecture || '(unknown)']);
      if (d.description) rows.push(['GPU description', d.description]);
      rows.push(['WebGPU features', d.features.length > 0 ? d.features.join(', ') : '(none)']);
    }
    if (b) {
      rows.push(['User agent', b.userAgent]);
      if (b.platform) rows.push(['Platform', b.platform]);
      if (b.hardwareConcurrency !== undefined) {
        rows.push(['Hardware concurrency', String(b.hardwareConcurrency)]);
      }
      if (b.deviceMemoryGB !== undefined) {
        rows.push(['Device memory (GB)', String(b.deviceMemoryGB)]);
      }
    }
    parts.push(DIVIDER + section('Device', kvTable(rows)));
  }

  // ── Engine config
  if (input.engineConfig) {
    const c = input.engineConfig;
    const rows: [string, string][] = [
      ['num_layers', String(c.num_layers)],
      ['hidden_size', String(c.hidden_size)],
      ['vocab_size', String(c.vocab_size)],
      ['context_length', String(c.context_length)],
      ['per_layer_input_dim', String(c.per_layer_input_dim)],
    ];
    if (input.initTimeSec !== undefined) {
      rows.push(['init time (s)', input.initTimeSec.toFixed(1)]);
    }
    parts.push(DIVIDER + section('Engine config', kvTable(rows)));
  }

  // ── Tuning profile
  if (input.tuning) {
    const t = input.tuning;
    const rows: [string, string][] = [
      ['profile id', `\`${t.id}\`${t.verified ? ' (verified)' : ' (unverified)'}`],
      ['description', t.description],
      ['selected because', t.reason],
      ['ffn.linearGateUp R', String(t.rowsPerWorkgroup_ffnGateUp)],
      ['decode depth', String(t.decodeDepth)],
      ['greedy fast path', String(t.greedyFastPath)],
    ];
    parts.push(DIVIDER + section('Tuning profile', kvTable(rows)));
  }

  // ── Bench
  if (input.bench && input.bench.length > 0) {
    const body = input.bench.map(benchBlock).join('\n\n');
    parts.push(DIVIDER + section('Benchmark', body));
  }

  // ── Correctness sweep
  if (input.crossLabSweep && input.crossLabSweep.length > 0) {
    const cosineValues = input.crossLabSweep.map((r) => r.cos).filter((c) => Number.isFinite(c));
    const minCos = Math.min(...cosineValues);
    const headline = `Min cosine similarity across ${input.crossLabSweep.length} capture points: **${minCos.toFixed(6)}**`;
    let body = `${headline}\n\n${sweepTable(input.crossLabSweep)}`;
    if (input.crossLabDetail) {
      const d = input.crossLabDetail;
      body += `\n\n**Detail (${d.capturePoint})**\n\n`;
      body += `- cosine: ${d.cosineSimilarity.toFixed(6)}\n`;
      body += `- maxAbsDiff: ${d.maxAbsDiff.toFixed(4)}\n`;
      body += `- ourNorm / refNorm: ${d.ourNorm.toFixed(2)} / ${d.refNorm.toFixed(2)}\n`;
      body += `- concentration: top1%=${fmtPct(d.concentration.top1PctFraction)}, top10%=${fmtPct(d.concentration.top10PctFraction)}, top50%=${fmtPct(d.concentration.top50PctFraction)}, participation=${d.concentration.participationRatio.toFixed(3)}\n`;
    }
    parts.push(DIVIDER + section('Correctness (crossLabDiff sweep)', body));
  }

  // ── Profile
  if (input.profile) {
    parts.push(DIVIDER + section('GPU profile (timestamp-query)', profileTable(input.profile)));
  }
  if (input.cpuProfile) {
    parts.push(DIVIDER + section('CPU profile (phase breakdown)', cpuProfileTable(input.cpuProfile)));
  }

  // ── Free-form notes
  if (input.notes && input.notes.trim().length > 0) {
    parts.push(DIVIDER + section('Notes', input.notes));
  }

  return parts.join('\n');
}
