// Tuning panel — profile picker + curated override controls.
//
// Shows the active device profile (auto-detected or user-chosen), the
// reason it was selected, and per-knob override controls for the knobs
// that actually move perf. Applying a change triggers a full engine
// dispose + re-init behind a confirmation dialog — there is no in-place
// re-tune path.

import {
  subscribe,
  getState,
  reinitWithTuning,
  type EngineState,
} from '../lib/engine.js';
import { PROFILES } from '../../src/index.js';
import type { TuningProfile, TuningProfileOverrides, FeatureIntent } from '../../src/index.js';
import { confirmDialog } from '../lib/dialog.js';

interface DraftOverrides {
  baseProfileId: string;
  workgroupSize: number;
  defaultRowsPerWorkgroup: number;
  ffnGateUpRows: number;
  decodeDepth: 1 | 2;
  greedyFastPath: boolean;
  subgroups: FeatureIntent;
  subgroupMatrix: FeatureIntent;
}

function draftFromProfile(p: TuningProfile): DraftOverrides {
  return {
    baseProfileId: p.id,
    workgroupSize: p.matmul.workgroupSize,
    defaultRowsPerWorkgroup: p.matmul.defaultRowsPerWorkgroup,
    ffnGateUpRows: p.matmul.rowsPerWorkgroupByKernel?.['ffn.linearGateUp']
      ?? p.matmul.defaultRowsPerWorkgroup,
    decodeDepth: p.pipeline.decodeDepth,
    greedyFastPath: p.pipeline.greedyFastPath,
    subgroups: p.features.subgroups,
    subgroupMatrix: p.features.subgroupMatrix,
  };
}

function draftToOverrides(d: DraftOverrides): TuningProfileOverrides {
  return {
    id: `${d.baseProfileId}+custom`,
    matmul: {
      workgroupSize: d.workgroupSize,
      defaultRowsPerWorkgroup: d.defaultRowsPerWorkgroup,
      rowsPerWorkgroupByKernel: { 'ffn.linearGateUp': d.ffnGateUpRows },
    },
    pipeline: {
      decodeDepth: d.decodeDepth,
      greedyFastPath: d.greedyFastPath,
    },
    features: {
      subgroups: d.subgroups,
      subgroupMatrix: d.subgroupMatrix,
    },
  };
}

export function mountTuningPanel(root: HTMLElement): void {
  root.innerHTML = `
    <div class="card">
      <h3>active profile</h3>
      <div id="active-detail" class="empty">engine not loaded</div>
    </div>

    <div class="card">
      <h3>available profiles</h3>
      <p class="card-sub">
        Three shipped profiles in v1. <span class="good">verified</span>
        profiles have been calibrated on matching hardware;
        <span class="warn">educated-guess</span> profiles await calibration
        and may be ~20–30% slower than their ceiling. Contribute via a
        device report.
      </p>
      <div id="profile-list"></div>
    </div>

    <details class="card">
      <summary>override &amp; re-initialize</summary>
      <p class="card-sub" style="margin-top: var(--s-3);">
        Applying overrides disposes the engine, re-creates it with the new
        tuning, and wipes the KV cache + conversation transcript. Knobs are
        documented in <code>src/tuning/profile.ts</code>; only the ones with
        measured impact are surfaced here.
      </p>
      <div id="overrides-form"></div>
    </details>
  `;

  const elActive = root.querySelector('#active-detail') as HTMLElement;
  const elProfiles = root.querySelector('#profile-list') as HTMLElement;
  const elOverrides = root.querySelector('#overrides-form') as HTMLElement;

  const renderProfiles = (activeId: string | null) => {
    elProfiles.innerHTML = Object.values(PROFILES).map((p) => `
      <div class="card" style="background: var(--bg); margin-bottom: var(--s-3);">
        <div class="row" style="justify-content: space-between; align-items: baseline;">
          <div class="mono-sm">
            <b>${escapeHtml(p.id)}</b>
            ${p.verified ? '<span class="good" style="margin-left: var(--s-2);">verified</span>' : '<span class="warn" style="margin-left: var(--s-2);">educated-guess</span>'}
            ${p.id === activeId ? '<span class="accent" style="margin-left: var(--s-2);">(active)</span>' : ''}
          </div>
          <button type="button" class="ghost" data-profile-id="${escapeHtml(p.id)}" ${p.id === activeId ? 'disabled' : ''}>
            ${p.id === activeId ? 'active' : 'switch to this'}
          </button>
        </div>
        <div class="mono-sm muted" style="margin-top: var(--s-2);">${escapeHtml(p.description)}</div>
        ${p.notes ? `<div class="mono-sm muted" style="margin-top: var(--s-2);">${escapeHtml(p.notes)}</div>` : ''}
        <table class="data" style="margin-top: var(--s-3);">
          <tbody>
            <tr><th>matmul workgroup</th><td class="num">${p.matmul.workgroupSize}</td></tr>
            <tr><th>default rowsPerWorkgroup</th><td class="num">${p.matmul.defaultRowsPerWorkgroup}</td></tr>
            <tr><th>ffn.linearGateUp rows</th><td class="num">${p.matmul.rowsPerWorkgroupByKernel?.['ffn.linearGateUp'] ?? p.matmul.defaultRowsPerWorkgroup}</td></tr>
            <tr><th>decodeDepth</th><td class="num">${p.pipeline.decodeDepth}</td></tr>
            <tr><th>greedyFastPath</th><td class="num">${p.pipeline.greedyFastPath}</td></tr>
            <tr><th>subgroups intent</th><td class="num">${p.features.subgroups}</td></tr>
            <tr><th>subgroup-matrix intent</th><td class="num">${p.features.subgroupMatrix}</td></tr>
          </tbody>
        </table>
      </div>
    `).join('');

    elProfiles.querySelectorAll<HTMLButtonElement>('[data-profile-id]').forEach((btn) => {
      btn.addEventListener('click', async () => {
        const id = btn.dataset.profileId!;
        await applyTuning(id);
      });
    });
  };

  const renderActive = (s: EngineState) => {
    if (!s.tuning) {
      elActive.classList.add('empty');
      elActive.textContent = 'engine not loaded';
      return;
    }
    elActive.classList.remove('empty');
    const p = s.tuning;
    elActive.innerHTML = `
      <div class="mono-sm">
        <b>${escapeHtml(p.id)}</b>
        ${p.verified ? '<span class="good" style="margin-left: var(--s-2);">verified</span>' : '<span class="warn" style="margin-left: var(--s-2);">educated-guess</span>'}
      </div>
      <div class="mono-sm muted" style="margin-top: var(--s-2);">${escapeHtml(p.description)}</div>
      <div class="mono-sm muted" style="margin-top: var(--s-2);">
        <b>selected because:</b> ${escapeHtml(s.tuningReason ?? 'unknown')}
      </div>
    `;
  };

  const renderOverrides = (s: EngineState) => {
    const base = s.tuning ?? PROFILES['generic'];
    const draft = draftFromProfile(base);

    elOverrides.innerHTML = `
      <div class="field-row">
        <label>
          base profile
          <select id="ov-base">
            ${Object.keys(PROFILES).map((id) => `<option value="${id}" ${id === draft.baseProfileId ? 'selected' : ''}>${escapeHtml(id)}</option>`).join('')}
          </select>
        </label>
        <label>
          matmul.workgroupSize
          <select id="ov-wg">
            ${[64, 128, 256].map((n) => `<option value="${n}" ${n === draft.workgroupSize ? 'selected' : ''}>${n}</option>`).join('')}
          </select>
        </label>
      </div>
      <div class="field-row">
        <label>
          matmul.defaultRowsPerWorkgroup
          <select id="ov-default-rows">
            ${[1, 2, 4].map((n) => `<option value="${n}" ${n === draft.defaultRowsPerWorkgroup ? 'selected' : ''}>${n}</option>`).join('')}
          </select>
        </label>
        <label>
          ffn.linearGateUp rows
          <select id="ov-gateup-rows">
            ${[1, 2, 4].map((n) => `<option value="${n}" ${n === draft.ffnGateUpRows ? 'selected' : ''}>${n}</option>`).join('')}
          </select>
        </label>
      </div>
      <div class="field-row">
        <label>
          pipeline.decodeDepth
          <select id="ov-depth">
            <option value="1" ${draft.decodeDepth === 1 ? 'selected' : ''}>1 (serial)</option>
            <option value="2" ${draft.decodeDepth === 2 ? 'selected' : ''}>2 (pipelined)</option>
          </select>
        </label>
        <label>
          pipeline.greedyFastPath
          <select id="ov-fastpath">
            <option value="true"  ${draft.greedyFastPath ? 'selected' : ''}>true</option>
            <option value="false" ${!draft.greedyFastPath ? 'selected' : ''}>false</option>
          </select>
        </label>
      </div>
      <div class="field-row">
        <label>
          features.subgroups
          <select id="ov-sg">
            ${(['use', 'avoid', 'auto'] as FeatureIntent[]).map((v) => `<option value="${v}" ${v === draft.subgroups ? 'selected' : ''}>${v}</option>`).join('')}
          </select>
        </label>
        <label>
          features.subgroupMatrix
          <select id="ov-sgm">
            ${(['use', 'avoid', 'auto'] as FeatureIntent[]).map((v) => `<option value="${v}" ${v === draft.subgroupMatrix ? 'selected' : ''}>${v}</option>`).join('')}
          </select>
        </label>
      </div>
      <div class="card-actions">
        <button type="button" id="ov-apply" class="primary" ${s.status !== 'ready' ? 'disabled' : ''}>apply &amp; re-init</button>
        <span class="mono-sm muted">Engine dispose + re-init. Conversation cleared.</span>
      </div>
    `;

    elOverrides.querySelector('#ov-apply')!.addEventListener('click', async () => {
      const read = <T>(id: string, parse: (v: string) => T): T =>
        parse((elOverrides.querySelector(`#${id}`) as HTMLSelectElement).value);
      const d: DraftOverrides = {
        baseProfileId: read('ov-base', String),
        workgroupSize: read('ov-wg', Number),
        defaultRowsPerWorkgroup: read('ov-default-rows', Number),
        ffnGateUpRows: read('ov-gateup-rows', Number),
        decodeDepth: read('ov-depth', (v) => (Number(v) as 1 | 2)),
        greedyFastPath: read('ov-fastpath', (v) => v === 'true'),
        subgroups: read('ov-sg', (v) => v as FeatureIntent),
        subgroupMatrix: read('ov-sgm', (v) => v as FeatureIntent),
      };
      const ok = await confirmDialog({
        title: 'Apply tuning override?',
        message:
          `This will dispose the engine, reload it with the new tuning (${d.baseProfileId}+custom), and wipe your conversation. First-load was already paid; re-init is faster but still a few seconds.`,
        confirmLabel: 'Apply',
        danger: true,
      });
      if (!ok) return;
      try {
        const override = draftToOverrides(d);
        await reinitWithTuning(override);
      } catch {
        // status bar will show the error
      }
    });
  };

  const applyTuning = async (profileId: string) => {
    const ok = await confirmDialog({
      title: `Switch to ${profileId}?`,
      message: 'The engine will be disposed and reloaded. Your conversation will be cleared.',
      confirmLabel: 'Switch',
      danger: true,
    });
    if (!ok) return;
    try {
      await reinitWithTuning(profileId);
    } catch {
      // status bar shows error
    }
  };

  subscribe((s) => {
    renderActive(s);
    renderProfiles(s.tuning?.id ?? null);
    renderOverrides(s);
  });
}

function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
