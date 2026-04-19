// Entry point for the human-first demo. Wires shell chrome (status bar,
// load hero, tabs) and mounts each panel into its container. The engine
// singleton lives in `lib/engine.ts`; panels subscribe there directly.

import './styles.css';

// Self-hosted fonts — bundled through Vite so they serve same-origin
// (COEP `require-corp` blocks cross-origin font fetches). Two weights of
// each; the stylesheet uses variable weight via the @fontsource packages.
import '@fontsource/jetbrains-mono/400.css';
import '@fontsource/jetbrains-mono/500.css';
import '@fontsource/jetbrains-mono/700.css';
import '@fontsource/ibm-plex-mono/400.css';
import '@fontsource/ibm-plex-mono/500.css';
import '@fontsource/ibm-plex-mono/600.css';

import { subscribe, getState, init, installWindowChat } from './lib/engine.js';
import type { EngineState } from './lib/engine.js';
import { REPO_SLUG } from './lib/github.js';

import { mountChatPanel }         from './panels/chat.js';
import { mountBenchPanel }        from './panels/bench.js';
import { mountProfilePanel }      from './panels/profile.js';
import { mountCorrectnessPanel }  from './panels/correctness.js';
import { mountTuningPanel }       from './panels/tuning.js';
import { mountDeviceReportPanel } from './panels/device-report.js';
import { mountWeightProbesPanel } from './panels/weight-probes.js';

installWindowChat();

function $(id: string): HTMLElement {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing element #${id}`);
  return el;
}

function bindShell(): void {
  // Hydrate every `[data-repo-link]` anchor from the single `REPO_SLUG`
  // constant in `app/lib/github.ts`. Avoids three-way drift between the
  // two header anchors and the github module.
  const repoUrl = `https://github.com/${REPO_SLUG}`;
  document.querySelectorAll<HTMLAnchorElement>('a[data-repo-link]').forEach((a) => {
    a.href = repoUrl;
  });

  // Load button.
  const btn = $('btn-load') as HTMLButtonElement;
  btn.addEventListener('click', () => {
    init().catch(() => { /* error shown via status line */ });
  });

  // Tabs — plus the sliding underline driven by CSS vars on the tablist.
  const tablist = $('tabs');
  const tabs = document.querySelectorAll<HTMLButtonElement>('.tab[data-tab]');
  const panels = document.querySelectorAll<HTMLElement>('.panel[data-panel]');

  const moveUnderlineTo = (el: HTMLElement) => {
    const parentRect = tablist.getBoundingClientRect();
    const r = el.getBoundingClientRect();
    tablist.style.setProperty('--tab-underline-x', `${r.left - parentRect.left + tablist.scrollLeft}px`);
    tablist.style.setProperty('--tab-underline-w', `${r.width}px`);
  };

  tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      const name = tab.dataset.tab!;
      tabs.forEach((t) => t.classList.toggle('active', t === tab));
      panels.forEach((p) => p.classList.toggle('active', p.dataset.panel === name));
      moveUnderlineTo(tab);
      window.scrollTo({ top: 0, behavior: 'instant' as ScrollBehavior });
    });
  });

  // Initial underline placement, and re-position on resize.
  const placeInitial = () => {
    const active = Array.from(tabs).find((t) => t.classList.contains('active')) ?? tabs[0];
    if (active) moveUnderlineTo(active);
  };
  requestAnimationFrame(placeInitial);
  window.addEventListener('resize', placeInitial);
}

function renderStatusBar(s: EngineState): void {
  const chip = $('status-chip');
  const chipLbl = chip.querySelector('.status-chip-label') as HTMLElement;
  chip.className = `status-chip ${s.status}`;
  chipLbl.textContent = s.status;

  $('status-line').textContent = s.statusLine;

  const meta = $('status-meta');
  if (s.config && s.tuning) {
    meta.innerHTML = [
      `<span>layers <b>${s.config.num_layers}</b></span>`,
      `<span>ctx <b>${s.config.context_length}</b></span>`,
      `<span>tuning <b>${s.tuning.id}</b></span>`,
    ].join('');
  } else {
    meta.textContent = '';
  }
}

function renderLoadHero(s: EngineState): void {
  const hero = $('load-hero');
  const btn = $('btn-load') as HTMLButtonElement;
  const progressWrap = $('load-progress');
  const progressLabel = $('load-progress-label');
  const progressFill = $('load-progress-fill');

  if (s.status === 'uninit' || s.status === 'loading' || s.status === 'error') {
    hero.classList.remove('hidden');
  } else {
    hero.classList.add('hidden');
  }

  btn.disabled = s.status === 'loading';
  btn.textContent =
    s.status === 'loading' ? 'Loading…' :
    s.status === 'error'   ? 'Retry load' :
                             'Load Gemma 4 E2B';

  if (s.status === 'loading' && s.loadProgress) {
    progressWrap.classList.remove('hidden');
    const { loaded, total, status } = s.loadProgress;
    const pct = total ? (loaded / total) * 100 : 0;
    progressLabel.textContent = total
      ? `${status} · ${(pct).toFixed(1)}%`
      : status;
    progressFill.style.width = `${Math.max(1, Math.min(100, pct))}%`;
  } else {
    progressWrap.classList.add('hidden');
  }
}

function mount(): void {
  bindShell();

  // Mount every panel. They subscribe to engine state internally.
  mountChatPanel($('panel-chat'));
  mountBenchPanel($('panel-bench'));
  mountProfilePanel($('panel-profile'));
  mountCorrectnessPanel($('panel-correctness'));
  mountTuningPanel($('panel-tuning'));
  mountDeviceReportPanel($('panel-device-report'));
  mountWeightProbesPanel($('panel-weight-probes'));

  // Shell chrome subscribes to the same stream as panels.
  subscribe((s) => {
    renderStatusBar(s);
    renderLoadHero(s);
  });

  // Prime render.
  renderStatusBar(getState());
  renderLoadHero(getState());
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', mount);
} else {
  mount();
}
