// Minimal confirm() replacement styled to match the brutalist aesthetic.
// Used everywhere a panel is about to do something destructive (reset the
// conversation, re-init the engine, etc.) — native `confirm()` is ugly and
// doesn't respect dark mode.

export interface ConfirmOptions {
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  danger?: boolean;
}

export function confirmDialog(opts: ConfirmOptions): Promise<boolean> {
  return new Promise((resolve) => {
    const root = document.getElementById('dialog-root');
    if (!root) { resolve(window.confirm(`${opts.title}\n\n${opts.message}`)); return; }

    const backdrop = document.createElement('div');
    backdrop.className = 'dialog-backdrop';
    backdrop.innerHTML = `
      <div class="dialog" role="dialog" aria-modal="true">
        <h3>${escapeHtml(opts.title)}</h3>
        <p>${escapeHtml(opts.message)}</p>
        <div class="dialog-actions">
          <button type="button" class="ghost" data-act="cancel">${escapeHtml(opts.cancelLabel ?? 'Cancel')}</button>
          <button type="button" class="${opts.danger ? 'danger' : 'primary'}" data-act="confirm">${escapeHtml(opts.confirmLabel ?? 'Confirm')}</button>
        </div>
      </div>
    `;
    root.append(backdrop);

    const confirm = backdrop.querySelector<HTMLButtonElement>('[data-act="confirm"]')!;
    const cancel = backdrop.querySelector<HTMLButtonElement>('[data-act="cancel"]')!;
    const close = (result: boolean) => {
      backdrop.remove();
      document.removeEventListener('keydown', onKey);
      resolve(result);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close(false);
      if (e.key === 'Enter') close(true);
    };
    confirm.addEventListener('click', () => close(true));
    cancel.addEventListener('click', () => close(false));
    backdrop.addEventListener('click', (e) => { if (e.target === backdrop) close(false); });
    document.addEventListener('keydown', onKey);
    cancel.focus();
  });
}

function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
