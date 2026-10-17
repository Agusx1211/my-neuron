// Small DOM helpers shared by the UI controllers.

export function $<T extends HTMLElement = HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as T;
}

export function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  props: { class?: string; text?: string; attrs?: Record<string, string> } = {},
  ...children: (Node | string)[]
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  if (props.class) el.className = props.class;
  if (props.text !== undefined) el.textContent = props.text;
  if (props.attrs) for (const [k, v] of Object.entries(props.attrs)) el.setAttribute(k, v);
  el.append(...children);
  return el;
}

/** Unhide with an enter transition (the element's CSS defines `.entering`). */
export function reveal(el: HTMLElement) {
  el.classList.remove('leaving');
  el.classList.add('entering');
  el.hidden = false;
  void el.offsetWidth; // commit the start state so the transition runs
  el.classList.remove('entering');
}

/** Leave transition, then hide. */
export function conceal(el: HTMLElement, ms = 500): Promise<void> {
  if (el.hidden) return Promise.resolve();
  el.classList.add('leaving');
  return new Promise((r) => setTimeout(() => {
    if (el.classList.contains('leaving')) { el.hidden = true; el.classList.remove('leaving'); }
    r();
  }, ms));
}

export const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

export function fmtBytes(n: number) {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  return `${(n / 1e6).toFixed(0)} MB`;
}

export function fmtNum(n: number) {
  return n.toLocaleString('en-US');
}

/** Visible text for a token: show leading spaces as a faint dot. */
export function tokenLabel(t: string) {
  return t.replace(/\n/g, '⏎').replace(/^ /, '·');
}

let toastTimer = 0;
export function toast(msg: string, ms = 5000) {
  const el = $('toast');
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => { el.hidden = true; }, ms);
}
