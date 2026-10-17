// The neuron card: identity, stats, token heat, layer map, live checks, runners-up.

import type { Analysis, AtlasConcept, ModelInfo, NeuronHit, ProbeResult } from '../engine/api';
import { $, conceal, fmtNum, h, reveal, tokenLabel } from './dom';

export interface CardHandlers {
  onExplore: () => void;
  onAgain: () => void;
  onRunner: (i: number) => void;
  onTest: (text: string) => void;
  onToken: (at: number | null) => void;
}

const signed = (v: number, d = 2) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d);

export class Card {
  private root = $('card');
  private analysis: Analysis | null = null;
  private info: ModelInfo | null = null;
  private current = 0;
  private rows = new Map<string, { li: HTMLLIElement; bar: HTMLElement; val: HTMLElement }>();
  private scale = 1;

  constructor(private handlers: CardHandlers) {
    $('explore-btn').addEventListener('click', () => handlers.onExplore());
    $('again-btn').addEventListener('click', () => handlers.onAgain());
    $('test-form').addEventListener('submit', (e) => {
      e.preventDefault();
      const input = $<HTMLInputElement>('test-input');
      const text = input.value.trim();
      if (!text) return;
      input.value = '';
      handlers.onTest(text);
    });
  }

  get hitIndex() {
    return this.current;
  }

  show(analysis: Analysis, info: ModelInfo, keepScroll = false) {
    this.analysis = analysis;
    this.info = info;
    $('test-results').innerHTML = '';
    this.setHit(0);
    reveal(this.root);
    if (!keepScroll) this.root.querySelector('.card-scroll')!.scrollTop = 0;
  }

  setHit(i: number) {
    const a = this.analysis, info = this.info;
    if (!a || !info) return;
    this.current = i;
    const hit = a.top[i];
    const pos = hit.pole > 0;
    $('card-eyebrow').textContent = i > 0 ? `Runner-up #${i + 1}`
      : a.at === null ? 'Your neuron'
      : a.at === a.tokens.length ? 'Your neuron at the end'
      : `Your neuron at “${a.tokens[a.at].text.trim()}”`;
    const title = $('card-title');
    title.innerHTML = '';
    title.append(
      'Layer ', h('span', { class: 'mono', text: String(hit.layer) }),
      h('span', { class: 'sep', text: '·' }),
      'Neuron ', h('span', { class: 'mono', text: `#${fmtNum(hit.index)}` }),
    );
    const badges = $('card-badges');
    badges.innerHTML = '';
    badges.append(
      h('span', { class: `badge ${pos ? 'badge-pos' : 'badge-neg'}`, text: pos ? '+ positive pole' : '− negative pole' }),
      h('span', { class: 'badge badge-kind', text: info.layerTypes[hit.layer] === 'full' ? 'after full attention' : 'after Gated DeltaNet' }),
    );
    // the neuron's theme at a glance (its strongest atlas concepts for this pole)
    let theme = document.getElementById('card-theme');
    if (!theme) {
      theme = h('p', { class: 'card-theme', attrs: { id: 'card-theme' } });
      badges.after(theme);
    }
    const known = (hit.preview ?? []).filter((p) => p.toLowerCase() !== this.analysis!.text.trim().toLowerCase()).slice(0, 4);
    theme.hidden = known.length === 0;
    theme.innerHTML = '';
    theme.append(h('span', { class: 'muted', text: pos ? 'Fires for ' : 'Driven down by ' }), known.join(' · '));
    const z = $('stat-z');
    z.textContent = `${hit.z.toFixed(1)}σ`;
    z.className = `stat-num mono ${pos ? 'pos' : 'neg'}`;
    $('stat-z-label').textContent = pos ? 'above its baseline' : 'below its baseline';
    $('stat-act').textContent = signed(hit.act);
    this.renderTokens(hit);
    this.renderLayerMap(hit);
    this.renderRunners();
    $('verify-section').hidden = true;
    $('test-results').innerHTML = '';
    const btn = $<HTMLButtonElement>('explore-btn');
    btn.disabled = false;
    btn.textContent = 'What else lights it up?';
  }

  private renderTokens(hit: NeuronHit) {
    const strip = $('token-strip');
    strip.innerHTML = '';
    const a = this.analysis!;
    const T = a.tokens.length;
    const m = Math.max(1e-6, ...hit.perToken.map((v) => Math.abs(v)));
    const tint = hit.pole > 0 ? '255,181,71' : '63,184,255';
    // the user's tokens, then the end of the message (where the model has read everything)
    [...a.tokens.map((t) => tokenLabel(t.text)), 'end'].forEach((label, j) => {
      const v = hit.perToken[j] ?? 0;
      const k = Math.max(0, Math.min(1, (v * hit.pole) / m));
      const cls = ['tok'];
      if (j === T) cls.push('tok-end');
      if (a.reading.includes(j)) cls.push('reading');
      const el = h('button', { class: cls.join(' '), text: label, attrs: { type: 'button', 'data-v': signed(v), 'aria-pressed': String(a.at === j) } });
      el.style.background = `rgba(${tint},${(0.05 + 0.5 * k * k).toFixed(3)})`;
      el.style.borderColor = `rgba(${tint},${(0.1 + 0.6 * k).toFixed(3)})`;
      if (k > 0.6) {
        el.style.color = '#fff';
        el.style.boxShadow = `0 0 ${Math.round(14 * k)}px rgba(${tint},${(0.5 * k).toFixed(2)})`;
      }
      el.addEventListener('click', () => this.handlers.onToken(a.at === j ? null : j));
      strip.append(el);
    });
    const note = $('token-note');
    note.innerHTML = '';
    note.append(h('i', { class: 'reading-key' }));
    if (a.at === null) {
      note.append('Read at the end of your text, where the model has seen all of it. Tap any token to see the neuron that fires there.');
    } else {
      const back = h('button', { class: 'link-btn', text: '↺ whole text', attrs: { type: 'button' } });
      back.addEventListener('click', () => this.handlers.onToken(null));
      note.append(a.at === T ? 'Read at the end of your message only · ' : `Read at “${a.tokens[a.at].text.trim()}” only · `, back);
    }
  }

  private renderLayerMap(hit: NeuronHit) {
    const map = $('layer-map');
    map.innerHTML = '';
    this.info!.layerTypes.forEach((t, l) => {
      const cls = ['lm'];
      if (t === 'full') cls.push('full');
      if (l === hit.layer) cls.push('on', hit.pole > 0 ? 'pos' : 'neg');
      map.append(h('span', { class: cls.join(' '), attrs: { title: `Layer ${l}${t === 'full' ? ' · full attention' : ''}` } }));
    });
    const depth = Math.round(((hit.layer + 0.5) / this.info!.layers) * 100);
    $('layer-caption').textContent = `${depth}% of the way up · ${this.info!.layers} layers`;
  }

  private renderRunners() {
    const list = $('runners');
    list.innerHTML = '';
    this.analysis!.top.slice(0, 10).forEach((r, i) => {
      const theme = (r.preview ?? []).filter((p) => p.toLowerCase() !== this.analysis!.text.trim().toLowerCase()).slice(0, 3);
      const b = h('button', { class: 'runner', attrs: { type: 'button' } },
        h('span', { class: `rdot ${r.pole > 0 ? 'pos' : 'neg'}` }),
        h('span', { class: 'rname' },
          h('span', { text: `L${r.layer} · #${fmtNum(r.index)}` }),
          ...(theme.length ? [h('span', { class: 'rtheme', text: theme.join(' · ') })] : []),
        ),
        h('span', { class: 'rz', text: `${r.z.toFixed(1)}σ` }),
      );
      if (i === this.current) b.setAttribute('aria-current', 'true');
      b.addEventListener('click', () => { if (i !== this.current) this.handlers.onRunner(i); });
      list.append(h('li', {}, b));
    });
  }

  exploring(state: 'loading' | 'open') {
    const btn = $<HTMLButtonElement>('explore-btn');
    btn.disabled = true;
    btn.textContent = state === 'loading' ? 'Reading the atlas…' : 'Constellation open';
  }

  /** Show the atlas list; values fill in as live probes return. */
  startVerify(primary: AtlasConcept[], pole: 1 | -1, you?: { text: string; act: number }) {
    const sec = $('verify-section');
    sec.hidden = false;
    const list = $('concept-list');
    list.innerHTML = '';
    this.rows.clear();
    this.scale = Math.max(1e-6, Math.abs(primary[0]?.act ?? 1));
    const color = pole > 0 ? 'var(--amber)' : 'var(--cyan)';
    const add = (text: string, atlasAct: number, you = false) => {
      const bar = h('i');
      bar.style.background = color;
      bar.style.opacity = '0.35';
      bar.style.width = `${Math.min(100, (Math.abs(atlasAct) / this.scale) * 100)}%`;
      const val = h('span', { class: 'cval', text: you ? signed(atlasAct) : '…' });
      const li = h('li', { class: you ? 'you live' : '' }, h('span', { class: 'cname', text: you ? `${text} (you)` : text }), h('span', { class: 'cbar' }, bar), val);
      if (you) bar.style.opacity = '1';
      list.append(li);
      if (!you) this.rows.set(text, { li, bar, val });
    };
    if (you) add(you.text, you.act, true);
    primary.slice(0, 16).forEach((c) => add(c.text, c.act));
    this.verifyStatus(0, Math.min(16, primary.length));
  }

  verifyStatus(done: number, total: number) {
    $('verify-status').textContent = done < total
      ? `Re-measuring on your GPU… ${done}/${total}`
      : `Verified live: ${total} concepts re-run through the model`;
  }

  liveResult(r: ProbeResult) {
    const row = this.rows.get(r.text);
    if (!row) return;
    row.li.classList.add('live');
    row.bar.style.opacity = '1';
    row.bar.style.width = `${Math.min(100, (Math.abs(r.act) / this.scale) * 100)}%`;
    row.val.textContent = signed(r.act);
  }

  testResult(r: ProbeResult) {
    const strong = r.z >= 3;
    const line = h('div', { class: 'tr' },
      h('span', { text: r.text }),
      h('span', { class: 'mono', text: `${signed(r.act)} · ${r.z.toFixed(1)}σ${strong ? ' · fires' : ''}` }),
    );
    $('test-results').prepend(line);
  }

  hide() {
    return conceal(this.root, 600);
  }
}
