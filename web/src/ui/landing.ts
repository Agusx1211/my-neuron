// Landing panel (model picker, load button) and the loading readout it turns into.

import type { LoadProgress, ModelSpec } from '../engine/api';
import { $, conceal, fmtBytes, fmtNum, h } from './dom';

export const PREVIEW_DIMS: Record<string, { layers: number; neurons: number; params: string }> = {
  'qwen3.5-4b': { layers: 32, neurons: 9216, params: '4B' },
  'qwen3.5-0.8b': { layers: 24, neurons: 3584, params: '0.8B' },
};

/** The full model by default; the lite one on phones and low-memory machines. */
export function defaultModel(models: ModelSpec[]): ModelSpec {
  const nav = navigator as Navigator & { deviceMemory?: number; userAgentData?: { mobile?: boolean } };
  const mobile = nav.userAgentData?.mobile ?? /Android|iPhone|iPad|Mobile/i.test(navigator.userAgent);
  const lowMem = nav.deviceMemory !== undefined && nav.deviceMemory < 8;
  return (mobile || lowMem) && models.length > 1 ? models[models.length - 1] : models[0];
}

const PHASE_TEXT: Record<LoadProgress['phase'], string> = {
  checking: 'Checking your browser cache…',
  download: 'Downloading weights from Hugging Face',
  cached: 'Loading from cache',
  upload: 'Uploading to GPU',
  atlas: 'Loading concept atlas',
  ready: 'Ready',
};

export class Landing {
  private panel = $('landing');
  private picker = $('model-picker');
  private loadBtn = $<HTMLButtonElement>('load-btn');
  private selected: ModelSpec;
  private cached = new Set<string>();
  private lastBytes = 0;
  private lastT = 0;
  private speed = 0;

  constructor(private models: ModelSpec[], private onSelect: (m: ModelSpec) => void, onLoad: (m: ModelSpec) => void) {
    this.selected = defaultModel(models);
    this.renderPicker();
    this.loadBtn.addEventListener('click', () => onLoad(this.selected));
  }

  get model() {
    return this.selected;
  }

  setCached(id: string, yes: boolean) {
    if (yes) this.cached.add(id); else this.cached.delete(id);
    this.renderPicker();
  }

  noWebGPU(reason: string) {
    const box = $('nogpu');
    box.hidden = false;
    box.innerHTML = '';
    box.append(
      h('strong', { text: 'WebGPU isn’t available here. ' }),
      document.createTextNode(reason || 'Try a recent Chrome, Edge or Safari on a computer with a GPU.'),
    );
    this.loadBtn.disabled = true;
  }

  private renderPicker() {
    this.picker.querySelectorAll('.model-opt').forEach((n) => n.remove());
    for (const m of this.models) {
      const dims = PREVIEW_DIMS[m.id];
      const input = h('input', { attrs: { type: 'radio', name: 'model', value: m.id } });
      input.checked = m.id === this.selected.id;
      input.addEventListener('change', () => {
        this.selected = m;
        this.updateMeta();
        this.onSelect(m);
      });
      const name = h('span', { class: 'model-name', text: m.name });
      if (this.cached.has(m.id)) name.append(h('span', { class: 'pill pill-cached', text: 'cached' }));
      if (m.id.includes('0.8b')) name.append(h('span', { class: 'pill pill-lite', text: 'lite' }));
      const sub = h('span', {
        class: 'model-sub',
        text: dims ? `${dims.layers} layers · ${fmtNum(dims.layers * dims.neurons)} neurons` : '',
      });
      const label = h('label', { class: 'model-opt' },
        input,
        h('span', { class: 'radio', attrs: { 'aria-hidden': 'true' } }),
        h('span', {}, name, sub),
        h('span', { class: 'model-size', text: fmtBytes(m.bytes) }),
      );
      this.picker.append(label);
    }
    this.updateMeta();
  }

  private updateMeta() {
    $('load-meta').textContent = this.cached.has(this.selected.id) ? 'cached' : fmtBytes(this.selected.bytes);
  }

  showLoading(model: ModelSpec) {
    $('landing-body').hidden = true;
    $('loading-body').hidden = false;
    const dims = PREVIEW_DIMS[model.id];
    $('load-layers').textContent = `Layer 0 / ${dims?.layers ?? '?'} on GPU`;
    this.lastBytes = 0;
    this.lastT = performance.now();
    this.speed = 0;
  }

  progress(p: LoadProgress, layers: number) {
    $('load-phase').textContent = p.message ?? PHASE_TEXT[p.phase];
    const pct = p.total ? Math.min(100, (p.loaded / p.total) * 100) : 0;
    const shown = p.phase === 'upload' || p.phase === 'atlas' || p.phase === 'ready' ? 100 : pct;
    $('load-pct').textContent = `${Math.floor(shown)}%`;
    $('load-fill').style.width = `${shown}%`;
    $('load-bar').setAttribute('aria-valuenow', String(Math.floor(shown)));
    $('load-bytes').textContent = `${fmtBytes(p.loaded)} / ${fmtBytes(p.total)}`;
    const now = performance.now();
    if (now - this.lastT > 400) {
      const inst = ((p.loaded - this.lastBytes) / ((now - this.lastT) / 1000)) / 1e6;
      this.speed = this.speed ? this.speed * 0.6 + inst * 0.4 : inst;
      this.lastBytes = p.loaded;
      this.lastT = now;
    }
    $('load-speed').textContent = (p.phase === 'download' || p.phase === 'cached') && this.speed > 0.1 ? `${this.speed.toFixed(1)} MB/s` : '';
    $('load-layers').textContent = `Layer ${Math.min(layers, p.layersReady)} / ${layers} on GPU`;
  }

  hide() {
    return conceal(this.panel, 650);
  }
}
