// Fake engine with realistic shapes and timings, so the UI is fully demoable
// without WebGPU or a 2.7 GB download.

import type {
  Analysis, AtlasConcept, Engine, LoadProgress, ModelInfo, ModelSpec, NeuronAtlas, NeuronHit, ProbeResult,
} from './api';

const DIMS: Record<string, { layers: number; neurons: number }> = {
  'qwen3.5-4b': { layers: 32, neurons: 9216 },
  'qwen3.5-0.8b': { layers: 24, neurons: 3584 },
};

const THEMES: string[][] = [
  ['Buenos Aires', 'tango', 'Lionel Messi', 'mate', 'Patagonia', 'Diego Maradona', 'asado', 'Córdoba', 'gaucho', 'Rosario', 'dulce de leche', 'Mendoza', 'peso', 'Borges', 'La Boca', 'Evita', 'Pampas', 'empanada', 'River Plate', 'Malbec', 'Ushuaia', 'bandoneón', 'alfajor', 'Iguazú'],
  ['pizza', 'lasagna', 'mozzarella', 'Naples', 'spaghetti', 'risotto', 'espresso', 'gelato', 'pesto', 'tiramisu', 'focaccia', 'parmesan', 'carbonara', 'Rome', 'basil', 'ravioli', 'olive oil', 'prosciutto', 'Tuscany', 'calzone', 'oregano', 'trattoria', 'gnocchi', 'Venice'],
  ['nostalgia', 'childhood', 'memories', 'old photographs', 'homesickness', 'vinyl records', 'summer camp', 'grandmother', 'yearbook', 'cassette tape', 'reunion', 'lullaby', 'hometown', 'first love', 'polaroid', 'scrapbook', 'diary', 'sunset', 'autumn', 'letters', 'heirloom', 'old friends', 'treehouse', 'fairy tale'],
  ['hacker', 'username', 'GitHub', 'password', 'terminal', 'Linux', 'open source', 'blockchain', 'Ethereum', 'compiler', 'Discord', 'Reddit', 'avatar', 'Bitcoin', 'server', 'firewall', 'Python', 'repository', 'API', 'encryption', 'debugging', 'Stack Overflow', 'kernel', 'cryptography'],
  ['ocean', 'whale', 'coral reef', 'tide', 'dolphin', 'lighthouse', 'sailor', 'shipwreck', 'seagull', 'harbor', 'octopus', 'surfing', 'Atlantis', 'anchor', 'pearl', 'lagoon', 'mermaid', 'seaweed', 'submarine', 'Titanic', 'jellyfish', 'compass', 'island', 'storm'],
  ['dragon', 'wizard', 'castle', 'Gandalf', 'unicorn', 'spellbook', 'knight', 'prophecy', 'potion', 'Hogwarts', 'elf', 'enchanted forest', 'sorcerer', 'griffin', 'Middle-earth', 'phoenix', 'quest', 'goblin', 'amulet', 'troll', 'crystal ball', 'Merlin', 'rune', 'dungeon'],
];

const OPPOSITES: string[][] = [
  ['spreadsheet', 'invoice', 'tax form', 'warehouse', 'regulation', 'quarterly report', 'parking lot', 'stapler', 'compliance', 'fax machine', 'audit', 'insurance claim'],
  ['silence', 'emptiness', 'vacuum', 'void', 'blank page', 'zero', 'nothing', 'static', 'white noise', 'gray', 'idle', 'pause'],
];

function hash(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}

function rng(seed: number) {
  let a = seed || 1;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function fakeTokens(text: string): string[] {
  const words = text.match(/\s*[^\s]+/g) ?? [text];
  const out: string[] = [];
  for (const w of words) {
    for (let i = 0; i < w.length; i += 6) out.push(w.slice(i, i + 6));
  }
  return out.length ? out : [text];
}

export class MockEngine implements Engine {
  private info: ModelInfo | null = null;
  private lastText = '';

  async init() {
    await sleep(150);
    return { webgpu: true, adapter: 'Mock GPU (demo mode)' };
  }

  async isCached(model: ModelSpec) {
    try { return localStorage.getItem(`mock-cached:${model.id}`) === '1'; } catch { return false; }
  }

  async load(model: ModelSpec, onProgress: (p: LoadProgress) => void): Promise<ModelInfo> {
    const dims = DIMS[model.id] ?? DIMS['qwen3.5-4b'];
    const cached = await this.isCached(model);
    const total = model.bytes;
    const steps = cached ? 40 : 110;
    onProgress({ phase: 'checking', loaded: 0, total, layersReady: 0 });
    await sleep(300);
    for (let s = 1; s <= steps; s++) {
      const f = s / steps;
      const eased = cached ? f : f * f * (3 - 2 * f) * 0.6 + f * 0.4;
      onProgress({
        phase: cached ? 'cached' : 'download',
        loaded: Math.round(total * eased),
        total,
        layersReady: Math.floor(dims.layers * eased),
      });
      await sleep(cached ? 30 : 55);
    }
    onProgress({ phase: 'upload', loaded: total, total, layersReady: dims.layers });
    await sleep(400);
    onProgress({ phase: 'atlas', loaded: total, total, layersReady: dims.layers });
    await sleep(500);
    onProgress({ phase: 'ready', loaded: total, total, layersReady: dims.layers });
    try { localStorage.setItem(`mock-cached:${model.id}`, '1'); } catch { /* private mode */ }
    this.info = {
      id: model.id,
      name: model.name,
      layers: dims.layers,
      neurons: dims.neurons,
      layerTypes: Array.from({ length: dims.layers }, (_, i) => ((i + 1) % 4 === 0 ? 'full' : 'linear')),
      adapter: 'Mock GPU (demo mode)',
    };
    return this.info;
  }

  async analyze(text: string): Promise<Analysis> {
    const info = this.info ?? (await this.load({ id: 'qwen3.5-4b', name: 'Qwen3.5-4B', url: '', bytes: 1, atlas: '' }, () => {}));
    const t0 = performance.now();
    this.lastText = text;
    const rand = rng(hash(text));
    const L = info.layers, N = info.neurons;
    const tokens = fakeTokens(text).map((t, i) => ({ text: t, id: 1000 + i }));
    const heat = new Float32Array(L * N);
    // background: mostly calm with mild noise
    for (let i = 0; i < heat.length; i++) {
      const r = rand();
      heat[i] = r < 0.985 ? (rand() - 0.5) * 1.2 : (rand() - 0.35) * 7;
    }
    // a few thousand mid-strength neurons concentrated in mid/late layers
    for (let k = 0; k < 2500; k++) {
      const l = Math.min(L - 1, Math.floor(L * (0.3 + 0.7 * Math.pow(rand(), 0.7))));
      const i = Math.floor(rand() * N);
      heat[l * N + i] = (rand() < 0.75 ? 1 : -1) * (2.5 + rand() * 3.5);
    }
    const top: NeuronHit[] = [];
    let z = 9 + rand() * 4;
    for (let k = 0; k < 24; k++) {
      const layer = Math.min(L - 1, Math.floor(L * (0.35 + 0.6 * rand())));
      const index = Math.floor(rand() * N);
      const pole: 1 | -1 = rand() < 0.8 ? 1 : -1;
      const peak = Math.floor(rand() * tokens.length);
      const act = pole * (1.2 + z * 0.28);
      const perToken = tokens.map((_, j) => (j === peak ? act : act * Math.max(0, 0.6 - Math.abs(j - peak) * 0.25) * rand() + (rand() - 0.5) * 0.15));
      const theme = (pole > 0 ? THEMES : OPPOSITES)[Math.floor(rand() * (pole > 0 ? THEMES : OPPOSITES).length)];
      const preview = theme.slice(0, 4);
      top.push({ layer, index, pole, z, act, perToken, match: 0.4 + rand() * 0.3, preview });
      heat[layer * N + index] = pole * z;
      z -= 0.25 + rand() * 0.35;
    }
    await sleep(500 + tokens.length * 25);
    return { text, tokens, top, heat, ms: performance.now() - t0 };
  }

  private conceptsFor(layer: number, index: number) {
    const rand = rng(hash(`${layer}:${index}`));
    const theme = THEMES[Math.floor(rand() * THEMES.length)];
    const opp = OPPOSITES[Math.floor(rand() * OPPOSITES.length)];
    const scale = 2.2 + rand() * 2;
    const positive: AtlasConcept[] = theme
      .map((text, i) => ({ text, act: scale * (1 - i / (theme.length * 1.25)) * (0.92 + rand() * 0.08) }))
      .sort((a, b) => b.act - a.act);
    const negative: AtlasConcept[] = opp
      .map((text, i) => ({ text, act: -(scale * 0.45) * (1 - i / (opp.length * 1.4)) }))
      .sort((a, b) => a.act - b.act);
    return { positive, negative, scale };
  }

  async atlas(layer: number, index: number): Promise<NeuronAtlas> {
    await sleep(250);
    const { positive, negative, scale } = this.conceptsFor(layer, index);
    return {
      layer, index, positive, negative,
      baseline: { meanMax: scale * 0.12, stdMax: scale * 0.11, meanMin: -scale * 0.08, stdMin: scale * 0.07 },
    };
  }

  async probe(layer: number, index: number, pole: 1 | -1, texts: string[], onResult?: (r: ProbeResult) => void): Promise<ProbeResult[]> {
    const { positive, negative, scale } = this.conceptsFor(layer, index);
    const known = new Map([...positive, ...negative].map((c) => [c.text.toLowerCase(), c.act]));
    const base = pole > 0 ? { m: scale * 0.12, s: scale * 0.11 } : { m: -scale * 0.08, s: scale * 0.07 };
    const out: ProbeResult[] = [];
    for (const text of texts) {
      await sleep(70 + Math.random() * 40);
      const r = rng(hash(`${layer}:${index}:${text}`));
      let act: number;
      const k = known.get(text.toLowerCase());
      if (k !== undefined) act = pole > 0 ? Math.max(k, 0) * (0.9 + r() * 0.2) : Math.min(k, 0) * (0.9 + r() * 0.2);
      else if (text === this.lastText) act = pole * scale * (0.9 + r() * 0.3);
      else act = pole * scale * (r() < 0.3 ? 0.4 + r() * 0.5 : r() * 0.3);
      const z = pole > 0 ? (act - base.m) / base.s : (base.m - act) / base.s;
      const res = { text, act, z };
      out.push(res);
      onResult?.(res);
    }
    return out;
  }
}
