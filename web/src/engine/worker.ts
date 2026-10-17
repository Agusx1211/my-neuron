/// <reference lib="webworker" />
// Inference worker: owns the GPU device, the WASM core, the model and the atlas.

import initWasm, { Tokenizer, dequant, f16ToF32, heatMap, selectNeurons, semEmbed, spanExtrema } from '../wasm-pkg/neuron_core.js';
import type { Analysis, ModelInfo, ModelSpec, NeuronAtlas, NeuronHit, ProbeResult } from './api';
import { isCached, loadModel, type RowSource } from './loader';
import { GpuModel, configFromMeta, isFull, type ModelConfig } from './model';

interface AtlasMeta {
  layers: number;
  neurons: number;
  kPos: number;
  kNeg: number;
  prefix: string;
  suffix: string;
  extra: number;        // tokens after the text that count toward the span
  minLayer: number;
  floorPos: number[];
  floorNeg: number[];
  concepts: number;
  statCols: number;     // mean/std max, mean/std min, top-list entry thresholds
  semLayer: number;     // residual layer used for the semantic embedding
  semDim: number;
  hidden: number;
}

// selection knobs (see selectNeurons in wasm/src/score.rs; tuned in py/exp/select.py)
const Z_MIN = 6;
const REL_POW = 3;
const TOP_K = 24;

// Function words are left out of the user's span so "my dog" is about the dog,
// not about "my" (atlas concepts rarely contain them either).
const STOP = new Set(`a an the my your his her its our their i me we you he she it they this that these those
of in on at to for from by with and or but is am are was were be been 's ' s mine yours myself
im i'm about as into`.split(/\s+/));

let device: GPUDevice | null = null;
let initP: Promise<{ webgpu: boolean; adapter?: string; reason?: string }> | null = null;
let adapterName = '';
let wasmReady: Promise<unknown> | null = null;
let model: GpuModel | null = null;
let cfg: ModelConfig | null = null;
let tok: Tokenizer | null = null;
let rows: RowSource | null = null;
let atlasBase = '';
let meta: AtlasMeta | null = null;
let stats: Float32Array | null = null;   // [L][F][statCols]
let top8: Uint16Array | null = null;     // [L][F][16]
let semE: Float32Array | null = null;    // [concepts][semDim]
let semProj: Float32Array | null = null; // [hidden][semDim]
let semMu: Float32Array | null = null;   // [hidden]
let lexicon: { concepts: string[]; categories: string[]; cat: Uint16Array | number[] } | null = null;
const layerCache = new Map<number, Promise<ArrayBuffer>>();
let prefixIds: number[] = [];
let suffixIds: number[] = [];
let queue: Promise<unknown> = Promise.resolve();
const PAD = 248044;
const MAX_TOKENS = 512; // activation buffers are reserved once for this many tokens

/** Serialize GPU work: one forward pass at a time. */
function exclusive<T>(fn: () => Promise<T>): Promise<T> {
  const run = queue.then(fn, fn);
  queue = run.catch(() => undefined);
  return run;
}

async function init() {
  wasmReady ??= initWasm();
  await wasmReady;
  if (!('gpu' in navigator)) return { webgpu: false, reason: 'This browser does not support WebGPU.' };
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return { webgpu: false, reason: 'No WebGPU adapter is available (GPU blocklisted or disabled).' };
  const lim = adapter.limits;
  device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize,
      maxBufferSize: lim.maxBufferSize,
    },
  });
  device.lost.then((info) => console.error('WebGPU device lost', info.message));
  device.addEventListener('uncapturederror', (e) => console.error('WebGPU error:', (e as GPUUncapturedErrorEvent).error.message));
  const info = adapter.info;
  adapterName = [info?.vendor, info?.architecture, info?.description].filter(Boolean).join(' ') || 'WebGPU device';
  return { webgpu: true, adapter: adapterName };
}

async function load(spec: ModelSpec, base: string, progress: (p: unknown) => void): Promise<ModelInfo> {
  const ok = await (initP ??= init());
  if (!ok.webgpu || !device) throw new Error(ok.reason ?? 'WebGPU is not available');
  atlasBase = base;
  const metaP = fetch(`${atlasBase}/meta.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null);
  const res = await loadModel(spec.id, spec.url, spec.bytes, async (h) => {
    cfg = configFromMeta(h.meta);
    const m = new GpuModel(device!, cfg);
    await m.compile();
    m.reserve(MAX_TOKENS);
    progress({ phase: 'download', loaded: 0, total: spec.bytes, layersReady: 0 });
    tok = new Tokenizer(h.bytes);
    return m;
  }, (loaded, total, layersReady, fromCache) => {
    progress({ phase: fromCache ? 'cached' : 'download', loaded, total, layersReady });
  });
  model = res.model;
  rows = res.rows;
  progress({ phase: 'upload', loaded: spec.bytes, total: spec.bytes, layersReady: cfg!.nLayer });
  await device.queue.onSubmittedWorkDone();

  progress({ phase: 'atlas', loaded: spec.bytes, total: spec.bytes, layersReady: cfg!.nLayer });
  meta = await metaP;
  if (meta) {
    const bin = (f: string) => fetch(`${atlasBase}/${f}`).then((r) => {
      if (!r.ok) throw new Error(`atlas ${f}: HTTP ${r.status}`);
      return r.arrayBuffer();
    });
    const [s, t8, sem, lex] = await Promise.all([
      bin('stats.bin'), bin('top8.bin'), bin('sem.bin'),
      fetch(`${atlasBase}/lexicon.json`).then((r) => r.json()),
    ]);
    stats = f16ToF32(new Uint8Array(s));
    top8 = new Uint16Array(t8);
    const semAll = f16ToF32(new Uint8Array(sem));
    const ek = meta.concepts * meta.semDim;
    semE = semAll.subarray(0, ek);
    semProj = semAll.subarray(ek, ek + meta.hidden * meta.semDim);
    semMu = semAll.subarray(ek + meta.hidden * meta.semDim);
    lexicon = lex;
    prefixIds = Array.from(tok!.encode(meta.prefix));
    suffixIds = Array.from(tok!.encode(meta.suffix));
  }
  progress({ phase: 'ready', loaded: spec.bytes, total: spec.bytes, layersReady: cfg!.nLayer });
  return {
    id: spec.id,
    name: spec.name,
    layers: cfg!.nLayer,
    neurons: cfg!.dFF,
    layerTypes: [...Array(cfg!.nLayer).keys()].map((l) => (isFull(cfg!, l) ? 'full' : 'linear')),
    adapter: adapterName,
  };
}

function embed(ids: number[]): Float32Array {
  return dequant(14, rows!.read(ids));
}

function decode(id: number): string {
  return new TextDecoder().decode(tok!.piece(id));
}

/** Wrap a text in the atlas template: returns ids and the span of tokens that count. */
function frame(text: string, maxTokens = 64) {
  // user text must never smuggle in control tokens like <|im_end|>
  const safe = text.replace(/<(?=\||\/?(think|tool_call|tool_response)>)/g, '< ');
  const body = Array.from(tok!.encode(safe)).slice(0, maxTokens);
  const ids = [...prefixIds, ...body, ...suffixIds];
  const start = prefixIds.length;
  return { ids, body, start, end: start + body.length + Math.min(meta?.extra ?? 0, suffixIds.length) };
}

// The last forward pass, so reading another token of the same text needs no GPU work.
let last: { text: string; f: ReturnType<typeof frame>; n: number; acts: Float32Array; resid: Float32Array; words: number[] } | null = null;

/**
 * Find the neurons for a text. By default they are read at the end of the text
 * (its last word plus the end-of-message token), where the model has seen all
 * of it; `at` reads a single token instead (`tokens.length` = end of message).
 */
async function analyze(text: string, at: number | null = null): Promise<Analysis> {
  if (!model || !cfg) throw new Error('model not loaded');
  if (!meta || !stats) throw new Error('concept atlas missing');
  return exclusive(async () => {
    const c = cfg!;
    const M = meta!;
    let ms = 0;
    if (last?.text !== text) {
      last = null;
      const f = frame(text.slice(0, 300), 48); // bounded so the full capture stays < 64 MB
      const t0 = performance.now();
      const { acts, resid } = await model!.forward(embed(f.ids), f.ids.length, { kind: 'all' }, M.semLayer);
      ms = performance.now() - t0;
      let words = f.body.map((_, i) => i).filter((i) => !STOP.has(decode(f.body[i]).trim().toLowerCase()));
      if (!words.length) words = f.body.map((_, i) => i);
      last = { text, f, n: f.ids.length, acts, resid: resid!, words };
    }
    const { f, n, acts, resid, words } = last;
    const T = f.body.length;
    if (at !== null && !(at >= 0 && at <= T)) throw new Error(`no token at ${at}`);
    // positions are relative to the text; T stands for the end-of-message token(s)
    const toMask = (pos: number[]) => {
      const mask = new Uint8Array(n);
      for (const p of pos) {
        if (p < T) mask[f.start + p] = 1;
        else mask.fill(1, f.start + T, f.end);
      }
      return mask;
    };
    const reading = at !== null ? [at] : words.length ? [words[words.length - 1], T] : [T];
    const mask = toMask(reading);
    // what the input means: the whole text, or just the context up to a single token
    const semMask = at === null || at === T ? toMask([...words, T]) : mask;
    const ext = spanExtrema(acts, c.nLayer, n, c.dFF, mask);
    const fp = new Float32Array(M.floorPos);
    const fn = new Float32Array(M.floorNeg);
    const q = semEmbed(resid, c.d, semMask, semMu!, semProj!, M.semDim);
    let rows = selectNeurons(ext, stats!, c.dFF, fp, fn, top8!, semE!, q, M.minLayer, Z_MIN, true, REL_POW, TOP_K);
    if (rows.length < 7 * 8) {
      // unusual input that enters few top lists: fall back to any strongly excited neuron
      rows = selectNeurons(ext, stats!, c.dFF, fp, fn, top8!, semE!, q, M.minLayer, Z_MIN, false, REL_POW, TOP_K);
    }
    const top: NeuronHit[] = [];
    for (let i = 0; i < rows.length; i += 7) {
      const [layer, index, pole, , z, act, rel] = rows.subarray(i, i + 7);
      const perToken: number[] = [];
      for (let t = f.start; t < f.start + T + 1; t++) perToken.push(acts[(layer * n + t) * c.dFF + index]);
      const t8 = (layer * c.dFF + index) * 16 + (pole > 0 ? 0 : 8);
      const preview = [...top8!.subarray(t8, t8 + 4)].map((ci) => lexicon!.concepts[ci]);
      top.push({ layer, index, pole: pole > 0 ? 1 : -1, z, act, perToken, match: Math.max(0, rel), preview });
    }
    const heat = heatMap(ext, stats!, M.statCols, c.dFF, fp, fn);
    return {
      text,
      tokens: f.body.map((id) => ({ id, text: decode(id) })),
      at,
      reading,
      top,
      heat,
      ms,
    };
  });
}

function layerFile(layer: number): Promise<ArrayBuffer> {
  let p = layerCache.get(layer);
  if (!p) {
    p = fetch(`${atlasBase}/L${layer}.bin`).then((r) => {
      if (!r.ok) throw new Error(`atlas layer ${layer}: HTTP ${r.status}`);
      return r.arrayBuffer();
    });
    p.catch(() => layerCache.delete(layer));
    layerCache.set(layer, p);
  }
  return p;
}

async function atlas(layer: number, index: number): Promise<NeuronAtlas> {
  if (!meta || !stats || !lexicon) throw new Error('concept atlas missing');
  const buf = await layerFile(layer);
  const k = meta.kPos + meta.kNeg;
  const rec = new Uint8Array(buf, index * k * 4, k * 4);
  const ids = new Uint16Array(rec.slice(0, k * 2).buffer);
  const vals = f16ToF32(rec.subarray(k * 2));
  const item = (j: number) => {
    const ci = ids[j];
    const cat = lexicon!.cat[ci];
    return { text: lexicon!.concepts[ci], act: vals[j], category: lexicon!.categories[cat] };
  };
  const j = (layer * meta.neurons + index) * meta.statCols;
  const s = stats.subarray(j, j + 4);
  return {
    layer,
    index,
    positive: [...Array(meta.kPos).keys()].map(item),
    negative: [...Array(meta.kNeg).keys()].map((j) => item(meta!.kPos + j)),
    baseline: { meanMax: s[0], stdMax: s[1], meanMin: s[2], stdMin: s[3] },
  };
}

async function probe(layer: number, index: number, pole: 1 | -1, texts: string[], onResult: (r: ProbeResult) => void): Promise<ProbeResult[]> {
  if (!model || !cfg || !meta || !stats) throw new Error('model not loaded');
  const c = cfg;
  const s = stats.subarray((layer * c.dFF + index) * meta.statCols, (layer * c.dFF + index) * meta.statCols + 4);
  const sd = pole > 0 ? Math.max(s[1], meta.floorPos[layer]) : Math.max(s[3], meta.floorNeg[layer]);
  const framed = texts.map((t) => ({ text: t, ...frame(t.slice(0, 200), 48) }));
  const out: ProbeResult[] = [];
  // group by length so padding stays small, then pack batches under MAX_TOKENS
  const order = framed.map((_, i) => i).sort((a, b) => framed[a].ids.length - framed[b].ids.length);
  let i = 0;
  while (i < order.length) {
    // small first batch so results start streaming quickly
    const cap = out.length === 0 ? 12 : 256;
    const group: number[] = [];
    let Tb = 0;
    while (i < order.length && group.length < cap) {
      const len = framed[order[i]].ids.length;
      if (group.length && (group.length + 1) * Math.max(Tb, len) > MAX_TOKENS) break;
      group.push(order[i++]);
      Tb = Math.max(Tb, len);
    }
    const ids: number[] = [];
    for (const j of group) ids.push(...framed[j].ids, ...Array(Tb - framed[j].ids.length).fill(PAD));
    const run = () => exclusive(() => model!.forward(embed(ids), Tb, { kind: 'neurons', layer, idx: [index] }));
    let { acts } = await run();
    if (acts.every((v) => v === 0)) {
      // real activations are never all exactly zero: the readback was lost, retry once
      console.warn('probe readback was empty, retrying');
      await device!.queue.onSubmittedWorkDone();
      ({ acts } = await run());
      if (acts.every((v) => v === 0)) throw new Error('the GPU returned no data for this probe');
    }
    group.forEach((j, b) => {
      const f = framed[j];
      let v = pole > 0 ? -Infinity : Infinity;
      for (let t = f.start; t < f.end; t++) {
        const a = acts[b * Tb + t];
        v = pole > 0 ? Math.max(v, a) : Math.min(v, a);
      }
      const z = pole > 0 ? (v - s[0]) / sd : (s[2] - v) / sd;
      const r = { text: f.text, act: v, z };
      out.push(r);
      onResult(r);
    });
  }
  return out;
}

// --- message plumbing -----------------------------------------------------------

type Req = { id: number; op: string; args: unknown[] };

self.onmessage = async (e: MessageEvent<Req>) => {
  const { id, op, args } = e.data;
  const event = (data: unknown) => (self as unknown as Worker).postMessage({ id, event: data });
  try {
    let result: unknown;
    switch (op) {
      case 'init': result = await (initP ??= init()); break;
      case 'isCached': { const s = args[0] as ModelSpec; result = await isCached(s.id, s.bytes); break; }
      case 'load': result = await load(args[0] as ModelSpec, args[1] as string, event); break;
      case 'analyze': {
        const a = await analyze(args[0] as string, (args[1] as number | null) ?? null);
        (self as unknown as Worker).postMessage({ id, result: a }, [a.heat.buffer]);
        return;
      }
      case 'atlas': result = await atlas(args[0] as number, args[1] as number); break;
      case 'probe': result = await probe(args[0] as number, args[1] as number, args[2] as 1 | -1, args[3] as string[], event); break;
      default: throw new Error(`unknown op ${op}`);
    }
    (self as unknown as Worker).postMessage({ id, result });
  } catch (err) {
    (self as unknown as Worker).postMessage({ id, error: err instanceof Error ? err.message : String(err) });
  }
};
