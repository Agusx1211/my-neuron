// Dev harness: load a local GGUF, run one forward pass, expose activations for comparison.
import initWasm, { Tokenizer, dequant } from '../src/wasm-pkg/neuron_core.js';
import { loadModel } from '../src/engine/loader';
import { GpuModel, configFromMeta } from '../src/engine/model';

const log = (s: string) => { document.getElementById('log')!.textContent += s + '\n'; console.log(s); };
const q = new URLSearchParams(location.search);
const file = q.get('file')!;
const layers = (q.get('layers') ?? '0,3,10').split(',').map(Number);

(async () => {
  try {
    await initWasm();
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('no adapter');
    log('adapter: ' + JSON.stringify({ vendor: adapter.info.vendor, arch: adapter.info.architecture, desc: adapter.info.description }));
    const device = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
    device.addEventListener('uncapturederror', (e) => log('GPU ERROR: ' + (e as GPUUncapturedErrorEvent).error.message));
    const head = await fetch(file, { method: 'HEAD' });
    const total = Number(head.headers.get('content-length'));
    let tok: Tokenizer | null = null;
    const t0 = performance.now();
    const res = await loadModel('test', file, total, async (h) => {
      const m = new GpuModel(device, configFromMeta(h.meta));
      await m.compile();
      m.reserve(256);
      tok = new Tokenizer(h.bytes);
      return m;
    }, () => {});
    await device.queue.onSubmittedWorkDone();
    log(`loaded in ${(performance.now() - t0).toFixed(0)} ms`);
    const text = q.get('text') ?? '<|im_start|>user\nAgustin, also known as agusx1211, loves Buenos Aires and mate.<|im_end|>';
    const ids = Array.from(tok!.encode(text));
    log('ids ' + JSON.stringify(ids));
    const emb = dequant(14, res.rows.read(ids));
    const t1 = performance.now();
    const { acts } = await res.model.forward(emb, ids.length, { kind: 'layers', layers });
    log(`forward ${ids.length} tokens in ${(performance.now() - t1).toFixed(0)} ms`);
    // a second batched run: same text twice plus padding, to check batching
    const B = 3;
    const T = ids.length + 2;
    const batch: number[] = [];
    for (let b = 0; b < B; b++) batch.push(...ids, ...Array(T - ids.length).fill(248044));
    const t2 = performance.now();
    // batching/padding check up to the first full-attention layer in the list (cheap)
    const bl = layers.find((l) => (l + 1) % 4 === 0) ?? layers[0];
    const { acts: actsB } = await res.model.forward(dequant(14, res.rows.read(batch)), T, { kind: 'neurons', layer: bl, idx: [0, 1, 2] });
    log(`batched forward ${batch.length} tokens in ${(performance.now() - t2).toFixed(0)} ms`);
    (window as any).__result = { ids, layers, acts: Array.from(acts), batched: Array.from(actsB), batchedLayer: bl, T, B };
    log('DONE');
  } catch (e) {
    log('FAIL ' + (e as Error).stack);
    (window as any).__result = { error: String(e) };
  }
})();
