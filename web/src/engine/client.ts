// Main-thread Engine backed by the inference worker.

import type { Analysis, Engine, LoadProgress, ModelInfo, ModelSpec, NeuronAtlas, ProbeResult } from './api';

type Pending = { resolve: (v: any) => void; reject: (e: Error) => void; onEvent?: (e: any) => void };

export class WorkerEngine implements Engine {
  private worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
  private next = 1;
  private pending = new Map<number, Pending>();

  constructor() {
    this.worker.onmessage = (e: MessageEvent) => {
      const { id, result, error, event } = e.data;
      const p = this.pending.get(id);
      if (!p) return;
      if (event !== undefined) { p.onEvent?.(event); return; }
      this.pending.delete(id);
      if (error !== undefined) p.reject(new Error(error));
      else p.resolve(result);
    };
    this.worker.onerror = (e) => {
      for (const p of this.pending.values()) p.reject(new Error(e.message || 'worker crashed'));
      this.pending.clear();
    };
  }

  private call<T>(op: string, args: unknown[], onEvent?: (e: any) => void): Promise<T> {
    const id = this.next++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onEvent });
      this.worker.postMessage({ id, op, args });
    });
  }

  init() { return this.call<{ webgpu: boolean; adapter?: string; reason?: string }>('init', []); }

  isCached(model: ModelSpec) { return this.call<boolean>('isCached', [model]); }

  async load(model: ModelSpec, onProgress: (p: LoadProgress) => void) {
    const atlas = new URL(model.atlas, new URL(import.meta.env.BASE_URL, location.href)).href.replace(/\/$/, '');
    // testing: `?gguf=<url>` loads the weights from another location (e.g. a local mirror)
    const override = new URLSearchParams(location.search).get('gguf');
    if (override) {
      const head = await fetch(override, { method: 'HEAD' });
      model = { ...model, id: `${model.id}-dev`, url: override, bytes: Number(head.headers.get('content-length')) };
    }
    return this.call<ModelInfo>('load', [model, atlas], onProgress);
  }

  analyze(text: string) { return this.call<Analysis>('analyze', [text]); }

  atlas(layer: number, index: number) { return this.call<NeuronAtlas>('atlas', [layer, index]); }

  probe(layer: number, index: number, pole: 1 | -1, texts: string[], onResult?: (r: ProbeResult) => void) {
    return this.call<ProbeResult[]>('probe', [layer, index, pole, texts], onResult);
  }
}
