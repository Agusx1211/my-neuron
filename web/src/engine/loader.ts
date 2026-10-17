// Streams a GGUF file (OPFS cache or network, resumable), converts each tensor
// to its GPU format in WASM and uploads it as soon as its bytes arrive.

import { dequant, padBlocks, parseHeader } from '../wasm-pkg/neuron_core.js';
import type { WFormat } from './kernels';
import type { GpuModel } from './model';

export interface TensorInfo { name: string; dims: number[]; type: number; offset: number; nbytes: number }
export interface Header { dataStart: number; meta: Record<string, unknown>; tensors: TensorInfo[]; bytes: Uint8Array }

const T = { F32: 0, Q8_0: 8, Q4_K: 12, Q5_K: 13, Q6_K: 14 };
const CHUNK = 32 << 20;

/** Random access to embedding rows (Q6_K) without holding them on the GPU. */
export interface RowSource {
  rowBytes: number;
  read(ids: ArrayLike<number>): Uint8Array; // concatenated raw rows
}

// --- OPFS cache -------------------------------------------------------------

type SyncHandle = {
  read(b: Uint8Array, o: { at: number }): number;
  write(b: Uint8Array, o: { at: number }): number;
  getSize(): number;
  truncate(n: number): void;
  flush(): void;
  close(): void;
};

async function opfsDir(): Promise<FileSystemDirectoryHandle | null> {
  try {
    const root = await navigator.storage.getDirectory();
    return await root.getDirectoryHandle('models', { create: true });
  } catch {
    return null;
  }
}

async function openSync(dir: FileSystemDirectoryHandle, name: string, mode: 'readwrite' | 'read-only'): Promise<SyncHandle> {
  const fh = await dir.getFileHandle(name, { create: true });
  const open = (fh as unknown as { createSyncAccessHandle(o?: unknown): Promise<SyncHandle> }).createSyncAccessHandle.bind(fh);
  try {
    return await open({ mode });
  } catch {
    return await open(); // older engines: exclusive handle only
  }
}

export async function isCached(id: string, bytes: number): Promise<boolean> {
  const dir = await opfsDir();
  if (!dir) return false;
  try {
    await dir.getFileHandle(`${id}.done`);
    const f = await (await dir.getFileHandle(`${id}.gguf`)).getFile();
    return f.size === bytes;
  } catch {
    return false;
  }
}

// --- chunk sources ----------------------------------------------------------

interface Progress { loaded: number; total: number; fromCache: boolean }

async function* networkChunks(url: string, start: number, total: number): AsyncGenerator<Uint8Array> {
  let pos = start;
  let tries = 0;
  while (pos < total) {
    try {
      const res = await fetch(url, { headers: pos > 0 ? { Range: `bytes=${pos}-` } : {} });
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
      if (pos > 0 && res.status !== 206) throw new Error('server ignored range request');
      const reader = res.body.getReader();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        pos += value.byteLength;
        tries = 0;
        yield value;
      }
    } catch (e) {
      if (++tries > 6) throw e;
      await new Promise((r) => setTimeout(r, 1000 * tries));
    }
  }
}

/**
 * Produce the whole file as sequential chunks. Uses the OPFS copy when complete,
 * otherwise resumes the download from whatever prefix is cached, writing new
 * bytes through to OPFS. Returns the open handle for later embedding reads.
 */
async function openStream(id: string, url: string, total: number, onBytes: (p: Progress) => void) {
  const dir = await opfsDir();
  let handle: SyncHandle | null = null;
  let cachedBytes = 0;
  let complete = false;
  if (dir) {
    try {
      complete = await isCached(id, total);
      if (!complete) {
        // a partial download from an earlier visit already counts toward usage
        const partial = await dir.getFileHandle(`${id}.gguf`).then((f) => f.getFile()).then((f) => f.size).catch(() => 0);
        const est = await navigator.storage.estimate();
        if ((est.quota ?? 0) - (est.usage ?? 0) < (total - partial) * 1.05) throw new Error('not enough storage quota');
        await navigator.storage.persist?.();
      }
      handle = await openSync(dir, `${id}.gguf`, complete ? 'read-only' : 'readwrite');
      cachedBytes = complete ? total : Math.min(handle.getSize(), total);
    } catch (e) {
      console.warn('OPFS cache unavailable, streaming without cache:', e);
      handle?.close();
      handle = null;
      cachedBytes = 0;
    }
  }
  let loaded = 0;
  let writesOk = true;
  async function* chunks(): AsyncGenerator<Uint8Array> {
    // cached prefix
    while (handle && loaded < cachedBytes) {
      const n = Math.min(CHUNK, cachedBytes - loaded);
      const b = new Uint8Array(n);
      handle.read(b, { at: loaded });
      loaded += n;
      onBytes({ loaded, total, fromCache: true });
      yield b;
      await new Promise((r) => setTimeout(r, 0));
    }
    if (loaded < total) {
      for await (const b of networkChunks(url, loaded, total)) {
        if (handle) {
          try { handle.write(b, { at: loaded }); } catch (e) {
            // e.g. quota in private windows: keep streaming, just without the cache
            console.warn('OPFS write failed, continuing uncached', e);
            writesOk = false;
            handle.close();
            handle = null;
          }
        }
        loaded += b.byteLength;
        onBytes({ loaded, total, fromCache: false });
        yield b;
      }
    }
    if (handle && dir && !complete) {
      try {
        handle.flush();
        await dir.getFileHandle(`${id}.done`, { create: true });
      } catch (e) {
        // the bytes are all there; without the marker the next visit re-validates from the file
        console.warn('could not mark the model cache complete', e);
      }
    }
  }
  return {
    chunks,
    /** Open cache handle that holds the whole file (null when streaming uncached). */
    getHandle: () => (handle && writesOk ? handle : null),
    /** True when the file came entirely from a complete cache. */
    fromCompleteCache: complete,
    cachedBytes,
  };
}

// --- tensor conversion ---------------------------------------------------------

function toGpu(t: TensorInfo, bytes: Uint8Array): { data: Uint8Array; fmt: WFormat; rows: number; cols: number } {
  const cols = t.dims[0];
  const rows = t.dims.length > 1 ? t.dims.slice(1).reduce((a, b) => a * b, 1) : 1;
  switch (t.type) {
    case T.F32: return { data: bytes, fmt: 'f32', rows, cols };
    case T.Q4_K: return { data: bytes, fmt: 'q4k', rows, cols };
    case T.Q5_K: return { data: bytes, fmt: 'q5k', rows, cols };
    case T.Q6_K: return { data: padBlocks(bytes, 210, 212), fmt: 'q6k', rows, cols };
    case T.Q8_0: {
      const f = dequant(T.Q8_0, bytes);
      return { data: new Uint8Array(f.buffer, f.byteOffset, f.byteLength), fmt: 'f32', rows, cols };
    }
  }
  throw new Error(`unsupported tensor type ${t.type} (${t.name})`);
}

/**
 * Stream the model, building it as tensors arrive. `makeModel` is called once
 * the header is parsed so the caller can size the GPU model.
 */
export async function loadModel(
  id: string, url: string, total: number,
  makeModel: (h: Header) => Promise<GpuModel>,
  onProgress: (loaded: number, total: number, layersReady: number, fromCache: boolean) => void,
): Promise<{ header: Header; model: GpuModel; rows: RowSource }> {
  let lastReport = 0;
  let layersReady = 0;
  let fromCache = false;
  let loaded = 0;
  const report = (force = false) => {
    const now = performance.now();
    if (force || now - lastReport > 80) { lastReport = now; onProgress(loaded, total, layersReady, fromCache); }
  };
  const stream = await openStream(id, url, total, (p) => { loaded = p.loaded; fromCache = p.fromCache; report(); });

  let headBuf = new Uint8Array(0);
  let header: Header | null = null;
  let model: GpuModel | null = null;
  let order: TensorInfo[] = [];
  let embed: TensorInfo | undefined;
  let embedMem: Uint8Array | null = null;
  const perLayer = new Map<number, number>();
  let next = 0;                         // index into order of the tensor being filled
  let pending: Uint8Array | null = null; // its bytes (null when skipping)
  let pos = 0;                          // file offset of the next incoming byte

  const finish = (t: TensorInfo, bytes: Uint8Array | null) => {
    if (t === embed) {
      if (bytes) embedMem = bytes;
    } else if (bytes && t.name !== 'output_norm.weight' && t.name !== 'output.weight') {
      const g = toGpu(t, bytes);
      model!.addWeight(t.name, g.data, g.fmt, g.rows, g.cols);
    }
    const m = /^blk\.(\d+)\./.exec(t.name);
    if (m) {
      perLayer.set(+m[1], perLayer.get(+m[1])! - 1);
      while (perLayer.get(layersReady) === 0) layersReady++;
    }
  };
  const begin = () => {
    const t = order[next];
    // the embedding table is read row by row from the cache file when the file is
    // already complete; during a download it is kept in memory in case caching fails
    pending = t && !(t === embed && stream.fromCompleteCache) ? new Uint8Array(t.nbytes) : null;
  };
  const consume = (chunk: Uint8Array) => {
    const cs = pos;
    const ce = pos + chunk.byteLength;
    while (next < order.length) {
      const t = order[next];
      const te = t.offset + t.nbytes;
      if (pending) {
        const a = Math.max(cs, t.offset);
        const b = Math.min(ce, te);
        if (b > a) pending.set(chunk.subarray(a - cs, b - cs), a - t.offset);
      }
      if (te > ce) break;
      finish(t, pending);
      next++;
      begin();
    }
    pos = ce;
    report();
  };

  for await (const chunk of stream.chunks()) {
    if (header) { consume(chunk); continue; }
    const nb = new Uint8Array(headBuf.byteLength + chunk.byteLength);
    nb.set(headBuf);
    nb.set(chunk, headBuf.byteLength);
    headBuf = nb;
    try {
      const h = JSON.parse(parseHeader(headBuf)) as Omit<Header, 'bytes'>;
      header = { ...h, bytes: headBuf.slice(0, h.dataStart) };
    } catch (e) {
      if (String((e as Error)?.message ?? e).includes('NEED_MORE')) continue;
      throw e;
    }
    model = await makeModel(header);
    order = header.tensors.slice().sort((a, b) => a.offset - b.offset);
    embed = order.find((t) => t.name === 'token_embd.weight');
    if (!embed) throw new Error('model has no token embedding table');
    for (const t of order) {
      const m = /^blk\.(\d+)\./.exec(t.name);
      if (m) perLayer.set(+m[1], (perLayer.get(+m[1]) ?? 0) + 1);
    }
    begin();
    const all = headBuf;
    headBuf = new Uint8Array(0);
    consume(all);
  }
  if (!header || !model) throw new Error('model file ended before the header');
  if (next < order.length) throw new Error(`model file truncated at tensor ${order[next].name}`);
  report(true);

  const e = embed!;
  const rowBytes = (e.dims[0] / 256) * 210;
  const handle = stream.getHandle();
  const mem = handle ? null : (embedMem as Uint8Array | null); // drop the RAM copy once the cache is good
  if (!handle && !mem) throw new Error('lost the embedding table; please reload');
  const rows: RowSource = {
    rowBytes,
    read(ids) {
      const out = new Uint8Array(ids.length * rowBytes);
      for (let i = 0; i < ids.length; i++) {
        const dst = out.subarray(i * rowBytes, (i + 1) * rowBytes);
        if (handle) handle.read(dst, { at: e.offset + ids[i] * rowBytes });
        else dst.set(mem!.subarray(ids[i] * rowBytes, (ids[i] + 1) * rowBytes));
      }
      return out;
    },
  };
  return { header, model, rows };
}
