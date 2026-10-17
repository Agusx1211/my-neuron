// Qwen3.5 text decoder on WebGPU: weight registry + batched prefill forward pass
// that exposes every MLP neuron (silu(gate) * up).

import {
  attentionWGSL, convSiluWGSL, gatedNormWGSL, gatherWGSL, gdnScanWGSL, l2normWGSL,
  matmulWGSL, qkNormRopeWGSL, rmsnormWGSL, swigluWGSL, type WFormat,
} from './kernels';

export interface ModelConfig {
  nLayer: number;
  d: number;
  dFF: number;
  nHead: number;
  nKV: number;
  headDim: number;
  ropeDim: number;
  ropeBase: number;
  eps: number;
  convK: number;
  dk: number;   // linear-attention key head dim
  nKH: number;  // linear-attention key heads
  nVH: number;  // linear-attention value heads
  dv: number;   // linear-attention value head dim
  interval: number; // every interval-th layer is full attention
}

export function configFromMeta(meta: Record<string, unknown>): ModelConfig {
  const a = meta['general.architecture'] as string;
  const g = (k: string) => meta[`${a}.${k}`] as number;
  if (a !== 'qwen35') throw new Error(`unsupported architecture ${a}`);
  const nVH = g('ssm.time_step_rank');
  return {
    nLayer: g('block_count'),
    d: g('embedding_length'),
    dFF: g('feed_forward_length'),
    nHead: g('attention.head_count'),
    nKV: g('attention.head_count_kv'),
    headDim: g('attention.key_length'),
    ropeDim: g('rope.dimension_count'),
    ropeBase: g('rope.freq_base'),
    eps: g('attention.layer_norm_rms_epsilon'),
    convK: g('ssm.conv_kernel'),
    dk: g('ssm.state_size'),
    nKH: g('ssm.group_count'),
    nVH,
    dv: g('ssm.inner_size') / nVH,
    interval: g('full_attention_interval'),
  };
}

export const isFull = (cfg: ModelConfig, layer: number) => (layer + 1) % cfg.interval === 0;

interface Weight { buf: GPUBuffer; fmt: WFormat; rows: number; cols: number }

/** What to read back from a forward pass. */
export type Capture =
  | { kind: 'all' }                                   // every neuron of every layer, all tokens
  | { kind: 'neurons'; layer: number; idx: number[] } // a few neurons of one layer (early exit)
  | { kind: 'layers'; layers: number[] };             // all neurons of selected layers

const SLOT = 256;

export class GpuModel {
  readonly cfg: ModelConfig;
  private device: GPUDevice;
  private w = new Map<string, Weight>();
  private pipes = new Map<string, GPUComputePipeline>();
  private cap = 0; // token capacity of the activation buffers
  private bufs: Record<string, GPUBuffer> = {};
  private uni!: GPUBuffer;
  private uniData = new ArrayBuffer(SLOT * 1024);
  private uniCount = 0;
  private pass!: GPUComputePassEncoder;

  constructor(device: GPUDevice, cfg: ModelConfig) {
    this.device = device;
    this.cfg = cfg;
  }

  async compile(): Promise<void> {
    const make = async (name: string, code: string) => {
      const module = this.device.createShaderModule({ code, label: name });
      this.pipes.set(name, await this.device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' }, label: name }));
    };
    const jobs: Promise<void>[] = [];
    for (const f of ['q4k', 'q5k', 'q6k', 'f32'] as WFormat[]) {
      jobs.push(make(`mm_${f}`, matmulWGSL(f, false)), make(`mma_${f}`, matmulWGSL(f, true)));
    }
    jobs.push(
      make('rms', rmsnormWGSL(false)), make('rms_ip', rmsnormWGSL(true)), make('swiglu', swigluWGSL),
      make('gather', gatherWGSL), make('qknorm', qkNormRopeWGSL), make('attn', attentionWGSL),
      make('conv', convSiluWGSL), make('l2', l2normWGSL), make('gdn', gdnScanWGSL), make('gnorm', gatedNormWGSL),
    );
    await Promise.all(jobs);
    this.uni = this.device.createBuffer({ size: this.uniData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  /** Register a tensor already converted to a GPU format (see loader). */
  addWeight(name: string, data: Uint8Array, fmt: WFormat, rows: number, cols: number) {
    const size = Math.ceil(data.byteLength / 4) * 4;
    const buf = this.device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: name });
    if (data.byteLength % 4 === 0) this.device.queue.writeBuffer(buf, 0, data);
    else { const p = new Uint8Array(size); p.set(data); this.device.queue.writeBuffer(buf, 0, p); }
    this.w.set(name, { buf, fmt, rows, cols });
  }

  has(name: string) { return this.w.has(name); }

  private W(name: string): Weight {
    const x = this.w.get(name);
    if (!x) throw new Error(`missing weight ${name}`);
    return x;
  }

  /**
   * Allocate the activation buffers once, for batches of up to `tokens`. They
   * are never reallocated afterwards: swapping buffers between passes produced
   * all-zero readbacks on some WebGPU stacks.
   */
  reserve(tokens: number) {
    if (this.cap) throw new Error('activation buffers already reserved');
    const c = this.cfg;
    const cap = Math.ceil(tokens / 64) * 64;
    const f = (count: number, label: string, extra = 0) =>
      this.device.createBuffer({ size: count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | extra, label });
    const conv = 2 * c.nKH * c.dk + c.nVH * c.dv;
    this.bufs = {
      x: f(cap * c.d, 'x'),
      h: f(cap * c.d, 'h'),
      big: f(cap * Math.max(conv, 2 * c.nHead * c.headDim), 'big'),
      conv: f(cap * conv, 'conv'),
      z: f(cap * Math.max(c.nVH * c.dv, c.nHead * c.headDim), 'z'),
      k: f(cap * c.nKV * c.headDim, 'k'),
      v: f(cap * c.nKV * c.headDim, 'v'),
      beta: f(cap * c.nVH, 'beta'),
      alpha: f(cap * c.nVH, 'alpha'),
      g: f(cap * c.dFF, 'gate'),
      u: f(cap * c.dFF, 'up'),
    };
    this.cap = cap;
  }

  // --- encoding helpers -----------------------------------------------------
  private slot(values: (number | { f: number })[]): number {
    if (this.uniCount >= this.uniData.byteLength / SLOT) throw new Error('uniform arena full');
    const off = this.uniCount++ * SLOT;
    const u = new Uint32Array(this.uniData, off, 16);
    const fl = new Float32Array(this.uniData, off, 16);
    values.forEach((v, i) => { if (typeof v === 'number') u[i] = v; else fl[i] = v.f; });
    return off;
  }

  private run(pipe: string, params: (number | { f: number })[], buffers: GPUBuffer[], x: number, y = 1) {
    const p = this.pipes.get(pipe)!;
    const off = this.slot(params);
    const entries: GPUBindGroupEntry[] = [{ binding: 0, resource: { buffer: this.uni, offset: off, size: 64 } }];
    buffers.forEach((b, i) => entries.push({ binding: i + 1, resource: { buffer: b } }));
    this.pass.setPipeline(p);
    this.pass.setBindGroup(0, this.device.createBindGroup({ layout: p.getBindGroupLayout(0), entries }));
    this.pass.dispatchWorkgroups(x, y);
  }

  private grid(threads: number): [number, number] {
    const wg = Math.ceil(threads / 256);
    return wg <= 65535 ? [wg, 1] : [65535, Math.ceil(wg / 65535)];
  }

  private matmul(name: string, X: GPUBuffer, Y: GPUBuffer, n: number, acc = false) {
    const w = this.W(name);
    this.run(`${acc ? 'mma' : 'mm'}_${w.fmt}`, [n, w.rows, w.cols, 0], [X, w.buf, Y], Math.ceil(w.rows / 64), Math.ceil(n / 64));
  }

  private rms(X: GPUBuffer, wname: string, rows: number, len: number, H: number, SN: number, SH: number, Y?: GPUBuffer) {
    if (rows > 65535) throw new Error('too many rows for rmsnorm');
    const params = [len, H, SN, SH, { f: this.cfg.eps }, 0, 0, 0];
    if (Y) this.run('rms', params, [X, this.W(wname).buf, Y], rows);
    else this.run('rms_ip', params, [X, this.W(wname).buf], rows);
  }

  /**
   * Run a padded batch: `embeds` holds n = B*T token embeddings (B sequences of
   * T tokens each, right-padded). Returns the requested neuron activations:
   *  - all:     [layers][n][dFF]
   *  - layers:  [layers.length][n][dFF]
   *  - neurons: [idx.length][n]
   * plus, when residualAt >= 0, the residual stream [n][d] after that layer.
   */
  async forward(embeds: Float32Array, T: number, capture: Capture, residualAt = -1): Promise<{ acts: Float32Array; resid: Float32Array | null }> {
    const c = this.cfg;
    const n = embeds.length / c.d;
    if (!Number.isInteger(n) || n % T !== 0) throw new Error('bad batch shape');
    if (n * Math.max(2 * c.nKH, c.nVH, c.nHead) > 65535) throw new Error('batch too large');
    if (T > 1024) throw new Error('sequence too long');
    if (n > this.cap) throw new Error(`batch of ${n} tokens exceeds the reserved ${this.cap}`);
    const B = this.bufs;
    this.device.queue.writeBuffer(B.x, 0, embeds);

    const lastLayer = Math.max(residualAt, capture.kind === 'neurons' ? capture.layer
      : capture.kind === 'layers' ? Math.max(...capture.layers) : c.nLayer - 1);
    const capLayers = capture.kind === 'all' ? [...Array(c.nLayer).keys()]
      : capture.kind === 'layers' ? capture.layers : [];
    const capFloats = capture.kind === 'neurons' ? capture.idx.length * n : capLayers.length * n * c.dFF;
    const capBuf = this.device.createBuffer({ size: Math.max(16, capFloats * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const residBytes = residualAt >= 0 ? n * c.d * 4 : 0;
    const staging = this.device.createBuffer({ size: Math.max(16, capFloats * 4 + residBytes), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    // surface GPU validation / OOM errors instead of silently reading back zeros
    this.device.pushErrorScope('validation');
    this.device.pushErrorScope('out-of-memory');
    this.uniCount = 0;
    const enc = this.device.createCommandEncoder();
    this.pass = enc.beginComputePass();
    const conv = 2 * c.nKH * c.dk + c.nVH * c.dv;

    for (let L = 0; L <= lastLayer; L++) {
      const p = `blk.${L}.`;
      this.rms(B.x, p + 'attn_norm.weight', n, c.d, 1, c.d, 0, B.h);
      if (isFull(c, L)) {
        const qg = 2 * c.nHead * c.headDim;
        this.matmul(p + 'attn_q.weight', B.h, B.big, n);
        this.matmul(p + 'attn_k.weight', B.h, B.k, n);
        this.matmul(p + 'attn_v.weight', B.h, B.v, n);
        const qn = [c.nHead, qg, 2 * c.headDim, T, c.headDim, c.ropeDim, { f: c.eps }, { f: c.ropeBase }];
        this.run('qknorm', qn, [B.big, this.W(p + 'attn_q_norm.weight').buf], n * c.nHead);
        const kn = [c.nKV, c.nKV * c.headDim, c.headDim, T, c.headDim, c.ropeDim, { f: c.eps }, { f: c.ropeBase }];
        this.run('qknorm', kn, [B.k, this.W(p + 'attn_k_norm.weight').buf], n * c.nKV);
        this.run('attn', [c.nHead, c.nKV, c.headDim, T, { f: 1 / Math.sqrt(c.headDim) }, 0, 0, 0], [B.big, B.k, B.v, B.z], n * c.nHead);
        this.matmul(p + 'attn_output.weight', B.z, B.x, n, true);
      } else {
        this.matmul(p + 'attn_qkv.weight', B.h, B.big, n);
        this.matmul(p + 'attn_gate.weight', B.h, B.z, n);
        this.matmul(p + 'ssm_beta.weight', B.h, B.beta, n);
        this.matmul(p + 'ssm_alpha.weight', B.h, B.alpha, n);
        this.run('conv', [n, conv, T, c.convK], [B.big, this.W(p + 'ssm_conv1d.weight').buf, B.conv], ...this.grid(n * conv));
        this.run('l2', [c.nKH, c.dk, conv, 0], [B.conv], n * 2 * c.nKH);
        this.run('gdn', [c.nKH, c.nVH, conv, T], [B.conv, B.beta, B.alpha, this.W(p + 'ssm_a').buf, this.W(p + 'ssm_dt.bias').buf, B.big], (n / T) * c.nVH);
        this.run('gnorm', [c.dv, { f: c.eps }, 0, 0], [B.big, B.z, this.W(p + 'ssm_norm.weight').buf], n * c.nVH);
        this.matmul(p + 'ssm_out.weight', B.big, B.x, n, true);
      }
      this.rms(B.x, p + 'post_attention_norm.weight', n, c.d, 1, c.d, 0, B.h);
      this.matmul(p + 'ffn_gate.weight', B.h, B.g, n);
      this.matmul(p + 'ffn_up.weight', B.h, B.u, n);
      const ci = capLayers.indexOf(L);
      this.run('swiglu', [n * c.dFF, ci >= 0 ? ci * n * c.dFF : 0, ci >= 0 ? 1 : 0, 0], [B.g, B.u, capBuf], ...this.grid(n * c.dFF));
      if (capture.kind === 'neurons' && L === capture.layer) {
        capture.idx.forEach((idx, j) => this.run('gather', [n, c.dFF, idx, j * n], [B.g, capBuf], Math.ceil(n / 256)));
      }
      if (L < lastLayer || L === residualAt) this.matmul(p + 'ffn_down.weight', B.g, B.x, n, true);
      if (L === residualAt) {
        this.pass.end();
        enc.copyBufferToBuffer(B.x, 0, staging, capFloats * 4, residBytes);
        this.pass = enc.beginComputePass();
      }
    }
    this.pass.end();
    this.device.queue.writeBuffer(this.uni, 0, this.uniData, 0, this.uniCount * SLOT);
    enc.copyBufferToBuffer(capBuf, 0, staging, 0, Math.max(16, capFloats * 4));
    this.device.queue.submit([enc.finish()]);
    const oom = await this.device.popErrorScope();
    const invalid = await this.device.popErrorScope();
    if (oom || invalid) {
      staging.destroy();
      capBuf.destroy();
      throw new Error(`GPU error in forward pass: ${(oom ?? invalid)!.message}`);
    }
    await staging.mapAsync(GPUMapMode.READ);
    const range = staging.getMappedRange();
    const acts = new Float32Array(range.slice(0, capFloats * 4));
    const resid = residBytes ? new Float32Array(range.slice(capFloats * 4, capFloats * 4 + residBytes)) : null;
    staging.unmap();
    staging.destroy();
    capBuf.destroy();
    return { acts, resid };
  }
}
