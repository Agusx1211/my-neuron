// WGSL compute kernels for the Qwen3.5 forward pass (prefill only, f32 activations).
//
// Weight formats on the GPU (all u32-aligned):
//   Q4_K  144-byte blocks of 256 (native ggml layout)
//   Q5_K  176-byte blocks of 256 (native)
//   Q6_K  210-byte blocks padded to 212 bytes (repacked in WASM)
//   F32   plain floats
// Every activation tensor is row-major [tokens, features].

export type WFormat = 'q4k' | 'q5k' | 'q6k' | 'f32';

// --- quantized-weight GEMM -------------------------------------------------
// Y[n, m] (+)= sum_k X[n, k] * W[m, k]. 64x64 output tile, 256 threads, K step 32.
// Each thread dequantizes 8 consecutive weights of one row into shared memory.

const DEQ: Record<WFormat, string> = {
  // Q4_K: d,dmin | scales[12] | qs[128]
  q4k: /* wgsl */ `
const BLOCK_U32 = 36u;
fn sbyte(b: u32, i: u32) -> u32 { return (W[b + (i >> 2u)] >> ((i & 3u) * 8u)) & 0xffu; }
fn scale_min(b: u32, j: u32) -> vec2<f32> {
  // scales live at byte 4..15 of the block
  let s = b + 1u;
  if (j < 4u) {
    return vec2<f32>(f32(sbyte(s, j) & 63u), f32(sbyte(s, j + 4u) & 63u));
  }
  let sc = (sbyte(s, j + 4u) & 0xfu) | ((sbyte(s, j - 4u) >> 6u) << 4u);
  let mn = (sbyte(s, j + 4u) >> 4u) | ((sbyte(s, j) >> 6u) << 4u);
  return vec2<f32>(f32(sc), f32(mn));
}
fn load8(row: u32, k: u32, out: ptr<function, array<f32, 8>>) {
  let b = (row * KB + (k >> 8u)) * BLOCK_U32;
  let kb = k & 255u;
  let j = kb >> 5u;
  let l = kb & 31u;
  let dm = unpack2x16float(W[b]);
  let sm = scale_min(b, j);
  let d = dm.x * sm.x;
  let m = dm.y * sm.y;
  let shift = (j & 1u) * 4u;
  let w0 = W[b + 4u + ((32u * (j >> 1u) + l) >> 2u)];
  let w1 = W[b + 4u + ((32u * (j >> 1u) + l) >> 2u) + 1u];
  for (var e = 0u; e < 4u; e++) {
    (*out)[e] = d * f32((w0 >> (e * 8u + shift)) & 0xfu) - m;
    (*out)[e + 4u] = d * f32((w1 >> (e * 8u + shift)) & 0xfu) - m;
  }
}`,
  // Q5_K: d,dmin | scales[12] | qh[32] | qs[128]
  q5k: /* wgsl */ `
const BLOCK_U32 = 44u;
fn sbyte(b: u32, i: u32) -> u32 { return (W[b + (i >> 2u)] >> ((i & 3u) * 8u)) & 0xffu; }
fn scale_min(b: u32, j: u32) -> vec2<f32> {
  let s = b + 1u;
  if (j < 4u) {
    return vec2<f32>(f32(sbyte(s, j) & 63u), f32(sbyte(s, j + 4u) & 63u));
  }
  let sc = (sbyte(s, j + 4u) & 0xfu) | ((sbyte(s, j - 4u) >> 6u) << 4u);
  let mn = (sbyte(s, j + 4u) >> 4u) | ((sbyte(s, j) >> 6u) << 4u);
  return vec2<f32>(f32(sc), f32(mn));
}
fn load8(row: u32, k: u32, out: ptr<function, array<f32, 8>>) {
  let b = (row * KB + (k >> 8u)) * BLOCK_U32;
  let kb = k & 255u;
  let j = kb >> 5u;
  let l = kb & 31u;
  let dm = unpack2x16float(W[b]);
  let sm = scale_min(b, j);
  let d = dm.x * sm.x;
  let m = dm.y * sm.y;
  let shift = (j & 1u) * 4u;
  let q0 = W[b + 12u + ((32u * (j >> 1u) + l) >> 2u)];
  let q1 = W[b + 12u + ((32u * (j >> 1u) + l) >> 2u) + 1u];
  let h0 = W[b + 4u + (l >> 2u)];
  let h1 = W[b + 4u + (l >> 2u) + 1u];
  for (var e = 0u; e < 4u; e++) {
    let hb0 = ((h0 >> (e * 8u + j)) & 1u) << 4u;
    let hb1 = ((h1 >> (e * 8u + j)) & 1u) << 4u;
    (*out)[e] = d * f32(((q0 >> (e * 8u + shift)) & 0xfu) | hb0) - m;
    (*out)[e + 4u] = d * f32(((q1 >> (e * 8u + shift)) & 0xfu) | hb1) - m;
  }
}`,
  // Q6_K (padded to 212 bytes): ql[128] | qh[64] | scales[16] i8 | d f16 | pad
  q6k: /* wgsl */ `
const BLOCK_U32 = 53u;
fn load8(row: u32, k: u32, out: ptr<function, array<f32, 8>>) {
  let b = (row * KB + (k >> 8u)) * BLOCK_U32;
  let kb = k & 255u;
  let n = kb >> 7u;
  let r = kb & 127u;
  let qd = r >> 5u;
  let l = r & 31u;
  let d = unpack2x16float(W[b + 52u]).x;
  let si = 8u * n + (l >> 4u) + 2u * qd;
  let sw = W[b + 48u + (si >> 2u)];
  let sc = f32(extractBits(bitcast<i32>(sw), (si & 3u) * 8u, 8u));
  let ds = d * sc;
  let qlb = 64u * n + l + (qd & 1u) * 32u;
  let nsh = (qd >> 1u) * 4u;
  let qhb = 128u + 32u * n + l;
  let hsh = 2u * qd;
  let l0 = W[b + (qlb >> 2u)];
  let l1 = W[b + (qlb >> 2u) + 1u];
  let g0 = W[b + (qhb >> 2u)];
  let g1 = W[b + (qhb >> 2u) + 1u];
  for (var e = 0u; e < 4u; e++) {
    let a0 = ((l0 >> (e * 8u + nsh)) & 0xfu) | (((g0 >> (e * 8u + hsh)) & 3u) << 4u);
    let a1 = ((l1 >> (e * 8u + nsh)) & 0xfu) | (((g1 >> (e * 8u + hsh)) & 3u) << 4u);
    (*out)[e] = ds * (f32(a0) - 32.0);
    (*out)[e + 4u] = ds * (f32(a1) - 32.0);
  }
}`,
  f32: /* wgsl */ `
fn load8(row: u32, k: u32, out: ptr<function, array<f32, 8>>) {
  let b = row * P.K + k;
  for (var e = 0u; e < 8u; e++) { (*out)[e] = bitcast<f32>(W[b + e]); }
}`,
};

export function matmulWGSL(fmt: WFormat, accumulate: boolean): string {
  return /* wgsl */ `
struct Params { N: u32, M: u32, K: u32, _p: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;

var<workgroup> Ws: array<f32, 2048>; // [32 k][64 m]
var<workgroup> Xs: array<f32, 2048>; // [32 k][64 n]

// override-free constant: blocks per row for 256-element block formats
var<private> KB: u32;
${DEQ[fmt]}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  KB = P.K >> 8u;
  let m0 = wg.x * 64u;
  let n0 = wg.y * 64u;
  let lr = t >> 2u;          // row within tile (0..63) for loads
  let lp = (t & 3u) * 8u;    // k offset within the 32-chunk
  let tm = t & 15u;
  let tn = t >> 4u;
  var acc: array<f32, 16>;
  for (var i = 0u; i < 16u; i++) { acc[i] = 0.0; }
  var w8: array<f32, 8>;
  for (var k0 = 0u; k0 < P.K; k0 += 32u) {
    let wr = m0 + lr;
    if (wr < P.M) {
      load8(wr, k0 + lp, &w8);
    } else {
      for (var e = 0u; e < 8u; e++) { w8[e] = 0.0; }
    }
    for (var e = 0u; e < 8u; e++) { Ws[(lp + e) * 64u + lr] = w8[e]; }
    let xn = n0 + lr;
    for (var e = 0u; e < 8u; e++) {
      Xs[(lp + e) * 64u + lr] = select(0.0, X[xn * P.K + k0 + lp + e], xn < P.N);
    }
    workgroupBarrier();
    for (var kk = 0u; kk < 32u; kk++) {
      let a = vec4<f32>(Ws[kk * 64u + tm], Ws[kk * 64u + tm + 16u], Ws[kk * 64u + tm + 32u], Ws[kk * 64u + tm + 48u]);
      let b = vec4<f32>(Xs[kk * 64u + tn], Xs[kk * 64u + tn + 16u], Xs[kk * 64u + tn + 32u], Xs[kk * 64u + tn + 48u]);
      for (var j = 0u; j < 4u; j++) {
        acc[j * 4u + 0u] += a.x * b[j];
        acc[j * 4u + 1u] += a.y * b[j];
        acc[j * 4u + 2u] += a.z * b[j];
        acc[j * 4u + 3u] += a.w * b[j];
      }
    }
    workgroupBarrier();
  }
  for (var j = 0u; j < 4u; j++) {
    let n = n0 + tn + 16u * j;
    if (n >= P.N) { continue; }
    for (var i = 0u; i < 4u; i++) {
      let m = m0 + tm + 16u * i;
      if (m >= P.M) { continue; }
      ${accumulate ? 'Y[n * P.M + m] += acc[j * 4u + i];' : 'Y[n * P.M + m] = acc[j * 4u + i];'}
    }
  }
}`;
}

// --- RMSNorm ---------------------------------------------------------------
// Rows of length LEN located at base(r) = (r / H) * SN + (r % H) * SH.
// Out-of-place variant writes a dense [rows, LEN] output; in-place rewrites X.
export function rmsnormWGSL(inplace: boolean): string {
  return /* wgsl */ `
struct Params { LEN: u32, H: u32, SN: u32, SH: u32, eps: f32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, ${inplace ? 'read_write' : 'read'}> X: array<f32>;
@group(0) @binding(2) var<storage, read> Wt: array<f32>;
${inplace ? '' : '@group(0) @binding(3) var<storage, read_write> Y: array<f32>;'}
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let r = wg.x + wg.y * 65535u;
  let base = (r / P.H) * P.SN + (r % P.H) * P.SH;
  var s = 0.0;
  for (var i = t; i < P.LEN; i += 256u) { let v = X[base + i]; s += v * v; }
  red[t] = s;
  workgroupBarrier();
  for (var w = 128u; w > 0u; w >>= 1u) {
    if (t < w) { red[t] += red[t + w]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(red[0] / f32(P.LEN) + P.eps);
  for (var i = t; i < P.LEN; i += 256u) {
    ${inplace ? 'X[base + i] = X[base + i] * inv * Wt[i];' : 'Y[r * P.LEN + i] = X[base + i] * inv * Wt[i];'}
  }
}`;
}

// --- SwiGLU: G <- silu(G) * U, optionally captured --------------------------
export const swigluWGSL = /* wgsl */ `
struct Params { total: u32, capOffset: u32, capture: u32, _p: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> G: array<f32>;
@group(0) @binding(2) var<storage, read> U: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let i = id.x + id.y * nw.x * 256u;
  if (i >= P.total) { return; }
  let g = G[i];
  let a = g / (1.0 + exp(-g)) * U[i];
  G[i] = a;
  if (P.capture != 0u) { C[P.capOffset + i] = a; }
}`;

// --- gather one neuron's activation for every token ------------------------
export const gatherWGSL = /* wgsl */ `
struct Params { N: u32, F: u32, idx: u32, outOffset: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read_write> O: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= P.N) { return; }
  O[P.outOffset + id.x] = A[id.x * P.F + P.idx];
}`;

// --- full attention: per-head RMSNorm + partial NeoX RoPE, in place ---------
// Rows are heads of HD dims at base = n * SN + h * SH; position = n % T.
export const qkNormRopeWGSL = /* wgsl */ `
struct Params { H: u32, SN: u32, SH: u32, T: u32, HD: u32, ROT: u32, eps: f32, theta: f32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> X: array<f32>;
@group(0) @binding(2) var<storage, read> Wt: array<f32>;
var<workgroup> red: array<f32, 64>;
var<workgroup> v: array<f32, 512>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let r = wg.x + wg.y * 65535u;
  let n = r / P.H;
  let base = n * P.SN + (r % P.H) * P.SH;
  var s = 0.0;
  for (var i = t; i < P.HD; i += 64u) { let x = X[base + i]; v[i] = x; s += x * x; }
  red[t] = s;
  workgroupBarrier();
  for (var w = 32u; w > 0u; w >>= 1u) {
    if (t < w) { red[t] += red[t + w]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(red[0] / f32(P.HD) + P.eps);
  for (var i = t; i < P.HD; i += 64u) { v[i] = v[i] * inv * Wt[i]; }
  workgroupBarrier();
  let pos = f32(n % P.T);
  let half = P.ROT / 2u;
  for (var i = t; i < P.HD; i += 64u) {
    var o = v[i];
    if (i < P.ROT) {
      let fi = f32(i % half);
      let ang = pos * pow(P.theta, -2.0 * fi / f32(P.ROT));
      let c = cos(ang);
      let sn = sin(ang);
      if (i < half) { o = v[i] * c - v[i + half] * sn; } else { o = v[i] * c + v[i - half] * sn; }
    }
    X[base + i] = o;
  }
}`;

// --- causal softmax attention with GQA and sigmoid output gate --------------
// QG: [N, H, 2*HD] (query | gate per head), K/V: [N, KVH, HD]; O: [N, H*HD].
export const attentionWGSL = /* wgsl */ `
struct Params { H: u32, KVH: u32, HD: u32, T: u32, scale: f32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> QG: array<f32>;
@group(0) @binding(2) var<storage, read> Kb: array<f32>;
@group(0) @binding(3) var<storage, read> Vb: array<f32>;
@group(0) @binding(4) var<storage, read_write> O: array<f32>;
var<workgroup> q: array<f32, 256>;
var<workgroup> sc: array<f32, 1024>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let r = wg.x + wg.y * 65535u;
  let n = r / P.H;
  let h = r % P.H;
  let kvh = h / (P.H / P.KVH);
  let pos = n % P.T;
  let s0 = n - pos;
  let qb = (n * P.H + h) * 2u * P.HD;
  for (var i = t; i < P.HD; i += 64u) { q[i] = QG[qb + i]; }
  workgroupBarrier();
  var mx = -1e30;
  for (var j = t; j <= pos; j += 64u) {
    let kb = ((s0 + j) * P.KVH + kvh) * P.HD;
    var d = 0.0;
    for (var i = 0u; i < P.HD; i++) { d += q[i] * Kb[kb + i]; }
    d *= P.scale;
    sc[j] = d;
    mx = max(mx, d);
  }
  red[t] = mx;
  workgroupBarrier();
  for (var w = 32u; w > 0u; w >>= 1u) {
    if (t < w) { red[t] = max(red[t], red[t + w]); }
    workgroupBarrier();
  }
  let gmx = red[0];
  workgroupBarrier();
  var sum = 0.0;
  for (var j = t; j <= pos; j += 64u) { let e = exp(sc[j] - gmx); sc[j] = e; sum += e; }
  red[t] = sum;
  workgroupBarrier();
  for (var w = 32u; w > 0u; w >>= 1u) {
    if (t < w) { red[t] += red[t + w]; }
    workgroupBarrier();
  }
  let inv = 1.0 / red[0];
  for (var i = t; i < P.HD; i += 64u) {
    var o = 0.0;
    for (var j = 0u; j <= pos; j++) { o += sc[j] * Vb[((s0 + j) * P.KVH + kvh) * P.HD + i]; }
    let g = QG[qb + P.HD + i];
    O[(n * P.H + h) * P.HD + i] = o * inv / (1.0 + exp(-g));
  }
}`;

// --- Gated DeltaNet ----------------------------------------------------------
// Causal depthwise conv (kernel CK) over each sequence + SiLU. X, Y: [N, C]; Wc: [C, CK].
export const convSiluWGSL = /* wgsl */ `
struct Params { N: u32, C: u32, T: u32, CK: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> Wc: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let i = id.x + id.y * nw.x * 256u;
  if (i >= P.N * P.C) { return; }
  let n = i / P.C;
  let c = i % P.C;
  let pos = n % P.T;
  var s = 0.0;
  for (var k = 0u; k < P.CK; k++) {
    let back = P.CK - 1u - k;
    if (back <= pos) { s += Wc[c * P.CK + k] * X[(n - back) * P.C + c]; }
  }
  Y[i] = s / (1.0 + exp(-s));
}`;

// L2-normalize q and k heads in place (q additionally scaled by 1/sqrt(DK)).
// Row r covers token n = r / (2*NKH); first NKH rows of a token are q heads.
export const l2normWGSL = /* wgsl */ `
struct Params { NKH: u32, DK: u32, C: u32, _p: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> X: array<f32>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let r = wg.x + wg.y * 65535u;
  let n = r / (2u * P.NKH);
  let hh = r % (2u * P.NKH);
  let base = n * P.C + hh * P.DK;
  var s = 0.0;
  for (var i = t; i < P.DK; i += 64u) { let v = X[base + i]; s += v * v; }
  red[t] = s;
  workgroupBarrier();
  for (var w = 32u; w > 0u; w >>= 1u) {
    if (t < w) { red[t] += red[t + w]; }
    workgroupBarrier();
  }
  var inv = inverseSqrt(red[0] + 1e-6);
  if (hh < P.NKH) { inv = inv / sqrt(f32(P.DK)); }
  for (var i = t; i < P.DK; i += 64u) { X[base + i] = X[base + i] * inv; }
}`;

// Recurrent delta rule. One workgroup per (sequence, value head); thread j owns
// column j of the [DK x DV] state. QKV: conv output [N, C] laid out
// q(NKH*DK) | k(NKH*DK) | v(NVH*DV); value head h reads key head h % NKH.
export const gdnScanWGSL = /* wgsl */ `
struct Params { NKH: u32, NVH: u32, C: u32, T: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> QKV: array<f32>;
@group(0) @binding(2) var<storage, read> Bt: array<f32>;   // [N, NVH] raw beta logits
@group(0) @binding(3) var<storage, read> Al: array<f32>;   // [N, NVH] raw alpha
@group(0) @binding(4) var<storage, read> A: array<f32>;    // [NVH] = -exp(A_log)
@group(0) @binding(5) var<storage, read> DT: array<f32>;   // [NVH] dt bias
@group(0) @binding(6) var<storage, read_write> O: array<f32>; // [N, NVH*128]
var<workgroup> kq: array<f32, 256>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) j: u32) {
  let seq = wg.x / P.NVH;
  let h = wg.x % P.NVH;
  let kh = h % P.NKH;
  let qoff = kh * 128u;
  let koff = P.NKH * 128u + kh * 128u;
  let voff = 2u * P.NKH * 128u + h * 128u;
  var S: array<f32, 128>;
  for (var i = 0u; i < 128u; i++) { S[i] = 0.0; }
  for (var t = 0u; t < P.T; t++) {
    let n = seq * P.T + t;
    let row = n * P.C;
    kq[j] = QKV[row + koff + j];
    kq[128u + j] = QKV[row + qoff + j];
    workgroupBarrier();
    let a = Al[n * P.NVH + h] + DT[h];
    let sp = select(log(1.0 + exp(a)), a, a > 20.0);
    let decay = exp(A[h] * sp);
    let beta = 1.0 / (1.0 + exp(-Bt[n * P.NVH + h]));
    var kv = 0.0;
    for (var i = 0u; i < 128u; i++) { S[i] *= decay; kv += S[i] * kq[i]; }
    let delta = (QKV[row + voff + j] - kv) * beta;
    var o = 0.0;
    for (var i = 0u; i < 128u; i++) { S[i] += kq[i] * delta; o += S[i] * kq[128u + i]; }
    O[n * P.NVH * 128u + h * 128u + j] = o;
    workgroupBarrier();
  }
}`;

// Gated RMSNorm over each value head: O <- rms(O) * w * silu(Z), in place.
export const gatedNormWGSL = /* wgsl */ `
struct Params { DV: u32, eps: f32, _a: u32, _b: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> O: array<f32>;
@group(0) @binding(2) var<storage, read> Z: array<f32>;
@group(0) @binding(3) var<storage, read> Wt: array<f32>;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let r = wg.x + wg.y * 65535u;
  let base = r * P.DV;
  var s = 0.0;
  for (var i = t; i < P.DV; i += 128u) { let v = O[base + i]; s += v * v; }
  red[t] = s;
  workgroupBarrier();
  for (var w = 64u; w > 0u; w >>= 1u) {
    if (t < w) { red[t] += red[t + w]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(red[0] / f32(P.DV) + P.eps);
  for (var i = t; i < P.DV; i += 128u) {
    let z = Z[base + i];
    O[base + i] = O[base + i] * inv * Wt[i] * (z / (1.0 + exp(-z)));
  }
}`;
