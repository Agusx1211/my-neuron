//! CPU-side ggml block formats: dequantization (embeddings, small tensors,
//! test oracles) and repacking into 4-byte aligned blocks for WGSL.

use half::f16;

#[inline]
fn h(b: &[u8], o: usize) -> f32 {
    f16::from_le_bytes([b[o], b[o + 1]]).to_f32()
}

#[inline]
fn scale_min_k4(j: usize, q: &[u8]) -> (f32, f32) {
    if j < 4 {
        ((q[j] & 63) as f32, (q[j + 4] & 63) as f32)
    } else {
        (
            ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)) as f32,
            ((q[j + 4] >> 4) | ((q[j] >> 6) << 4)) as f32,
        )
    }
}

fn q4k(b: &[u8], y: &mut [f32]) {
    let (d, dmin) = (h(b, 0), h(b, 2));
    let sc = &b[4..16];
    let mut q = &b[16..144];
    for (n, j) in (0..256).step_by(64).enumerate() {
        let (s1, m1) = scale_min_k4(2 * n, sc);
        let (s2, m2) = scale_min_k4(2 * n + 1, sc);
        for l in 0..32 {
            y[j + l] = d * s1 * (q[l] & 0xF) as f32 - dmin * m1;
            y[j + 32 + l] = d * s2 * (q[l] >> 4) as f32 - dmin * m2;
        }
        q = &q[32..];
    }
}

fn q5k(b: &[u8], y: &mut [f32]) {
    let (d, dmin) = (h(b, 0), h(b, 2));
    let sc = &b[4..16];
    let qh = &b[16..48];
    let mut ql = &b[48..176];
    let (mut u1, mut u2) = (1u8, 2u8);
    for (n, j) in (0..256).step_by(64).enumerate() {
        let (s1, m1) = scale_min_k4(2 * n, sc);
        let (s2, m2) = scale_min_k4(2 * n + 1, sc);
        for l in 0..32 {
            let hi1 = if qh[l] & u1 != 0 { 16 } else { 0 };
            let hi2 = if qh[l] & u2 != 0 { 16 } else { 0 };
            y[j + l] = d * s1 * ((ql[l] & 0xF) + hi1) as f32 - dmin * m1;
            y[j + 32 + l] = d * s2 * ((ql[l] >> 4) + hi2) as f32 - dmin * m2;
        }
        ql = &ql[32..];
        u1 <<= 2;
        u2 <<= 2;
    }
}

fn q6k(b: &[u8], y: &mut [f32]) {
    let d = h(b, 208);
    for n in 0..2 {
        let ql = &b[64 * n..];
        let qh = &b[128 + 32 * n..];
        let sc = &b[192 + 8 * n..];
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
            let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
            let o = 128 * n + l;
            y[o] = d * (sc[is] as i8) as f32 * q1 as f32;
            y[o + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
            y[o + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
            y[o + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
        }
    }
}

fn q8_0(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0);
    for i in 0..32 {
        y[i] = d * (b[2 + i] as i8) as f32;
    }
}

/// Dequantize a contiguous run of blocks of ggml type `ty` into f32.
pub fn dequant(ty: u32, src: &[u8]) -> Result<Vec<f32>, String> {
    match ty {
        0 => Ok(src.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()),
        1 => Ok(src.chunks_exact(2).map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32()).collect()),
        30 => Ok(src.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect()),
        _ => {
            let (n, bytes, f): (usize, usize, fn(&[u8], &mut [f32])) = match ty {
                8 => (32, 34, q8_0),
                12 => (256, 144, q4k),
                13 => (256, 176, q5k),
                14 => (256, 210, q6k),
                _ => return Err(format!("dequant: unsupported ggml type {ty}")),
            };
            let blocks = src.len() / bytes;
            let mut y = vec![0f32; blocks * n];
            for (i, b) in src.chunks_exact(bytes).enumerate() {
                f(b, &mut y[i * n..(i + 1) * n]);
            }
            Ok(y)
        }
    }
}

/// Copy fixed-size blocks into a wider stride (e.g. Q6_K 210 -> 212 bytes) so
/// every block starts on a u32 boundary for the GPU.
pub fn pad_blocks(src: &[u8], from: usize, to: usize) -> Vec<u8> {
    let blocks = src.len() / from;
    let mut out = vec![0u8; blocks * to];
    for (i, b) in src.chunks_exact(from).enumerate() {
        out[i * to..i * to + from].copy_from_slice(b);
    }
    out
}
