//! Neuron scoring: span extrema over captured activations and z-scored ranking
//! against the per-neuron concept-atlas baseline.

use wasm_bindgen::prelude::*;

/// acts: [layers][tokens][ff]. Returns [layers][ff][2] = (max, min) over the
/// tokens where mask[t] != 0.
#[wasm_bindgen(js_name = spanExtrema)]
pub fn span_extrema(acts: &[f32], layers: usize, tokens: usize, ff: usize, mask: &[u8]) -> Vec<f32> {
    let mut out = vec![0f32; layers * ff * 2];
    for l in 0..layers {
        let o = &mut out[l * ff * 2..(l + 1) * ff * 2];
        for i in 0..ff {
            o[2 * i] = f32::NEG_INFINITY;
            o[2 * i + 1] = f32::INFINITY;
        }
        for t in (0..tokens).filter(|&t| mask.get(t).copied().unwrap_or(0) != 0) {
            let row = &acts[(l * tokens + t) * ff..(l * tokens + t + 1) * ff];
            for (i, &a) in row.iter().enumerate() {
                if a > o[2 * i] {
                    o[2 * i] = a;
                }
                if a < o[2 * i + 1] {
                    o[2 * i + 1] = a;
                }
            }
        }
    }
    out
}

/// Signed z-score of every neuron for the visualization: [layers][ff], positive
/// when the positive pole wins, negative when the negative pole does.
#[wasm_bindgen(js_name = heatMap)]
pub fn heat_map(ext: &[f32], stats: &[f32], cols: usize, ff: usize, std_floor_pos: &[f32], std_floor_neg: &[f32]) -> Vec<f32> {
    let n = ext.len() / 2;
    let mut out = vec![0f32; n];
    for j in 0..n {
        let l = j / ff;
        let s = &stats[j * cols..j * cols + 4];
        let zp = (ext[2 * j] - s[0]) / s[1].max(std_floor_pos.get(l).copied().unwrap_or(0.0)).max(1e-6);
        let zn = (s[2] - ext[2 * j + 1]) / s[3].max(std_floor_neg.get(l).copied().unwrap_or(0.0)).max(1e-6);
        out[j] = if zp >= zn { zp.max(0.0) } else { -zn.max(0.0) };
    }
    out
}

/// IEEE half floats (little endian) to f32.
#[wasm_bindgen(js_name = f16ToF32)]
pub fn f16_to_f32(src: &[u8]) -> Vec<f32> {
    src.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect()
}

/// Theme-aware neuron selection.
///
/// A neuron (pole) is a candidate when this input would enter its top list
/// (activation past the stored entry threshold) and its z-score is at least
/// `z_min`. Candidates are ranked by `z * max(rel, 0)^rel_pow`, where `rel` is
/// the mean cosine similarity between the input's semantic embedding `q` and
/// the embeddings of the neuron's top-8 atlas concepts, i.e. how much the
/// neuron's theme is about this input.
///
/// stats: [layers][ff][6] = mean_max, std_max, mean_min, std_min, entry_pos, entry_neg
/// top8:  [layers][ff][16] concept ids (8 positive, then 8 negative)
/// emb:   [concepts][dim] unit vectors; q: [dim] unit vector
/// Returns k rows of [layer, neuron, pole, score, z, act, rel].
#[wasm_bindgen(js_name = selectNeurons)]
#[allow(clippy::too_many_arguments)]
pub fn select_neurons(
    ext: &[f32], stats: &[f32], ff: usize, floor_pos: &[f32], floor_neg: &[f32],
    top8: &[u16], emb: &[f32], q: &[f32], min_layer: usize, z_min: f32, require_entry: bool,
    rel_pow: f32, k: usize,
) -> Vec<f32> {
    let dim = q.len();
    let csim: Vec<f32> = emb.chunks_exact(dim).map(|e| e.iter().zip(q).map(|(a, b)| a * b).sum()).collect();
    let layers = ext.len() / (ff * 2);
    let mut best: Vec<[f32; 7]> = Vec::with_capacity(k + 1);
    for l in min_layer..layers {
        let (fp, fneg) = (floor_pos.get(l).copied().unwrap_or(0.0), floor_neg.get(l).copied().unwrap_or(0.0));
        for i in 0..ff {
            let j = l * ff + i;
            let s = &stats[j * 6..j * 6 + 6];
            let t8 = &top8[j * 16..j * 16 + 16];
            for pole in 0..2 {
                let (a, z, entered) = if pole == 0 {
                    let a = ext[2 * j];
                    (a, (a - s[0]) / s[1].max(fp).max(1e-6), a >= s[4])
                } else {
                    let a = ext[2 * j + 1];
                    (a, (s[2] - a) / s[3].max(fneg).max(1e-6), a <= s[5])
                };
                if z < z_min || (require_entry && !entered) {
                    continue;
                }
                let ids = &t8[pole * 8..pole * 8 + 8];
                let rel = ids.iter().map(|&c| csim.get(c as usize).copied().unwrap_or(0.0)).sum::<f32>() / 8.0;
                let score = z * rel.max(0.0).powf(rel_pow);
                if best.len() == k && score <= best[k - 1][3] {
                    continue;
                }
                let row = [l as f32, i as f32, if pole == 0 { 1.0 } else { -1.0 }, score, z, a, rel];
                let pos = best.partition_point(|b| b[3] >= score);
                best.insert(pos, row);
                best.truncate(k);
            }
        }
    }
    best.concat()
}

/// Semantic embedding of an input: mean residual over the masked tokens,
/// centered by `mu`, projected by `proj` ([d][k]) and L2-normalized.
#[wasm_bindgen(js_name = semEmbed)]
pub fn sem_embed(resid: &[f32], d: usize, mask: &[u8], mu: &[f32], proj: &[f32], k: usize) -> Vec<f32> {
    let mut pooled = vec![0f32; d];
    let toks: Vec<usize> = (0..resid.len() / d).filter(|&t| mask.get(t).copied().unwrap_or(0) != 0).collect();
    let n = toks.len().max(1) as f32;
    for &t in &toks {
        for (p, &x) in pooled.iter_mut().zip(&resid[t * d..(t + 1) * d]) {
            *p += x / n;
        }
    }
    let mut q = vec![0f32; k];
    for (i, p) in pooled.iter().enumerate() {
        let c = p - mu[i];
        for (qj, w) in q.iter_mut().zip(&proj[i * k..(i + 1) * k]) {
            *qj += c * w;
        }
    }
    let norm = q.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
    q.iter().map(|v| v / norm).collect()
}
