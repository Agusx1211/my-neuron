//! Byte-level BPE over token ids, matching HF `tokenizers` for Qwen3.5
//! (NFC normalizer, regex Split pre-tokenizer, ByteLevel, BPE merges).

use crate::gguf::{Header, Value};
use fancy_regex::Regex;
use rustc_hash::FxHashMap;
use unicode_normalization::UnicodeNormalization;

const SPLIT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

fn byte_to_unicode() -> [char; 256] {
    let mut bs: Vec<u32> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut out = ['\0'; 256];
    for (b, c) in bs.iter().zip(cs.iter()) {
        out[*b as usize] = char::from_u32(*c).unwrap();
    }
    out
}

pub struct Bpe {
    byte_tok: [u32; 256],
    merges: FxHashMap<(u32, u32), (u32, u32)>, // pair -> (rank, merged id)
    specials: Vec<(String, u32)>,
    pieces: Vec<Vec<u8>>, // raw bytes per token id
    re: Regex,
    cache: FxHashMap<String, Vec<u32>>,
}

impl Bpe {
    pub fn from_header(h: &Header) -> Result<Bpe, String> {
        let tokens = match h.get("tokenizer.ggml.tokens") {
            Some(Value::StrArray(v)) => v,
            _ => return Err("gguf has no tokenizer.ggml.tokens".into()),
        };
        let merges = match h.get("tokenizer.ggml.merges") {
            Some(Value::StrArray(v)) => v,
            _ => return Err("gguf has no tokenizer.ggml.merges".into()),
        };
        let types: &[i64] = match h.get("tokenizer.ggml.token_type") {
            Some(Value::IntArray(v)) => v,
            _ => &[],
        };
        let b2u = byte_to_unicode();
        let mut u2b: FxHashMap<char, u8> = FxHashMap::default();
        for (b, c) in b2u.iter().enumerate() {
            u2b.insert(*c, b as u8);
        }
        let mut vocab: FxHashMap<&str, u32> = FxHashMap::default();
        let mut pieces = Vec::with_capacity(tokens.len());
        let mut specials = Vec::new();
        for (i, t) in tokens.iter().enumerate() {
            vocab.insert(t, i as u32);
            let ty = types.get(i).copied().unwrap_or(1);
            if ty == 3 || ty == 4 {
                specials.push((t.to_string(), i as u32));
                pieces.push(t.as_bytes().to_vec());
            } else {
                pieces.push(t.chars().filter_map(|c| u2b.get(&c).copied()).collect());
            }
        }
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        let mut byte_tok = [u32::MAX; 256];
        for b in 0..256 {
            let s = b2u[b].to_string();
            byte_tok[b] = *vocab.get(s.as_str()).ok_or("byte token missing")?;
        }
        let mut mm = FxHashMap::default();
        mm.reserve(merges.len());
        for (rank, m) in merges.iter().enumerate() {
            let Some((a, b)) = m.split_once(' ') else { continue };
            let (Some(&ia), Some(&ib)) = (vocab.get(a), vocab.get(b)) else { continue };
            let joined = format!("{a}{b}");
            let Some(&ij) = vocab.get(joined.as_str()) else { continue };
            mm.insert((ia, ib), (rank as u32, ij));
        }
        Ok(Bpe {
            byte_tok,
            merges: mm,
            specials,
            pieces,
            re: Regex::new(SPLIT).map_err(|e| e.to_string())?,
            cache: FxHashMap::default(),
        })
    }

    fn bpe_word(&mut self, w: &str) -> Vec<u32> {
        if let Some(v) = self.cache.get(w) {
            return v.clone();
        }
        let mut sym: Vec<u32> = w.bytes().map(|b| self.byte_tok[b as usize]).collect();
        loop {
            let mut best = (u32::MAX, 0usize, 0u32);
            for i in 0..sym.len().saturating_sub(1) {
                if let Some(&(rank, id)) = self.merges.get(&(sym[i], sym[i + 1])) {
                    if rank < best.0 {
                        best = (rank, i, id);
                    }
                }
            }
            if best.0 == u32::MAX {
                break;
            }
            // merge every occurrence of the winning pair, left to right
            let (a, b) = (sym[best.1], sym[best.1 + 1]);
            let mut out = Vec::with_capacity(sym.len());
            let mut i = 0;
            while i < sym.len() {
                if i + 1 < sym.len() && sym[i] == a && sym[i + 1] == b {
                    out.push(best.2);
                    i += 2;
                } else {
                    out.push(sym[i]);
                    i += 1;
                }
            }
            sym = out;
        }
        if self.cache.len() < 100_000 {
            self.cache.insert(w.to_string(), sym.clone());
        }
        sym
    }

    fn encode_plain(&mut self, text: &str, out: &mut Vec<u32>) {
        let norm: String = text.nfc().collect();
        let words: Vec<String> = self
            .re
            .find_iter(&norm)
            .filter_map(|m| m.ok().map(|m| m.as_str().to_string()))
            .collect();
        for w in words {
            out.extend(self.bpe_word(&w));
        }
    }

    pub fn encode(&mut self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut start = 0;
        let mut i = 0;
        let bytes = text.as_bytes();
        while i < bytes.len() {
            if bytes[i] == b'<' {
                let hit = self
                    .specials
                    .iter()
                    .find(|(s, _)| text[i..].starts_with(s.as_str()))
                    .map(|(s, id)| (s.len(), *id));
                if let Some((len, id)) = hit {
                    if start < i {
                        self.encode_plain(&text[start..i], &mut out);
                    }
                    out.push(id);
                    i += len;
                    start = i;
                    continue;
                }
            }
            i += 1;
        }
        if start < bytes.len() {
            self.encode_plain(&text[start..], &mut out);
        }
        out
    }

    pub fn piece(&self, id: u32) -> &[u8] {
        self.pieces.get(id as usize).map(|v| v.as_slice()).unwrap_or(&[])
    }

    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }
}
