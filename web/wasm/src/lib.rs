mod gguf;
mod quant;
mod score;
mod tokenizer;

use wasm_bindgen::prelude::*;

fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn gguf_err(e: gguf::Error) -> JsError {
    match e {
        gguf::Error::NeedMore => JsError::new("NEED_MORE"),
        gguf::Error::Bad(m) => JsError::new(&m),
    }
}

/// Parse a GGUF header prefix. Returns JSON {dataStart, meta, tensors}; throws
/// "NEED_MORE" when the prefix is too short. String arrays longer than 64
/// entries (vocab, merges, chat template pieces) are summarized by length.
#[wasm_bindgen(js_name = parseHeader)]
pub fn parse_header(bytes: &[u8]) -> Result<String, JsError> {
    let h = gguf::parse(bytes).map_err(gguf_err)?;
    let mut o = format!("{{\"dataStart\":{},\"meta\":{{", h.data_start);
    for (i, (k, v)) in h.meta.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        o.push_str(&json_str(k));
        o.push(':');
        match v {
            gguf::Value::Int(x) => o.push_str(&x.to_string()),
            gguf::Value::Float(x) => o.push_str(&format!("{x:e}")),
            gguf::Value::Bool(x) => o.push_str(if *x { "true" } else { "false" }),
            gguf::Value::Str(s) => o.push_str(&json_str(s)),
            gguf::Value::StrArray(a) if a.len() > 64 => o.push_str(&format!("{{\"length\":{}}}", a.len())),
            gguf::Value::StrArray(a) => {
                o.push('[');
                o.push_str(&a.iter().map(|s| json_str(s)).collect::<Vec<_>>().join(","));
                o.push(']');
            }
            gguf::Value::IntArray(a) if a.len() > 64 => o.push_str(&format!("{{\"length\":{}}}", a.len())),
            gguf::Value::IntArray(a) => o.push_str(&format!("{a:?}")),
            gguf::Value::FloatArray(a) => o.push_str(&format!("{a:?}")),
        }
    }
    o.push_str("},\"tensors\":[");
    for (i, t) in h.tensors.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        let n: u64 = t.dims.iter().product();
        let nbytes = gguf::block(t.ty).map(|(be, bb)| n / be * bb).unwrap_or(0);
        o.push_str(&format!(
            "{{\"name\":{},\"dims\":{:?},\"type\":{},\"offset\":{},\"nbytes\":{}}}",
            json_str(&t.name),
            t.dims,
            t.ty,
            t.offset,
            nbytes
        ));
    }
    o.push_str("]}");
    Ok(o)
}

#[wasm_bindgen]
pub struct Tokenizer {
    bpe: tokenizer::Bpe,
}

#[wasm_bindgen]
impl Tokenizer {
    /// Build from the raw GGUF header bytes (vocab + merges never leave WASM).
    #[wasm_bindgen(constructor)]
    pub fn new(header: &[u8]) -> Result<Tokenizer, JsError> {
        let h = gguf::parse(header).map_err(gguf_err)?;
        Ok(Tokenizer { bpe: tokenizer::Bpe::from_header(&h).map_err(|e| JsError::new(&e))? })
    }

    pub fn encode(&mut self, text: &str) -> Vec<u32> {
        self.bpe.encode(text)
    }

    /// Encode many strings at once: returns [n, len_0..len_{n-1}, ids...].
    #[wasm_bindgen(js_name = encodeMany)]
    pub fn encode_many(&mut self, texts: Vec<String>) -> Vec<u32> {
        let enc: Vec<Vec<u32>> = texts.iter().map(|t| self.bpe.encode(t)).collect();
        let mut out = Vec::with_capacity(1 + enc.len() + enc.iter().map(|e| e.len()).sum::<usize>());
        out.push(enc.len() as u32);
        out.extend(enc.iter().map(|e| e.len() as u32));
        for e in enc {
            out.extend(e);
        }
        out
    }

    /// Raw bytes of a token (may be a partial UTF-8 sequence).
    pub fn piece(&self, id: u32) -> Vec<u8> {
        self.bpe.piece(id).to_vec()
    }

    #[wasm_bindgen(getter, js_name = vocabSize)]
    pub fn vocab_size(&self) -> usize {
        self.bpe.vocab_size()
    }
}

#[wasm_bindgen]
pub fn dequant(ty: u32, src: &[u8]) -> Result<Vec<f32>, JsError> {
    quant::dequant(ty, src).map_err(|e| JsError::new(&e))
}

#[wasm_bindgen(js_name = padBlocks)]
pub fn pad_blocks(src: &[u8], from: usize, to: usize) -> Vec<u8> {
    quant::pad_blocks(src, from, to)
}

pub use score::*;
