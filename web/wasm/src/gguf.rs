//! GGUF v3 header reader. Big string arrays (vocab, merges) are kept as raw
//! slices so the tokenizer can build itself without crossing into JS.

pub enum Value<'a> {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(&'a str),
    StrArray(Vec<&'a str>),
    IntArray(Vec<i64>),
    FloatArray(Vec<f64>),
}

pub struct Tensor {
    pub name: String,
    pub dims: Vec<u64>,
    pub ty: u32,
    pub offset: u64, // absolute
}

pub struct Header<'a> {
    pub meta: Vec<(String, Value<'a>)>,
    pub tensors: Vec<Tensor>,
    pub data_start: u64,
}

impl<'a> Header<'a> {
    pub fn get(&self, key: &str) -> Option<&Value<'a>> {
        self.meta.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
}

#[derive(Debug)]
pub enum Error {
    NeedMore,
    Bad(String),
}

struct Rd<'a> {
    b: &'a [u8],
    o: usize,
}

impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        if self.o + n > self.b.len() {
            return Err(Error::NeedMore);
        }
        let s = &self.b[self.o..self.o + n];
        self.o += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn str(&mut self) -> Result<&'a str, Error> {
        let n = self.u64()? as usize;
        let s = self.take(n)?;
        std::str::from_utf8(s).map_err(|_| Error::Bad("invalid utf8 in gguf string".into()))
    }
    fn scalar(&mut self, t: u32) -> Result<Value<'a>, Error> {
        Ok(match t {
            0 => Value::Int(self.take(1)?[0] as i64),
            1 => Value::Int(self.take(1)?[0] as i8 as i64),
            2 => Value::Int(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64),
            3 => Value::Int(i16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64),
            4 => Value::Int(self.u32()? as i64),
            5 => Value::Int(self.u32()? as i32 as i64),
            6 => Value::Float(f32::from_le_bytes(self.take(4)?.try_into().unwrap()) as f64),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.str()?),
            10 => Value::Int(self.u64()? as i64),
            11 => Value::Int(self.u64()? as i64),
            12 => Value::Float(f64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            _ => return Err(Error::Bad(format!("bad gguf value type {t}"))),
        })
    }
    fn value(&mut self, t: u32) -> Result<Value<'a>, Error> {
        if t != 9 {
            return self.scalar(t);
        }
        let at = self.u32()?;
        let n = self.u64()? as usize;
        if at == 8 {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(self.str()?);
            }
            return Ok(Value::StrArray(v));
        }
        let mut ints = Vec::new();
        let mut floats = Vec::new();
        for _ in 0..n {
            match self.scalar(at)? {
                Value::Int(i) => ints.push(i),
                Value::Bool(b) => ints.push(b as i64),
                Value::Float(f) => floats.push(f),
                _ => return Err(Error::Bad("nested gguf array".into())),
            }
        }
        Ok(if at == 6 || at == 12 { Value::FloatArray(floats) } else { Value::IntArray(ints) })
    }
}

pub fn parse(b: &[u8]) -> Result<Header<'_>, Error> {
    let mut r = Rd { b, o: 0 };
    if r.take(4)? != b"GGUF" {
        return Err(Error::Bad("not a GGUF file".into()));
    }
    let version = r.u32()?;
    if version < 2 {
        return Err(Error::Bad(format!("unsupported gguf version {version}")));
    }
    let n_tensors = r.u64()? as usize;
    let n_kv = r.u64()? as usize;
    let mut meta = Vec::with_capacity(n_kv);
    for _ in 0..n_kv {
        let k = r.str()?.to_string();
        let t = r.u32()?;
        meta.push((k, r.value(t)?));
    }
    let mut raw = Vec::with_capacity(n_tensors);
    for _ in 0..n_tensors {
        let name = r.str()?.to_string();
        let nd = r.u32()? as usize;
        let mut dims = Vec::with_capacity(nd);
        for _ in 0..nd {
            dims.push(r.u64()?);
        }
        let ty = r.u32()?;
        let offset = r.u64()?;
        raw.push(Tensor { name, dims, ty, offset });
    }
    let align = meta
        .iter()
        .find(|(k, _)| k == "general.alignment")
        .and_then(|(_, v)| if let Value::Int(i) = v { Some(*i as u64) } else { None })
        .unwrap_or(32);
    let data_start = (r.o as u64).div_ceil(align) * align;
    for t in raw.iter_mut() {
        t.offset += data_start;
    }
    Ok(Header { meta, tensors: raw, data_start })
}

/// (elements per block, bytes per block)
pub fn block(ty: u32) -> Option<(u64, u64)> {
    Some(match ty {
        0 => (1, 4),
        1 | 30 => (1, 2),
        8 => (32, 34),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        _ => return None,
    })
}
