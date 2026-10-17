"""Compare neuron-selection rules on an atlas: for probe inputs, print the chosen
neurons and their top concepts."""
import json, sys, os
import numpy as np
import torch
from transformers import AutoTokenizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qwen35 import Qwen35

atlas, gguf_path, tok_dir = sys.argv[1], sys.argv[2], sys.argv[3]
meta = json.load(open(f"{atlas}/meta.json"))
lex = json.load(open(f"{atlas}/lexicon.json"))
L, F, C, K = meta["layers"], meta["neurons"], meta["concepts"], meta["kPos"] + meta["kNeg"]
stats = torch.from_numpy(np.fromfile(f"{atlas}/stats.bin", dtype=np.float16).astype(np.float32).reshape(L, F, 6)).cuda()
top8 = torch.from_numpy(np.fromfile(f"{atlas}/top8.bin", dtype=np.uint16).astype(np.int64).reshape(L, F, 16)).cuda()
sem = np.fromfile(f"{atlas}/sem.bin", dtype=np.float16).astype(np.float32)
k, D = meta["semDim"], meta["hidden"]
E = torch.from_numpy(sem[: C * k].reshape(C, k)).cuda()
proj = torch.from_numpy(sem[C * k: C * k + D * k].reshape(D, k)).cuda()
mu = torch.from_numpy(sem[C * k + D * k:]).cuda()
floor_p = torch.tensor(meta["floorPos"]).cuda()[:, None]
floor_n = torch.tensor(meta["floorNeg"]).cuda()[:, None]

tok = AutoTokenizer.from_pretrained(tok_dir)
enc = lambda s: tok(s, add_special_tokens=False).input_ids
m = None  # loaded lazily: only needed for probes missing from the cache


def layer_rows(l):
    return np.fromfile(f"{atlas}/L{l}.bin", dtype=np.uint8).reshape(F, 4 * K)


STOP = set("""a an the my your his her its our their i me we you he she it they this that these those
of in on at to for from by with and or but is am are was were be been 's ' s mine yours myself
im i'm about as into""".split())


def run(text):
    global m
    if m is None:
        m = Qwen35(gguf_path, dtype=torch.bfloat16)
    P, S = enc(meta["prefix"]), enc(meta["suffix"])
    body = enc(text)
    ids = torch.tensor([P + body + S])
    a, b = len(P), len(P) + len(body) + min(meta["extra"], len(S))
    keep = torch.ones(b - a, dtype=torch.bool)
    if os.environ.get("MASK_STOP"):
        for i, t in enumerate(body):
            if tok.decode([t]).strip().lower() in STOP:
                keep[i] = False
        if not keep[: len(body)].any():
            keep[:] = True
    mx, mn = [], []
    q = None
    x = m.embed[ids].to("cuda", m.dtype)
    for l in range(L):
        p = f"blk.{l}."
        h = m.rms(x, m.w[p + "attn_norm.weight"])
        x = x + (m.full_attn(l, h) if m.is_full(l) else m.linear_attn(l, h))
        h = m.rms(x, m.w[p + "post_attention_norm.weight"])
        g = torch.nn.functional.silu(h @ m.w[p + "ffn_gate.weight"].T) * (h @ m.w[p + "ffn_up.weight"].T)
        act = g.float()[0, a:b][keep.cuda()]
        mx.append(act.amax(0)); mn.append(act.amin(0))
        x = x + g @ m.w[p + "ffn_down.weight"].T
        if l == meta["semLayer"]:
            pooled = x.float()[0, a:b][keep.cuda()].mean(0)
            q = (pooled - mu) @ proj
            q = q / q.norm()
    return torch.stack(mx), torch.stack(mn), q, [tok.decode([i]) for i in body]


def show(name, scores, mx, mn, q, n=3):
    flat = scores.flatten().topk(n)
    print(f"  -- {name}")
    for v, idx in zip(flat.values.tolist(), flat.indices.tolist()):
        pole, rest = divmod(idx, L * F)
        l, f = divmod(rest, F)
        rec = layer_rows(l)[f]
        cid = rec[: 2 * K].view(np.uint16)
        vals = rec[2 * K:].view(np.float16)
        sel = range(0, 10) if pole == 0 else range(meta["kPos"], meta["kPos"] + 10)
        a = (mx if pole == 0 else mn)[l, f].item()
        z = ((mx[l, f] - stats[l, f, 0]) / torch.maximum(stats[l, f, 1], floor_p[l, 0])).item() if pole == 0 else \
            ((stats[l, f, 2] - mn[l, f]) / torch.maximum(stats[l, f, 3], floor_n[l, 0])).item()
        print(f"    L{l:02d} N{f:5d} {'+-'[pole]} score={v:6.2f} z={z:5.1f} act={a:6.2f} | " +
              ", ".join(f"{lex['concepts'][cid[j]]}({float(vals[j]):.1f})" for j in sel))


probes = sys.argv[4:] or ["Lionel Messi", "agusx1211", "Agustin Aguilar", "pizza", "nostalgia", "Buenos Aires",
                          "my dog Rocky", "I love programming in Rust", "Taylor Swift", "a rainy Sunday afternoon",
                          "Ethereum smart contracts", "my grandmother's kitchen"]
import pickle
cache_path = os.environ.get("PROBE_CACHE", "/tmp/probe_cache.pkl")
cache = pickle.load(open(cache_path, "rb")) if os.path.exists(cache_path) else {}
FORMULAS = [f.split("::", 1) for f in os.environ.get("FORMULAS", "").split(";") if f] or None
for text in probes:
    if text not in cache:
        mx, mn, q, toks = run(text)
        cache[text] = (mx.cpu(), mn.cpu(), q.cpu(), toks)
        pickle.dump(cache, open(cache_path, "wb"))
    mx, mn, q, toks = (t.cuda() if torch.is_tensor(t) else t for t in cache[text])
    zp = (mx - stats[..., 0]) / torch.maximum(stats[..., 1], floor_p)
    zn = (stats[..., 2] - mn) / torch.maximum(stats[..., 3], floor_n)
    z = torch.stack([zp, zn])
    z[:, : meta["minLayer"]] = -1e9
    entry = torch.stack([mx >= stats[..., 4], mn <= stats[..., 5]])
    csim = E @ q                                                     # [C]
    rel = torch.stack([csim[top8[..., :8]].mean(-1), csim[top8[..., 8:]].mean(-1)])  # [2,L,F]
    coh = rel
    print(f"\n[{text}] tokens={toks}")
    if FORMULAS:
        NEG = torch.tensor(-1e9, device="cuda")
        for name, expr in FORMULAS:
            show(name, eval(expr), mx, mn, q, n=2)
        continue
    show("z", z, mx, mn, q)
    cand = entry & (z > 3)
    print(f"  ({int(cand.sum())} entry candidates)")
    show("entry+relevance", torch.where(cand, rel, torch.tensor(-1e9, device='cuda')), mx, mn, q)
    show("entry: z*relevance^2", torch.where(cand, z * rel.clamp_min(0) ** 2, torch.tensor(-1e9, device='cuda')), mx, mn, q)
    show("all: z*relevance^2", torch.where(z > 3, z * rel.clamp_min(0) ** 2, torch.tensor(-1e9, device='cuda')), mx, mn, q)
