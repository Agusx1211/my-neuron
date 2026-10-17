"""Which token positions should the neuron choice look at? Compares the whole
span (current), the end of the text (last content token + <|im_end|>), and each
single token, printing the winning neuron and its top concepts.

usage: positions.py <atlas> <hf tokenizer dir> <gguf> <probes.txt>
"""
import json, os, sys
import numpy as np
import torch
from transformers import AutoTokenizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qwen35 import Qwen35

atlas = sys.argv[1]
meta = json.load(open(f"{atlas}/meta.json"))
lex = json.load(open(f"{atlas}/lexicon.json"))
L, F, C = meta["layers"], meta["neurons"], meta["concepts"]
K = meta["kPos"] + meta["kNeg"]
k, D = meta["semDim"], meta["hidden"]
stats = torch.from_numpy(np.fromfile(f"{atlas}/stats.bin", dtype=np.float16).astype(np.float32).reshape(L, F, 6)).cuda()
top8 = torch.from_numpy(np.fromfile(f"{atlas}/top8.bin", dtype=np.uint16).astype(np.int64).reshape(L, F, 16)).cuda()
floor_p = torch.tensor(meta["floorPos"]).cuda()[:, None]
floor_n = torch.tensor(meta["floorNeg"]).cuda()[:, None]
s = np.fromfile(f"{atlas}/sem.bin", dtype=np.float16).astype(np.float32)
E = torch.from_numpy(s[: C * k].reshape(C, k)).cuda()
P = torch.from_numpy(s[C * k: C * k + D * k].reshape(D, k)).cuda()
mu = torch.from_numpy(s[C * k + D * k:]).cuda()
rows = {l: np.fromfile(f"{atlas}/L{l}.bin", dtype=np.uint8).reshape(F, 4 * K) for l in range(L)}

STOP = set("""a an the my your his her its our their i me we you he she it they this that these those
of in on at to for from by with and or but is am are was were be been 's ' s mine yours myself
im i'm about as into""".split())
tok = AutoTokenizer.from_pretrained(sys.argv[2])
enc = lambda s: tok(s, add_special_tokens=False).input_ids
m = Qwen35(sys.argv[3], dtype=torch.bfloat16)
NEG = torch.tensor(-1e9, device="cuda")


def forward(text):
    P_, S_ = enc(meta["prefix"]), enc(meta["suffix"])
    body = enc(text)
    ids = torch.tensor([P_ + body + S_])
    a, b = len(P_), len(P_) + len(body) + 1
    acts = []
    x = m.embed[ids].to("cuda", m.dtype)
    with torch.no_grad():
        for l in range(L):
            p = f"blk.{l}."
            h = m.rms(x, m.w[p + "attn_norm.weight"])
            x = x + (m.full_attn(l, h) if m.is_full(l) else m.linear_attn(l, h))
            h = m.rms(x, m.w[p + "post_attention_norm.weight"])
            g = torch.nn.functional.silu(h @ m.w[p + "ffn_gate.weight"].T) * (h @ m.w[p + "ffn_up.weight"].T)
            acts.append(g.float()[0, a:b])
            x = x + g @ m.w[p + "ffn_down.weight"].T
            if l == meta["semLayer"]:
                resid = x.float()[0, a:b]
    return torch.stack(acts), resid, [tok.decode([t]) for t in body]   # acts [L][T][F]


def pick(acts, resid, sel, qsel):
    sel = torch.tensor(sel, device="cuda")
    qs = torch.tensor(qsel, device="cuda")
    mx, mn = acts[:, sel].amax(1), acts[:, sel].amin(1)
    zp = (mx - stats[..., 0]) / torch.maximum(stats[..., 1], floor_p)
    zn = (stats[..., 2] - mn) / torch.maximum(stats[..., 3], floor_n)
    z = torch.stack([zp, zn]); z[:, : meta["minLayer"]] = -1e9
    entry = torch.stack([mx >= stats[..., 4], mn <= stats[..., 5]])
    q = (resid[qs].mean(0) - mu) @ P
    q = q / q.norm()
    csim = E @ q
    rel = torch.stack([csim[top8[..., :8]].mean(-1), csim[top8[..., 8:]].mean(-1)])
    score = rel.clamp_min(0) ** 3 * z
    sc = torch.where(entry & (z >= 6), score, NEG)
    if (sc > 0).sum() < 8:
        sc = torch.where(z >= 6, score, NEG)
    idx = sc.flatten().argmax().item()
    pole, rest = divmod(idx, L * F)
    l, f = divmod(rest, F)
    rec = rows[l][f]
    cid = rec[: 2 * K].view(np.uint16)
    js = range(0, 6) if pole == 0 else range(meta["kPos"], meta["kPos"] + 6)
    col = acts[l, :, f] * (1 if pole == 0 else -1)
    peak = int(sel[col[sel].argmax()].item())
    return l, f, pole, z[pole, l, f].item(), peak, ", ".join(lex["concepts"][cid[j]] for j in js)


for text in [l.strip() for l in open(sys.argv[4]) if l.strip() and not l.startswith("#")]:
    acts, resid, toks = forward(text)
    T = len(toks)
    keep = [i for i, t in enumerate(toks) if t.strip().lower() not in STOP] or list(range(T))
    names = toks + ["<end>"]
    print(f"\n[{text}] {toks}")
    modes = {
        "all  ": (keep + [T], keep + [T]),
        "tail ": ([keep[-1], T], keep + [T]),
        "tailq": ([keep[-1], T], [keep[-1], T]),
        "end  ": ([T], keep + [T]),
    }
    if T > 1:
        for i in range(T):
            modes[f"tok{i} "] = ([i], [i])
    for name, (sel, qsel) in modes.items():
        l, f, pole, z, peak, top = pick(acts, resid, sel, qsel)
        print(f"  {name} L{l:02d} N{f:5d} {'+-'[pole]} z={z:5.1f} @{names[peak]!r:12s} | {top}")
