"""Which residual depth makes the best semantic embedding for neuron selection?
For each probe, show the formula-B winner under each candidate sem layer."""
import glob, json, os, re, sys
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


def load_sem(path):
    s = np.fromfile(path, dtype=np.float16).astype(np.float32)
    E = torch.from_numpy(s[: C * k].reshape(C, k)).cuda()
    P = torch.from_numpy(s[C * k: C * k + D * k].reshape(D, k)).cuda()
    mu = torch.from_numpy(s[C * k + D * k:]).cuda()
    return E, P, mu


sems = {meta["semLayer"]: load_sem(f"{atlas}/sem.bin")}
for p in glob.glob(f"{atlas}/sem_L*.bin"):
    sems[int(re.findall(r"sem_L(\d+)", p)[0])] = load_sem(p)
layers = sorted(sems)
print("sem layers:", layers)

STOP = set("""a an the my your his her its our their i me we you he she it they this that these those
of in on at to for from by with and or but is am are was were be been 's ' s mine yours myself
im i'm about as into""".split())
tok = AutoTokenizer.from_pretrained(sys.argv[2])
enc = lambda s: tok(s, add_special_tokens=False).input_ids
m = Qwen35(sys.argv[3], dtype=torch.bfloat16)
rows = {l: np.fromfile(f"{atlas}/L{l}.bin", dtype=np.uint8).reshape(F, 4 * K) for l in range(L)}

probes = [l.strip() for l in open(sys.argv[4]) if l.strip()]
for text in probes:
    P_, S_ = enc(meta["prefix"]), enc(meta["suffix"])
    body = enc(text)
    ids = torch.tensor([P_ + body + S_])
    a, b = len(P_), len(P_) + len(body) + 1
    keep = torch.tensor([tok.decode([t]).strip().lower() not in STOP for t in body] + [True])
    if not keep[:-1].any():
        keep[:] = True
    keep = keep.cuda()
    mx, mn, pooled = [], [], {}
    x = m.embed[ids].to("cuda", m.dtype)
    with torch.no_grad():
        for l in range(L):
            p = f"blk.{l}."
            h = m.rms(x, m.w[p + "attn_norm.weight"])
            x = x + (m.full_attn(l, h) if m.is_full(l) else m.linear_attn(l, h))
            h = m.rms(x, m.w[p + "post_attention_norm.weight"])
            g = torch.nn.functional.silu(h @ m.w[p + "ffn_gate.weight"].T) * (h @ m.w[p + "ffn_up.weight"].T)
            act = g.float()[0, a:b][keep]
            mx.append(act.amax(0)); mn.append(act.amin(0))
            x = x + g @ m.w[p + "ffn_down.weight"].T
            if l in sems:
                pooled[l] = x.float()[0, a:b][keep].mean(0)
    mx, mn = torch.stack(mx), torch.stack(mn)
    zp = (mx - stats[..., 0]) / torch.maximum(stats[..., 1], floor_p)
    zn = (stats[..., 2] - mn) / torch.maximum(stats[..., 3], floor_n)
    z = torch.stack([zp, zn]); z[:, :2] = -1e9
    entry = torch.stack([mx >= stats[..., 4], mn <= stats[..., 5]])
    print(f"\n[{text}]")
    for sl in layers:
        E, P, mu = sems[sl]
        q = (pooled[sl] - mu) @ P
        q = q / q.norm()
        csim = E @ q
        rel = torch.stack([csim[top8[..., :8]].mean(-1), csim[top8[..., 8:]].mean(-1)])
        score = torch.where(entry & (z >= 6), rel.clamp_min(0) ** 3 * z, torch.tensor(-1e9, device="cuda"))
        idx = score.flatten().argmax().item()
        pole, rest = divmod(idx, L * F)
        l, f = divmod(rest, F)
        rec = rows[l][f]
        cid = rec[: 2 * K].view(np.uint16)
        sel = range(0, 7) if pole == 0 else range(meta["kPos"], meta["kPos"] + 7)
        print(f"  sem@{sl:2d}: L{l:02d} N{f:5d} {'+-'[pole]} z={z[pole, l, f].item():5.1f} | " + ", ".join(lex["concepts"][cid[j]] for j in sel))
