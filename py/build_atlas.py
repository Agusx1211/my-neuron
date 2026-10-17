"""Build the concept atlas for one model.

Runs every lexicon concept through the model inside the same template the
browser uses and records, for every MLP neuron:
  * mean/std of the span-max (positive pole) and span-min (negative pole)
  * the top-K concepts for each pole
and, per concept, a compact semantic embedding (mean residual stream over the
span at a middle layer, centered + PCA) used to judge which neurons have a
theme related to the user's input.

Outputs (in OUT dir): meta.json, lexicon.json, stats.bin, L{l}.bin, sem.bin.
"""
import argparse, json, math, os, time
import numpy as np
import torch
from transformers import AutoTokenizer
from qwen35 import Qwen35

ap = argparse.ArgumentParser()
ap.add_argument("--gguf", required=True)
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--lexicon", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--kpos", type=int, default=32)
ap.add_argument("--kneg", type=int, default=16)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--sem-layer", type=float, default=0.3, help="fraction of depth for the semantic embedding (see py/exp/semlayers.py)")
ap.add_argument("--sem-dim", type=int, default=128)
ap.add_argument("--sem-extra", default="", help="extra depth fractions to export as sem_L{l}.bin (experiments)")
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

PREFIX, SUFFIX, EXTRA = "<|im_start|>user\n", "<|im_end|>", 1
PAD = 248044

tok = AutoTokenizer.from_pretrained(args.tokenizer)
enc = lambda s: tok(s, add_special_tokens=False).input_ids
concepts, cats = [], []
seen = set()
for line in open(args.lexicon, encoding="utf-8"):
    parts = line.rstrip("\n").split("\t")
    if not parts[0].strip() or parts[0].lower() in seen:
        continue
    seen.add(parts[0].lower())
    concepts.append(parts[0].strip())
    cats.append(parts[1].strip() if len(parts) > 1 else "other")
if args.limit:
    concepts, cats = concepts[: args.limit], cats[: args.limit]
C = len(concepts)
print(f"{C} concepts")

t0 = time.time()
m = Qwen35(args.gguf, dtype=torch.bfloat16)
print(f"model loaded in {time.time() - t0:.0f}s")
L, F, D = m.n_layer, m.d_ff, m.d
sem_layer = int(round(args.sem_layer * (L - 1)))
dev = "cuda"

P, S = enc(PREFIX), enc(SUFFIX)
seqs = [(P + enc(c) + S) for c in concepts]
spans = [(len(P), len(s) - len(S) + min(EXTRA, len(S))) for s in seqs]

KP, KN = args.kpos, args.kneg
top_v = torch.full((L, F, KP), -1e9, device=dev)
top_i = torch.zeros((L, F, KP), dtype=torch.int32, device=dev)
bot_v = torch.full((L, F, KN), 1e9, device=dev)
bot_i = torch.zeros((L, F, KN), dtype=torch.int32, device=dev)
s_mx = torch.zeros((L, F), dtype=torch.float64, device=dev)
s_mx2 = torch.zeros((L, F), dtype=torch.float64, device=dev)
s_mn = torch.zeros((L, F), dtype=torch.float64, device=dev)
s_mn2 = torch.zeros((L, F), dtype=torch.float64, device=dev)
sem = torch.zeros((C, D), dtype=torch.float32)
extra_layers = sorted({int(round(float(f) * (L - 1))) for f in args.sem_extra.split(",") if f} - {sem_layer})
sem_x = {l: torch.zeros((C, D), dtype=torch.float32) for l in extra_layers}

order = sorted(range(C), key=lambda i: len(seqs[i]))
t0 = time.time()
with torch.no_grad():
    for bi in range(0, C, args.batch):
        idx = order[bi: bi + args.batch]
        T = max(len(seqs[i]) for i in idx)
        ids = torch.tensor([seqs[i] + [PAD] * (T - len(seqs[i])) for i in idx])
        mask = torch.zeros(len(idx), T, dtype=torch.bool)
        for r, i in enumerate(idx):
            a, b = spans[i]
            mask[r, a:b] = True
        mask = mask.to(dev)[..., None]
        gidx = torch.tensor(idx, dtype=torch.int32, device=dev)

        def on_act(l, act):
            a = act.float()
            mx = torch.where(mask, a, -1e9).amax(1)   # [B, F]
            mn = torch.where(mask, a, 1e9).amin(1)
            s_mx[l] += mx.sum(0).double(); s_mx2[l] += (mx.double() ** 2).sum(0)
            s_mn[l] += mn.sum(0).double(); s_mn2[l] += (mn.double() ** 2).sum(0)
            v = torch.cat([top_v[l], mx.T], 1)
            ii = torch.cat([top_i[l], gidx[None].expand(F, -1)], 1)
            k = v.topk(KP, dim=1)
            top_v[l], top_i[l] = k.values, ii.gather(1, k.indices)
            v = torch.cat([bot_v[l], mn.T], 1)
            ii = torch.cat([bot_i[l], gidx[None].expand(F, -1)], 1)
            k = v.topk(KN, dim=1, largest=False)
            bot_v[l], bot_i[l] = k.values, ii.gather(1, k.indices)

        x = m.embed[ids.cpu()].to(dev, m.dtype)
        for l in range(L):
            p = f"blk.{l}."
            h = m.rms(x, m.w[p + "attn_norm.weight"])
            x = x + (m.full_attn(l, h) if m.is_full(l) else m.linear_attn(l, h))
            h = m.rms(x, m.w[p + "post_attention_norm.weight"])
            act = torch.nn.functional.silu(h @ m.w[p + "ffn_gate.weight"].T) * (h @ m.w[p + "ffn_up.weight"].T)
            on_act(l, act)
            x = x + act @ m.w[p + "ffn_down.weight"].T
            if l == sem_layer or l in sem_x:
                xs = x.float()
                pooled = ((xs * mask).sum(1) / mask.sum(1)).cpu()
                (sem if l == sem_layer else sem_x[l])[idx] = pooled
        done = min(bi + args.batch, C)
        if (bi // args.batch) % 5 == 0:
            el = time.time() - t0
            print(f"  {done}/{C} concepts  {el:.0f}s  eta {el / done * (C - done):.0f}s", flush=True)

os.makedirs(args.out, exist_ok=True)
mean_mx = (s_mx / C).float(); std_mx = (s_mx2 / C - (s_mx / C) ** 2).clamp_min(0).sqrt().float()
mean_mn = (s_mn / C).float(); std_mn = (s_mn2 / C - (s_mn / C) ** 2).clamp_min(0).sqrt().float()
# columns: mean/std of span-max, mean/std of span-min, entry thresholds of the top-16 lists
thr = 16
stats = torch.stack([mean_mx, std_mx, mean_mn, std_mn, top_v[:, :, thr - 1], bot_v[:, :, thr - 1]], -1)
stats = stats.cpu().numpy().astype(np.float16)  # [L,F,6]
stats.tofile(os.path.join(args.out, "stats.bin"))
floor_pos = std_mx.median(1).values.cpu().tolist()
floor_neg = std_mn.median(1).values.cpu().tolist()

for l in range(L):
    ids_ = torch.cat([top_i[l], bot_i[l]], 1).cpu().numpy().astype(np.uint16)          # [F, KP+KN]
    vals = torch.cat([top_v[l], bot_v[l]], 1).cpu().numpy().astype(np.float16)
    rec = np.concatenate([ids_.view(np.uint8), vals.view(np.uint8)], 1)                 # [F, 4*(KP+KN)]
    rec.tofile(os.path.join(args.out, f"L{l}.bin"))

# compact top-8 lists of every neuron (for in-browser theme matching)
top8 = torch.cat([top_i[:, :, :8], bot_i[:, :, :8]], 2).cpu().numpy().astype(np.uint16)  # [L,F,16]
top8.tofile(os.path.join(args.out, "top8.bin"))

# semantic embedding: center, PCA, L2 normalize
def save_sem(X, path):
    mu = X.mean(0, keepdim=True)
    Xc = X - mu
    U, Sv, V = torch.pca_lowrank(Xc, q=args.sem_dim, center=False)
    proj = V[:, : args.sem_dim]                                      # [D, k]
    E = Xc @ proj
    E = E / E.norm(dim=1, keepdim=True).clamp_min(1e-6)
    np.concatenate([E.numpy().astype(np.float16).ravel(),
                    proj.numpy().astype(np.float16).ravel(),
                    mu.numpy().astype(np.float16).ravel()]).tofile(path)


save_sem(sem, os.path.join(args.out, "sem.bin"))
for l, X in sem_x.items():
    save_sem(X, os.path.join(args.out, f"sem_L{l}.bin"))

cat_names = sorted(set(cats))
json.dump({"concepts": concepts, "categories": cat_names, "cat": [cat_names.index(c) for c in cats]},
          open(os.path.join(args.out, "lexicon.json"), "w"), ensure_ascii=False)
json.dump({
    "layers": L, "neurons": F, "kPos": KP, "kNeg": KN, "concepts": C, "statCols": 6, "entryRank": thr,
    "prefix": PREFIX, "suffix": SUFFIX, "extra": EXTRA, "minLayer": 2,
    "floorPos": floor_pos, "floorNeg": floor_neg,
    "semLayer": sem_layer, "semDim": args.sem_dim, "hidden": D,
}, open(os.path.join(args.out, "meta.json"), "w"), indent=1)
print("done", time.time() - t0)
