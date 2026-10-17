"""Pick a prompt template + scoring rule: for probe inputs, find the top neuron by
z-score against a concept baseline and print what else drives that neuron."""
import sys, torch
from transformers import AutoTokenizer
sys.path.insert(0, "..")
from qwen35 import Qwen35

tok = AutoTokenizer.from_pretrained("../../models/hf4b")
m = Qwen35("../../models/q4b-Q4_K_M.gguf", dtype=torch.bfloat16)
concepts = [l.strip() for l in open("quick_concepts.txt") if l.strip()]
probes = ["Lionel Messi", "agusx1211", "Agustin Aguilar", "pizza", "nostalgia", "Buenos Aires",
          "my dog Rocky", "I love programming in Rust", "Taylor Swift", "a rainy Sunday afternoon"]

TEMPLATES = {
    "chat": ("<|im_start|>user\n", "<|im_end|>"),
    "about": ("Tell me about ", "."),
    "chat_about": ("<|im_start|>user\nTell me about ", ".<|im_end|>"),
}
enc = lambda s: tok(s, add_special_tokens=False).input_ids


@torch.no_grad()
def extrema(texts, prefix, suffix, extra):
    """span max/min per neuron: [len(texts), L, F] for tokens of the text (+ `extra` following tokens)."""
    P, S = enc(prefix), enc(suffix)
    seqs, spans = [], []
    for t in texts:
        c = enc(t)
        seqs.append(P + c + S)
        spans.append((len(P), len(P) + len(c) + min(extra, len(S))))
    T = max(len(s) for s in seqs)
    ids = torch.tensor([s + [248044] * (T - len(s)) for s in seqs])
    mask = torch.zeros(len(seqs), T, dtype=torch.bool)
    for i, (a, b) in enumerate(spans):
        mask[i, a:b] = True
    mask = mask.cuda()[..., None]
    mx, mn = [], []

    def hook(i, act):
        a = act.float()
        mx.append(torch.where(mask, a, -1e9).amax(1))
        mn.append(torch.where(mask, a, 1e9).amin(1))

    m.forward(ids, on_act=hook)
    return torch.stack(mx, 1), torch.stack(mn, 1)


def batched(texts, *a):
    outs = [extrema(texts[i:i + 128], *a) for i in range(0, len(texts), 128)]
    return torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs])


for name, (pre, suf) in TEMPLATES.items():
    for extra in [0, 1]:
        cmx, cmn = batched(concepts, pre, suf, extra)          # [C, L, F]
        mu_p, sd_p = cmx.mean(0), cmx.std(0)
        mu_n, sd_n = cmn.mean(0), cmn.std(0)
        floor = sd_p.median(-1, keepdim=True).values          # per-layer floor
        floor_n = sd_n.median(-1, keepdim=True).values
        pmx, pmn = batched(probes, pre, suf, extra)
        print(f"\n===== template={name} extra={extra}")
        for qi, q in enumerate(probes):
            zp = (pmx[qi] - mu_p) / torch.maximum(sd_p, floor)
            zn = (mu_n - pmn[qi]) / torch.maximum(sd_n, floor_n)
            z = torch.stack([zp, zn])                          # [2, L, F]
            z[:, :2] = -1e9                                   # skip the first two layers
            flat = z.flatten().topk(3)
            lines = []
            for v, idx in zip(flat.values.tolist(), flat.indices.tolist()):
                pole, rest = divmod(idx, z.shape[1] * z.shape[2])
                L, F = divmod(rest, z.shape[2])
                col = cmx[:, L, F] if pole == 0 else -cmn[:, L, F]
                top = col.topk(8).indices.tolist()
                lines.append(f"  L{L:02d} N{F:5d} {'+-'[pole]} z={v:5.1f} | " + ", ".join(concepts[i] for i in top))
            print(f"[{q}]")
            print("\n".join(lines))
