"""Compare WebGPU engine activations (from dev/run-engine-test.mjs) with the torch spec."""
import json, sys
import numpy as np
import torch
sys.path.insert(0, "py")
from qwen35 import Qwen35

r = json.load(open(sys.argv[1]))
# the 4B spec only fits the 16 GB test GPU in bf16, so its tolerance is looser
big = "q4b" in sys.argv[2]
m = Qwen35(sys.argv[2], dtype=torch.bfloat16 if big else torch.float32)
min_cos = 0.999 if big else 0.9999
ids = torch.tensor([r["ids"]])
acts = {}
m.forward(ids, on_act=lambda l, a: acts.__setitem__(l, a[0].float().cpu().numpy()))
n, F = len(r["ids"]), m.d_ff
got = np.array(r["acts"], dtype=np.float32).reshape(len(r["layers"]), n, F)
ok = True
for i, l in enumerate(r["layers"]):
    a, b = acts[l], got[i]
    cos = float((a * b).sum() / np.linalg.norm(a) / np.linalg.norm(b))
    rel = float(np.linalg.norm(a - b) / np.linalg.norm(a))
    same_top = list(np.argsort(-a[1:].max(0))[:5]) == list(np.argsort(-b[1:].max(0))[:5])
    print(f"layer {l:2d}: cos={cos:.6f} rel={rel:.5f} maxabs={np.abs(a - b).max():.4f} top5 same={same_top}")
    ok &= cos > min_cos
# batching/padding must not change anything: compare against the engine's own unbatched run
bat = np.array(r["batched"], dtype=np.float32).reshape(3, r["B"], r["T"])
for j in range(3):
    for b in range(r["B"]):
        d = np.abs(bat[j, b, :n] - got[r["layers"].index(r.get("batchedLayer", r["layers"][-1]))][:, j]).max()
        if d > 1e-3:
            ok = False
            print(f"batched mismatch neuron {j} seq {b}: {d}")
print("PASS" if ok else "FAIL")
