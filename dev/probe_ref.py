"""Reference activations for dev/probe-check.mjs (same framing as the worker)."""
import json, sys, torch
sys.path.insert(0, "py")
from transformers import AutoTokenizer
from qwen35 import Qwen35
r = json.load(open(sys.argv[1]))
tok = AutoTokenizer.from_pretrained(sys.argv[2])
m = Qwen35(sys.argv[3], dtype=torch.float32)
enc = lambda s: tok(s, add_special_tokens=False).input_ids
P, S = enc("<|im_start|>user\n"), enc("<|im_end|>")
print("analysis top:", r["res"].pop("analysisTop"))
for key, got in r["res"].items():
    l, i = map(int, key.split(":"))
    for (text, act) in got:
        body = enc(text)
        ids = torch.tensor([P + body + S])
        acts = {}
        m.forward(ids, upto=l, on_act=lambda L, a: acts.__setitem__(L, a[0]))
        ref = acts[l][len(P): len(P) + len(body) + 1, i].max().item()
        print(f"L{l} N{i} {text!r:28} browser={act:8.4f} ref={ref:8.4f}")
