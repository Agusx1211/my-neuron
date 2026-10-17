"""Sanity-check the 4B GGUF spec: next-token loss with both V-head mappings."""
import sys, torch, torch.nn.functional as F
from transformers import AutoTokenizer
from qwen35 import Qwen35

tok = AutoTokenizer.from_pretrained(sys.argv[1])
m = Qwen35(sys.argv[2], dtype=torch.bfloat16)
text = "The capital of France is Paris. The capital of Germany is Berlin. The quick brown fox jumps over the lazy dog. Water boils at 100 degrees Celsius at sea level."
ids = tok(text, return_tensors="pt").input_ids
for grouped in [False, True]:
    m.grouped_v = grouped
    lg = m.logits(ids)[0, :-1]
    print("grouped" if grouped else "tiled", "loss", F.cross_entropy(lg, ids[0, 1:].to(lg.device)).item())
m.grouped_v = False
pred = m.logits(tok("The capital of Japan is", return_tensors="pt").input_ids)[0, -1].topk(5).indices
print("next tokens:", [tok.decode([int(t)]) for t in pred])
