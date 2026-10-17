"""Compare the GGUF-spec forward pass against HF transformers on Qwen3.5-0.8B."""
import sys, torch
from transformers import AutoTokenizer, AutoConfig
import transformers
from qwen35 import Qwen35

hf_dir, gguf_path = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(hf_dir)
text = "<|im_start|>user\nAgustin, also known as agusx1211, loves Buenos Aires and mate.<|im_end|>"
ids = tok(text, return_tensors="pt").input_ids
print("tokens", ids.shape, tok.convert_ids_to_tokens(ids[0])[:20])

cfg = AutoConfig.from_pretrained(hf_dir)
cls = getattr(transformers, "Qwen3_5ForConditionalGeneration")
hf = cls.from_pretrained(hf_dir, torch_dtype=torch.float32).cuda().eval()
lm = hf.model.language_model
hf_acts = {}
for i, layer in enumerate(lm.layers):
    layer.mlp.down_proj.register_forward_hook(lambda m, inp, out, i=i: hf_acts.__setitem__(i, inp[0].detach()))
with torch.no_grad():
    hf(input_ids=ids.cuda())
del hf; torch.cuda.empty_cache()

m = Qwen35(gguf_path)
my = {}
m.forward(ids, on_act=lambda i, a: my.__setitem__(i, a))
for i in range(m.n_layer):
    a, b = hf_acts[i][0].float(), my[i][0].float()
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    rel = ((a - b).norm() / a.norm()).item()
    top_a = a[1:].max(0).values.topk(5).indices.tolist(); top_b = b[1:].max(0).values.topk(5).indices.tolist()
    print(f"layer {i:2d} {'full' if m.is_full(i) else 'lin '} cos={cos:.5f} rel={rel:.4f} maxabs={a.abs().max():.2f} top5 hf={top_a} gguf={top_b}")
