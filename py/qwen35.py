"""Plain-torch Qwen3.5 text forward pass that reads weights straight from a GGUF file.

This is the executable spec for the WebGPU engine: same tensor layouts (llama.cpp
conversion: norm weights pre-shifted by +1, ssm_a = -exp(A_log), V heads in tiled
order), same math, and it exposes every MLP neuron activation silu(gate) * up.
"""
import math
import os
import numpy as np
import torch
import torch.nn.functional as F
import gguf


class Qwen35:
    def __init__(self, path, device="cuda", dtype=torch.float32):
        r = gguf.GGUFReader(path)
        self.device, self.dtype = device, dtype
        kv = {f.name: f for f in r.fields.values()}

        def val(k):
            f = kv[k]
            v = f.parts[f.data[0]]
            return v.tolist()[0] if len(v) == 1 else v.tolist()

        a = "qwen35"
        self.n_layer = val(f"{a}.block_count")
        self.d = val(f"{a}.embedding_length")
        self.d_ff = val(f"{a}.feed_forward_length")
        self.n_head = val(f"{a}.attention.head_count")
        self.n_kv = val(f"{a}.attention.head_count_kv")
        self.head_dim = val(f"{a}.attention.key_length")
        self.rope_dim = val(f"{a}.rope.dimension_count")
        self.rope_base = val(f"{a}.rope.freq_base")
        self.eps = val(f"{a}.attention.layer_norm_rms_epsilon")
        self.conv_k = val(f"{a}.ssm.conv_kernel")
        self.dk = val(f"{a}.ssm.state_size")          # linear key head dim
        self.n_kh = val(f"{a}.ssm.group_count")       # linear key heads
        self.n_vh = val(f"{a}.ssm.time_step_rank")    # linear value heads
        self.dv = val(f"{a}.ssm.inner_size") // self.n_vh
        self.interval = val(f"{a}.full_attention_interval")
        self.w = {}
        # per-tensor cache of the dequantized weights (keeps peak host RAM at one tensor)
        cache = f"{path}.{str(dtype).split('.')[-1]}.cache"
        if os.path.exists(os.path.join(cache, "done")):
            for fn in os.listdir(cache):
                if fn.endswith(".pt"):
                    t = torch.load(os.path.join(cache, fn), mmap=True)
                    if fn == "token_embd.weight.pt":
                        self.embed = t  # f16, memory-mapped
                    else:
                        self.w[fn[:-3]] = t.to(device)
            return
        os.makedirs(cache, exist_ok=True)
        for t in r.tensors:
            arr = gguf.quants.dequantize(t.data, t.tensor_type)
            shape = [int(x) for x in reversed(t.shape.tolist())]
            arr = np.asarray(arr, dtype=np.float32).reshape(shape)
            if t.name == "token_embd.weight":
                self.embed = torch.from_numpy(arr)  # keep on CPU
                continue
            self.w[t.name] = torch.from_numpy(arr).to(device=device, dtype=dtype)
        for k, v in self.w.items():
            torch.save(v.cpu(), os.path.join(cache, k + ".pt"))
        torch.save(self.embed.half(), os.path.join(cache, "token_embd.weight.pt"))
        open(os.path.join(cache, "done"), "w").close()
        self.embed = self.embed.half()  # match the cached path exactly

    def is_full(self, i):
        return (i + 1) % self.interval == 0

    def rms(self, x, w):
        x32 = x.float()
        y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * w.float()).to(x.dtype)

    def full_attn(self, i, h):
        B, T, _ = h.shape
        W = self.w
        p = f"blk.{i}."
        qg = (h @ W[p + "attn_q.weight"].T).view(B, T, self.n_head, 2 * self.head_dim)
        q, gate = qg[..., : self.head_dim], qg[..., self.head_dim:]
        k = (h @ W[p + "attn_k.weight"].T).view(B, T, self.n_kv, self.head_dim)
        v = (h @ W[p + "attn_v.weight"].T).view(B, T, self.n_kv, self.head_dim)
        q = self.rms(q, W[p + "attn_q_norm.weight"])
        k = self.rms(k, W[p + "attn_k_norm.weight"])
        # NeoX-style rotary on the first rope_dim dims of each head
        half = self.rope_dim // 2
        inv = 1.0 / (self.rope_base ** (torch.arange(0, half, device=h.device, dtype=torch.float32) * 2 / self.rope_dim))
        ang = torch.arange(T, device=h.device, dtype=torch.float32)[:, None] * inv[None]
        cos, sin = ang.cos()[None, :, None, :], ang.sin()[None, :, None, :]

        def rope(x):
            x1, x2, rest = x[..., :half], x[..., half: self.rope_dim], x[..., self.rope_dim:]
            return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest], -1).to(x.dtype)

        q, k = rope(q), rope(k)
        rep = self.n_head // self.n_kv
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        o = o.transpose(1, 2) * torch.sigmoid(gate)
        return o.reshape(B, T, -1) @ W[p + "attn_output.weight"].T

    def linear_attn(self, i, h):
        B, T, _ = h.shape
        W = self.w
        p = f"blk.{i}."
        qkv = h @ W[p + "attn_qkv.weight"].T                     # [B,T,2*nkh*dk + nvh*dv]
        z = h @ W[p + "attn_gate.weight"].T                       # [B,T,nvh*dv]
        b = h @ W[p + "ssm_beta.weight"].T                        # [B,T,nvh]
        a = h @ W[p + "ssm_alpha.weight"].T
        cw = W[p + "ssm_conv1d.weight"]                           # [C, K]
        C, K = cw.shape
        xp = F.pad(qkv.transpose(1, 2), (K - 1, 0))
        qkv = F.silu(F.conv1d(xp, cw[:, None, :], groups=C)).transpose(1, 2)
        nk, nv = self.n_kh * self.dk, self.n_vh * self.dv
        q = qkv[..., :nk].reshape(B, T, self.n_kh, self.dk).float()
        k = qkv[..., nk: 2 * nk].reshape(B, T, self.n_kh, self.dk).float()
        v = qkv[..., 2 * nk:].reshape(B, T, self.n_vh, self.dv).float()
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) / math.sqrt(self.dk)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        beta = torch.sigmoid(b.float())
        g = W[p + "ssm_a"].float() * F.softplus(a.float() + W[p + "ssm_dt.bias"].float())
        # tiled V-head order: value head j uses key head j % n_kh
        if getattr(self, "grouped_v", False):
            idx = torch.arange(self.n_vh, device=h.device) // (self.n_vh // self.n_kh)
        else:
            idx = torch.arange(self.n_vh, device=h.device) % self.n_kh
        q, k = q[:, :, idx], k[:, :, idx]
        S = torch.zeros(B, self.n_vh, self.dk, self.dv, device=h.device)
        outs = []
        for t in range(T):
            S = S * g[:, t].exp()[..., None, None]
            kt = k[:, t]
            kv_mem = (S * kt[..., None]).sum(-2)
            delta = (v[:, t] - kv_mem) * beta[:, t][..., None]
            S = S + kt[..., None] * delta[..., None, :]
            outs.append((S * q[:, t][..., None]).sum(-2))
        o = torch.stack(outs, 1)                                    # [B,T,nvh,dv]
        o = self.rms(o, W[p + "ssm_norm.weight"]) * F.silu(z.view(B, T, self.n_vh, self.dv).float())
        return o.reshape(B, T, -1).to(h.dtype) @ W[p + "ssm_out.weight"].T

    @torch.no_grad()
    def forward(self, ids, upto=None, on_act=None):
        """ids: LongTensor [B,T]. on_act(layer, act[B,T,d_ff]) receives each layer's MLP neurons."""
        x = self.embed[ids.cpu()].to(self.device, self.dtype)
        n = self.n_layer if upto is None else upto + 1
        for i in range(n):
            p = f"blk.{i}."
            h = self.rms(x, self.w[p + "attn_norm.weight"])
            x = x + (self.full_attn(i, h) if self.is_full(i) else self.linear_attn(i, h))
            h = self.rms(x, self.w[p + "post_attention_norm.weight"])
            act = F.silu(h @ self.w[p + "ffn_gate.weight"].T) * (h @ self.w[p + "ffn_up.weight"].T)
            if on_act is not None:
                on_act(i, act)
            x = x + act @ self.w[p + "ffn_down.weight"].T
        return x

    @torch.no_grad()
    def logits(self, ids):
        x = self.forward(ids)
        x = self.rms(x, self.w["output_norm.weight"])
        return x.float() @ self.embed.to(self.device).float().T
