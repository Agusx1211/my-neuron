import pickle, sys, numpy as np, torch
cache = pickle.load(open(sys.argv[1], "rb"))
for i, text in enumerate(sys.argv[3:]):
    mx, mn, q, toks = cache[text]
    ext = torch.stack([mx, mn], -1).float().numpy().astype(np.float32)  # [L,F,2]
    ext.tofile(f"{sys.argv[2]}/ext{i}.bin")
    q.float().numpy().astype(np.float32).tofile(f"{sys.argv[2]}/q{i}.bin")
    print(i, text)
