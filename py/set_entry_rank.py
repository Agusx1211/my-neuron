"""Change the top-list entry rank of an existing atlas without rebuilding it.

stats.bin columns 4 and 5 hold, per neuron, the activation a concept needs to
enter the neuron's top-N list (positive and negative pole). Both are read back
from the stored top lists in L{layer}.bin, so any N up to kPos / kNeg works.
"""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("atlas")
ap.add_argument("rank", type=int)
args = ap.parse_args()

meta = json.load(open(os.path.join(args.atlas, "meta.json")))
L, F, KP, KN = meta["layers"], meta["neurons"], meta["kPos"], meta["kNeg"]
if not 1 <= args.rank <= min(KP, KN):
    raise SystemExit(f"rank must be between 1 and {min(KP, KN)}")
K = KP + KN
stats = np.fromfile(os.path.join(args.atlas, "stats.bin"), dtype=np.float16).reshape(L, F, meta["statCols"])
for l in range(L):
    vals = np.fromfile(os.path.join(args.atlas, f"L{l}.bin"), dtype=np.uint8).reshape(F, 4 * K)[:, 2 * K:].copy().view(np.float16)
    stats[l, :, 4] = vals[:, args.rank - 1]
    stats[l, :, 5] = vals[:, KP + args.rank - 1]
stats.tofile(os.path.join(args.atlas, "stats.bin"))
meta["entryRank"] = args.rank
json.dump(meta, open(os.path.join(args.atlas, "meta.json"), "w"), indent=1)
print(f"{args.atlas}: entry rank set to {args.rank}")
