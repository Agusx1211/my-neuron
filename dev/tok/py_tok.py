import json, sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1])
xs = json.load(open(sys.argv[2]))
ref = [tok.encode(s, add_special_tokens=False).ids for s in xs]
got = json.load(open(sys.argv[3]))
bad = 0
for s, a, b in zip(xs, ref, got):
    if a != b:
        bad += 1
        if bad <= 10: print("MISMATCH", repr(s), a, b)
print(f"{len(xs) - bad}/{len(xs)} match")
