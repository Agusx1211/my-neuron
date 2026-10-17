"""Merge the lexicon sources into exactly N concepts (default 65,535, the most
that fits the atlas's 16-bit concept ids).

Priority: the original core list, then the per-country packs (capped per
country and interleaved across categories so every culture gets similar
depth), then common English words by frequency, then Wikipedia Vital Articles
Level 5 interleaved across its sections to fill the remaining slots.

Output: lexicon/concepts.tsv with columns concept, category, source.
"""
import argparse
import collections
import glob
import itertools
import os
import re
import unicodedata

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.path.join(os.path.dirname(__file__), "..", "lexicon"))
ap.add_argument("--total", type=int, default=65535)
ap.add_argument("--country-cap", type=int, default=480)
ap.add_argument("--words-cap", type=int, default=8000)
ap.add_argument("--out", default=None, help="output file (default: lexicon/concepts.tsv)")
args = ap.parse_args()

src = os.path.join(args.dir, "sources")


def norm(s):
    return unicodedata.normalize("NFC", " ".join(s.split()))


def ok(c):
    return bool(c) and len(c) <= 60 and len(c.split()) <= 6 and "\t" not in c


def read(path, ncols):
    rows = []
    for line in open(path, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        parts = [norm(p) for p in parts] + [""] * ncols
        rows.append(parts[:ncols])
    return rows


def interleave(groups):
    """Round-robin over a dict of lists, preserving the order inside each list."""
    out = []
    for batch in itertools.zip_longest(*groups.values()):
        out.extend(x for x in batch if x is not None)
    return out


out, seen = [], set()
drop = os.path.join(src, "drop.txt")
if os.path.exists(drop):
    seen.update(norm(l).casefold() for l in open(drop, encoding="utf-8") if l.strip() and not l.startswith("#"))
counts = collections.Counter()


def add(concept, category, source, limit=None):
    k = concept.casefold()
    if not ok(concept) or k in seen or len(out) >= args.total or (limit is not None and counts[source] >= limit):
        return False
    seen.add(k)
    out.append((concept, category or "other", source))
    counts[source] += 1
    return True


# 1. core list (the original 17.7k, kept verbatim)
for c, cat in read(os.path.join(src, "core.tsv"), 2):
    add(c, cat, "core")

# 2. country packs, interleaved by category within each country, capped per country
packs = collections.defaultdict(lambda: collections.defaultdict(list))
for f in sorted(glob.glob(os.path.join(src, "countries_*.tsv"))):
    for c, cat, country in read(f, 3):
        packs[country or "unknown"][cat].append((c, cat))
for country, cats in sorted(packs.items()):
    for c, cat in interleave(cats):
        add(c, cat, f"country:{country}", args.country_cap)

# 3. common words by frequency
words = os.path.join(src, "words.tsv")
if os.path.exists(words):
    for c, cat in read(words, 2):
        add(c, cat, "words", args.words_cap)

# 4. Vital Articles Level 5, interleaved across sections to fill the rest
va5 = os.path.join(src, "va5.tsv")
if os.path.exists(va5):
    sections = collections.defaultdict(list)
    # Latin taxon names (binomials, families, orders) are encyclopedic noise here
    taxon = re.compile(r"^[A-Z][a-z]+ [a-z]+$|^[A-Z][a-z]+(idae|inae|aceae|ales|iformes|opsida|phyta|mycota|oidea)$")
    for c, cat, sec in read(va5, 3):
        if cat in ("animal", "plant-nature", "science") and taxon.match(c):
            continue
        sections[sec.split("/")[0] + "/" + (sec.split("/")[1] if "/" in sec else "")].append((c, cat))
    for c, cat in interleave(sections):
        if len(out) >= args.total:
            break
        add(c, cat, "va5")

with open(args.out or os.path.join(args.dir, "concepts.tsv"), "w", encoding="utf-8") as f:
    for c, cat, s in out:
        f.write(f"{c}\t{cat}\t{s}\n")

print(f"{len(out)} concepts (target {args.total})")
by_kind = collections.Counter(s.split(":")[0] for _, _, s in out)
print("by source:", dict(by_kind))
print("countries:", ", ".join(f"{k.split(':')[1]} {v}" for k, v in sorted(counts.items()) if k.startswith("country:")))
if len(out) < args.total:
    print(f"WARNING: short by {args.total - len(out)}")
