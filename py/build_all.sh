#!/bin/bash
# Rebuild both concept atlases from lexicon/concepts.tsv (run inside the container, GPU).
set -e
cd "$(dirname "$0")"
.venv/bin/python build_atlas.py --gguf ../models/q4b-Q4_K_M.gguf --tokenizer ../models/hf4b --lexicon ../lexicon/concepts.tsv --out ../web/public/atlas/qwen3.5-4b --batch 128
.venv/bin/python build_atlas.py --gguf ../models/q08-Q4_K_M.gguf --tokenizer ../models/hf08 --lexicon ../lexicon/concepts.tsv --out ../web/public/atlas/qwen3.5-0.8b --batch 256
echo ALL_ATLASES_DONE
