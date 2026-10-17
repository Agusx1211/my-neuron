#!/bin/bash
cd "$(dirname "$0")"
MASK_STOP=1 FORMULAS="$1" PROBE_CACHE=/home/agusx1211/workspaces/my-neuron/py/exp/probe_cache_stop.pkl ../.venv/bin/python select.py ../../web/public/atlas/qwen3.5-4b ../../models/q4b-Q4_K_M.gguf ../../models/hf4b "${@:2}" 2>&1 | grep -vE "Warn|warn|from_numpy|self.w"
