#!/bin/bash
# Serve the frozen 4B bundle (dev/poster-serve.sh) and capture README/social images
# with dev/poster.mjs (inside the container).
set -e
cd "$(dirname "$0")/.."
dev/poster-serve.sh
rm -rf /tmp/poster && mkdir -p /tmp/poster
nohup node dev/poster.mjs /tmp/poster http://localhost:8660/ /dev-models/q4b-Q4_K_M.gguf "${1:-Lionel Messi}" > /tmp/poster.log 2>&1 &
echo "poster run started"
