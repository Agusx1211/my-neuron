#!/bin/bash
# Rebuild the frozen bundle (keeping the 4B GGUF beside it) and serve it on :8660.
set -e
cd "$(dirname "$0")/.."
for p in $(pgrep -f "vite preview --outDir /tmp/dist-poste[r]" || true); do kill "$p" 2>/dev/null || true; done
(cd web && npx vite build --outDir /tmp/dist-poster-new --emptyOutDir 2>&1 | grep -E "built in|error" || true)
mkdir -p /tmp/dist-poster-new/dev-models
if [ -f /tmp/dist-poster/dev-models/q4b-Q4_K_M.gguf ]; then mv /tmp/dist-poster/dev-models/q4b-Q4_K_M.gguf /tmp/dist-poster-new/dev-models/; else cp models/q4b-Q4_K_M.gguf /tmp/dist-poster-new/dev-models/; fi
rm -rf /tmp/dist-poster && mv /tmp/dist-poster-new /tmp/dist-poster
(cd web && nohup npx vite preview --outDir /tmp/dist-poster --port 8660 --strictPort > /tmp/preview-poster.log 2>&1 &)
for i in $(seq 1 30); do curl -sf -o /dev/null http://localhost:8660/ && break; sleep 0.5; done
echo "served on :8660"
