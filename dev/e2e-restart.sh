#!/bin/bash
# Rebuild the frozen e2e bundle, restart its preview server on :8657 and rerun dev/e2e.mjs.
# usage (inside the container, from the repo root): dev/e2e-restart.sh <model-id> <gguf-file> [text]
set -e
cd "$(dirname "$0")/.."
for p in $(pgrep -f "node dev/e2e[.]mjs" || true) $(pgrep -f "vite preview --outDir /tmp/dist-e2[e]" || true); do kill "$p" 2>/dev/null || true; done
(cd web && npx vite build --outDir /tmp/dist-e2e-new --emptyOutDir 2>&1 | grep -E "built in|error" || true)
mkdir -p /tmp/dist-e2e-new/dev-models
cp "models/$2" /tmp/dist-e2e-new/dev-models/
rm -rf /tmp/dist-e2e && mv /tmp/dist-e2e-new /tmp/dist-e2e
(cd web && nohup npx vite preview --outDir /tmp/dist-e2e --port 8657 --strictPort > /tmp/preview-e2e.log 2>&1 &)
for i in $(seq 1 30); do curl -sf -o /dev/null http://localhost:8657/ && break; sleep 0.5; done
[ -n "$NO_E2E" ] && { echo "served on :8657"; exit 0; }
rm -rf /tmp/e2e && mkdir -p /tmp/e2e
nohup node dev/e2e.mjs /tmp/e2e "$1" "/dev-models/$2" "${3:-Lionel Messi}" > /tmp/e2e.log 2>&1 &
echo "e2e started"
