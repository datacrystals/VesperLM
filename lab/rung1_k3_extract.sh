#!/bin/bash
# Laptop-side K3 core extraction (box's Kimi-K3 is LFS stubs — real shards
# live here) + rsync to the box core_bank.  CPU only, skip-done, detached-safe.
set -u
REPO=/home/tliao/VesperLM
LOCAL=/home/tliao
OUT=$LOCAL/rung1
mkdir -p "$OUT/results" "$OUT/logs"
cd "$REPO"
PY=python3
log() { echo "[$(date +%H:%M:%S)] $*" >> "$OUT/logs/k3_extract.log"; }

log "manifest"
$PY lab/bank_rung1.py --phase manifest --root "$OUT" --local-dir "$LOCAL" \
  >> "$OUT/logs/manifest.log" 2>&1

log "k3 extract start"
$PY lab/bank_rung1.py --phase extract --root "$OUT" --donor kimi-k3 \
  --local-dir "$LOCAL" --model-dir "$LOCAL/Desktop/Kimi-K3" \
  >> "$OUT/logs/k3_extract_run.log" 2>&1
n=$(find "$LOCAL/core_bank/kimi-k3" -name core.safetensors 2>/dev/null | wc -l)
log "extracted $n k3 cores locally"

log "rsync -> box"
for a in $(seq 1 30); do
  rsync -a --partial --timeout=120 "$LOCAL/core_bank/kimi-k3/" \
    tliao@100.64.0.7:/data/tliao/core_bank/kimi-k3/ \
    >> "$OUT/logs/k3_rsync.log" 2>&1 && break
  log "rsync retry $a"
  sleep 60
done
log "K3 EXTRACT+SHIP DONE ($n cores)"
