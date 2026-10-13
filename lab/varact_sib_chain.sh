#!/bin/bash
# varact sibling-recal chain — claim-aware GPU grab, OOM-retry, incremental JSON.
# Same claim protocol as door10_chain.sh. Backs up joint_state.pt before any
# write (sibrecal itself does it); never touches user tmux sessions.
set -u
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm-6.2.4/lib/rocblas/library
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
PY=/data/tliao/vesper_sft/venv/bin/python
OUT=/data/tliao/varact_out
CLAIM_DIR=/data/hackathon/.tmp/mr_gpu_claims
mkdir -p "$OUT" "$CLAIM_DIR"
cd /data/tliao/vesper_sft/VesperLM
log() { echo "[$(date +%H:%M:%S)] $*" >> "$OUT/sib_chain.log"; }

NEED=$((20 * 1024 * 1024 * 1024))

free_of() {
  local g=$1 used total
  used=$(rocm-smi -d "$g" --showmeminfo vram 2>/dev/null | awk '/Total Used/{print $NF}')
  total=$(rocm-smi -d "$g" --showmeminfo vram 2>/dev/null | awk '/Total Memory/{print $NF}')
  [[ -z "$used" || -z "$total" ]] && { echo 0; return; }
  echo $((10#$total - 10#$used))
}

claimed() {
  local f="$CLAIM_DIR/gpu$1"
  [[ -f "$f" ]] && (( $(date +%s) - $(stat -c %Y "$f") < 600 ))
}

pick_and_claim() {
  exec 9>"$CLAIM_DIR/lock"
  flock 9
  local best=-1 best_free=0 g f
  for g in $(rocm-smi -i 2>/dev/null | grep -E '^GPU\[[0-9]+\]' | awk -F'[][]' '{print $2}' | sort -n | uniq); do
    claimed "$g" && continue
    f=$(free_of "$g")
    if (( f > best_free )); then best_free=$f; best=$g; fi
  done
  if (( best_free >= NEED )); then
    touch "$CLAIM_DIR/gpu$best"
    echo "$best"
  fi
  exec 9>&-
}

for a in $(seq 1 30); do
  gpu=$(pick_and_claim)
  if [[ -z "$gpu" ]]; then
    log "waiting for free GPU (attempt $a)"
    sleep 30
    continue
  fi
  log "attempt $a gpu=$gpu"
  ( while true; do touch "$CLAIM_DIR/gpu$gpu" 2>/dev/null; sleep 240; done ) &
  refpid=$!
  HIP_VISIBLE_DEVICES=$gpu $PY lab/varact_sib_recal.py --phase all \
    --out "$OUT/sibrecal.json" --device cuda --dtype float16 "$@" \
    >> "$OUT/sibrecal.log" 2>&1
  rc=$?
  kill $refpid 2>/dev/null
  rm -f "$CLAIM_DIR/gpu$gpu"
  if [ $rc -eq 0 ]; then log "OK gpu=$gpu"; exit 0; fi
  log "attempt $a rc=$rc gpu=$gpu"
  tail -5 "$OUT/sibrecal.log" >> "$OUT/sib_chain.log"
  sleep 20
done
log "GAVE UP"
exit 1
