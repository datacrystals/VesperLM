#!/bin/bash
# persona_lane.sh <spine> <tag> [steps] — one persona wrap fit with
# door10 claim-lock protocol + OOM retry (bank2n/v05 lane lessons).
set -u
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm-6.2.4/lib/rocblas/library
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
SPINE=$1
TAG=$2
STEPS=${3:-300}
PY=/data/tliao/vesper_sft/venv/bin/python
ROOT=/data/tliao/vesper_sft/VesperLM
CLAIM_DIR=/data/hackathon/.tmp/mr_gpu_claims
mkdir -p "$CLAIM_DIR"
cd "$ROOT"
log() { echo "[$(date +%H:%M:%S)] lane $SPINE $TAG $*" >> /data/tliao/persona_out/lanes.log; }

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
  if (( best_free >= NEED )); then touch "$CLAIM_DIR/gpu$best"; echo "$best"; fi
  exec 9>&-
}

mkdir -p /data/tliao/persona_out
for attempt in 1 2 3 4 5 6; do
  gpu=""
  for a in $(seq 1 60); do
    gpu=$(pick_and_claim)
    [[ -n "$gpu" ]] && break
    log "waiting gpu attempt $a"
    sleep 20
  done
  [[ -z "$gpu" ]] && { log "FATAL no GPU"; exit 1; }
  log "claimed gpu=$gpu attempt $attempt"
  ( while true; do touch "$CLAIM_DIR/gpu$gpu" 2>/dev/null; sleep 240; done ) &
  REFPID=$!
  HIP_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu \
    $PY lab/persona_build.py --spine "$SPINE" --phase fit \
      --wrap-subset "$TAG" --expert-steps "$STEPS" --skip-done \
      --out /data/tliao/persona_out/fit_${SPINE}_${TAG}.json \
      > /data/tliao/persona_out/fit_${SPINE}_${TAG}.log 2>&1
  rc=$?
  kill $REFPID 2>/dev/null
  rm -f "$CLAIM_DIR/gpu$gpu" 2>/dev/null
  log "fit rc=$rc"
  if [[ $rc -eq 0 ]]; then exit 0; fi
  tail -3 /data/tliao/persona_out/fit_${SPINE}_${TAG}.log >> /data/tliao/persona_out/lanes.log
  sleep 5
done
log "GAVE UP"
exit 1
