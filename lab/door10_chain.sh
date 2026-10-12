#!/bin/bash
# Door 10 calibration-scale sweep — per-budget arms, claim-aware GPU grab.
#
# Ops: detached setsid, incremental JSON per arm, no pgrep gates.
# Coexistence: the box's margin_router launchers pick "GPU with >=19GiB free
# not claimed" via /data/hackathon/.tmp/mr_gpu_claims (10-min claim TTL).
# We use the SAME claim protocol: flock-pick+claim a GPU with >=20GiB free,
# refresh the claim while our job runs, release on exit. That is what stops
# the co-tenant race that OOM'd the first smoke (they claimed the card
# between our pick and model load).
set -u
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm-6.2.4/lib/rocblas/library
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/data/tliao/vesper_sft/venv/bin/python
OUT=/data/tliao/graft_door10_out
SLICE=/data/tliao/graft_door10_slice
CLAIM_DIR=/data/hackathon/.tmp/mr_gpu_claims
mkdir -p "$OUT" "$CLAIM_DIR"
cd /data/tliao/vesper_sft/VesperLM
log() { echo "[$(date +%H:%M:%S)] $*" >> "$OUT/door10_chain.log"; }

MODEL=/data/tliao/OLMoE-1B-7B-0924
REG=/data/tliao/graft_d0_out/registry_olmoe_6d84c485.pt
CORE=/data/tliao/graft_door25_out/door25_core.pt
DOM=/data/tliao/graft_d0_slice
NEED=$((20 * 1024 * 1024 * 1024))   # ours peaks ~18GiB; leave headroom

free_of() {
  local g=$1 used total
  used=$(rocm-smi -d "$g" --showmeminfo vram 2>/dev/null | awk '/Total Used/{print $NF}')
  total=$(rocm-smi -d "$g" --showmeminfo vram 2>/dev/null | awk '/Total Memory/{print $NF}')
  [[ -z "$used" || -z "$total" ]] && { echo 0; return; }
  echo $((10#$total - 10#$used))
}

# claimed by ANYONE else (we keep our own claims fresh ourselves)
claimed() {
  local f="$CLAIM_DIR/gpu$1"
  [[ -f "$f" ]] && (( $(date +%s) - $(stat -c %Y "$f") < 600 ))
}

# flock-pick a GPU, claim it, print id (empty if none)
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

run_arm() {
  local name=$1 budget=$2
  local gpu="" refpid=""
  for a in $(seq 1 60); do
    gpu=$(pick_and_claim)
    if [[ -z "$gpu" ]]; then
      log "$name waiting for free GPU (attempt $a)"
      sleep 30
      continue
    fi
    log "$name attempt $a gpu=$gpu budget=$budget"
    # keep OUR claim fresh while the arm runs (the 10-min TTL would
    # otherwise let the vultures pile back onto our card mid-run)
    ( while [[ ! -f "$OUT/door10_b${budget}.done" ]]; do
        touch "$CLAIM_DIR/gpu$gpu" 2>/dev/null
        sleep 240
      done ) &
    refpid=$!
    HIP_VISIBLE_DEVICES=$gpu $PY lab/door10_calscale.py \
      --model $MODEL --registry $REG --core-from $CORE \
      --slices-dir $SLICE --domains-dir $DOM \
      --budgets $budget \
      --out "$OUT/door10_b${budget}.json" \
      --device cuda --dtype float16 \
      >> "$OUT/door10_b${budget}.log" 2>&1
    local rc=$?
    touch "$OUT/door10_b${budget}.done"
    kill $refpid 2>/dev/null
    rm -f "$CLAIM_DIR/gpu$gpu"
    if [ $rc -eq 0 ]; then log "$name OK gpu=$gpu"; return 0; fi
    log "$name attempt $a rc=$rc gpu=$gpu"
    sleep 20
  done
  log "$name GAVE UP"
  return 1
}

log "chain start"
run_arm b11k  11000  & P1=$!
sleep 3
run_arm b50k  50000  & P2=$!
sleep 3
run_arm b200k 200000 & P3=$!
sleep 3
run_arm b500k 500000 & P4=$!
wait $P1; R1=$?
wait $P2; R2=$?
wait $P3; R3=$?
wait $P4; R4=$?
log "lanes r11k=$R1 r50k=$R2 r200k=$R3 r500k=$R4"

$PY lab/door10_calscale.py --merge \
  "$OUT/door10_b11000.json" "$OUT/door10_b50000.json" \
  "$OUT/door10_b200000.json" "$OUT/door10_b500000.json" \
  --out "$OUT/door10_merged.json" >> "$OUT/door10_merge.log" 2>&1
log "merge rc=$?"
log "chain done"
