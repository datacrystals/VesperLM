#!/bin/bash
# Door 10 recovery chain — SEQUENTIAL per-budget arms (one card at a time;
# parallel arms thrash against co-tenant memory grabs) + merge.
# Ops: detached setsid, incremental JSON, no pgrep gates, claim-aware GPU
# grab (/data/hackathon/.tmp/mr_gpu_claims) with claim refresh during runs.
set -u
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm-6.2.4/lib/rocblas/library
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/data/tliao/vesper_sft/venv/bin/python
OUT=/data/tliao/graft_door10_out
SLICE=/data/tliao/graft_door10_slice
CLAIM_DIR=/data/hackathon/.tmp/mr_gpu_claims
mkdir -p "$OUT" "$CLAIM_DIR"
cd /data/tliao/vesper_sft/VesperLM
log() { echo "[$(date +%H:%M:%S)] $*" >> "$OUT/door10_chain2.log"; }

MODEL=/data/tliao/OLMoE-1B-7B-0924
REG=/data/tliao/graft_d0_out/registry_olmoe_6d84c485.pt
CORE=/data/tliao/graft_door25_out/door25_core.pt
DOM=/data/tliao/graft_d0_slice
NEED=$((19 * 1024 * 1024 * 1024))

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

run_arm() {
  local name=$1 budget=$2
  # already done? (resume-friendly)
  if [[ -f "$OUT/door10_b${budget}.done" ]]; then
    log "$name skip (done marker)"; return 0
  fi
  local gpu="" refpid=""
  for a in $(seq 1 200); do
    gpu=$(pick_and_claim)
    if [[ -z "$gpu" ]]; then
      log "$name waiting for free GPU (attempt $a)"
      sleep 30
      continue
    fi
    log "$name attempt $a gpu=$gpu budget=$budget free=$(free_of $gpu)"
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
    rm -f "$OUT/door10_b${budget}.done"   # allow retry
    sleep 20
  done
  log "$name GAVE UP"
  return 1
}

log "chain2 start (sequential)"
run_arm b50k  50000;  log "b50k rc=$?"
run_arm b200k 200000; log "b200k rc=$?"
run_arm b500k 500000; log "b500k rc=$?"

$PY lab/door10_calscale.py --merge \
  "$OUT/door10_b11000.json" "$OUT/door10_b50000.json" \
  "$OUT/door10_b200000.json" "$OUT/door10_b500000.json" \
  --out "$OUT/door10_merged.json" >> "$OUT/door10_merge.log" 2>&1
log "merge rc=$?"
log "chain2 done"
