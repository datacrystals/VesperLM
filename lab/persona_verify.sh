#!/bin/bash
# persona_verify.sh — wait for both persona assembles, then run the shipped
# bars on both spines: persona_probe (single/multi/canary/live) + the real
# chat CLI --probe path (bar c) with bare default flags.
set -u
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm-6.2.4/lib/rocblas/library
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/data/tliao/vesper_sft/venv/bin/python
ROOT=/data/tliao/vesper_sft/VesperLM
OUT=/data/tliao/persona_out
CLAIM_DIR=/data/hackathon/.tmp/mr_gpu_claims
mkdir -p "$OUT" "$CLAIM_DIR"
cd "$ROOT"
log() { echo "[$(date +%H:%M:%S)] verify $*" >> "$OUT/verify.log"; }

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

run_claimed() {
  local name=$1; shift
  local gpu=""
  for a in $(seq 1 90); do
    gpu=$(pick_and_claim)
    [[ -n "$gpu" ]] && break
    log "waiting gpu for $name attempt $a"
    sleep 20
  done
  [[ -z "$gpu" ]] && { log "FATAL no GPU for $name"; return 1; }
  ( while true; do touch "$CLAIM_DIR/gpu$gpu" 2>/dev/null; sleep 240; done ) &
  local refpid=$!
  log "start $name gpu=$gpu"
  HIP_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu "$@" \
    > "$OUT/$name.log" 2>&1
  local rc=$?
  kill $refpid 2>/dev/null
  rm -f "$CLAIM_DIR/gpu$gpu" 2>/dev/null
  log "done $name rc=$rc"
}

# wait for the two assemble verdicts (file-based; no pgrep)
for a in $(seq 1 240); do
  ok=1
  for sp in phi qwen; do
    [[ -f "$OUT/assemble_$sp.json" ]] || ok=0
    if [[ -f "$OUT/assemble_$sp.json" ]]; then
      $PY -c "import json,sys; d=json.load(open('$OUT/assemble_$sp.json')); sys.exit(0 if d.get('phase')=='assemble_done' else 1)" || ok=0
    fi
  done
  [[ $ok -eq 1 ]] && break
  sleep 30
done
log "assembles observed (or timed out) — verifying"

run_claimed probe_phi $PY lab/persona_probe.py --spine phi --phase all \
  --out "$OUT/probe_phi.json"
run_claimed probe_qwen $PY lab/persona_probe.py --spine qwen --phase all --kv \
  --out "$OUT/probe_qwen.json"
run_claimed chat_probe_phi $PY lab/chat_v05.py --probe --gpu 0 \
  --max-new-tokens 60
run_claimed chat_probe_qwen $PY lab/chat_v06.py --probe --gpu 0 \
  --probe-tokens 60
log "verify complete"
