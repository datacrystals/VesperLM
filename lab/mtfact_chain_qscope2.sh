#!/bin/bash
# mtfact chain_qscope2 — live-path probes (bar c) for the two scoped ship
# candidates after chain_qscope:
#   probe_qsov_tok     question-scoped bonus ONLY (no row boost) — the
#                      minimal-diff default-on candidate
#   probe_qsov_tok_b4  same at beta=4 (row-selective scope should make
#                      higher beta safe: only the matched row's taught ids
#                      ever see the bias)
# joint_state.pt is NEVER written.  Claim protocol = mtfact_chain.sh.
set -u
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm-6.2.4/lib/rocblas/library
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
PY=/data/tliao/vesper_sft/venv/bin/python
OUT=/data/tliao/mtfact_out
CLAIM_DIR=/data/hackathon/.tmp/mr_gpu_claims
mkdir -p "$OUT" "$CLAIM_DIR"
cd /data/tliao/vesper_sft/VesperLM
log() { echo "[$(date +%H:%M:%S)] $*" >> "$OUT/chain_qscope2.log"; }

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
gpu=""
for a in $(seq 1 40); do
  gpu=$(pick_and_claim)
  [[ -n "$gpu" ]] && break
  log "waiting gpu attempt $a"
  sleep 20
done
[[ -z "$gpu" ]] && { log "FATAL no GPU"; exit 1; }
log "claimed gpu=$gpu"
( while true; do touch "$CLAIM_DIR/gpu$gpu" 2>/dev/null; sleep 240; done ) &
REFPID=$!
run_phase() {
  local name=$1; shift
  log "start $name"
  HIP_VISIBLE_DEVICES=$gpu $PY "$@" >> "$OUT/$name.log" 2>&1
  log "done $name rc=$?"
}

run_phase probe_qsov_tok lab/chat_v05.py --probe --gpu 0 --max-new-tokens 80 \
  --mem-bonus 3.0 --no-mem-boost --qscope overlap
run_phase probe_qsov_tok_b4 lab/chat_v05.py --probe --gpu 0 --max-new-tokens 80 \
  --mem-bonus 4.0 --no-mem-boost --qscope overlap

kill $REFPID 2>/dev/null
log "chain_qscope2 complete"
