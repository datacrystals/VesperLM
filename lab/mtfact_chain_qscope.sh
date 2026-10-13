#!/bin/bash
# mtfact chain_qscope — row-selective retrieval scope (MODULAR_MOE 10.10
# rung 1): question-anchored activation bars.  Cheapest-first falsifier:
#   qdiag            question-side evidence separation (overlap + qspan p)
#   eval overlap     mechanism 1a: token-overlap question-match gate
#   eval qspan       mechanism 1b: row p_home on question tokens
#   eval ov+bonus    1a with bonus only (no row boost)
#   eval margin      mechanism 3: 1a + row-selective margin 0.2
#   eval slot        mechanism 2: 1a + answer-slot-only bonus (slot-k 8)
#   canary           no taught-name leaks
#   probe live       chat_v05 --probe with the scope (bar c) + baseline
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
log() { echo "[$(date +%H:%M:%S)] $*" >> "$OUT/chain_qscope.log"; }

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

gpu=""
for a in $(seq 1 40); do
  gpu=$(pick_and_claim)
  [[ -n "$gpu" ]] && break
  log "waiting for free GPU (attempt $a)"
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

# --- mechanism 1 falsifier: question-side evidence separation -----------
run_phase qdiag lab/mtfact.py --phase qdiag --gpu 0 --turns 0,4,8 \
  --out "$OUT/qdiag.json"

# --- bar (a)+(b): controlled grid, mechanism 1a (overlap) ---------------
run_phase eval_qsov_both lab/mtfact.py --phase eval --gpu 0 --fix both \
  --beta 3.0 --qscope overlap --turns 0,4,8 --gen-new 24 \
  --out "$OUT/eval_qsov_both.json"

# --- mechanism 1b (question-span p_home) --------------------------------
run_phase eval_qsv_both lab/mtfact.py --phase eval --gpu 0 --fix both \
  --beta 3.0 --qscope qspan --turns 0,4,8 --gen-new 24 \
  --out "$OUT/eval_qsv_both.json"

# --- 1a with bonus only (no row boost) ----------------------------------
run_phase eval_qsov_tok lab/mtfact.py --phase eval --gpu 0 --fix tokenbonus \
  --beta 3.0 --qscope overlap --turns 0,4,8 --gen-new 24 \
  --out "$OUT/eval_qsov_tok.json"

# --- mechanism 3: row-selective margin ---------------------------------
run_phase eval_qsov_marg lab/mtfact.py --phase eval --gpu 0 --fix both \
  --beta 3.0 --qscope overlap --qmargin 0.2 --turns 0,4,8 --gen-new 24 \
  --out "$OUT/eval_qsov_marg.json"

# --- mechanism 2: answer-slot-only -------------------------------------
run_phase eval_qsov_slot lab/mtfact.py --phase eval --gpu 0 --fix both \
  --beta 3.0 --qscope overlap --slot-k 8 --turns 0,4,8 --gen-new 24 \
  --out "$OUT/eval_qsov_slot.json"

# --- bar (d): canary, winner candidate 1a ------------------------------
run_phase canary_qsov lab/mtfact.py --phase canary --gpu 0 --fix both \
  --beta 3.0 --qscope overlap --gen-new 24 \
  --out "$OUT/canary_qsov.json"

# --- bar (c): live path (generated fillers) + shipped baseline ----------
run_phase probe_qsov lab/chat_v05.py --probe --gpu 0 --max-new-tokens 80 \
  --mem-bonus 3.0 --mem-boost --qscope overlap
run_phase probe_base lab/chat_v05.py --probe --gpu 0 --max-new-tokens 80 \
  --mem-bonus off --no-mem-boost --qscope none

kill $REFPID 2>/dev/null
log "chain_qscope complete"
