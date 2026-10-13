#!/bin/bash
# persona_chain.sh — the whole persona pack build, detached:
#   12 fit lanes (6 persona wraps x phi/qwen) on claim-locked GPUs, then
#   per-spine assemble (additive install + guard-checkpointed sibling recal +
#   lam grid + probe battery) once that spine's 6 plugins exist and are finite.
# Launch: setsid nohup bash lab/persona_chain.sh </dev/null \
#           >> /data/tliao/persona_out/chain.log 2>&1 &
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
log() { echo "[$(date +%H:%M:%S)] chain $*" >> "$OUT/chain.log"; }

TAGS="mem_poi,mem_machine,mem_gundam,mem_cars,mem_trains,mem_igbt"
declare -a TAG_ARR=(${TAGS//,/ })

# ---- fit lanes (background; each claims its own GPU) ----
for spine in phi qwen; do
  for tag in "${TAG_ARR[@]}"; do
    log "launch lane $spine $tag"
    bash lab/persona_lane.sh "$spine" "$tag" 300 &
  done
done

# ---- per-spine assemble watcher (file-based gate; no pgrep) ----
plugins_ready() {
  local dir=$1 tag
  for tag in "${TAG_ARR[@]}"; do
    [[ -f "$dir/${tag}_plugin.pt" ]] || return 1
  done
  $PY - "$dir" <<'EOF'
import sys, torch, os
sys.path.insert(0, "lab")
import door6_expert_mix as d6
d = sys.argv[1]
for t in ("mem_poi","mem_machine","mem_gundam","mem_cars","mem_trains","mem_igbt"):
    w, r, _, _ = d6.load_wrap_plugin(os.path.join(d, t + "_plugin.pt"))
    if not d6.wrap_is_finite(w, r):
        sys.exit(1)
sys.exit(0)
EOF
}

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

assemble_spine() {
  local spine=$1 dir=$2
  log "watch $spine ($dir)"
  for a in $(seq 1 240); do
    if plugins_ready "$dir"; then break; fi
    sleep 30
  done
  if ! plugins_ready "$dir"; then
    log "$spine plugins NEVER ready — skip assemble"
    return 1
  fi
  log "$spine plugins ready — assemble"
  local gpu=""
  for a in $(seq 1 60); do
    gpu=$(pick_and_claim)
    [[ -n "$gpu" ]] && break
    sleep 20
  done
  [[ -z "$gpu" ]] && { log "$spine FATAL no GPU for assemble"; return 1; }
  ( while true; do touch "$CLAIM_DIR/gpu$gpu" 2>/dev/null; sleep 240; done ) &
  local refpid=$!
  HIP_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu \
    $PY lab/persona_build.py --spine "$spine" --phase assemble \
      --out "$OUT/assemble_$spine.json" \
      > "$OUT/assemble_$spine.log" 2>&1
  local rc=$?
  kill $refpid 2>/dev/null
  rm -f "$CLAIM_DIR/gpu$gpu" 2>/dev/null
  log "$spine assemble rc=$rc"
  return $rc
}

assemble_spine phi /data/tliao/v05_out &
assemble_spine qwen /data/tliao/v06_out &
wait
log "chain complete"
