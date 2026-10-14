#!/bin/bash
# RUNG-0 RE-FIT — 3 specialty + 9 mem plugs on vk-base, now that the
# row-space bug (6c89e7a) is fixed.  Claim-locked, skip-done, detached.
# Impress runs HEADLESS (bank_rung1.py --phase impress, verdict-only JSON) —
# chat_vfinal.py decode bug is agent-58's lane; we do not touch that file.
set -u
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm-6.2.4/lib/rocblas/library
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
PY=/data/tliao/vesper_sft/venv/bin/python
REPO=/data/tliao/vesper_sft/VesperLM
OUT=/data/tliao/vk_final
CLAIM_DIR=/data/hackathon/.tmp/mr_gpu_claims
mkdir -p "$OUT/logs" "$OUT/locks" "$CLAIM_DIR"
cd "$REPO"
log() { echo "[$(date +%H:%M:%S)] $*" >> "$OUT/logs/rung0_chain.log"; }

NEED=$((6 * 1024 * 1024 * 1024))
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
  exec 9>"$CLAIM_DIR/lock"; flock 9
  local best=-1 best_free=0 g f
  for g in 0 1 2 3 4 5 6 7; do
    claimed "$g" && continue
    f=$(free_of "$g")
    if (( f > best_free )); then best_free=$f; best=$g; fi
  done
  if (( best_free >= NEED )); then touch "$CLAIM_DIR/gpu$best"; echo "$best"; fi
  exec 9>&-
}

ARGS="--model $OUT/chat_model --registry $OUT/registry_vk.pt --plugins-dir $OUT \
  --specialties-dir $REPO/Dataset/data/specialties --cores-dir /data/tliao/v05_cores \
  --device cuda --dtype float16 --skip-done --out $OUT/vk_results.json"

run_claimed() {
  local name=$1; shift
  local gpu=""
  for a in $(seq 1 300); do
    gpu=$(pick_and_claim)
    [[ -n "$gpu" ]] && break
    log "$name waiting for GPU (attempt $a)"
    sleep 120
  done
  [[ -z "$gpu" ]] && { log "$name no GPU — giving up"; return 1; }
  log "$name on GPU $gpu"
  ( while true; do touch "$CLAIM_DIR/gpu$gpu" 2>/dev/null; sleep 240; done ) &
  local refpid=$!
  HIP_VISIBLE_DEVICES=$gpu "$@" >> "$OUT/logs/${name}.log" 2>&1
  kill "$refpid" 2>/dev/null
  rm -f "$CLAIM_DIR/gpu$gpu" 2>/dev/null
  log "$name done"
}

log "rung0 chain start (row-space fix 6c89e7a)"

# three fit lanes in parallel (wrap subsets from VK_WRAPS)
( run_claimed fit_A $PY lab/vk_final_build.py --phase fit $ARGS \
    --wrap-subset k3_hack,glm_cybersec,mimo_writing --out $OUT/fit_A.json ) &
( run_claimed fit_B $PY lab/vk_final_build.py --phase fit $ARGS \
    --wrap-subset mem_pet,mem_edit,mem_words,mem_poi --out $OUT/fit_B.json ) &
( run_claimed fit_C $PY lab/vk_final_build.py --phase fit $ARGS \
    --wrap-subset mem_machine,mem_gundam,mem_cars,mem_trains,mem_igbt \
    --out $OUT/fit_C.json ) &
wait
log "fit lanes done"

run_claimed assemble $PY lab/vk_final_build.py --phase assemble $ARGS
log "assemble done"

# headless impress (verdict-only JSON) — NOT the chat_vfinal CLI
mkdir -p /data/tliao/rung0_impress
run_claimed impress $PY lab/bank_rung1.py --phase impress \
  --root /data/tliao/rung0_impress --model $OUT/chat_model \
  --registry $OUT/registry_vk.pt --extra-plugins-dir $OUT \
  --device cuda --dtype float16 --varact-k 4
log "RUNG0 CHAIN ALL DONE"
