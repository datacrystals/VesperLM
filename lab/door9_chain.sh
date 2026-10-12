#!/bin/bash
# Door 9 chain (emb -> depth -> wrap arms) — detached ops rule; watch JSONs.
set -u
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm-6.2.4/lib/rocblas/library
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/data/tliao/vesper_sft/venv/bin/python
OUT=/data/tliao/graft_door9_out
mkdir -p "$OUT"
cd /data/tliao/vesper_sft/VesperLM
T0=$(date +%s)
log() { echo "[$(date +%H:%M:%S)] $*" >> "$OUT/door9_chain.log"; }

run() {
  local name=$1 gpu=$2; shift 2
  for a in 1 2 3; do
    log "$name attempt $a gpu=$gpu"
    HIP_VISIBLE_DEVICES=$gpu $PY lab/door9_embedding_w.py "$@" \
      >> "$OUT/${name}.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then log "$name OK"; return 0; fi
    log "$name attempt $a rc=$rc"
  done
  return 1
}

MODEL_A=/data/tliao/OLMoE-1B-7B-0924
MODEL_B=/data/tliao/Phi-mini-MoE-instruct
REG=/data/tliao/graft_d0_out/registry_olmoe_6d84c485.pt
CORE=/data/tliao/graft_door25_out/door25_core.pt
DOM=/data/tliao/graft_d0_slice

log "chain start"

run emb 2 --phase emb --model-a $MODEL_A --model-b $MODEL_B \
  --out-dir "$OUT" --out "$OUT/door9_emb.json" --mlp-steps 800 --device cuda
log "emb rc=$?"

run depth 2 --phase depth --model-a $MODEL_A --model-b $MODEL_B \
  --out-dir "$OUT" --out "$OUT/door9_depth.json" --depth-tokens 512 --device cuda
log "depth rc=$?"

run wrap_rand 2 --phase wrap --arm rand --model $MODEL_A --registry $REG \
  --core-from $CORE --domains-dir $DOM --out-dir "$OUT" \
  --out "$OUT/door9_wrap_rand.json" --plugin-out "$OUT/door9_rand_plugin.pt" &
PID1=$!
run wrap_winit 5 --phase wrap --arm winit --model $MODEL_A --registry $REG \
  --core-from $CORE --domains-dir $DOM --out-dir "$OUT" \
  --out "$OUT/door9_wrap_winit.json" --plugin-out "$OUT/door9_winit_plugin.pt" &
PID2=$!
wait $PID1; R1=$?
wait $PID2; R2=$?
log "arms r1=$R1 r2=$R2"

ELAPSED=$(( $(date +%s) - T0 ))
if [ $R1 -eq 0 ] && [ $R2 -eq 0 ] && [ $ELAPSED -lt 3600 ]; then
  run wrap_wboth 2 --phase wrap --arm wboth --model $MODEL_A --registry $REG \
    --core-from $CORE --domains-dir $DOM --out-dir "$OUT" \
    --out "$OUT/door9_wrap_wboth.json" --plugin-out "$OUT/door9_wboth_plugin.pt"
  log "wboth rc=$?"
fi
log "chain done elapsed=${ELAPSED}s"
