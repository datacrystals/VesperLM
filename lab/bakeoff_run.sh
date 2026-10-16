#!/bin/bash
# Bake-off delivery: mimo-flash arm + glm-flash arm + compare.
# Both arms frozen protocol (spine_bakeoff.py): convo_test.DIALOGUES,
# score_turn rubric, T=0.8 top-p 0.95 rep 1.1 max-new 128, seed 0.
set -u
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm-6.2.4/lib/rocblas/library
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
CLAIM_DIR=/data/hackathon/.tmp/mr_gpu_claims
LAB=/data/tliao/mimo_flash_lab
OUT=/data/tliao/bakeoff
PY=/home/tliao/vesper_sft_venv/bin/python
mkdir -p "$OUT" "$CLAIM_DIR" /home/tliao/mimo_flash/logs

run_arm() {
  local spine=$1 gpu=$2 plugins=$3
  touch "$CLAIM_DIR/gpu$gpu"
  ( while true; do touch "$CLAIM_DIR/gpu$gpu" 2>/dev/null; sleep 240; done ) &
  local ref=$!
  echo "[bake] $spine on GPU$gpu $(date -Is)" | tee -a /home/tliao/mimo_flash/logs/bake.log
  cd "$LAB"
  HIP_VISIBLE_DEVICES=$gpu $PY "$LAB/spine_bakeoff.py" \
      --spine "$spine" --backend graft --plugins "$plugins" \
      --out "$OUT/${spine//-/_}.json" \
      >> "/home/tliao/mimo_flash/logs/bake_${spine}.log" 2>&1
  echo "[bake] $spine exit=$? $(date -Is)" | tee -a /home/tliao/mimo_flash/logs/bake.log
  kill $ref 2>/dev/null
}

# run both arms in parallel — one GPU each (fits have released theirs)
run_arm mimo-flash 0 /home/tliao/mimo_flash/plugs &
run_arm glm-flash 1 /data/tliao/glmflash_out &
wait

$PY "$LAB/spine_bakeoff.py" --compare "$OUT/mimo_flash.json" "$OUT/glm_flash.json" \
    2>&1 | tee /home/tliao/mimo_flash/logs/compare.txt
echo "[bake] COMPLETE $(date -Is)" | tee -a /home/tliao/mimo_flash/logs/bake.log
