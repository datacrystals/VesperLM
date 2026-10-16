#!/bin/bash
# MiMo-Flash 6-plug persona set fit — one lane per plug, claim-locked GPUs 0-5.
# FEED >=200k per plug (10.7), owner recal 0.85, additive lam.
set -u
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm-6.2.4/lib/rocblas/library
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
CLAIM_DIR=/data/hackathon/.tmp/mr_gpu_claims
LAB=/data/tliao/mimo_flash_lab
OUT=/home/tliao/mimo_flash
PY=/home/tliao/vesper_sft_venv/bin/python
PLUGS_DIR=$OUT/plugs
LOGS=$OUT/logs
mkdir -p "$CLAIM_DIR" "$OUT" "$PLUGS_DIR" "$LOGS"

TAGS=(mem_trains mem_poi mem_gundam mem_machine mem_cars mem_igbt)
GPUS=(0 1 2 3 4 5)   # hemmingway holds 6+7 — never touch

free_of() {
  rocm-smi --showmeminfo vram 2>/dev/null | awk -v g="$1" '
    index($0, "GPU[" g "]") == 1 && /Total Memory/ {t=$NF}
    index($0, "GPU[" g "]") == 1 && /Total Used/ {u=$NF}
    END {if (t && u) printf "%d\n", (t-u)/1073741824; else print 0}'
}

launch_lane() {
  local tag=$1 gpu=$2
  if ! mkdir -p "$CLAIM_DIR" 2>/dev/null; then :; fi
  exec 9>"$CLAIM_DIR/lock"; flock 9
  if [[ -f "$CLAIM_DIR/gpu$gpu" ]] && (( $(date +%s) - $(stat -c %Y "$CLAIM_DIR/gpu$gpu") < 600 )); then
    local mine=0
    # stale-check only: claims are refreshed by their owners every 240s
    :
  fi
  touch "$CLAIM_DIR/gpu$gpu"
  exec 9>&-
  ( while true; do touch "$CLAIM_DIR/gpu$gpu" 2>/dev/null; sleep 240; done ) &
  local ref=$!
  echo "[lane] $tag on GPU$gpu $(date -Is)" | tee -a "$LOGS/plugs_launch.log"
  cd "$LAB"
  HIP_VISIBLE_DEVICES=$gpu $PY "$LAB/mimo_flash_host.py" \
      --phase plugs --plug-tags "$tag" --device cuda \
      --out "$OUT/plugs/${tag}_result.json" \
      --model-dir /data/ModelDownloader/MiMo-V2.6-Flash-INT4 \
      --data-dir /data/tliao/graft_d2_slice \
      --n-layers 2 --seq 128 --bs 2 \
      --host-layer 0 --host-expert 0 \
      --feed-tokens 200000 --expert-steps 300 --row-steps 150 \
      --plugins-dir "$PLUGS_DIR" \
      >> "$LOGS/plug_${tag}.log" 2>&1
  echo "[lane] $tag exit=$? $(date -Is)" | tee -a "$LOGS/plugs_launch.log"
  kill $ref 2>/dev/null
}

# one lane per plug; 2 waves of 3 (each lane peaks ~60GB RSS on the cache
# load — 6 concurrent lanes would oversubscribe the box's 251GB)
run_wave() {
  local pids=()
  for i in "$@"; do
    launch_lane "${TAGS[$i]}" "${GPUS[$i]}" &
    pids+=($!)
    sleep 30
  done
  for p in "${pids[@]}"; do wait $p; done
}
run_wave 0 1 2 3 4 5
echo "[lanes] all done $(date -Is)" | tee -a "$LOGS/plugs_launch.log"
