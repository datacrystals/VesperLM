#!/bin/bash
# MiMo-Flash spine rung-1 lane — claim-aware, detached, <=2 GPUs.
# usage: setsid bash /data/tliao/mimo_flash_lab/launch.sh
set -u
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm-6.2.4/lib/rocblas/library
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
CLAIM_DIR=/data/hackathon/.tmp/mr_gpu_claims
LAB=/data/tliao/mimo_flash_lab
OUT=/home/tliao/mimo_flash
PY=/home/tliao/vesper_sft_venv/bin/python
mkdir -p "$CLAIM_DIR" "$OUT/logs"
exec 9>"$CLAIM_DIR/lock"; flock 9

free_of() {
  /home/tliao/vesper_sft_venv/bin/python - "$1" <<'PY'
import subprocess, sys
g = sys.argv[1]
try:
    txt = subprocess.run(["rocm-smi", "--showmeminfo", "vram"],
                         capture_output=True, text=True, timeout=20).stdout
except Exception:
    print(0); sys.exit()
tot = used = None
for line in txt.splitlines():
    if not line.startswith(f"GPU[{g}]"):
        continue
    if "Total Used" in line:
        used = int(line.rsplit(None, 1)[-1])
    elif "Total Memory" in line:
        tot = int(line.rsplit(None, 1)[-1])
print((tot - used) // 1073741824 if tot and used is not None else 0)
PY
}

claimed() {
  local f="$CLAIM_DIR/gpu$1"
  [[ -f "$f" ]] && (( $(date +%s) - $(stat -c %Y "$f") < 600 ))
}
pick() {
  local best=-1 best_free=0 g f
  for g in "$@"; do
    claimed "$g" && continue
    f=$(free_of "$g")
    if (( f > best_free )); then best_free=$f; best=$g; fi
  done
  if (( best_free >= 18 )); then touch "$CLAIM_DIR/gpu$best"; echo "$best"; fi
}

GPU=""
for a in $(seq 1 120); do
  exec 9>"$CLAIM_DIR/lock"; flock 9
  GPU=$(pick 5 6 1 4 0 2 3 7)
  exec 9>&-
  [[ -n "$GPU" ]] && break
  echo "[launch] claim-wait $a $(date -Is)" | tee -a "$OUT/logs/launch.log"
  sleep 60
done
if [[ -z "$GPU" ]]; then echo "[launch] no free GPU" | tee -a "$OUT/logs/launch.log"; exit 1; fi
exec 9>&-
( while true; do touch "$CLAIM_DIR/gpu$GPU" 2>/dev/null; sleep 240; done ) &
REF=$!
trap "kill $REF 2>/dev/null" EXIT
echo "[launch] mimo-flash rung1 GPU=$GPU pid $$ $(date -Is)" | tee -a "$OUT/logs/launch.log"

cd "$LAB"
echo "[launch] phase=all $(date -Is)" | tee -a "$OUT/logs/launch.log"
HIP_VISIBLE_DEVICES=$GPU $PY "$LAB/mimo_flash_host.py" \
    --phase wrap --out "$OUT/mimo_host_wrap.json" \
    --model-dir /data/ModelDownloader/MiMo-V2.6-Flash-INT4 \
    --data-dir /data/tliao/graft_d2_slice \
    --n-layers 2 --seq 128 --bs 2 --n-batches 6 \
    --host-layer 0 --host-expert 0 \
    --feed-tokens 50000 --expert-steps 300 \
    >> "$OUT/logs/all.log" 2>&1
echo "[launch] phase=all exit=$? $(date -Is)" | tee -a "$OUT/logs/launch.log"
echo "[launch] complete $(date -Is)" | tee -a "$OUT/logs/launch.log"
