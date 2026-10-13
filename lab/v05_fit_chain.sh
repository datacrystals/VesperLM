#!/bin/bash
# v0.5 fit lanes — dynamic freest-GPU claim, bs1x512 (bank2n co-tenant shape),
# 600 steps x 512 = 307200 tok-slots fed (>=200k door10 budget).
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm-6.2.4/lib/rocblas/library
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/tliao/vesper_sft/VesperLM
PY=/data/tliao/vesper_sft/venv/bin/python
OUT=/data/tliao/v05_out
mkdir -p $OUT
echo "=== launch $(date +%H:%M:%S) ===" >> $OUT/launcher.log
rocm-smi --showmeminfo vram 2>/dev/null | grep "Memory (B)" >> $OUT/launcher.log
GPUS=($(rocm-smi --showmeminfo vram 2>/dev/null | awk '
  /Used Memory/{u=$NF; printf "%d %d\n", t-u, idx; next}
  /Total Memory/{t=$NF; match($1,/[0-9]+/); idx=substr($1,RSTART,RLENGTH)}
' | sort -rn | awk '$1>20000000000{print $2}' | head -3))
echo "picked GPUs: ${GPUS[*]}" >> $OUT/launcher.log
i=0
for name in k3_hack glm_cybersec mimo_writing; do
  g=${GPUS[$i]:-7}; i=$((i+1))
  echo "launch $name on gpu $g $(date +%H:%M:%S)" >> $OUT/launcher.log
  setsid env CUDA_VISIBLE_DEVICES=$g HIP_VISIBLE_DEVICES=$g \
    $PY lab/v05_build.py --phase fit --wrap-subset $name \
    --batch-size 1 --expert-steps 500 \
    --out $OUT/fit_$name.json > $OUT/fit_$name.log 2>&1 &
done
echo "all launched $(date +%H:%M:%S)" >> $OUT/launcher.log
