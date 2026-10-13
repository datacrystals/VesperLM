#!/bin/bash
# one v0.5 fit lane with OOM-retry across freest GPUs (bank2n lesson)
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm-6.2.4/lib/rocblas/library
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /data/tliao/vesper_sft/VesperLM
PY=/data/tliao/vesper_sft/venv/bin/python
OUT=/data/tliao/v05_out
name=$1
steps=${2:-500}
for attempt in 1 2 3; do
  if [ -f $OUT/${name}_plugin.pt ]; then
    $PY -c "import torch,sys; sys.path.insert(0,'lab'); import door6_expert_mix as d6; w,r,_,_=d6.load_wrap_plugin('$OUT/${name}_plugin.pt'); sys.exit(0 if d6.wrap_is_finite(w,r) else 1)" \
      && { echo "lane $name plugin ok" >> $OUT/launcher.log; exit 0; }
  fi
  G=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '
    /Used Memory/{u=$NF; printf "%d %d\n", t-u, idx; next}
    /Total Memory/{t=$NF; match($1,/[0-9]+/); idx=substr($1,RSTART,RLENGTH)}
  ' | sort -rn | head -1 | awk '{print $2}')
  echo "lane $name attempt $attempt gpu $G $(date +%H:%M:%S)" >> $OUT/launcher.log
  CUDA_VISIBLE_DEVICES=$G HIP_VISIBLE_DEVICES=$G \
    $PY lab/v05_build.py --phase fit --wrap-subset $name \
    --batch-size 1 --expert-steps $steps --skip-done \
    --out $OUT/fit_$name.json > $OUT/fit_$name.log 2>&1 && exit 0
  echo "lane $name attempt $attempt FAILED $(date +%H:%M:%S)" >> $OUT/launcher.log
  tail -3 $OUT/fit_$name.log >> $OUT/launcher.log
  sleep 5
done
echo "lane $name GAVE UP" >> $OUT/launcher.log
