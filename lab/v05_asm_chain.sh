#!/bin/bash
# v0.5 assemble — wait on the 3 fit JSONs (file-based, no pgrep gates), then
# run the 8-plug additive joint on the freest GPU.
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm-6.2.4/lib/rocblas/library
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/tliao/vesper_sft/VesperLM
PY=/data/tliao/vesper_sft/venv/bin/python
OUT=/data/tliao/v05_out
while true; do
  ok=1
  for name in k3_hack glm_cybersec mimo_writing; do
    python3 -c "
import json,sys
d=json.load(open('$OUT/fit_$name.json'))
ph=d.get('phase','')
fi=d.get('fitted',{})
done = any(v.get('skipped') or v.get('plugin') for v in fi.values()) or ph=='fit_done' or ph.startswith('failed')
sys.exit(0 if done else 1)
" 2>/dev/null || ok=0
  done
  [ $ok -eq 1 ] && break
  sleep 30
done
echo "fits complete $(date +%H:%M:%S)" >> $OUT/launcher.log
for name in k3_hack glm_cybersec mimo_writing; do
  echo "== $name:" >> $OUT/launcher.log
  tail -2 $OUT/fit_$name.log >> $OUT/launcher.log
done
G=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '
  /Used Memory/{u=$NF; printf "%d %d\n", t-u, idx; next}
  /Total Memory/{t=$NF; match($1,/[0-9]+/); idx=substr($1,RSTART,RLENGTH)}
' | sort -rn | head -1 | awk '{print $2}')
echo "assemble on gpu $G $(date +%H:%M:%S)" >> $OUT/launcher.log
setsid env CUDA_VISIBLE_DEVICES=$G HIP_VISIBLE_DEVICES=$G \
  $PY lab/v05_build.py --phase assemble \
  --out $OUT/v05_results.json > $OUT/assemble.log 2>&1 &
echo "assemble pid $!" >> $OUT/launcher.log
