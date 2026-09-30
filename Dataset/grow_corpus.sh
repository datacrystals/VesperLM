#!/bin/bash
# Grow the pretrain corpus from the locally cached HF datasets.
# Writes fineweb_pretrain_v2.bin / code_pretrain_v2.bin, then repoints
# the phase symlinks so data/index.txt picks up the new bins.
set -u
cd "$(dirname "$0")"
PY=/home/tliao/venvs/gen/bin/python

echo "[$(date '+%H:%M:%S')] fineweb start"
$PY 03_fineweb.py >> corpus_v2.log 2>&1
echo "[$(date '+%H:%M:%S')] fineweb done rc=$?"

echo "[$(date '+%H:%M:%S')] code start"
$PY 04_prepare_code.py >> corpus_v2.log 2>&1
echo "[$(date '+%H:%M:%S')] code done rc=$?"

ln -sfn ../fineweb_pretrain_v2.bin data/pretrain/phase1_pretrain.bin
ln -sfn ../code_pretrain_v2.bin data/pretrain/phase2_pretrain.bin
ls -la data/pretrain/ data/*_v2.bin
echo "[$(date '+%H:%M:%S')] corpus v2 ready"
