#!/bin/bash
# Build the Nemotron pretrain blend after the code corpus job releases the CPU.
set -u
cd "$(dirname "$0")"
PY=/home/tliao/venvs/gen/bin/python

while pgrep -f "04_prepare_code" > /dev/null; do
    sleep 30
done
echo "[$(date '+%H:%M:%S')] code corpus done; starting nemotron blend"
$PY 05_nemotron_dataset.py > nemotron.log 2>&1
echo "[$(date '+%H:%M:%S')] nemotron rc=$?"
ls -la data/pretrain/nemotron_phase*.bin 2>/dev/null
