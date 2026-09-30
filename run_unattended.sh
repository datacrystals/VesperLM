#!/bin/bash
# run_unattended.sh — supervisor for the VesperLM tiny-agent pipeline.
#
# Keeps the whole train->SFT->eval pipeline alive without a human or the
# Kimi session: restarts crashed stages, and when SFT finishes it runs
# the agent-harness eval automatically.
#
# Usage: nohup bash run_unattended.sh > unattended.log 2>&1 &

cd "$(dirname "$0")"   # repo root
PY=/home/tliao/venvs/vesper/bin/python
TORCHRUN=/home/tliao/venvs/vesper/bin/torchrun

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------- 1. SFT (with restart-on-crash, max 5 tries) ----------
sft_done=0
for attempt in 1 2 3 4 5; do
    # Skip if an SFT checkpoint already exists (resume covers the rest)
    latest=$(ls -d SFT/sft_checkpoints/step_* 2>/dev/null | sed 's/.*step_//' | sort -n | tail -1)
    if [ -n "$latest" ] && [ "$latest" -ge 2900 ]; then
        log "SFT already complete (step_$latest)"
        sft_done=1
        break
    fi
    log "SFT attempt $attempt (latest ckpt: ${latest:-none})"
    cd SFT && $TORCHRUN --nproc_per_node=3 01_sft_train.py >> sft_run_unattended.log 2>&1
    rc=$?
    cd ..
    log "SFT exited rc=$rc"
    if [ $rc -eq 0 ]; then sft_done=1; break; fi
    sleep 30   # cool down before restart
done

# ---------- 2. Eval with the agent harness ----------
if [ "$sft_done" = "1" ]; then
    log "Running harness eval on latest SFT checkpoint..."
    cd Agent
    $PY agent_harness.py "List all files in the current directory, including hidden ones." \
        >> ../Agent/eval_unattended.log 2>&1
    $PY agent_harness.py "What is 17 * 23 + 145?" \
        >> ../Agent/eval_unattended.log 2>&1
    cd ..
    log "Harness eval done — see Agent/eval_unattended.log"
fi

log "Unattended pipeline finished."
