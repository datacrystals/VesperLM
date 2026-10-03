#!/bin/bash
# run_overnight.sh — 2026-10-02 overnight chain (set up by Kimi session).
#
# 1. Wait for the manually-launched 118M (tiny_agent_v2) SFT to finish
#    (it is ALREADY RUNNING, log: /home/tliao/sft_v2_118m.log), restarting
#    it if it crashed before step 3000.
# 2. Run the agent-harness eval on the finished SFT checkpoint.
# 3. Continue the v2 pretrain (resumes step_best=4100 -> total 6000) so the
#    GPUs stay busy after SFT. Log: Pretrain/pretrain_v2_continue.log
#
# Launch: nohup bash run_overnight.sh > /home/tliao/overnight_chain.log 2>&1 &

cd "$(dirname "$0")"   # repo root
PY=/home/tliao/venvs/vesper/bin/python
TORCHRUN=/home/tliao/venvs/vesper/bin/torchrun
log() { echo "[$(date '+%H:%M:%S')] $*"; }

latest_sft() { ls -d SFT/sft_checkpoints/step_* 2>/dev/null | sed 's/.*step_//' | sort -n | tail -1; }

# ---------- 1. Wait for the running SFT, then ensure completion ----------
log "Waiting for running SFT (torchrun 01_sft_train.py)..."
while pgrep -f "torchrun --nproc_per_node=3 01_sft_train.py" >/dev/null 2>&1; do
    sleep 120
done
log "SFT process exited (latest ckpt: $(latest_sft))"

for attempt in 1 2 3; do
    latest=$(latest_sft)
    if [ -n "$latest" ] && [ "$latest" -ge 2900 ]; then break; fi
    log "SFT incomplete (latest ${latest:-none}) — relaunch attempt $attempt/3"
    cd SFT && $TORCHRUN --nproc_per_node=3 01_sft_train.py >> sft_run_overnight.log 2>&1
    cd ..
    log "SFT attempt $attempt exited rc=$? (latest $(latest_sft))"
    sleep 20
done

# ---------- 2. Harness eval ----------
latest=$(latest_sft)
if [ -n "$latest" ] && [ "$latest" -ge 2900 ]; then
    log "SFT complete (step_$latest) — running harness eval..."
    cd Agent
    $PY agent_harness.py "List all files in the current directory, including hidden ones." \
        >> ../Agent/eval_v2_118m.log 2>&1
    $PY agent_harness.py "What is 17 * 23 + 145?" \
        >> ../Agent/eval_v2_118m.log 2>&1
    cd ..
    log "Harness eval done -> Agent/eval_v2_118m.log"
else
    log "WARNING: SFT never reached step 3000 — going to pretrain anyway to keep GPUs busy."
fi

# ---------- 3. Continue pretrain (tiny_agent_v2, resumes step_best=4100) ----------
log "Starting pretrain continuation (v2, step 4101 -> 6000)..."
cd Pretrain && $TORCHRUN --nproc_per_node=3 02_pretrain_linear.py >> pretrain_v2_continue.log 2>&1
log "Pretrain exited rc=$? — overnight chain done."
