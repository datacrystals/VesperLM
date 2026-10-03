#!/bin/bash
# run_refresh_chain.sh — 2026-10-03 SFT-refresh chain (set up by Kimi session).
#
# 1. Wait for the running SFT refresh (tiny_agent_v2 from pretrain step_best@5200,
#    log /home/tliao/sft_refresh_5200.log), relaunching it if it crashed before 2900
#    (relaunch auto-resumes from latest step_* — that is desired for crash recovery).
# 2. Run agent-harness eval with HELD-OUT prompts (disjoint from EVAL_PROMPTS in
#    SFT/01_sft_train.py and from the templated tooluse training data).
# Nothing chained after: pretrain v2 is COMPLETE (1.42B tokens, step_best@5200).

cd "$(dirname "$0")"   # repo root
PY=/home/tliao/venvs/vesper/bin/python
TORCHRUN=/home/tliao/venvs/vesper/bin/torchrun
log() { echo "[$(date '+%H:%M:%S')] $*"; }

latest_sft() { ls -d SFT/sft_checkpoints/step_* 2>/dev/null | sed 's/.*step_//' | grep -E '^[0-9]+$' | sort -n | tail -1; }

# ---------- 1. Wait for the running SFT refresh ----------
log "Waiting for SFT refresh (torchrun 01_sft_train.py)..."
while pgrep -f "torchrun --nproc_per_node=3 01_sft_train.py" >/dev/null 2>&1; do
    sleep 180
done
log "SFT process exited (latest ckpt: $(latest_sft))"

for attempt in 1 2 3; do
    latest=$(latest_sft)
    if [ -n "$latest" ] && [ "$latest" -ge 2900 ]; then break; fi
    log "SFT incomplete (latest ${latest:-none}) — relaunch attempt $attempt/3 (resumes)"
    cd SFT && $TORCHRUN --nproc_per_node=3 01_sft_train.py >> sft_refresh_restart.log 2>&1
    cd ..
    log "SFT attempt $attempt exited rc=$? (latest $(latest_sft))"
    sleep 20
done

# ---------- 2. Harness eval (held-out prompts only) ----------
latest=$(latest_sft)
if [ -n "$latest" ] && [ "$latest" -ge 2900 ]; then
    log "SFT refresh complete (step_$latest) — running harness eval..."
    # Classics (compare against v1 run; note 17*23+145 matches tooluse templates,
    # treat its success as memorization signal, not generalization):
    $PY Agent/agent_harness.py "List all files in the current directory, including hidden ones." \
        >> Agent/eval_refresh_5200.log 2>&1
    $PY Agent/agent_harness.py "What is 17 * 23 + 145?" \
        >> Agent/eval_refresh_5200.log 2>&1
    # Fresh held-out (invented 2026-10-03, in no training data or eval list):
    $PY Agent/agent_harness.py "Create a file called notes.txt containing the word hello." \
        >> Agent/eval_refresh_5200.log 2>&1
    $PY Agent/agent_harness.py "What is 13 * 12?" \
        >> Agent/eval_refresh_5200.log 2>&1
    log "Harness eval done -> Agent/eval_refresh_5200.log"
else
    log "WARNING: SFT refresh never reached step 2900."
fi
log "chain done."
