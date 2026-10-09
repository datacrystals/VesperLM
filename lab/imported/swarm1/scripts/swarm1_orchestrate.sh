#!/bin/bash
# swarm1 batch orchestrator: run all experiments in parallel on the droplet.
# Each job logs to /root/swarm1_logs/<name>.log; exit codes in /root/swarm1_logs/exit.<name>.
set -u
PY=/root/venvs/pod/bin/python
REPO=/root/VesperLM
LOGS=/root/swarm1_logs
mkdir -p "$LOGS"
cd "$REPO"

run() {
  local name="$1"; shift
  echo "[orch] start $name $(date -u +%H:%M:%S)"
  ( "$@" > "$LOGS/$name.log" 2>&1; echo $? > "$LOGS/exit.$name" ) &
}

# --- 1. t1 second-seed head-to-head (lab_small, seed 2) ---
run t1s_farm $PY lab/runner.py --slots 2 --once

# --- 2. reject_w curve at t1: rw10 and rw12 ---
run plugin_rw10 env PLUGIN_T1_REJECT_W=10 PLUGIN_T1_PHASE_B_STEPS=600 \
    PLUGIN_T1_OUT=/root/plugin_t1_rw10.json \
    $PY lab/plugin_expert_test_t1.py
run plugin_rw12 env PLUGIN_T1_REJECT_W=12 PLUGIN_T1_PHASE_B_STEPS=600 \
    PLUGIN_T1_OUT=/root/plugin_t1_rw12.json \
    $PY lab/plugin_expert_test_t1.py

# --- 3. expert-expansion live test at t0 (expand + control) ---
run expand_run $PY lab/swarm_expand_t0.py
run expand_ctrl env SWARM_EXPAND_CONTROL=1 SWARM_EXPAND_OUT=/root/swarm_expand_ctrl.json \
    $PY lab/swarm_expand_t0.py

# --- 4. multi-plug-in at t0 ---
run multiplug $PY lab/swarm_multiplug_t0.py

wait
echo "[orch] all jobs done $(date -u +%H:%M:%S)"
echo "--- exit codes ---"
for f in "$LOGS"/exit.*; do echo "$(basename $f): $(cat $f)"; done
