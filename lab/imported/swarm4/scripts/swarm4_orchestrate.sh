#!/bin/bash
# swarm4 batch orchestrator: seed-2 t2 pair + passport_dim sweep (lab runner,
# 5 queue entries) and D55-at-t2 at owner_mass 0.55/0.50 — all parallel.
# Each job logs to /root/swarm4_logs/<name>.log; exit codes in /root/swarm4_logs/exit.<name>.
set -u
PY=/root/venvs/pod/bin/python
REPO=/root/VesperLM
LOGS=/root/swarm4_logs
mkdir -p "$LOGS"
cd "$REPO"

run() {
  local name="$1"; shift
  echo "[orch] start $name $(date -u +%H:%M:%S)"
  ( "$@" > "$LOGS/$name.log" 2>&1; echo $? > "$LOGS/exit.$name" ) &
}

# --- 1+2. t2s passport/topk pair (seed 2) and t1 passport_dim 32/64/128 sweep ---
run farm $PY lab/runner.py --slots 5 --once

# --- 3. D55 multi-plug-in at t2, two owner_mass arms ---
run d55_m55 env SWARM4_OWNER_MASS=0.55 SWARM4_OUT=/root/swarm4_d55_m55.json \
    $PY lab/swarm4_d55_t2.py
run d55_m50 env SWARM4_OWNER_MASS=0.50 SWARM4_OUT=/root/swarm4_d55_m50.json \
    $PY lab/swarm4_d55_t2.py

wait
echo "[orch] all jobs done $(date -u +%H:%M:%S)"
echo "--- exit codes ---"
for f in "$LOGS"/exit.*; do echo "$(basename $f): $(cat $f)"; done
