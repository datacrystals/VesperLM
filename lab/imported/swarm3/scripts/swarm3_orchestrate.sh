#!/bin/bash
# swarm3: (1) 470m_k passport-vs-topk A/B first for clean throughput,
#         (2) then D55 owner_mass ablation at t1.
set -u
PY=/root/venvs/pod/bin/python
REPO=/root/VesperLM
LOGS=/root/swarm3_logs
mkdir -p "$LOGS"
cd "$REPO"

run() {
  local name="$1"; shift
  echo "[orch] start $name $(date -u +%H:%M:%S)"
  ( "$@" > "$LOGS/$name.log" 2>&1; echo $? > "$LOGS/exit.$name" ) &
}

echo "=== GROUP 2 first: 470m_k A/B (clean throughput) ==="
run t3_470m $PY lab/runner.py --slots 2 --once
wait
echo "[orch] 470m group done $(date -u +%H:%M:%S)"

echo "=== GROUP 1: D55 owner_mass ablation at t1 (lab_small) ==="
for mass in 0.50 0.55 0.60; do
  tag=$(echo "$mass" | tr -d '.')
  run "d55_t1_m$tag" env \
      SWARM_P2_PROTOCOL=d SWARM_P2_CONFIG=lab_small SWARM_P2_SEQ_LEN=1024 \
      SWARM_P2_OWNER_MASS=$mass SWARM_P2_BASE_STEPS=400 SWARM_P2_EXP_STEPS=400 \
      SWARM_P2_CAL_STEPS=800 SWARM_P2_OUT=/root/swarm3_t1_m$tag.json \
      $PY lab/swarm3_purity.py
done
wait

echo "[orch] all done $(date -u +%H:%M:%S)"
for f in "$LOGS"/exit.*; do echo "$(basename $f): $(cat $f)"; done
