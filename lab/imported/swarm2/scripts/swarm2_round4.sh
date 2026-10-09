#!/bin/bash
# swarm2 round 4: bracket the owner-mass window with more expert training.
set -u
PY=/root/venvs/pod/bin/python
REPO=/root/VesperLM
LOGS=/root/swarm2_logs
mkdir -p "$LOGS"
cd "$REPO"

run() {
  local name="$1"; shift
  echo "[orch] start $name $(date -u +%H:%M:%S)"
  ( "$@" > "$LOGS/$name.log" 2>&1; echo $? > "$LOGS/exit.$name" ) &
}

run proto_d55 env SWARM_P2_PROTOCOL=d SWARM_P2_OWNER_MASS=0.55 SWARM_P2_EXP_STEPS=400 \
    SWARM_P2_CAL_STEPS=800 SWARM_P2_OUT=/root/swarm2_proto_d55.json $PY lab/swarm2_purity.py
run proto_d53 env SWARM_P2_PROTOCOL=d SWARM_P2_OWNER_MASS=0.53 SWARM_P2_EXP_STEPS=500 \
    SWARM_P2_CAL_STEPS=1000 SWARM_P2_OUT=/root/swarm2_proto_d53.json $PY lab/swarm2_purity.py

wait
echo "[orch] round4 done $(date -u +%H:%M:%S)"
for f in "$LOGS"/exit.proto_d55 "$LOGS"/exit.proto_d53; do
  [ -f "$f" ] && echo "$(basename $f): $(cat $f)"
done
