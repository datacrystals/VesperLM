#!/bin/bash
# swarm2 round 3: protocol D with stronger owner mass + longer final cal.
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

run proto_d60 env SWARM_P2_PROTOCOL=d SWARM_P2_OWNER_MASS=0.6 SWARM_P2_CAL_STEPS=800 \
    SWARM_P2_OUT=/root/swarm2_proto_d60.json $PY lab/swarm2_purity.py
run proto_d70 env SWARM_P2_PROTOCOL=d SWARM_P2_OWNER_MASS=0.7 SWARM_P2_CAL_STEPS=800 \
    SWARM_P2_OUT=/root/swarm2_proto_d70.json $PY lab/swarm2_purity.py

wait
echo "[orch] round3 done $(date -u +%H:%M:%S)"
for f in "$LOGS"/exit.proto_d60 "$LOGS"/exit.proto_d70; do
  [ -f "$f" ] && echo "$(basename $f): $(cat $f)"
done
