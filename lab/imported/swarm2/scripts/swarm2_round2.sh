#!/bin/bash
# swarm2 round 2: protocols C (mutual-exclusion target) and D (B + long final cal).
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

run proto_c2 env SWARM_P2_PROTOCOL=c SWARM_P2_OUT=/root/swarm2_proto_c2.json \
    $PY lab/swarm2_purity.py
run proto_d env SWARM_P2_PROTOCOL=d SWARM_P2_OUT=/root/swarm2_proto_d.json \
    $PY lab/swarm2_purity.py

wait
echo "[orch] round2 done $(date -u +%H:%M:%S)"
for f in "$LOGS"/exit.proto_c2 "$LOGS"/exit.proto_d; do
  [ -f "$f" ] && echo "$(basename $f): $(cat $f)"
done
