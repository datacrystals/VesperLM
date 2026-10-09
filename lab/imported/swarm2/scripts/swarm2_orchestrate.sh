#!/bin/bash
# swarm2: run purity-fix protocols A/B/C in parallel.
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

run proto_a env SWARM_P2_PROTOCOL=a SWARM_P2_OUT=/root/swarm2_proto_a.json \
    $PY lab/swarm2_purity.py
run proto_b env SWARM_P2_PROTOCOL=b SWARM_P2_OUT=/root/swarm2_proto_b.json \
    $PY lab/swarm2_purity.py
run proto_c env SWARM_P2_PROTOCOL=c SWARM_P2_OUT=/root/swarm2_proto_c.json \
    $PY lab/swarm2_purity.py

wait
echo "[orch] all protocols done $(date -u +%H:%M:%S)"
for f in "$LOGS"/exit.*; do echo "$(basename $f): $(cat $f)"; done
