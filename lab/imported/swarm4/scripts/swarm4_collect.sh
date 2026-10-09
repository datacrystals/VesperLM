#!/bin/bash
# swarm4 results collection: run ON THE LAPTOP after orchestrator finishes.
# Pulls jsons + logs from the droplet into lab/imported/swarm4/.
set -uo pipefail
IP="$1"
DEST=/home/tliao/VesperLM/lab/imported/swarm4
REPO_L=/home/tliao/VesperLM

mkdir -p "$DEST/results" "$DEST/logs"
# runner results (t2s pair + dim sweep)
scp -o ConnectTimeout=10 "root@$IP:/root/VesperLM/lab/results/t2s-*.json" "$DEST/results/" 2>&1
scp -o ConnectTimeout=10 "root@$IP:/root/VesperLM/lab/results/t1d*.json" "$DEST/results/" 2>&1
scp -o ConnectTimeout=10 "root@$IP:/root/VesperLM/lab/logs/t2s-*.log" "$DEST/logs/" 2>&1
scp -o ConnectTimeout=10 "root@$IP:/root/VesperLM/lab/logs/t1d*.log" "$DEST/logs/" 2>&1
# D55 results + logs
scp -o ConnectTimeout=10 "root@$IP:/root/swarm4_d55_m*.json" "$DEST/results/" 2>&1
scp -o ConnectTimeout=10 "root@$IP:/root/swarm4_logs/*.log" "$DEST/logs/" 2>&1
echo "--- collected ---"
ls -la "$DEST/results" "$DEST/logs"
