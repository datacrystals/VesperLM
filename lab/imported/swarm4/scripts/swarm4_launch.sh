#!/bin/bash
# swarm4 droplet bring-up driver — run on the LAPTOP as soon as create succeeds.
# Usage: bash swarm4_launch.sh <droplet_ip>
set -euo pipefail
IP="$1"
SCRIPTS=/home/tliao/VesperLM/lab/imported/swarm4/scripts
DATA=/tmp/swarm4_data

echo "=== uploading scripts + data to root@$IP ==="
ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "root@$IP" 'mkdir -p /root/swarm4_scripts /root/swarm4_data'
scp -o ConnectTimeout=10 "$SCRIPTS"/*.sh "$SCRIPTS"/*.py "$SCRIPTS"/*.json "root@$IP:/root/swarm4_scripts/"
rsync -a --partial --info=progress2 "$DATA/" "root@$IP:/root/swarm4_data/"

echo "=== running setup (venv, torch rocm, fla, patches, data placement) ==="
ssh "root@$IP" 'bash /root/swarm4_scripts/swarm4_setup.sh' 2>&1 | tee /tmp/swarm4_setup.log

echo "=== launching experiments ==="
ssh "root@$IP" 'nohup bash /root/swarm4_orchestrate.sh > /root/swarm4_orch.log 2>&1 < /dev/null & echo orch_pid=$!'

echo "=== launched. watch with: ssh root@$IP 'tail -f /root/swarm4_logs/*.log /root/swarm4_orch.log' ==="
