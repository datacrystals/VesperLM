#!/bin/bash
# ckpt_sync.sh — push new/updated checkpoints home through a reverse SSH tunnel.
#
# Runs ON the droplet. Requires:
#   - reverse tunnel from the laptop: ssh -N -R 2222:localhost:2222 root@<droplet>
#     (forwards droplet localhost:2222 -> laptop user-space sshd)
#   - /root/.ssh/config with a "home" Host entry (Port 2222, IdentityFile t3_home_key)
#
# Usage: ckpt_sync.sh <src_ckpt_root> <home_dest_dir>
# Watches <src_ckpt_root>/*/checkpoint.pt for mtime+size changes and pushes via
# scp to <home_dest_dir>/<name>.checkpoint.pt.tmp, then atomically mv's into
# <home_dest_dir>/<name>/checkpoint.pt. Pushed signatures are recorded in
# ~/.ckpt_sync_seen so a restart never re-pushes an already-synced checkpoint.
#
# NOTE: never pkill/pgrep this by unbracketed name from a shell whose own
# cmdline contains the pattern — use a bracket: pgrep -f "ckpt_syn[c].sh".

set -u
SRC=${1:-/root/VesperLM/Pretrain/vesper_linear_checkpoints_470m_k}
RPATH=${2:-/home/tliao/VesperLM/lab/imported/t3_ckpts}
DST="home:$RPATH"
SEEN=/root/.ckpt_sync_seen
touch "$SEEN"

while true; do
  for d in "$SRC"/*/; do
    name=$(basename "$d")
    f="$d/checkpoint.pt"
    [ -f "$f" ] || continue
    sig="$name $(stat -c %Y%s "$f")"
    grep -qx "$sig" "$SEEN" && continue
    if scp -o ConnectTimeout=15 "$f" "$DST/$name.checkpoint.pt.tmp"; then
      ssh home "mkdir -p '$RPATH/$name' && mv '$RPATH/$name.checkpoint.pt.tmp' '$RPATH/$name/checkpoint.pt'" && \
      scp -o ConnectTimeout=15 "$d"/*_snapshot.py "$DST/" 2>/dev/null && \
      echo "$sig" >> "$SEEN" && \
      echo "$(date -u +%H:%M:%S) synced $name"
    fi
  done
  sleep 60
done
