#!/usr/bin/env bash
# upload_ckpts.sh — run ON the cloud instance: stream checkpoints home.
# Polls the checkpoint dir; rsyncs every fully-written step_* dir home,
# then prunes local copies beyond the newest KEEP_LOCAL.
# Usage: HOME_SSH=user@home.host bash pod/upload_ckpts.sh <ckpt_dir> <home_dest_dir>
set -uo pipefail

CKPT_DIR="${1:?checkpoint dir}"
HOME_DEST="${2:?home destination dir, e.g. /home/tliao/vesper_runs/470m_k}"
HOME_SSH="${HOME_SSH:?set HOME_SSH=user@home.host}"
KEEP_LOCAL="${KEEP_LOCAL:-2}"
POLL="${POLL:-60}"

log() { echo "[uploader $(date -u +%H:%M:%S)] $*"; }
log "watching $CKPT_DIR -> $HOME_SSH:$HOME_DEST (keep $KEEP_LOCAL local)"

declare -A SENT
while true; do
    # A step dir is "complete" when checkpoint.pt exists and is older than 30s
    mapfile -t steps < <(ls -d "$CKPT_DIR"/step_* 2>/dev/null | sort -t_ -k2 -n)
    for d in "${steps[@]}"; do
        [[ -n "${SENT[$d]:-}" ]] && continue
        f="$d/checkpoint.pt"
        [[ -f "$f" ]] || continue
        # skip if modified in the last 30s (still being written)
        if [[ $(( $(date +%s) - $(stat -c %Y "$f") )) -lt 30 ]]; then continue; fi
        log "uploading $(basename "$d")"
        if rsync -a --partial --timeout=300 "$d" "$HOME_SSH:$HOME_DEST/" ; then
            SENT[$d]=1
            log "uploaded $(basename "$d")"
        else
            log "rsync FAILED for $(basename "$d") — will retry"
        fi
    done
    # prune local copies beyond KEEP_LOCAL (only ones already uploaded)
    uploaded_local=()
    for d in "${steps[@]}"; do [[ -n "${SENT[$d]:-}" ]] && uploaded_local+=("$d"); done
    excess=$(( ${#uploaded_local[@]} - KEEP_LOCAL ))
    if [[ $excess -gt 0 ]]; then
        for ((i=0; i<excess; i++)); do
            log "pruning local $(basename "${uploaded_local[$i]}")"
            rm -rf "${uploaded_local[$i]}"
        done
    fi
    sleep "$POLL"
done
