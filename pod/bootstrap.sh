#!/usr/bin/env bash
# bootstrap.sh — one-shot setup for a fresh cloud GPU instance (spot or not).
#
#   curl -sL <repo>/pod/bootstrap.sh | HOME_SSH=tliao@home.lan bash
# or, once the repo is checked out:
#   HOME_SSH=tliao@home.lan bash pod/bootstrap.sh
#
# What it does, in order:
#   1. env: python venv + torch (CUDA or ROCm autodetected) + repo deps + fla
#   2. repo: clone or fast-forward the VesperLM checkout, apply fla patches
#   3. data: rsync-pull corpus shards from home IN THE BACKGROUND; training
#      starts once every MANIFEST source has >=1 shard (or SYNC_GATE_SECS
#      elapse) — index.txt is rebuilt from whatever is present
#   4. state: pull latest checkpoint for $RUN_NAME from home (resume)
#   5. run:   trainer under an auto-resume loop + checkpoint uploader;
#      when the initial data sync completes, the trainer is restarted once
#      at the next checkpoint so it picks up the full corpus mix
#
# Required env: HOME_SSH (user@host reachable via key auth; the script can
#   generate a key and print it for you to authorize on home).
# Optional env: RUN_NAME (default 470m_k), VESPER_CONFIG (default 470m_k),
#   HOME_DATA (default ~/vesper_data), HOME_RUNS (default ~/vesper_runs),
#   REPO_URL, REPO_DIR (default ~/VesperLM), SYNC_GATE_SECS (default 900),
#   VESPER_MICRO_BATCH, VESPER_ACCUM.
set -uo pipefail

log() { echo "[bootstrap $(date -u +%H:%M:%S)] $*"; }

RUN_NAME="${RUN_NAME:-470m_k}"
VESPER_CONFIG="${VESPER_CONFIG:-470m_k}"
REPO_URL="${REPO_URL:-https://github.com/datacrystals/VesperLM.git}"
REPO_DIR="${REPO_DIR:-$HOME/VesperLM}"
HOME_DATA="${HOME_DATA:-vesper_data}"
HOME_RUNS="${HOME_RUNS:-vesper_runs}"
SYNC_GATE_SECS="${SYNC_GATE_SECS:-900}"
HOME_SSH="${HOME_SSH:-}"

# ---------- 0. SSH to home ----------
if [[ -z "$HOME_SSH" ]]; then
    echo "HOME_SSH is required (user@home.host)" >&2; exit 1
fi
if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$HOME_SSH" true 2>/dev/null; then
    log "no key auth to $HOME_SSH — generating a throwaway key"
    [[ -f ~/.ssh/pod_key ]] || ssh-keygen -t ed25519 -N "" -f ~/.ssh/pod_key
    echo
    echo ">>> AUTHORIZE THIS KEY ON HOME ($HOME_SSH), then re-run bootstrap:"
    cat ~/.ssh/pod_key.pub
    echo
    cat >> ~/.ssh/config <<EOF
Host pod-home
    HostName ${HOME_SSH#*@}
    User ${HOME_SSH%@*}
    IdentityFile ~/.ssh/pod_key
EOF
    exit 2
fi

# ---------- 1. Python env ----------
cd "$HOME"
if [[ ! -d ~/venvs/pod ]]; then
    python3 -m venv ~/venvs/pod
fi
source ~/venvs/pod/bin/activate
pip install -q --upgrade pip

if command -v rocm-smi >/dev/null 2>&1 || [[ -d /opt/rocm ]]; then
    log "ROCm detected — installing ROCm torch"
    pip install -q torch --index-url https://download.pytorch.org/whl/rocm6.3
else
    log "CUDA (or CPU) detected — installing default torch"
    pip install -q torch
fi

# ---------- 2. Repo ----------
if [[ -d "$REPO_DIR/.git" ]]; then
    git -C "$REPO_DIR" pull --ff-only || log "git pull failed — using existing checkout"
else
    git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
pip install -q -r requirements.txt
pip install -q --no-deps "git+https://github.com/fla-org/flash-linear-attention.git@v0.6.0"
python tools/patch_fla.py

# ---------- 3. Data sync (background) ----------
DATA_DIR="$REPO_DIR/Pretrain/data"
mkdir -p "$DATA_DIR"
SYNC_DONE=/tmp/pod_sync_done
rm -f "$SYNC_DONE"
(
    rsync -a --partial --timeout=600 "$HOME_SSH:$HOME_DATA/" "$DATA_DIR/" \
        && touch "$SYNC_DONE" \
        && bash pod/rebuild_index.sh "$DATA_DIR" >> /tmp/pod_sync.log 2>&1
    echo "sync loop exited rc=$?" >> /tmp/pod_sync.log
) >> /tmp/pod_sync.log 2>&1 &
log "data sync started in background (log: /tmp/pod_sync.log)"

# Gate: every MANIFEST source has >=1 shard, or timeout
bash pod/rebuild_index.sh "$DATA_DIR" || true
deadline=$(( $(date +%s) + SYNC_GATE_SECS ))
while [[ ! -f "$SYNC_DONE" && $(date +%s) -lt $deadline ]]; do
    missing=0
    while read -r source _; do
        [[ -z "$source" || "$source" == \#* ]] && continue
        if ! compgen -G "$DATA_DIR/vesperk/${source}_*.bin" >/dev/null \
           && [[ ! -f "$DATA_DIR/pretrain/${source}.bin" && ! -f "$DATA_DIR/${source}.bin" ]]; then
            missing=1; break
        fi
    done < pod/MANIFEST
    [[ $missing -eq 0 ]] && break
    sleep 20
done
bash pod/rebuild_index.sh "$DATA_DIR"
log "data gate passed (sync_done=$(test -f $SYNC_DONE && echo yes || echo no))"

# ---------- 4. Pull latest checkpoint ----------
CKPT_PARENT="$REPO_DIR/Pretrain"
rsync -a --partial --timeout=600 "$HOME_SSH:$HOME_RUNS/$RUN_NAME/" \
      "$CKPT_PARENT/vesper_linear_checkpoints_${RUN_NAME}/" 2>/dev/null \
    && log "pulled existing checkpoints for $RUN_NAME" \
    || log "no prior checkpoints for $RUN_NAME — fresh start"

# ---------- 5. Launch trainer + uploader ----------
export HOME_SSH
nohup bash pod/upload_ckpts.sh \
    "$CKPT_PARENT/vesper_linear_checkpoints_${RUN_NAME}" \
    "$HOME_DATA/../$HOME_RUNS/$RUN_NAME" \
    >> /tmp/pod_uploader.log 2>&1 &
log "uploader started (log: /tmp/pod_uploader.log)"

cat > /tmp/pod_train_loop.sh <<EOF
#!/usr/bin/env bash
cd "$REPO_DIR/Pretrain"
source ~/venvs/pod/bin/activate
export VESPER_CONFIG="$VESPER_CONFIG"
export TRITON_CACHE_DIR=/tmp/triton_cache FLA_CACHE_DIR=/tmp/fla_cache
NPROC=\$(python3 -c "import torch;print(max(1,torch.cuda.device_count()))")
restarted=0
while true; do
    torchrun --nproc_per_node=\$NPROC 02_pretrain_linear.py >> "pretrain_${RUN_NAME}.log" 2>&1
    echo "[train-loop \$(date -u +%H:%M:%S)] trainer exited rc=\$? — resuming in 30s" >> "pretrain_${RUN_NAME}.log"
    sleep 30
    # one-time restart after full data sync to pick up the complete mix
    if [[ -f "$SYNC_DONE" && \$restarted -eq 0 ]]; then
        restarted=1
        echo "[train-loop] full sync done — already restarting, new index active" >> "pretrain_${RUN_NAME}.log"
    fi
done
EOF
chmod +x /tmp/pod_train_loop.sh
nohup /tmp/pod_train_loop.sh >> /tmp/pod_train_loop.log 2>&1 &
log "trainer loop started (log: $REPO_DIR/Pretrain/pretrain_${RUN_NAME}.log)"
log "bootstrap complete — tail -f $REPO_DIR/Pretrain/pretrain_${RUN_NAME}.log"
