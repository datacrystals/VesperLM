#!/bin/bash
# swarm4 droplet bring-up: ROCm torch + fla pinned + patches + repo + data.
# Run as root on a fresh MI300X droplet, with swarm4_data/ and swarm4_scripts/
# already scp'd to /root/. Idempotent-ish; safe to re-run.
set -uo pipefail

log() { echo "[setup $(date -u +%H:%M:%S)] $*"; }

PYBIN=/root/venvs/pod/bin/python
REPO=/root/VesperLM

# ---------- 0. System packages ----------
log "apt: python3-venv + git"
apt-get update -qq
apt-get install -y -qq python3-venv python3.12-venv git rsync >/dev/null

# ---------- 1. Python env ----------
rm -rf /root/venvs/pod
python3 -m venv /root/venvs/pod
# shellcheck disable=SC1091
source /root/venvs/pod/bin/activate
pip install -q --upgrade pip

log "installing torch rocm6.3"
pip install -q torch --index-url https://download.pytorch.org/whl/rocm6.3

log "installing deps + fla pinned commit"
if [[ ! -d "$REPO/.git" ]]; then
    git clone https://github.com/datacrystals/VesperLM.git "$REPO"
fi
git -C "$REPO" pull --ff-only || log "git pull failed — using existing checkout"
pip install -q einops
pip install -q -r "$REPO/requirements.txt"
pip install -q --no-deps "git+https://github.com/fla-org/flash-linear-attention.git@37a6b1c6290e5240f6f0d80419d08a7aac27e548"

log "patching fla (3 patches incl ROCm num_stages cap)"
cd "$REPO" && python tools/patch_fla.py

# ---------- 2. Data placement ----------
log "placing real-data slices"
mkdir -p "$REPO/lab/data_synth"
cp -n /root/swarm4_data/phase1.bin "$REPO/lab/data_synth/phase1.bin"
cp -n /root/swarm4_data/phase2.bin "$REPO/lab/data_synth/phase2.bin"
cp -n /root/swarm4_data/swarm4_phase1.bin /root/swarm4_phase1.bin
cp -n /root/swarm4_data/swarm4_code.bin /root/swarm4_code.bin
cp -n /root/swarm4_data/swarm4_finemath.bin /root/swarm4_finemath.bin
cp -n /root/swarm4_data/swarm4_wikipedia.bin /root/swarm4_wikipedia.bin

# ---------- 3. Scripts ----------
cp /root/swarm4_scripts/*.json "$REPO/lab/queue/"
cp /root/swarm4_scripts/swarm4_d55_t2.py "$REPO/lab/"
cp /root/swarm4_scripts/swarm4_orchestrate.sh /root/swarm4_orchestrate.sh
chmod +x /root/swarm4_orchestrate.sh

# ---------- 4. Smoke ----------
log "smoke: GPU + fla + tiny model"
python - <<'EOF'
import torch, fla
from fla.ops.kda import chunk_kda
print("torch", torch.__version__, "hip", torch.version.hip, "gpus", torch.cuda.device_count())
print("fla ok", fla.__version__ if hasattr(fla, "__version__") else "import-ok")
assert torch.cuda.is_available()
print("VRAM free GB:", torch.cuda.mem_get_info()[0]/1e9)
EOF

log "setup complete"
