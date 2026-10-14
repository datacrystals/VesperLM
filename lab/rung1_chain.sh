#!/bin/bash
# RUNG-1 CHAIN — self-driving, detached (setsid), session-proof.
# extract (CPU) -> multi-lane wrap fits (claim-locked MI50s) -> growth
# checkpoints at 32/64/128/250 -> impress at endpoint.
# Ops (HANDOFF): claim locks mandatory, no pgrep gates, incremental JSON,
# skip-done everywhere, NEVER touch user tmux / rung-0 lanes.
set -u
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm-6.2.4/lib/rocblas/library
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
PY=/data/tliao/vesper_sft/venv/bin/python
REPO=/data/tliao/vesper_sft/VesperLM
OUT=/data/tliao/rung1
CLAIM_DIR=/data/hackathon/.tmp/mr_gpu_claims
mkdir -p "$OUT"/{logs,plugins,results,locks} "$CLAIM_DIR"
cd "$REPO"
log() { echo "[$(date +%H:%M:%S)] $*" >> "$OUT/logs/chain.log"; }

# ---- claim-aware GPU grab (door10_chain2.sh protocol, 10-min TTL) ----
NEED=$((6 * 1024 * 1024 * 1024))
free_of() {
  local g=$1 used total
  used=$(rocm-smi -d "$g" --showmeminfo vram 2>/dev/null | awk '/Total Used/{print $NF}')
  total=$(rocm-smi -d "$g" --showmeminfo vram 2>/dev/null | awk '/Total Memory/{print $NF}')
  [[ -z "$used" || -z "$total" ]] && { echo 0; return; }
  echo $((10#$total - 10#$used))
}
claimed() {
  local f="$CLAIM_DIR/gpu$1"
  [[ -f "$f" ]] && (( $(date +%s) - $(stat -c %Y "$f") < 600 ))
}
pick_and_claim() {
  exec 9>"$CLAIM_DIR/lock"; flock 9
  local best=-1 best_free=0 g f
  for g in 0 1 2 3 4 5 6 7; do
    claimed "$g" && continue
    f=$(free_of "$g")
    if (( f > best_free )); then best_free=$f; best=$g; fi
  done
  if (( best_free >= NEED )); then touch "$CLAIM_DIR/gpu$best"; echo "$best"; fi
  exec 9>&-
}

# ---- stage 1: extraction (CPU, skip-done) ----
log "extract start"
$PY lab/bank_rung1.py --phase manifest --root "$OUT" >> "$OUT/logs/manifest.log" 2>&1
$PY lab/bank_rung1.py --phase extract --root "$OUT" --donor glm \
  --local-dir /data/tliao >> "$OUT/logs/extract_glm.log" 2>&1
$PY lab/bank_rung1.py --phase extract --root "$OUT" --donor mimo \
  --local-dir /data/tliao >> "$OUT/logs/extract_mimo.log" 2>&1
# k3 cores arrive via laptop rsync (box holds LFS stubs) — background
# monitor only; fit lanes start immediately on cores already on disk and
# the driver's ready-first order picks up K3 as it lands (no pgrep).
( for a in $(seq 1 720); do
    n=$(find /data/tliao/core_bank/kimi-k3 -name core.safetensors 2>/dev/null | wc -l)
    log "k3 cores on box: $n / 175"
    (( n >= 175 )) && break
    sleep 100
  done
  log "k3 arrival monitor exit" ) &
log "extract stage done (lanes starting now)"

# ---- stage 2: fit lanes (5 lanes, claim-locked, skip-done) ----
NLANES=5
for lane in $(seq 0 $((NLANES - 1))); do
  (
    for attempt in $(seq 1 400); do
      gpu=$(pick_and_claim)
      if [ -z "$gpu" ]; then sleep 120; continue; fi
      log "lane $lane -> GPU $gpu (attempt $attempt)"
      # keep OUR claim fresh while the lane runs
      ( while true; do touch "$CLAIM_DIR/gpu$gpu" 2>/dev/null; sleep 240; done ) &
      REFPID=$!
      HIP_VISIBLE_DEVICES=$gpu $PY lab/bank_rung1.py --phase fit \
        --root "$OUT" --lane "$lane" --lanes "$NLANES" --skip-done \
        --device cuda --dtype float16 \
        >> "$OUT/logs/fit_lane${lane}.log" 2>&1
      kill "$REFPID" 2>/dev/null
      rm -f "$CLAIM_DIR/gpu$gpu" 2>/dev/null
      log "lane $lane fit pass done"
      # exit when every core has a plugin (skip-done makes re-entry cheap)
      left=$($PY - <<'EOF'
import json, os
m=json.load(open("/data/tliao/rung1/manifest.json"))
miss=sum(1 for c in m["cores"]
         if not os.path.exists(f"/data/tliao/rung1/plugins/{c['id']}_plugin.pt"))
print(miss)
EOF
)
      log "lane $lane remaining $left"
      (( left == 0 )) && break
      sleep 30
    done
  ) &
done

# ---- stage 3: growth checkpoints as fits land ----
( for n in 32 64 128 250; do
    while true; do
      done_n=$($PY - <<EOF
import json, os
m=json.load(open("/data/tliao/rung1/manifest.json"))
have=sum(1 for c in m["cores"][:$n]
         if os.path.exists(f"/data/tliao/rung1/plugins/{c['id']}_plugin.pt"))
print(have)
EOF
)
      log "growth $n: $done_n/$n plugins"
      (( done_n >= n )) && break
      sleep 600
    done
    [ -f "$OUT/results/growth_${n}.json" ] && continue
    gpu=""
    for ga in $(seq 1 200); do
      gpu=$(pick_and_claim)
      [ -n "$gpu" ] && break
      log "growth $n waiting for GPU (attempt $ga)"
      sleep 300
    done
    [ -z "$gpu" ] && { log "growth $n no GPU - leaving for relaunch"; continue; }
    log "growth $n on GPU $gpu"
    ( while true; do touch "$CLAIM_DIR/gpu$gpu" 2>/dev/null; sleep 240; done ) &
    REFPID=$!
    HIP_VISIBLE_DEVICES=$gpu $PY lab/bank_rung1.py --phase growth \
      --root "$OUT" --n "$n" --device cuda --dtype float16 \
      >> "$OUT/logs/growth_${n}.log" 2>&1
    kill "$REFPID" 2>/dev/null
    rm -f "$CLAIM_DIR/gpu$gpu" 2>/dev/null
    log "growth $n done"
  done
  # ---- stage 4: impress at endpoint ----
  gpu=$(pick_and_claim)
  while [ -z "$gpu" ]; do sleep 300; gpu=$(pick_and_claim); done
  log "impress on GPU $gpu"
  ( while true; do touch "$CLAIM_DIR/gpu$gpu" 2>/dev/null; sleep 240; done ) &
  REFPID=$!
  HIP_VISIBLE_DEVICES=$gpu $PY lab/bank_rung1.py --phase impress \
    --root "$OUT" --device cuda --dtype float16 --varact-k 4 \
    --extra-plugins-dir /data/tliao/vk_final \
    >> "$OUT/logs/impress.log" 2>&1
  kill "$REFPID" 2>/dev/null
  rm -f "$CLAIM_DIR/gpu$gpu" 2>/dev/null
  log "CHAIN ALL DONE"
) &
wait
log "rung1 parent exit"
