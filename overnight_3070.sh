#!/usr/bin/env bash
# overnight_3070.sh — first full Vesper-K (tiny_agent_k, 118M) pretrain on
# the local RTX 3070, unattended.
#
# Phase 1: fresh 6000-step run (config total_steps), micro-batch falls
#          back 4 -> 3 -> 2 if the dummy pass OOMs.
# Between:  rebuild data/index.txt (corpus builder finishes overnight —
#          phase 2 should see the full mix).
# Phase 2: resume-extension to 12000 steps so the GPU never idles.
#
# Logs: Pretrain/overnight_3070.log (driver) + trainer stdout inline.
set -uo pipefail
cd "$(dirname "$0")/Pretrain"

export TRITON_CACHE_DIR=/tmp/triton_cache_local
export FLA_CACHE_DIR=/tmp/fla_cache_local
export VESPER_CONFIG=tiny_agent_k
export VESPER_AMP=bf16
export VESPER_ACCUM=8

log() { echo "[overnight $(date -u +%H:%M:%S)] $*"; }

# Clean start: wipe the 200-step bf16 smoke-test checkpoints
rm -rf vesper_linear_checkpoints_tiny_agent_k

launch() {  # $1=micro_batch, $2=total_steps (optional)
    local mb=$1 ts=${2:-}
    log "trying micro_batch=$mb total_steps=${ts:-config-default}"
    if [[ -n "$ts" ]]; then
        VESPER_MICRO_BATCH=$mb VESPER_TOTAL_STEPS=$ts python3 -u 02_pretrain_linear.py
    else
        VESPER_MICRO_BATCH=$mb python3 -u 02_pretrain_linear.py
    fi
}

run_with_fallback() {  # $1=total_steps (optional)
    local mb
    for mb in 4 3 2; do
        if launch "$mb" "${1:-}"; then
            return 0
        fi
        log "micro_batch=$mb failed (rc=$?) — falling back"
        sleep 10
    done
    return 1
}

# ---------------- Phase 1 ----------------
log "phase 1: fresh tiny_agent_k pretrain (6000 steps)"
run_with_fallback
rc=$?
log "phase 1 exited rc=$rc"

# ---------------- Re-index with finished corpus ----------------
bash ../pod/rebuild_index.sh || log "index rebuild failed (non-fatal)"

# ---------------- Phase 2 ----------------
if [[ $rc -eq 0 ]]; then
    log "phase 2: resume-extension to 12000 steps"
    run_with_fallback 12000
    log "phase 2 exited rc=$?"
else
    log "phase 1 never succeeded — phase 2 skipped"
fi

log "overnight script done"
