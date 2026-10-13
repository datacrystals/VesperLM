#!/bin/bash
# v0.6 chain: registry (parity gate) -> fit lanes -> assemble -> impress test.
# Ops: detached setsid, incremental JSON, no pgrep gates, claim-aware GPU grab
# (the door10 claim protocol — same file, same TTL rules, stops the co-tenant
# race). bs1x512 co-tenant-safe shape. NEVER kills user tmux.
set -u
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm-6.2.4/lib/rocblas/library
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
PY=/data/tliao/vesper_sft/venv/bin/python
OUT=/data/tliao/v06_out
CLAIM_DIR=/data/hackathon/.tmp/mr_gpu_claims
mkdir -p "$OUT" "$CLAIM_DIR"
cd /data/tliao/vesper_sft/VesperLM
log() { echo "[$(date +%H:%M:%S)] $*" >> "$OUT/v06_chain.log"; }

MODEL=/data/tliao/v06_models/Qwen1.5-MoE-A2.7B-Chat
REG=$OUT/registry_qwen2moe.pt
NEED=$((28 * 1024 * 1024 * 1024))   # 14.3B fp16 = 26.6GiB weights; tight

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
  exec 9>"$CLAIM_DIR/lock"
  flock 9
  local best=-1 best_free=0 g f
  for g in $(rocm-smi -i 2>/dev/null | grep -E '^GPU\[[0-9]+\]' | awk -F'[][]' '{print $2}' | sort -n | uniq); do
    claimed "$g" && continue
    f=$(free_of "$g")
    if (( f > best_free )); then best_free=$f; best=$g; fi
  done
  if (( best_free >= NEED )); then
    touch "$CLAIM_DIR/gpu$best"
    echo "$best"
  fi
  exec 9>&-
}

refresh_claim() {  # $1=gpu $2=donefile
  # background claim-refresher must NOT inherit the command-substitution
  # stdout pipe (bash waits for pipe close -> deadlock; learned here)
  ( while [[ ! -f "$2" ]]; do
      touch "$CLAIM_DIR/gpu$1" 2>/dev/null
      sleep 240
    done ) >/dev/null 2>&1 &
  echo $!
}

# ---- phase 1: registry + parity gate (must pass 0.95 before fits) ----
if [[ ! -f "$OUT/registry_pass.marker" ]]; then
  reg_ok=0
  for attempt in 1 2 3 4 5 6; do
    log "registry: waiting for GPU (attempt $attempt)"
    gpu=""
    for a in $(seq 1 30); do
      gpu=$(pick_and_claim)
      [[ -n "$gpu" ]] && break
      sleep 20
    done
    [[ -z "$gpu" ]] && { log "registry no gpu yet — retry"; continue; }
    log "registry on gpu $gpu (attempt $attempt)"
    rm -f "$OUT/registry_done.json"
    ref=$(refresh_claim "$gpu" "$OUT/registry_done.json")
    HIP_VISIBLE_DEVICES=$gpu $PY lab/v06_build.py --phase registry \
      --model $MODEL --registry $REG --out $OUT/registry_done.json \
      --reg-seq-len 128 --reg-domain-batches 32 --reg-val-batches 8 \
      --max-cal-pairs 512 --recal-steps 150 --batch-size 1 \
      > $OUT/registry.log 2>&1
    rc=$?
    kill $ref 2>/dev/null
    log "registry exit $rc"
    tail -2 $OUT/registry.log >> $OUT/v06_chain.log
    if [[ $rc -eq 0 ]]; then reg_ok=1; break; fi
    sleep 10
  done
  if [[ $reg_ok -ne 1 ]]; then log "registry FAILED after retries"; exit 1; fi
  # numeric-only parity gate read (no python-in-shell parsing of strings)
  $PY - "$OUT/registry_done.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
g = r.get("parity_gate", {})
print("parity", g.get("logit_cosine_mean"), "pass", g.get("pass"))
sys.exit(0 if g.get("pass") else 2)
EOF
  if [[ $? -ne 0 ]]; then log "PARITY GATE FAIL — abort fits"; exit 2; fi
  log "parity gate PASS"
  touch "$OUT/registry_pass.marker"
else
  log "registry already done — skip"
fi

# ---- phase 2: fit lanes (3 specialty + 1 memory lane of 2 wraps) ----
run_lane() {
  local name=$1 subset=$2 steps=$3
  local donef="$OUT/${name}.done"
  [[ -f "$donef" ]] && { log "lane $name done — skip"; return 0; }
  local gpu=""
  for a in $(seq 1 90); do
    # all wraps of a lane done?
    local allok=1
    for w in ${subset//,/ }; do
      $PY - "$OUT/${w}_plugin.pt" <<'EOF' || allok=0
import sys, os
sys.path.insert(0, "lab")
import torch, door6_expert_mix as d6
p = sys.argv[1]
if not os.path.exists(p):
    raise SystemExit(1)
w, r, _, _ = d6.load_wrap_plugin(p)
raise SystemExit(0 if d6.wrap_is_finite(w, r) else 1)
EOF
    done
    if [[ $allok -eq 1 ]]; then
      log "lane $name plugins ok"
      touch "$donef"
      return 0
    fi
    gpu=$(pick_and_claim)
    if [[ -z "$gpu" ]]; then
      log "lane $name waiting for gpu (attempt $a)"
      sleep 30
      continue
    fi
    log "lane $name attempt $a gpu=$gpu subset=$subset"
    ref=$(refresh_claim "$gpu" "$donef")
    HIP_VISIBLE_DEVICES=$gpu $PY lab/v06_build.py --phase fit \
      --model $MODEL --registry $REG --wrap-subset "$subset" \
      --batch-size 1 --expert-steps $steps --skip-done \
      --plugins-dir $OUT --out "$OUT/fit_$name.json" \
      > "$OUT/fit_$name.log" 2>&1
    kill $ref 2>/dev/null
    log "lane $name attempt $a exit $?"
  done
  log "lane $name GAVE UP"
  return 1
}

run_lane k3_hack k3_hack 500 &
P1=$!
run_lane glm_cybersec glm_cybersec 500 &
P2=$!
run_lane mimo_writing mimo_writing 500 &
P3=$!
run_lane memories mem_pet,mem_edit 500 &
P4=$!
wait $P1; wait $P2; wait $P3; wait $P4
log "fit lanes finished"

# ---- phase 3: assemble (balanced λ) ----
if [[ ! -f "$OUT/assemble_done.marker" ]]; then
  gpu=""
  for a in $(seq 1 60); do
    gpu=$(pick_and_claim)
    [[ -n "$gpu" ]] && break
    log "assemble waiting (attempt $a)"
    sleep 20
  done
  [[ -z "$gpu" ]] && { log "assemble GAVE UP"; exit 1; }
  log "assemble on gpu $gpu"
  ref=$(refresh_claim "$gpu" "$OUT/assemble_done.marker")
  HIP_VISIBLE_DEVICES=$gpu $PY lab/v06_build.py --phase assemble \
    --model $MODEL --registry $REG --plugins-dir $OUT \
    --out $OUT/v06_results.json > $OUT/assemble.log 2>&1
  rc=$?
  kill $ref 2>/dev/null
  [[ $rc -eq 0 ]] && touch "$OUT/assemble_done.marker"
  log "assemble exit $rc"
  tail -5 $OUT/assemble.log >> $OUT/v06_chain.log
fi

# ---- phase 4: impress test ----
if [[ ! -f "$OUT/impress_done.marker" ]]; then
  gpu=""
  for a in $(seq 1 30); do
    gpu=$(pick_and_claim)
    [[ -n "$gpu" ]] && break
    sleep 20
  done
  [[ -z "$gpu" ]] && { log "impress GAVE UP"; exit 1; }
  log "impress on gpu $gpu"
  ref=$(refresh_claim "$gpu" "$OUT/impress_done.marker")
  HIP_VISIBLE_DEVICES=$gpu $PY lab/chat_v06.py --impress --impress-bare \
    --gpu $gpu --probe-tokens 80 --max-new-tokens 160 \
    > $OUT/impress.log 2>&1
  rc=$?
  kill $ref 2>/dev/null
  [[ $rc -eq 0 ]] && touch "$OUT/impress_done.marker"
  log "impress exit $rc"
fi
log "CHAIN COMPLETE"
