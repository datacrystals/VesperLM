#!/bin/bash
# OneClick smoke job: stage the lab sandbox, run a 30-step lab_tiny trainer
# (farm-style, Markov-synthetic corpus fallback), write result.json for
# RESULT_JSON exfil. GPU work runs on the OneClick box only — never on the laptop.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export REPO
export SANDBOX_ID="${SANDBOX_ID:-smoke}"
SANDBOX="$REPO/lab/sandbox/$SANDBOX_ID"
echo "[smoke] repo=$REPO sandbox_id=$SANDBOX_ID"

python3 - <<'PY'
import os, sys
sys.path.insert(0, os.path.join(os.environ["REPO"], "lab"))
import runner
sb = runner.setup_sandbox(os.environ["SANDBOX_ID"])
print("[smoke] sandbox ready:", sb)
PY
if [ $? -ne 0 ]; then
    echo "[smoke] sandbox staging failed"
    exit 1
fi

cd "$SANDBOX"
export VESPER_CONFIG="${VESPER_CONFIG:-lab_tiny}"
export VESPER_TOTAL_STEPS="${VESPER_TOTAL_STEPS:-30}"
export VESPER_AMP="${VESPER_AMP:-bf16}"
echo "[smoke] config=$VESPER_CONFIG steps=$VESPER_TOTAL_STEPS amp=$VESPER_AMP"

set -o pipefail
python3 -u "$REPO/Pretrain/02_pretrain_linear.py" 2>&1 | tee train.log
rc=${PIPESTATUS[0]}
echo "[smoke] trainer rc=$rc"

python3 - "$REPO" "$SANDBOX" "$rc" <<'PY'
import json, os, sys
repo, sandbox, rc = sys.argv[1], sys.argv[2], int(sys.argv[3])
sys.path.insert(0, os.path.join(repo, "lab"))
import runner
text = open(os.path.join(sandbox, "train.log"), errors="replace").read()
p = runner.parse_log(text)
res = {
    "job_id": "smoke_lab_tiny_30",
    "rc": rc,
    "config": os.environ.get("VESPER_CONFIG"),
    "total_steps": int(os.environ.get("VESPER_TOTAL_STEPS", "0") or 0),
    "steps_done": p["steps_done"],
    "ce_last": p["ce_last"],
    "val_loss_series": p["val_loss_series"],
    "val_loss_final": p["val_loss_final"],
    "tok_s_avg": p["tok_s_avg"],
    "log_tail": text.splitlines()[-30:],
}
out = os.path.join(sandbox, "result.json")
with open(out, "w") as f:
    json.dump(res, f, indent=2)
print("[smoke] wrote", out)
print("[smoke] summary:", json.dumps({k: res[k] for k in
      ("rc", "steps_done", "ce_last", "tok_s_avg")}))
PY
exit $rc
