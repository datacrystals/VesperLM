#!/bin/bash
# swarm5 speedrun-opts merge-gate canary (OneClick/ROCm box).
#
# Runs the lab_tiny KDA+MLA trainer 5x — flag matrix baseline / VESPER_COMPILE /
# VESPER_FUSED_CE / VESPER_VALUE_EMBED / all-three — 15 steps each, VESPER_AMP=bf16,
# VESPER_SEED=123 (paired: same init/data order wherever the flag allows).
# Per-run logs: lab/sandbox/canary_<name>/train.log ; summary: lab/sandbox/canary/result.json
# (cell 5 of the boot notebook exfils result.json as RESULT_JSON; stdout is the
# live exfil channel).
set -uo pipefail

REPO="${ONECLICK_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"
export REPO
echo "[canary] repo=$REPO host=$(hostname) utc=$(date -u +%FT%TZ) pwd=$(pwd)"

# ---- ground truth: what hardware/torch is this canary actually on? ----
python3 - <<'PY'
import torch
print("[canary] torch", torch.__version__, "cuda", torch.version.cuda, "hip",
      torch.version.hip, "avail", torch.cuda.is_available(), "n",
      torch.cuda.device_count() if torch.cuda.is_available() else 0, flush=True)
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"[canary] gpu{i} {torch.cuda.get_device_name(i)} {p}", flush=True)
import triton
print("[canary] triton", triton.__version__, flush=True)
PY
rocm-smi --showid 2>&1 | head -30 || true

# ---- deps: pinned fla (t3 pattern) + patches; best-effort python headers for triton ----
if [ "$(id -u)" = "0" ]; then
    apt-get install -y -q python3-dev >/dev/null 2>&1 && echo "[canary] apt python3-dev ok" \
        || echo "[canary] apt python3-dev unavailable (continuing)"
fi
python3 -m pip install -q --no-input einops matplotlib transformers 2>&1 | tail -3
python3 -m pip install -q --no-input --no-deps \
    "git+https://github.com/fla-org/flash-linear-attention.git@37a6b1c6290e5240f6f0d80419d08a7aac27e548" 2>&1 | tail -3
python3 tools/patch_fla.py && echo "[canary] patch_fla OK" || echo "[canary] patch_fla FAILED rc=$?"
python3 -c "import fla, einops; print('[canary] fla', getattr(fla,'__version__','?'), 'einops ok')"

# ---- shared trainer env (per-run flags layered on top) ----
export VESPER_CONFIG="${VESPER_CONFIG:-lab_tiny}"
export VESPER_TOTAL_STEPS="${VESPER_TOTAL_STEPS:-15}"
export VESPER_AMP="${VESPER_AMP:-bf16}"
export VESPER_SEED="${VESPER_SEED:-123}"
export MPLBACKEND=Agg

run_one() {
    local name="$1"; shift
    local sb="$REPO/lab/sandbox/canary_$name"
    local log="$REPO/lab/sandbox/canary_$name.log"
    echo "[canary] ===== RUN $name | flags: $* | steps=$VESPER_TOTAL_STEPS seed=$VESPER_SEED ====="
    rm -rf "$sb"
    python3 - "$name" <<'PY'
import os, sys
sys.path.insert(0, os.path.join(os.environ["REPO"], "lab"))
import runner
print("[canary] sandbox:", runner.setup_sandbox("canary_" + sys.argv[1]))
PY
    [ -d "$sb" ] || { echo "[canary] sandbox setup FAILED for $name"; return 1; }
    local t0=$SECONDS
    ( cd "$sb" && env "$@" python3 -u "$REPO/Pretrain/02_pretrain_linear.py" ) \
        >"$log" 2>&1
    local rc=$?
    local dt=$((SECONDS - t0))
    echo "[canary] RUN $name rc=$rc ${dt}s"
    grep -E "\[compile\]|\[fused-ce\]|\[value-embed\]|Max VRAM|Hybrid stack|Step [0-9]|CE Loss|Tok/s|Error|Traceback|error" \
        "$log" | tail -40
    return $rc
}

# a-e matrix (each isolated in its own sandbox so nothing resumes across flags)
run_one baseline || true
run_one compile VESPER_COMPILE=1 || true
run_one fused_ce VESPER_FUSED_CE=1 || true
run_one value_embed VESPER_VALUE_EMBED=1 || true

# e: all-three only if a-d all came back clean (checked in the summary pass below)
python3 - <<'PY'
import os, re, json, glob, sys
repo = os.environ["REPO"]

def parse(log):
    ce, tok, steps, dec, warns = [], [], 0, [], []
    for line in open(log, errors="replace"):
        m = re.search(r"Step\s+(\d+).*?CE Loss:\s*([0-9.eE+-]+)", line, re.S)
        m2 = re.search(r"CE Loss:\s*([0-9.eE+-]+)", line)
        m3 = re.search(r"Tok/s:\s*([0-9][0-9,]*)\s*\(GPU\)", line)
        m4 = re.search(r"Step\s+(\d+)", line)
        if m4:
            steps = max(steps, int(m4.group(1)))
        if m2:
            sm = re.search(r"Step\s+(\d+)", line)
            tokv = float(m3.group(1).replace(",", "")) if m3 else None
            ce.append({"step": int(sm.group(1)) if sm else steps,
                       "ce": float(m2.group(1)), "tok_s": tokv})
        for pat in (r"\[compile\].*", r"\[fused-ce\].*", r"\[value-embed\].*"):
            mm = re.search(pat, line)
            if mm:
                dec.append(mm.group(0)[:300])
        if re.search(r"warning|Warning|Traceback|CUDA error|hipError|inductor", line):
            warns.append(line.strip()[:300])
    return {"steps_done": steps, "ce_trace": ce, "decisions": dec,
            "warnings": warns[:20]}

matrix = ["baseline", "compile", "fused_ce", "value_embed"]
out = {"job_id": "swarm5_speedrun_canary", "runs": {}}
ok_all = True
for name in matrix:
    log = os.path.join(repo, "lab", "sandbox", f"canary_{name}.log")
    if not os.path.exists(log):
        out["runs"][name] = {"error": "no log"}
        ok_all = False
        continue
    p = parse(log)
    rc = 0
    p["rc_file_missing"] = False
    p["log_tail"] = open(log, errors="replace").read().splitlines()[-15:]
    ces = [c["ce"] for c in p["ce_trace"]]
    p["ce_finite"] = all(c == c and abs(c) < 1e6 for c in ces) and bool(ces)
    out["runs"][name] = p
    # a run "passed" if it stepped, CE finite, and no Traceback in the log
    txt = open(log, errors="replace").read()
    good = p["ce_finite"] and p["steps_done"] >= 10 and "Traceback" not in txt
    out["runs"][name]["pass"] = bool(good)
    ok_all = ok_all and good
out["all_of_a_d_pass"] = ok_all
print("[canary] a-d pass =", ok_all, "—", {k: v.get("pass") for k, v in out["runs"].items()})
os.makedirs(os.path.join(repo, "lab", "sandbox", "canary"), exist_ok=True)
with open(os.path.join(repo, "lab", "sandbox", "canary", "result.json"), "w") as f:
    json.dump(out, f, indent=2)
sys.exit(0 if ok_all else 10)
PY
A_D_RC=$?
echo "[canary] a-d summary rc=$A_D_RC"

if [ "$A_D_RC" = "0" ]; then
    run_one all3 VESPER_COMPILE=1 VESPER_FUSED_CE=1 VESPER_VALUE_EMBED=1 || true
else
    echo "[canary] SKIP all3 — one of a-d failed (per plan)"
fi

# ---- final result.json (include all3 + timing) ----
python3 - <<'PY'
import os, re, json
repo = os.environ["REPO"]
path = os.path.join(repo, "lab", "sandbox", "canary", "result.json")
out = json.load(open(path)) if os.path.exists(path) else {"runs": {}}

def parse(log):
    ce, steps, dec, warns = [], 0, [], []
    for line in open(log, errors="replace"):
        m2 = re.search(r"CE Loss:\s*([0-9.eE+-]+)", line)
        m3 = re.search(r"Tok/s:\s*([0-9][0-9,]*)\s*\(GPU\)", line)
        m4 = re.search(r"Step\s+(\d+)", line)
        if m4:
            steps = max(steps, int(m4.group(1)))
        if m2:
            sm = re.search(r"Step\s+(\d+)", line)
            tokv = float(m3.group(1).replace(",", "")) if m3 else None
            ce.append({"step": int(sm.group(1)) if sm else steps,
                       "ce": float(m2.group(1)), "tok_s": tokv})
        for pat in (r"\[compile\].*", r"\[fused-ce\].*", r"\[value-embed\].*"):
            mm = re.search(pat, line)
            if mm:
                dec.append(mm.group(0)[:300])
        if re.search(r"warning|Warning|Traceback|CUDA error|hipError|inductor", line):
            warns.append(line.strip()[:300])
    txt = open(log, errors="replace").read()
    ces = [c["ce"] for c in ce]
    return {"steps_done": steps, "ce_trace": ce, "decisions": dec,
            "warnings": warns[:20], "log_tail": txt.splitlines()[-15:],
            "ce_finite": all(c == c and abs(c) < 1e6 for c in ces) and bool(ces),
            "pass": ("Traceback" not in txt) and steps >= 10
                    and all(c == c and abs(c) < 1e6 for c in ces) and bool(ces)}

for name in ("baseline", "compile", "fused_ce", "value_embed", "all3"):
    log = os.path.join(repo, "lab", "sandbox", f"canary_{name}.log")
    if os.path.exists(log):
        out["runs"][name] = parse(log)
    elif name == "all3":
        out["runs"][name] = {"skipped": "a-d not all passing"}
print("[canary] FINAL:", json.dumps({k: {"pass": v.get("pass"), "ce": v.get("ce_trace", [])[-2:]}
                                    for k, v in out.get("runs", {}).items()})[:1500])
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print("[canary] wrote", path)
PY

echo "[canary] DONE utc=$(date -u +%FT%TZ)"
