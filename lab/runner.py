"""Queue executor for lab experiments.

Claims experiment JSONs from queue/, runs the trainer inside a per-run sandbox,
parses the log into a result JSON, and moves files along
queue/ -> running/ -> results/. Filesystem-only; nothing is deleted.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time

import numpy as np

LAB = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(LAB)
TRAINER = os.path.join(REPO, "Pretrain", "02_pretrain_linear.py")
TOKENIZER_SRC = os.path.join(REPO, "Dataset", "custom_tokenizer")

QUEUE_DIR = os.path.join(LAB, "queue")
RUNNING_DIR = os.path.join(LAB, "running")
RESULTS_DIR = os.path.join(LAB, "results")
LOGS_DIR = os.path.join(LAB, "logs")
SANDBOX_DIR = os.path.join(LAB, "sandbox")
SYNTH_DIR = os.path.join(LAB, "data_synth")

# t0 default budget is 30 minutes -> 400 steps; scale linearly from there.
STEPS_PER_30MIN = 400
MIN_FREE_VRAM_GB = 4.0
TIER_DEFAULT_BUDGET = {"t0": 30, "t1": 120, "t2": 480, "t3": 1440}

_RE_STEP = re.compile(r"Step\s+(\d+)")
_RE_VAL = re.compile(r"Val Loss:\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")
_RE_CE = re.compile(r"CE Loss:\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")
_RE_TOK = re.compile(r"Tok/s:\s*([0-9][0-9,]*)\s*\(GPU\)")


def parse_log(text):
    """Extract metrics from a trainer log. Pure function."""
    val_series = []
    ce_last = None
    tok_vals = []
    steps_done = 0
    for line in text.splitlines():
        m = _RE_STEP.search(line)
        if m:
            steps_done = max(steps_done, int(m.group(1)))
        m = _RE_VAL.search(line)
        if m:
            sm = _RE_STEP.search(line)
            step = int(sm.group(1)) if sm else steps_done
            val_series.append([step, float(m.group(1))])
        m = _RE_CE.search(line)
        if m:
            ce_last = float(m.group(1))
        for m in _RE_TOK.finditer(line):
            tok_vals.append(float(m.group(1).replace(",", "")))
    return {
        "val_loss_series": val_series,
        "val_loss_final": val_series[-1][1] if val_series else None,
        "ce_last": ce_last,
        "tok_s_avg": round(sum(tok_vals) / len(tok_vals), 1) if tok_vals else None,
        "steps_done": steps_done,
    }


def ensure_synth():
    """Generate the shared synthetic token bins once (uint16, rng(0))."""
    os.makedirs(SYNTH_DIR, exist_ok=True)
    for name, n_tokens in (("phase1.bin", 3_000_000), ("phase2.bin", 1_000_000)):
        path = os.path.join(SYNTH_DIR, name)
        if os.path.exists(path):
            continue
        tmp = path + ".tmp"
        rng = np.random.default_rng(0)
        mm = np.memmap(tmp, dtype=np.uint16, mode="w+", shape=(n_tokens,))
        mm[:] = rng.integers(0, 8192, size=n_tokens, dtype=np.uint16)
        mm.flush()
        del mm
        os.replace(tmp, path)


def setup_sandbox(exp_id):
    """Build sandbox/<id> with tokenizer link, data bins, and index.txt."""
    sandbox = os.path.join(SANDBOX_DIR, exp_id)
    data = os.path.join(sandbox, "data")
    os.makedirs(data, exist_ok=True)
    tok = os.path.join(sandbox, "custom_tokenizer")
    if not os.path.lexists(tok):
        os.symlink(os.path.abspath(TOKENIZER_SRC), tok)
    ensure_synth()
    for name in ("phase1.bin", "phase2.bin"):
        link = os.path.join(data, name)
        if not os.path.lexists(link):
            os.symlink(os.path.join(SYNTH_DIR, name), link)
    with open(os.path.join(data, "index.txt"), "w") as f:
        f.write("phase1.bin, 0.8\nphase2.bin, 0.2\n")
    return sandbox


def build_env(exp):
    budget = float(exp.get("budget_minutes") or TIER_DEFAULT_BUDGET.get(exp.get("tier"), 30))
    env = dict(os.environ)
    env["VESPER_AMP"] = "bf16"
    env["VESPER_CONFIG"] = exp["config"]
    env["VESPER_TOTAL_STEPS"] = str(max(1, int(round(budget * STEPS_PER_30MIN / 30))))
    for k, v in (exp.get("env") or {}).items():
        env[k] = str(v)
    return env


def gpu_free_gb():
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info()
    return free / 1e9


def claim_one():
    for name in sorted(os.listdir(QUEUE_DIR)):
        if not name.endswith(".json"):
            continue
        src = os.path.join(QUEUE_DIR, name)
        dst = os.path.join(RUNNING_DIR, name)
        try:
            os.rename(src, dst)
            return dst
        except OSError:
            continue
    return None


def load_exp(claim_path):
    with open(claim_path) as f:
        exp = json.load(f)
    if "id" not in exp:
        exp["id"] = os.path.splitext(os.path.basename(claim_path))[0]
    return exp


def start_job(exp, claim_path):
    sandbox = setup_sandbox(exp["id"])
    budget = float(exp.get("budget_minutes") or TIER_DEFAULT_BUDGET.get(exp.get("tier"), 30))
    log_path = os.path.join(LOGS_DIR, exp["id"] + ".log")
    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, "-u", os.path.abspath(TRAINER)],
        cwd=sandbox,
        env=build_env(exp),
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return {
        "exp": exp,
        "claim_path": claim_path,
        "log_path": log_path,
        "log_f": log_f,
        "proc": proc,
        "start": time.time(),
        "deadline": time.time() + budget * 60,
    }


def stop_job(job, sig=signal.SIGTERM):
    try:
        os.killpg(job["proc"].pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def finalize(job, rc, stopped):
    exp = job["exp"]
    try:
        job["log_f"].close()
    except Exception:
        pass
    with open(job["log_path"]) as f:
        text = f.read()
    parsed = parse_log(text)
    res = dict(exp)
    res.update({
        "val_loss_final": parsed["val_loss_final"],
        "val_loss_series": parsed["val_loss_series"],
        "ce_last": parsed["ce_last"],
        "tok_s_avg": parsed["tok_s_avg"],
        "steps_done": parsed["steps_done"],
        "seconds": round(time.time() - job["start"], 1),
        "rc": rc,
        "stopped": stopped,
    })
    if rc != 0 and not parsed["val_loss_series"]:
        res["failed"] = True
        res["log_tail"] = text.splitlines()[-40:]
    result_path = os.path.join(RESULTS_DIR, exp["id"] + ".json")
    if os.path.exists(result_path):
        result_path = os.path.join(RESULTS_DIR, "%s.%d.json" % (exp["id"], int(time.time())))
    with open(job["claim_path"], "w") as f:
        json.dump(res, f, indent=2)
        f.write("\n")
    os.rename(job["claim_path"], result_path)
    print("[runner] %s rc=%s stopped=%s val=%s -> %s" % (
        exp["id"], rc, stopped, parsed["val_loss_final"], os.path.basename(result_path)))


def reap(active):
    for job in active[:]:
        rc = job["proc"].poll()
        if rc is not None:
            finalize(job, rc, None)
            active.remove(job)
        elif time.time() > job["deadline"]:
            stop_job(job)
            try:
                rc = job["proc"].wait(timeout=30)
            except subprocess.TimeoutExpired:
                stop_job(job, signal.SIGKILL)
                rc = job["proc"].wait()
            finalize(job, rc, "budget")
            active.remove(job)


def fail_claim(claim_path, exp_id, err):
    res = {
        "id": exp_id,
        "tier": None,
        "config": None,
        "env": {},
        "val_loss_final": None,
        "val_loss_series": [],
        "ce_last": None,
        "tok_s_avg": None,
        "steps_done": 0,
        "seconds": 0.0,
        "rc": -1,
        "stopped": None,
        "failed": True,
        "log_tail": [str(err)],
    }
    result_path = os.path.join(RESULTS_DIR, exp_id + ".json")
    if os.path.exists(result_path):
        result_path = os.path.join(RESULTS_DIR, "%s.%d.json" % (exp_id, int(time.time())))
    with open(claim_path, "w") as f:
        json.dump(res, f, indent=2)
        f.write("\n")
    os.rename(claim_path, result_path)
    print("[runner] %s failed to start: %s" % (exp_id, err))


def run_once(slots, watch):
    active = []
    last_vram_warn = 0.0
    try:
        while True:
            while len(active) < slots:
                free = gpu_free_gb()
                if free is not None and free < MIN_FREE_VRAM_GB:
                    if time.time() - last_vram_warn > 60:
                        print("[runner] %.1f GB free VRAM < %.1f GB, waiting" % (free, MIN_FREE_VRAM_GB))
                        last_vram_warn = time.time()
                    break
                claim = claim_one()
                if claim is None:
                    break
                exp_id = os.path.splitext(os.path.basename(claim))[0]
                try:
                    exp = load_exp(claim)
                    exp_id = exp["id"]
                    job = start_job(exp, claim)
                except Exception as e:
                    fail_claim(claim, exp_id, e)
                    continue
                print("[runner] claimed %s (%s)" % (exp["id"], exp.get("config")))
                active.append(job)
            reap(active)
            queued = any(n.endswith(".json") for n in os.listdir(QUEUE_DIR))
            if not active and not queued:
                if not watch:
                    return
                time.sleep(30)
                continue
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[runner] interrupted, terminating children")
        for job in active:
            stop_job(job)
        for job in active:
            try:
                rc = job["proc"].wait(timeout=30)
            except subprocess.TimeoutExpired:
                stop_job(job, signal.SIGKILL)
                rc = job["proc"].wait()
            finalize(job, rc, "interrupt")
        raise SystemExit(130)


def main():
    ap = argparse.ArgumentParser(description="lab experiment queue runner")
    ap.add_argument("--slots", type=int, default=8, help="concurrent training runs")
    ap.add_argument("--once", action="store_true", help="process until the queue is empty")
    ap.add_argument("--watch", action="store_true", help="keep polling the queue every 30s")
    args = ap.parse_args()
    for d in (QUEUE_DIR, RUNNING_DIR, RESULTS_DIR, LOGS_DIR, SANDBOX_DIR):
        os.makedirs(d, exist_ok=True)
    run_once(args.slots, args.watch)


if __name__ == "__main__":
    main()
