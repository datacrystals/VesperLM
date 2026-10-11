"""Tiny end-of-path repro for the 3990/4000 trainer-vanish (lab/FAILURES.md).

t3 (470m_k) and the box 429M run both "vanished" at the end of the step
budget: last log line `Step 03990` of total_steps=4000, no traceback, no
completion message, and NO final checkpoint (latest save step_3500 — ~490
steps / ~450M tokens of weights lost on the box run).

Root cause (02_pretrain_linear.py): `for step in range(start_step, total_steps)`
never steps `total_steps` itself, while eval/save only fired on
`step % eval_interval == 0` and `step % 500 == 0` — so the scheduled end-of-run
eval+save (the step that would have been 4000) could never trigger, and the
process fell out of the loop into destroy_process_group()/exit with no
epilogue. The every-10-step log line at 3990 is the last thing written; steps
3991..3999 then run silently (~22 min at 429M/8k) and the process exits 0 —
indistinguishable from a crash.

This harness forces that end-of-path at tiny scale (VESPER_TOTAL_STEPS=20 on
the `tiny_agent` config — GLA hybrid, the same stack the box ran for the 429M
incident; eval_interval stays 100 so NO periodic eval/save can fire) and
checks the guarantees the fixed trainer must provide:

  complete: exit 0, explicit "[TRAINING COMPLETE]" banner, and a final
            eval+save at step 19 (step_19/checkpoint.pt,
            eval_samples_step19.json) — pre-fix these all silently miss.
  crash:    an injected RuntimeError at step 15 (monkeypatched p01.get_lr);
            must print the traceback AND leave an emergency checkpoint for the
            last completed step (step_14/checkpoint.pt) before exiting
            non-zero — pre-fix a crash loses everything since the last %500
            save.

Run on a CUDA box (the trainer is CUDA-only; laptop 3070 or the free P40s):
    python Pretrain/tests_end_of_loop.py            # both scenarios
    python Pretrain/tests_end_of_loop.py complete   # one scenario

The harness owns vesper_linear_checkpoints_tiny_agent/ and moves any
pre-existing copy aside before each scenario. Config choice: `tiny_agent`
(GLA hybrid — the stack the 429M box run used). The literal lab_tiny config is
KDA+MLA and cannot run unpatched here (fla's MLA hard-requires flash-attn;
tools/patch_fla.py is fla-version-drifted on this box), and KDA Triton kernels
are unproven on sm_61. First run pays the one-time FLA Triton JIT (~minutes);
later runs hit /tmp/triton_cache + /tmp/fla_cache.
"""
import os
import shutil
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
TRAINER = os.path.join(_HERE, "02_pretrain_linear.py")
CKPT_DIR = os.path.join(_HERE, "vesper_linear_checkpoints_tiny_agent")
TOTAL_STEPS = 20
CRASH_AT = 15

CRASH_DRIVER = f"""
import importlib.util
import os

spec = importlib.util.spec_from_file_location("p02", {TRAINER!r})
p02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p02)
_orig_get_lr = p02.p01.get_lr
_crash_at = int(os.environ["VESPER_TEST_CRASH_AT"])

def _get_lr(step, *a, **k):
    if step >= _crash_at:
        raise RuntimeError("injected end-of-path failure")
    return _orig_get_lr(step, *a, **k)

p02.p01.get_lr = _get_lr
p02.train()
"""


def _env():
    env = os.environ.copy()
    env.update({
        "VESPER_CONFIG": "tiny_agent",
        "VESPER_TOTAL_STEPS": str(TOTAL_STEPS),
        "VESPER_ACCUM": "2",
        "VESPER_SEED": "7",
    })
    return env


def _fresh_ckpt_dir():
    if os.path.exists(CKPT_DIR):
        bak = f"{CKPT_DIR}_bak_{int(time.time())}"
        print(f"[repro] moving existing {CKPT_DIR} -> {bak}")
        shutil.move(CKPT_DIR, bak)


def _run(cmd, env):
    print(f"[repro] $ {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=_HERE, env=env, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    print(out[-4000:])
    print(f"[repro] exit={proc.returncode} wall={time.time() - t0:.0f}s")
    return proc.returncode, out


def scenario_complete():
    _fresh_ckpt_dir()
    code, out = _run([sys.executable, TRAINER], _env())
    return {
        "exit 0": code == 0,
        "explicit [TRAINING COMPLETE] banner": "[TRAINING COMPLETE]" in out,
        "final eval ran at step 19": "Validation at Step 19" in out,
        "final checkpoint step_19/checkpoint.pt saved":
            os.path.exists(os.path.join(CKPT_DIR, "step_19", "checkpoint.pt")),
        "final eval_samples_step19.json written":
            os.path.exists(os.path.join(CKPT_DIR, "eval_samples_step19.json")),
    }


def scenario_crash():
    _fresh_ckpt_dir()
    env = _env()
    env["VESPER_TEST_CRASH_AT"] = str(CRASH_AT)
    code, out = _run([sys.executable, "-c", CRASH_DRIVER], env)
    return {
        "non-zero exit": code != 0,
        "traceback flushed to log": "injected end-of-path failure" in out,
        "emergency checkpoint step_14/checkpoint.pt saved":
            os.path.exists(os.path.join(CKPT_DIR, "step_14", "checkpoint.pt")),
        "emergency save message logged": "emergency checkpoint saved" in out,
    }


SCENARIOS = {
    "complete": scenario_complete,
    "crash": scenario_crash,
}


def main():
    which = sys.argv[1:] or list(SCENARIOS)
    unknown = [name for name in which if name not in SCENARIOS]
    if unknown:
        print(f"unknown scenario(s): {unknown}; choose from {list(SCENARIOS)}")
        return 2
    all_ok = True
    for name in which:
        print(f"\n=== scenario: {name} ===")
        checks = SCENARIOS[name]()
        for label, ok in checks.items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        all_ok = all_ok and all(checks.values())
    print(f"\n=== end-of-loop repro: {'ALL PASS' if all_ok else 'FAILURES PRESENT'} ===")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
