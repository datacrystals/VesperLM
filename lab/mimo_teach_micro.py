#!/usr/bin/env python3
"""mimo_teach_micro — per-session gated teach micro-session worker
(design: lab/MIMO_TEACH_HOOKS.md; policy: lab/TEACH_PIPELINE.md).

Spawned by vdemo /learn.  Reads the session JSONL (corrected triples +
labels), fits ONE rank-64 wrap at MoE block 0 from the wrap geometry
family (identity-at-birth, cumulative resume from the incumbent delta),
mild-CE on revised answers + out_pen NEG cadence, writes the candidate,
gates with consolidate.stub_gate semantics verbatim (tier-1), and
PROMOTEs atomically or ROLLBACKs.  Teaching never applies on a gate
REJECT.

Usage (spawned by the demo; HIP_VISIBLE_DEVICES is set by the parent):
  python mimo_teach_micro.py <user> <session_jsonl>
Env: TEACH_STEPS (24), TEACH_LR (3e-4), TEACH_LAM (0.5).
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/data/tliao/mimo_flash_lab")
sys.path.insert(0, "/home/tliao/mimo_flash/teach_lib")  # Hippocampus mirror
import mimo_full_host as fh  # noqa: E402
import mimo_flash_host as mh  # noqa: E402
import door25_wrap_expert as d25  # noqa: E402

USER = sys.argv[1]
SESS = sys.argv[2]
STEPS = int(os.environ.get("TEACH_STEPS", "24"))
LR = float(os.environ.get("TEACH_LR", "3e-4"))
LAM = float(os.environ.get("TEACH_LAM", "0.5"))
OUTPEN = 300.0
NEG_PROMPTS = [
    "What is the capital of France?",
    "What is 17 + 25?",
    "How do I bake a loaf of bread?",
]
BASEWIN = ("The history of economics is marked by long cycles of "
           "expansion and contraction, and each period leaves behind "
           "institutions that condition the next.")
ROOT = "/home/tliao/mimo_flash/teach_engines"
CLAIM_DIR = "/data/hackathon/.tmp/mr_gpu_claims"
GPU_INDEX = os.environ.get("HIP_VISIBLE_DEVICES", "0").split(",")[0].strip()
CLAIM_PATH = os.path.join(CLAIM_DIR, f"gpu{GPU_INDEX}")


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --- tier-1 gate: consolidate.stub_gate semantics verbatim -----------
def _stub_gate(candidate_dir: str) -> tuple:
    """Verbatim of Hippocampus/consolidate.stub_gate (TEACH_PIPELINE.md
    tripwires): mean_reward >= 0; top_response_fraction <= 0.5 (hard
    target-collapse rule); unique_response_ratio >= 0.25; final_kl <=
    10.0.  Imports the real module when the mirror is present."""
    try:
        from Hippocampus.consolidate import stub_gate  # type: ignore
        r = stub_gate(candidate_dir,
                      os.path.join(ROOT, USER))
        return r.decision, r.reason, r.metrics
    except Exception:
        pass
    stats = json.load(open(os.path.join(candidate_dir, "train_stats.json")))
    mean_reward = float(stats.get("mean_reward", 0.0))
    uniq_ratio = float(stats.get("unique_response_ratio", 1.0))
    top_frac = float(stats.get("top_response_fraction", 0.0))
    final_kl = float(stats.get("final_kl", 0.0))
    metrics = {"mean_reward": mean_reward,
               "unique_response_ratio": uniq_ratio,
               "top_response_fraction": top_frac, "final_kl": final_kl}
    if mean_reward < 0.0:
        return "ROLLBACK", f"mean reward {mean_reward:.3f} < 0", metrics
    if top_frac > 0.5:
        return "ROLLBACK", (f"target collapse: one response is "
                            f"{top_frac:.0%} of the batch"), metrics
    if uniq_ratio < 0.25:
        return "ROLLBACK", (f"degenerate batch: unique response ratio "
                            f"{uniq_ratio:.2f} < 0.25"), metrics
    if final_kl > 10.0:
        return "ROLLBACK", f"KL drift from base {final_kl:.3f} > 10.0", metrics
    return "PROMOTE", "stub gate checks passed", metrics


class Claim:
    def __init__(self):
        self.held = False

    def take(self) -> None:
        os.makedirs(CLAIM_DIR, exist_ok=True)
        if os.path.exists(CLAIM_PATH):
            log(f"ABORT: {CLAIM_PATH} claimed by another tenant")
            sys.exit(2)
        with open(CLAIM_PATH, "w") as f:
            f.write(f"pid={os.getpid()} user={os.environ.get('USER','?')}"
                    f"@{socket.gethostname()} prog=mimo_teach_micro "
                    f"started={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.held = True
        log(f"CLAIM ok: {CLAIM_PATH}")

    def drop(self) -> None:
        if self.held and os.path.exists(CLAIM_PATH):
            try:
                if f"pid={os.getpid()}" in open(CLAIM_PATH).read():
                    os.remove(CLAIM_PATH)
                    log(f"CLAIM released: {CLAIM_PATH}")
            except OSError:
                pass
        self.held = False


def main() -> None:
    claim = Claim()
    try:
        claim.take()
        inc = os.path.join(ROOT, USER, "delta.pt")
        udir = os.path.join(ROOT, USER)
        os.makedirs(udir, exist_ok=True)
        records = [json.loads(x) for x in open(SESS) if x.strip()]
        corr = [r for r in records if r.get("label") == "correct"]
        if not corr:
            log("nothing to teach (no corrected triples)")
            json.dump({"decision": "ROLLBACK", "reason": "empty batch",
                       "ts": time.strftime("%Y-%m-%d %H:%M:%S")},
                      open(os.path.join(udir, "last_gate.json"), "w"))
            sys.exit(4)
        rewards = [float(r.get("reward", 0)) for r in records
                   if "reward" in r]
        answers = [r["correction"] for r in corr]
        from collections import Counter
        counts = Counter(answers)
        stats = {
            "mean_reward": sum(rewards) / max(1, len(rewards)),
            "unique_response_ratio": len(set(answers)) / len(answers),
            "top_response_fraction": max(counts.values()) / len(answers),
        }
        log(f"batch: {len(corr)} corrections, rewards n={len(rewards)}, "
            f"uniq={stats['unique_response_ratio']:.2f} "
            f"top={stats['top_response_fraction']:.2f}")

        model, tok, cache = fh.build_full_host(fh.MODEL_DIR, "cuda",
                                               18.0, {})
        from torch.utils.checkpoint import checkpoint
        for layer in model.model.layers:
            orig = layer.forward
            def ck(hs, _o=orig, **kw):
                return checkpoint(_o, hs, use_reentrant=False, **kw)
            layer.forward = ck
        blk = mh.find_mimo_moe_blocks(model)[0]
        stock_fwd = blk.experts.forward
        ex = blk.experts[0]
        w_gu, w_down = ex.native_core()
        wrap = d25.WrappedForeignExpert(4096, 64, w_gu, w_down,
                                        seed=0, act="swiglu")

        inc = os.path.join(ROOT, USER, "delta.pt")
        resumed = False
        if os.path.exists(inc):
            wrap.load_state_dict(
                torch.load(inc, map_location="cpu",
                           weights_only=False)["wrap"])
            resumed = True
        wrap.to("cuda").train()
        log(f"teach wrap {'resumed from incumbent' if resumed else 'identity-at-birth'}; "
            f"steps={STEPS} lr={LR} lam={LAM}")

        def ce(ids, label_from):
            x = torch.tensor([ids], dtype=torch.long, device="cuda")
            o = model(input_ids=x, use_cache=False)
            return F.cross_entropy(
                o.logits[0, label_from - 1:-1].float(), x[0, label_from:])

        homes = []
        for r in corr:
            pids = tok.encode(tok.apply_chat_template(
                [{"role": "user", "content": r["question"]}],
                tokenize=False, add_generation_prompt=True),
                add_special_tokens=False)
            ans = r["correction"]
            aids = tok.encode(" " + ans if not ans.startswith(" ") else ans,
                              add_special_tokens=False)
            homes.append((pids + aids, len(pids)))

        def inject(h, i, w):
            return stock_fwd(h, i, w) + (LAM * wrap(h)).to(h.dtype)

        opt = torch.optim.AdamW(wrap.parameters(), lr=LR, weight_decay=0.0)
        blk.experts.forward = stock_fwd
        for step in range(STEPS):
            neg = step % 3 == 2
            if neg:
                ids = tok.encode(tok.apply_chat_template(
                    [{"role": "user", "content":
                      NEG_PROMPTS[step % len(NEG_PROMPTS)]}],
                    tokenize=False, add_generation_prompt=True),
                    add_special_tokens=False)
            else:
                batch, label_from = homes[step % len(homes)]

            def run_step():
                if neg:
                    blk.experts.forward = stock_fwd
                    ce_off = ce(ids, 1).detach()
                    blk.experts.forward = inject
                    ce_on = ce(ids, 1)
                    loss = OUTPEN * F.relu(ce_on - ce_off)
                    shown = float(ce_on)
                else:
                    blk.experts.forward = inject
                    loss = ce(batch, max(2, label_from))
                    shown = float(loss)
                opt.zero_grad()
                loss.backward()
                return shown
            for attempt in range(4):
                try:
                    shown = run_step()
                    break
                except RuntimeError:
                    blk.experts.forward = stock_fwd
                    if attempt == 3:
                        raise
                    torch.cuda.synchronize()
                    log(f"RETRY step {step} attempt {attempt + 1}")
            torch.nn.utils.clip_grad_norm_(wrap.parameters(), 1.0)
            opt.step()
            blk.experts.forward = stock_fwd
            if step % 4 == 3 or step == STEPS - 1:
                log(f"STEP {step:>3} {'NEG' if neg else 'HOME'} "
                    f"ce={shown:.4f}")

        blk.experts.forward = stock_fwd
        wrap.eval()
        ids = tok.encode(BASEWIN, add_special_tokens=False)[:65]
        with torch.no_grad():
            ce0 = float(ce(ids, 1))
            blk.experts.forward = inject
            ce1 = float(ce(ids, 1))
            blk.experts.forward = stock_fwd
        stats["final_kl"] = (ce1 - ce0) * len(ids)
        log(f"drift proxy: parity {ce0:.4f}->{ce1:.4f} "
            f"delta={ce1 - ce0:+.4f} nats={stats['final_kl']:+.2f}")

        udir = os.path.join(ROOT, USER)
        cand = os.path.join(udir, "candidate")
        os.makedirs(cand, exist_ok=True)
        torch.save({"format": "vesper-teach-delta-v1", "user": USER,
                    "lam": LAM, "seed": 0, "dose": {"steps": STEPS,
                    "lr": LR}, "wrap": wrap.state_dict()},
                   os.path.join(cand, "delta.pt"))
        json.dump(stats, open(os.path.join(cand, "train_stats.json"), "w"),
                  indent=1)

        decision, reason, metrics = _stub_gate(cand)
        json.dump({"decision": decision, "reason": reason,
                   "metrics": metrics,
                   "ts": time.strftime("%Y-%m-%d %H:%M:%S")},
                  open(os.path.join(udir, "last_gate.json"), "w"), indent=1)
        log(f"GATE {decision}: {reason} metrics={metrics}")
        if decision == "PROMOTE":
            if os.path.exists(inc):
                prev = os.path.join(udir, "delta_prev.pt")
                if os.path.exists(prev):
                    os.remove(prev)
                os.replace(inc, prev)
            os.replace(os.path.join(cand, "delta.pt"), inc)
            json.dump({"user": USER, "ts": time.time()},
                      open(os.path.join(udir, "pending_promote.json"), "w"))
            log(f"PROMOTED -> {inc}")
            rc = 0
        else:
            log("candidate discarded (rollback); incumbent untouched")
            rc = 4
        import shutil
        shutil.rmtree(cand, ignore_errors=True)
        print(f"TEACH_GATE={decision}", flush=True)
        print("TEACH_DONE", flush=True)
        claim.drop()
        sys.exit(rc)
    finally:
        claim.drop()


if __name__ == "__main__":
    main()
