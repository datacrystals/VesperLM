#!/usr/bin/env python3
"""mimo_teach_lora — per-session gated teach worker, final-block attention
LoRA geometry (design: lab/MIMO_TEACH_HOOKS.md; policy: lab/TEACH_PIPELINE.md;
vk-series recipe class: rank 8, alpha 16).

Successor geometry to the CLOSED block-0-wrap teach fit (FAILURES.md
mimo-teach-micro-divergence / mimo-teach-retry-lr-map: CE rises from step 0
at lr 3e-4/3e-5/3e-6 — recipe-shape unstable, do not retry that geometry).

Geometry (declared): LoRA rank 8, alpha 16 on the FINAL transformer block's
attention projections — q/k/v slice-wise inside the fused qkv_proj, plus
o_proj — of the MiMo full host (layer 47, full-attention per
hybrid_layer_pattern).  Trunk frozen; LoRA-B zeroed at birth
(identity-at-birth); cumulative resume from an existing promoted incumbent
delta if present.  305,152 trainable params (q 131,072 + k 38,912 +
v 36,864 + o 98,304).

Objective: mild-CE on corrected answers (prompt context, CE on the
revised-answer tail) + out_pen=300 NEG parity penalty every 3rd step on the
same 3 neutral prompts as mimo_teach_micro; grad-clip 1.0; 24 steps default
(TEACH_STEPS env).  Declared knob map: primary lr 2e-4; if CE rises from
step 0, ONE nested drop to 2e-5; if that also diverges -> TEACH-FAIL, close
the slot, document, stop.

Gate: tier-1 stub_gate semantics verbatim (identical to mimo_teach_micro).
PROMOTE -> atomic delta.pt under teach_engines/<user>/ (incumbent kept as
delta_prev.pt); ROLLBACK -> candidate discarded.  Teaching never applies on
a gate REJECT.

Usage (spawned by the demo or by hand; HIP_VISIBLE_DEVICES set by parent):
  python mimo_teach_lora.py <user> <session_jsonl>
Env: TEACH_STEPS (24), TEACH_LR (2e-4), TEACH_RANK (8), TEACH_ALPHA (16),
     TEACH_LAYER (47).
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/data/tliao/mimo_flash_lab")
sys.path.insert(0, "/home/tliao/mimo_flash/teach_lib")  # Hippocampus mirror
import mimo_full_host as fh  # noqa: E402

USER = sys.argv[1] if len(sys.argv) > 1 else None
SESS = sys.argv[2] if len(sys.argv) > 2 else None
if __name__ == "__main__" and (USER is None or SESS is None):
    sys.exit("usage: mimo_teach_lora.py <user> <session_jsonl>")
STEPS = int(os.environ.get("TEACH_STEPS", "24"))
LR = float(os.environ.get("TEACH_LR", "2e-4"))
RANK = int(os.environ.get("TEACH_RANK", "8"))
ALPHA = float(os.environ.get("TEACH_ALPHA", "16"))
LAYER = int(os.environ.get("TEACH_LAYER", "47"))
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
DELTA_FORMAT = "vesper-teach-lora-delta-v1"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --- LoRA wrappers (shared with the standalone verify script) ---------
class LoRAFusedQKV(nn.Module):
    """Slice-wise q/k/v LoRA on top of the frozen fused qkv_proj."""

    def __init__(self, base: nn.Linear, q_size: int, k_size: int,
                 v_size: int, rank: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.q_size, self.k_size, self.v_size = q_size, k_size, v_size
        self.scale = alpha / rank
        in_f = base.in_features
        dev = base.weight.device
        g = torch.Generator(device="cpu").manual_seed(0)

        def mk_a() -> nn.Parameter:
            return nn.Parameter(
                (torch.randn(rank, in_f, generator=g) * 0.02).to(dev))

        def mk_b(out_f: int) -> nn.Parameter:
            return nn.Parameter(torch.zeros(out_f, rank, device=dev))

        self.Aq, self.Bq = mk_a(), mk_b(q_size)
        self.Ak, self.Bk = mk_a(), mk_b(k_size)
        self.Av, self.Bv = mk_a(), mk_b(v_size)
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.enabled:
            xf = x.float()
            d = torch.cat([
                (xf @ self.Aq.t()) @ self.Bq.t(),
                (xf @ self.Ak.t()) @ self.Bk.t(),
                (xf @ self.Av.t()) @ self.Bv.t()], dim=-1)
            out = out + (d * self.scale).to(out.dtype)
        return out

    def lora_state(self) -> dict:
        return {k: v.detach().cpu() for k, v in (
            ("Aq", self.Aq), ("Bq", self.Bq), ("Ak", self.Ak),
            ("Bk", self.Bk), ("Av", self.Av), ("Bv", self.Bv))}

    def load_lora(self, state: dict) -> None:
        for k in ("Aq", "Bq", "Ak", "Bk", "Av", "Bv"):
            getattr(self, k).data.copy_(state[k].to(self.base.weight.device))


class LoRALinear(nn.Module):
    """Plain LoRA on top of a frozen Linear (o_proj)."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scale = alpha / rank
        dev = base.weight.device
        g = torch.Generator(device="cpu").manual_seed(1)
        self.A = nn.Parameter(
            (torch.randn(rank, base.in_features, generator=g) * 0.02)
            .to(dev))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank,
                                          device=dev))
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.enabled:
            d = (x.float() @ self.A.t()) @ self.B.t()
            out = out + (d * self.scale).to(out.dtype)
        return out

    def lora_state(self) -> dict:
        return {"A": self.A.detach().cpu(), "B": self.B.detach().cpu()}

    def load_lora(self, state: dict) -> None:
        self.A.data.copy_(state["A"].to(self.base.weight.device))
        self.B.data.copy_(state["B"].to(self.base.weight.device))


def attach_lora(model, delta: dict, layer_idx: int):
    """In-memory, additive, reversible attach.  Host weights on disk are
    never mutated; detach restores the exact pre-attach module references.
    Returns the detach handle."""
    attn = model.model.layers[layer_idx].self_attn
    orig_qkv, orig_o = attn.qkv_proj, attn.o_proj
    qkv = LoRAFusedQKV(orig_qkv, attn.q_size, attn.k_size, attn.v_size,
                       int(delta["rank"]), float(delta["alpha"]))
    o = LoRALinear(orig_o, int(delta["rank"]), float(delta["alpha"]))
    qkv.load_lora(delta["qkv"])
    o.load_lora(delta["o"])
    attn.qkv_proj, attn.o_proj = qkv, o
    return (attn, orig_qkv, orig_o)


def detach_lora(handle) -> None:
    attn, orig_qkv, orig_o = handle
    attn.qkv_proj, attn.o_proj = orig_qkv, orig_o


def set_lora_enabled(model, layer_idx: int, on: bool) -> None:
    attn = model.model.layers[layer_idx].self_attn
    attn.qkv_proj.enabled = on
    attn.o_proj.enabled = on


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
                    f"@{socket.gethostname()} prog=mimo_teach_lora "
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
        udir = os.path.join(ROOT, USER)
        os.makedirs(udir, exist_ok=True)
        inc = os.path.join(udir, "delta.pt")
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

        # geometry: fresh identity-at-birth LoRA on the final block's
        # attention projections; cumulative resume from the incumbent.
        attn = model.model.layers[LAYER].self_attn
        birth = {"rank": RANK, "alpha": ALPHA,
                 "qkv": LoRAFusedQKV(attn.qkv_proj, attn.q_size,
                                     attn.k_size, attn.v_size,
                                     RANK, ALPHA).lora_state(),
                 "o": LoRALinear(attn.o_proj, RANK, ALPHA).lora_state()}
        handle = attach_lora(model, birth, LAYER)
        n_params = sum(p.numel() for m in (attn.qkv_proj, attn.o_proj)
                       for p in m.parameters() if p.requires_grad)
        resumed = False
        if os.path.exists(inc):
            d = torch.load(inc, map_location="cpu", weights_only=False)
            if d.get("format") != DELTA_FORMAT:
                log(f"ABORT: incumbent {inc} is format "
                    f"{d.get('format')!r}, not {DELTA_FORMAT!r} — "
                    "refusing to resume across geometry classes")
                sys.exit(3)
            attn.qkv_proj.load_lora(d["qkv"])
            attn.o_proj.load_lora(d["o"])
            resumed = True
        log(f"teach LoRA {'resumed from incumbent' if resumed else 'identity-at-birth'}; "
            f"layer={LAYER} rank={RANK} alpha={ALPHA} params={n_params} "
            f"steps={STEPS} lr={LR}")

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

        params = [p for m in (attn.qkv_proj, attn.o_proj)
                  for p in m.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
        set_lora_enabled(model, LAYER, True)
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
                    set_lora_enabled(model, LAYER, False)
                    ce_off = ce(ids, 1).detach()
                    set_lora_enabled(model, LAYER, True)
                    ce_on = ce(ids, 1)
                    loss = OUTPEN * F.relu(ce_on - ce_off)
                    shown = float(ce_on)
                else:
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
                    if attempt == 3:
                        raise
                    torch.cuda.synchronize()
                    log(f"RETRY step {step} attempt {attempt + 1}")
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if step % 4 == 3 or step == STEPS - 1:
                log(f"STEP {step:>3} {'NEG' if neg else 'HOME'} "
                    f"ce={shown:.4f}")

        ids = tok.encode(BASEWIN, add_special_tokens=False)[:65]
        with torch.no_grad():
            set_lora_enabled(model, LAYER, False)
            ce0 = float(ce(ids, 1))
            set_lora_enabled(model, LAYER, True)
            ce1 = float(ce(ids, 1))
        stats["final_kl"] = (ce1 - ce0) * len(ids)
        log(f"drift proxy: parity {ce0:.4f}->{ce1:.4f} "
            f"delta={ce1 - ce0:+.4f} nats={stats['final_kl']:+.2f}")

        cand = os.path.join(udir, "candidate")
        os.makedirs(cand, exist_ok=True)
        torch.save({"format": DELTA_FORMAT, "user": USER, "rank": RANK,
                    "alpha": ALPHA, "layer": LAYER,
                    "dose": {"steps": STEPS, "lr": LR},
                    "qkv": attn.qkv_proj.lora_state(),
                    "o": attn.o_proj.lora_state()},
                   os.path.join(cand, "delta.pt"))
        json.dump(stats, open(os.path.join(cand, "train_stats.json"), "w"),
                  indent=1)

        decision, reason, metrics = _stub_gate(cand)
        json.dump({"decision": decision, "reason": reason,
                   "metrics": metrics, "geometry": DELTA_FORMAT,
                   "layer": LAYER, "rank": RANK, "alpha": ALPHA,
                   "lr": LR, "steps": STEPS, "session": os.path.basename(SESS),
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
        detach_lora(handle)
        print(f"TEACH_GATE={decision}", flush=True)
        print("TEACH_DONE", flush=True)
        claim.drop()
        sys.exit(rc)
    finally:
        claim.drop()


if __name__ == "__main__":
    main()
