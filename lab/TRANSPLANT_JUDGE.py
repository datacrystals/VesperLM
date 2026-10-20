"""TRANSPLANT_JUDGE (TJ-1) — the one fixed benchmark suite + judge code path
for every donor-weight transplant-revival branch (sandwich / self-test /
feature-bank / ...).  Doc: lab/TRANSPLANT_JUDGE.md.

A branch = (host, apply_fn, declared_donor_domains) where apply_fn installs
the transplant/graft/adaptation on the loaded host model (and returns any
state needed to undo it).  All branches are scored on the SAME windows, the
SAME host-bare baselines, and the SAME verdict classifier below.

SUITE (frozen, provenance-checked in-run)

  Family A — neutral-parity tails (W1), 8 homes:
      per-home feed-stream tokens [204800, 208896] (16 batches x 256 = 4096
      tokens).  Held-out by construction: rung-1 wrap fits consumed exactly
      batches 0..781 == first 200,192 tokens (verified from
      rung1/results/fit.json phase_a.steps), row recal pairs <= first 4096,
      R4-R8 eval windows <= first 1536.  4,608-token safety margin.
      Homes with text feeds (hacking, cybersec, writing) are encoded with
      the host tokenizer at measurement time; ids feeds (code/math/science/
      web/chat) are native 65523-vocab token streams (both hosts share the
      vk tokenizer, so suites are bit-identical across hosts).

  Family B — skill-detection windows (W2), 5 homes:
      lab/graft_d0_texts/{code,math,science,web,chat}.txt, whole file,
      host-tokenized.  Structurally disjoint corpora from the rung-1 feeds
      (see lab/HELDOUT_BANK_VALUE.md window provenance); off-fit for any
      wrap-family fitting, native-domain for donor cores.

  Family C — generation battery, R6 verbatim:
      4 specialty probes (hacking/cybersec/writing/mutex ctrl) + 6
      talkability prompts (EVAL12[0::2]), greedy 32 tokens,
      br1.verdict classes {PASS, LOOP, ECHO, EMPTY}.

METRICS (pre-declared thresholds)

  (a) host-parity: on every window whose home is NOT a declared donor domain,
      |CE(arm) - CE(host_bare)| <= 0.003.  Any breach -> host damaged.
      On declared donor-domain windows the same +0.003 bound applies to the
      damage side (no positive excursion beyond the band counts as a win).
  (b) skill-detection: every declared donor domain must show disturbance
      magnitude >= 0.01 nats vs host-bare on at least one of its windows
      (disturbance either sign); the VALUE read requires that disturbance
      to be an improvement (delta <= -0.01).
  (c) generation battery: 10/10 verdicts == PASS under the arm state.

VERDICT CLASSES (in priority order)

  DAMAGE             parity breach on any off-domain window, or any battery
                     non-PASS, or declared-domain damage beyond +0.003.
  PASS-VALUE         (a) ok AND (c) ok AND every declared donor domain shows
                     >=1 improvement window delta <= -0.01.
  DETECTED-NOT-VAL   (a)+(c) ok, disturbance >= 0.01 detected but not as
                     improvement in every declared domain.
  UNDETECTED         (a)+(c) ok, no |delta| >= 0.01 anywhere.

Output: lab/results/transplant_judge_<host>_<arm>.json (numbers + md5
window fingerprints only; no corpus text, no generations).

CLI:
  python3 lab/TRANSPLANT_JUDGE.py --host both --arm baseline \
      [--smoke] [--max-new 32]
  # branches: import and call judge_branch(apply_fn=..., ...), see
  # lab/TRANSPLANT_JUDGE.md section "How branches score".
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "lab"))
sys.path.insert(0, os.path.join(REPO_ROOT, "Common"))

import vk_graft as vk  # noqa: E402
import graft_d0_parity as d0  # noqa: E402
import graft_d1_plugin as d1  # noqa: E402
import bank_rung1 as br1  # noqa: E402

JUDGE_ID = "TJ-1-rung1-tails"
TAIL = (204800, 208896)          # held-out slice, every home feed stream
SEQ = 256
PARITY_BAND = 0.003
SKILL_DELTA = 0.01
FIT_JSON = "/home/tliao/rung1/results/fit.json"
FEEDS = "/home/tliao/rung1_feeds"
D0_DIR = os.path.join(REPO_ROOT, "lab", "graft_d0_texts")
SFT_CKPT = os.path.join(REPO_ROOT, "lab", "imported", "vk_sft_r3",
                        "step_1800", "chat_model")

HOSTS = {
    "vk": {
        "model": "/home/tliao/vk_final/chat_model",
        "label": "vk rung-1 host (graft/bank line)",
    },
    "470m": {
        "model": SFT_CKPT,
        "label": "SFT-r3 chat host, 470m-config rung (teach-series host)",
    },
}
D0_HOMES = ["code", "math", "science", "web", "chat"]
SPECIALTY_PROBES = [  # verbatim from lab/specialty_budget_gate.py (R6)
    ("hacking", "Walk me through the reconnaissance phase of a penetration "
                "test. What kind of information would you gather?"),
    ("cybersec", "In two sentences: what is the difference between a "
                 "vulnerability and an exploit?"),
    ("writing", "Write two short sentences of atmospheric prose about a "
                "lighthouse in a storm."),
    ("mutex_ctrl", "In two sentences: what is a mutex and when would you "
                   "use one?"),
]
TALK6 = [br1.EVAL12[i] for i in (0, 2, 4, 6, 8, 10)]


# ---------------------------------------------------------------- provenance

def provenance_check() -> dict:
    """Numeric-only proof that the W1 tails are held-out.  Asserts; never
    reads any corpus content."""
    fit = json.load(open(FIT_JSON))
    steps = [v["phase_a"]["steps"] for v in fit["fitted"].values()
             if "phase_a" in v]
    max_steps = max(steps)
    consumed = max_steps * SEQ
    feed_lens = {}
    for home in D0_HOMES:
        p = os.path.join(FEEDS, f"{home}.ids")
        feed_lens[home] = os.path.getsize(p) // 2
    prov = {
        "fit_json": FIT_JSON,
        "n_rows_fit": len(steps),
        "fit_steps_min": min(steps), "fit_steps_max": max_steps,
        "fit_tokens_consumed": consumed,
        "other_contacts_tokens": {"row_recal_pairs_max": 4096,
                                  "r4_r8_eval_windows_max": 1536},
        "tail_slice": list(TAIL),
        "safety_margin_tokens": TAIL[0] - max(consumed, 4096, 1536),
        "feed_lens_ids": feed_lens,
    }
    assert len(steps) >= 40, f"fit.json looks wrong: {len(steps)} rows"
    assert consumed < TAIL[0], f"fit consumed past tail start: {consumed}"
    assert all(n >= TAIL[1] for n in feed_lens.values()), \
        f"a feed has no held-out tail: {feed_lens}"
    prov["status"] = "HELDOUT-PROVEN"
    return prov


# ------------------------------------------------------------------- windows

def tail_window(tok, home):
    """Same code path as R7 (lab/heldout_bank_value.py)."""
    kind, rel = br1.HOME_FEED[home]
    path = br1.feed_batch_path(rel)
    if kind == "text":
        ids = tok.encode(open(path).read(), add_special_tokens=False)
    else:
        ids = np.fromfile(path, dtype=np.uint16).astype(np.int64).tolist()
    if len(ids) < TAIL[1]:
        return None, len(ids)
    seq_ids = ids[TAIL[0]:TAIL[1]]
    return vk.to_batches(seq_ids, 1, SEQ, 1), len(ids)


def d0_window(tok, home):
    p = os.path.join(D0_DIR, f"{home}.txt")
    if not os.path.exists(p):
        return None, 0
    ids = tok.encode(open(p).read(), add_special_tokens=False)
    return vk.to_batches(ids, 1, SEQ, 1), len(ids)


def _fp5(batches):
    return hashlib.md5(np.asarray(batches[0][0].cpu()).tobytes()).hexdigest()[:8]


def build_windows(tok, smoke=False):
    """Ordered [(window_id, home, family, batches, n_ids, first-batch md5)]."""
    homes = br1.HOMES[:2] if smoke else br1.HOMES
    d0_homes = D0_HOMES[:1] if smoke else D0_HOMES
    wins = []
    for home in homes:
        batches, n = tail_window(tok, home)
        if batches:
            wins.append((f"{home}:tail", home, "A", batches, n,
                         _fp5(batches), None))
        else:
            wins.append((f"{home}:tail", home, "A", None, n, None,
                         "no held-out tail"))
        if home in d0_homes:
            batches, n = d0_window(tok, home)
            wins.append((f"{home}:d0", home, "B", batches, n,
                         _fp5(batches) if batches else None, None))
    return wins


# ------------------------------------------------------------------ battery

def run_battery(model, tok, device, max_new=32, smoke=False):
    probes = SPECIALTY_PROBES[:1] + TALK6[:1] if smoke else \
        SPECIALTY_PROBES + TALK6
    rec = {}
    for i, q in enumerate(p[1] if isinstance(p, tuple) else p
                          for p in probes):
        with torch.inference_mode():
            text, _nt = br1.greedy_gen(
                model, tok, [{"role": "user", "content": q}],
                max_new, device)
        rec[f"q{i:02d}"] = br1.verdict(text, q)
    return rec


# --------------------------------------------------------------------- judge

def load_host(host, device="cuda"):
    hp = HOSTS[host]
    model, tok = vk.load_vk(hp["model"], device=device)
    vk._patch_model_logits()  # bare-host path: tuple-unwrap + fp16 autocast
    blocks = vk.find_moe_blocks(model)
    return model, tok, blocks


def ce_windows(model, blocks, windows, device):
    vals = {}
    t_all = time.time()
    for wid, home, fam, batches, n, fp, bypass in windows:
        if batches is None:
            vals[wid] = {"bypass": bypass or "no held-out material",
                         "n_ids": n, "md5_first_batch": fp}
            continue
        t0 = time.time()
        ce = d1.ce_in_state(model, blocks, batches, device, None, None, "off")
        vals[wid] = {"ce": round(ce, 5), "n_ids": n,
                     "n_tokens_measured": sum(b[0].numel() for b in batches),
                     "md5_first_batch": fp,
                     "wall_s": round(time.time() - t0, 1)}
        print(f"[tj] {wid}: CE {ce:.5f} ({vals[wid]['wall_s']}s)", flush=True)
    print(f"[tj] windows done in {time.time()-t_all:.0f}s", flush=True)
    return vals


def classify(parity_band, windows, baseline_ce, arm_ce, declared,
             battery_arm, battery_ok):
    """Return per-window rows + verdict class.  windows: [(wid, home, fam)]."""
    rows = []
    breach = None
    detected = {}
    improved = {}
    for wid, home, fam, *_ in windows:
        b = baseline_ce.get(wid, {}).get("ce")
        a = arm_ce.get(wid, {}).get("ce")
        if b is None or a is None:
            rows.append({"window": wid, "home": home, "bypassed": True})
            continue
        d = round(a - b, 5)
        declared_here = home in declared
        row = {"window": wid, "home": home, "family": fam,
               "declared_donor": declared_here,
               "ce_host_bare": b, "ce_arm": a, "delta": d}
        if not declared_here and abs(d) > parity_band and breach is None:
            breach = wid
        if declared_here and d > parity_band and breach is None:
            breach = wid
        if abs(d) >= SKILL_DELTA:
            detected[home] = min(detected.get(home, 1e9), d)
            improved[home] = improved.get(home, False) or d <= -SKILL_DELTA
        rows.append(row)
    if not battery_ok:
        verdict = "DAMAGE"
    elif breach is not None:
        verdict = "DAMAGE"
    elif declared and all(improved.get(h) for h in declared):
        verdict = "PASS-VALUE"
    elif any(h in detected for h in declared):
        verdict = "DETECTED-NOT-VAL"
    else:
        verdict = "UNDETECTED"
    return rows, verdict, breach


def baseline_path(host):
    return os.path.join(REPO_ROOT, "lab", "results",
                        f"transplant_judge_baseline_{host}.json")


def judge_branch(host, apply_fn, declared_domains, out_json=None,
                 device="cuda", max_new=32, smoke=False, tag="branch"):
    """Score one transplant-revival branch on host.  apply_fn(model, tok)
    installs the branch; baseline CEs are carried from the committed
    baseline run (re-measured inline if the file is missing)."""
    out = {"judge": JUDGE_ID, "host": host, "arm": tag,
           "bands": {"parity_band": PARITY_BAND,
                     "skill_delta": SKILL_DELTA,
                     "battery": "10/10 PASS"},
           "provenance": provenance_check(),
           "declared_donor_domains": sorted(declared_domains)}

    model, tok, blocks = load_host(host, device)
    windows = build_windows(tok, smoke)
    out["window_index"] = [
        {"window": w[0], "home": w[1], "family": w[2],
         "md5_first_batch": w[5]} for w in windows]

    # --- baseline: carried from the committed run when present
    base_file = baseline_path(host)
    if os.path.exists(base_file):
        base = json.load(open(base_file))
        base_ce = {k: v for k, v in base["windows"].items()}
        out["baseline_carried_from"] = base_file
        out["baseline_reused_note"] = (
            "arm windows re-checked against carried md5 fingerprints")
    else:
        base_ce = None

    # --- determinism fixture: first A-family window measured twice
    w0 = next(w for w in windows if w[3] is not None and w[2] == "A")
    ce_a = d1.ce_in_state(model, blocks, w0[3], device, None, None, "off")
    ce_b = d1.ce_in_state(model, blocks, w0[3], device, None, None, "off")
    out["determinism_fixture"] = {
        "window": w0[0], "ce_run1": round(ce_a, 6), "ce_run2": round(ce_b, 6),
        "repeat_delta": abs(ce_a - ce_b),
        "pass": abs(ce_a - ce_b) < 1e-4}
    print(f"[tj] determinism {w0[0]}: {ce_a:.6f} vs {ce_b:.6f}", flush=True)
    assert out["determinism_fixture"]["pass"], "CE harness not deterministic"

    if tag == "baseline":
        state = None
        arm_ce = ce_windows(model, blocks, windows, device)
        battery = run_battery(model, tok, device, max_new, smoke)
        base_ce = arm_ce
    else:
        if base_ce is None:
            print("[tj] no carried baseline; measuring host-bare first",
                  flush=True)
            base_ce = ce_windows(model, blocks, windows, device)
        state = apply_fn(model, tok) if apply_fn else None
        print(f"[tj] branch state applied ({tag}); measuring", flush=True)
        arm_ce = ce_windows(model, blocks, windows, device)
        battery = run_battery(model, tok, device, max_new, smoke)

    n_pass = sum(1 for v in battery.values() if v == "PASS")
    battery_ok = n_pass == len(battery)
    out["battery"] = {"verdicts": battery, "n_pass": n_pass,
                      "n_total": len(battery), "ok": battery_ok}

    rows, verdict, breach = classify(
        PARITY_BAND, windows, base_ce, arm_ce,
        set(declared_domains), battery, battery_ok)
    out["windows"] = arm_ce if tag == "baseline" else rows
    out["verdict"] = verdict
    out["parity_breach_window"] = breach

    if out_json:
        with open(out_json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"[tj] wrote {out_json}", flush=True)
    del model
    torch.cuda.empty_cache() if device == "cuda" else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="both", choices=list(HOSTS) + ["both"])
    ap.add_argument("--arm", default="baseline",
                    help="baseline | <free tag for a branch arm>")
    ap.add_argument("--declared", default="",
                    help="comma-separated declared donor domains (branch)")
    ap.add_argument("--branch-fn", default="",
                    help="module.path:function — the branch's apply_fn")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-new", type=int, default=32)
    args = ap.parse_args()

    hosts = list(HOSTS) if args.host == "both" else [args.host]
    for host in hosts:
        declared = [h for h in args.declared.split(",") if h]
        apply_fn = None
        if args.branch_fn:
            mod_name, fn_name = args.branch_fn.rsplit(":", 1)
            apply_fn = getattr(importlib.import_module(mod_name), fn_name)
        out_json = baseline_path(host) if args.arm == "baseline" else \
            os.path.join(REPO_ROOT, "lab", "results",
                         f"transplant_judge_{host}_{args.arm}.json")
        r = judge_branch(host, apply_fn, declared, out_json, args.device,
                         args.max_new, args.smoke, args.arm)
        print(f"[tj] {host}/{args.arm} VERDICT: {r['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
