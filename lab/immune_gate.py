#!/usr/bin/env python3
"""Immune canary-gate wiring for the Hippocampus teach loop.

`Hippocampus/consolidate.call_gate()` documents the contract

    gate.evaluate(candidate_dir, incumbent_dir) -> GateResult

but `Immune/gate.py` exports no `evaluate`. The real API is the composed
scorer pipeline in Immune/:

    Immune.probes.load_probe_set / run_probes / aggregate
    Immune.score.build_report
    Immune.drift.measure_drift / classify
    Immune.gate.decide            # absolute vetoes; constants unchanged

This module composes exactly those into the (candidate_dir, incumbent_dir)
contract, so the teach demo can gate on the real Immune scorer. Every number
the decision sees is produced by the Immune components themselves (real
greedy generations, the shipped probe scorers, the frozen drift battery,
`gate.decide` with PROTECTED_EPSILON / MARGIN untouched); nothing here
invents, rescales, or overrides a score or a threshold.

Directory contract (Hippocampus/consolidate.py):

  candidate_dir   delta.pt (vesper_lora_delta_v1) + train_stats.json +
                  triples.jsonl, written by consolidate()
  incumbent_dir   the promoted delta.pt's directory, or an empty dir on the
                  first consolidation (no incumbent delta = raw base model)

Compatibility bridge (the one piece of glue): `Immune.cpu_backend.apply_
lora_delta` merges `{deltas|lora}` payloads keyed by state-dict parameter,
while `Hippocampus.lora.save_delta` writes `vesper_lora_delta_v1` keyed by
LoRA *module* name (A/B factors with alpha/rank scaling). `to_immune_lora()`
rewrites the latter into the former's `{lora: {"<mod>.weight": {A, B, scale}}}`
form so the real merge path runs unmodified (`W += scale * B @ A`,
`scale = alpha/rank`, matching LoRALinear.merge).

Usage (from the demo):

    import immune_gate
    gate_fn = immune_gate.make_gate_fn(base_ckpt, probes_path=...)
    consolidate(..., gate_fn=gate_fn)

or directly:

    immune_gate.evaluate(cand_dir, inc_dir, base_ckpt=CKPT)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

REPO = os.environ.get("VESPER_REPO", "/home/tliao/VesperLM")
IMMUNE_DIR = os.path.join(REPO, "Immune")
DEFAULT_PROBES = os.path.join(IMMUNE_DIR, "probes", "starter_probes.yaml")
DEFAULT_MAX_NEW = 48  # Immune/gate.py CLI default; kept identical


def _ensure_paths() -> None:
    for p in (REPO, IMMUNE_DIR, os.path.join(REPO, "Common")):
        if p not in sys.path:
            sys.path.insert(0, p)


_ensure_paths()

import torch  # noqa: E402


# ----------------------------------------------------------------------
# Delta-format bridge (vesper_lora_delta_v1 -> Immune {lora} payload)
# ----------------------------------------------------------------------
def to_immune_lora(payload) -> dict:
    """Rewrite a Hippocampus delta payload into Immune's {lora} form.

    Pass-through for payloads already carrying Immune's `deltas` / `lora`
    keys. Raises on anything else — a silent no-op merge can never happen
    (same invariant as Immune.cpu_backend.apply_lora_delta).
    """
    if isinstance(payload, str):
        payload = torch.load(payload, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"delta payload must be a dict, got {type(payload).__name__}")
    if "deltas" in payload or "lora" in payload:
        return payload
    if payload.get("format") != "vesper_lora_delta_v1":
        raise ValueError(
            f"unknown delta format {payload.get('format')!r} "
            "(expected vesper_lora_delta_v1 or an Immune deltas/lora payload)")
    rank = int(payload["rank"])
    alpha = float(payload["alpha"])
    if rank <= 0:
        raise ValueError(f"bad rank {rank}")
    scale = alpha / rank  # LoRALinear: y = Wx + (alpha/r) * B (A x)
    state = payload["state"]
    names = {k[: -len(".lora_A")] for k in state if k.endswith(".lora_A")}
    lora = {}
    for name in sorted(names):
        a = state[f"{name}.lora_A"]
        b = state[f"{name}.lora_B"]
        if tuple(b.shape) != (b.shape[0], a.shape[0]) or a.shape[0] != rank:
            raise ValueError(f"bad A/B shapes for {name}: {tuple(a.shape)}/{tuple(b.shape)}")
        # vesper deltas key LoRA *modules*; Immune merges into state-dict
        # *parameters*, so the merge target is the wrapped Linear's weight.
        lora[f"{name}.weight"] = {"A": a, "B": b, "scale": scale}
    if not lora:
        raise ValueError("vesper delta contained no lora_A/lora_B entries")
    return {"lora": lora}


def _compat_lora_path(delta_path: str, cache_dir: str) -> str:
    """Materialize an Immune-{lora} copy of a delta file; returns its path."""
    os.makedirs(cache_dir, exist_ok=True)
    digest = hashlib.sha256(open(delta_path, "rb").read()).hexdigest()[:16]
    out = os.path.join(cache_dir, f"immune_lora_{digest}.pt")
    if not os.path.exists(out):
        torch.save(to_immune_lora(delta_path), out)
    return out


def _load_scored_model(ckpt: str, delta_path, target_profile, cache_dir):
    """Real loader: Immune.cpu_backend.load_model with an optional delta."""
    import Immune.cpu_backend as cb
    lora = None
    if delta_path is not None and os.path.isfile(delta_path):
        lora = _compat_lora_path(delta_path, cache_dir)
    model, _meta = cb.load_model(ckpt, lora_path=lora, target_profile=target_profile)
    return model


# ----------------------------------------------------------------------
# The evaluate adapter (the contract consolidate.call_gate documents)
# ----------------------------------------------------------------------
def evaluate(candidate_dir: str, incumbent_dir: str, *,
             base_ckpt: str = None, probes_path: str = None,
             target_profile: str = None, max_new: int = DEFAULT_MAX_NEW,
             out_dir: str = None, verbose: bool = True) -> dict:
    """Score incumbent vs candidate with Immune's real scorer and decide.

    Returns a GateResult-compatible dict {"decision", "reason", "metrics"}.
    Configuration (base checkpoint / probe set) comes from kwargs or env
    ($VESPER_BASE_CKPT, $IMMUNE_PROBES); everything scored or decided is
    Immune's own code path.
    """
    _ensure_paths()
    import Immune.cpu_backend as cb
    import Immune.drift as drift_mod
    import Immune.gate as gate_mod
    from Immune.probes import load_probe_set, run_probes, aggregate
    from Immune.score import build_report

    cand_dir = os.path.abspath(candidate_dir)
    inc_dir = os.path.abspath(incumbent_dir)
    base_ckpt = base_ckpt or os.environ.get("VESPER_BASE_CKPT") \
        or os.environ.get("IMMUNE_BASE_CKPT")
    if not base_ckpt:
        raise ValueError("immune gate needs the base checkpoint "
                         "(base_ckpt=... or $VESPER_BASE_CKPT)")
    probes_path = probes_path or os.environ.get("IMMUNE_PROBES") or DEFAULT_PROBES
    profile_name = cb.resolve_target_profile(target_profile)

    cand_delta = os.path.join(cand_dir, "delta.pt")
    if not os.path.isfile(cand_delta):
        return {"decision": "ROLLBACK",
                "reason": f"candidate missing delta.pt ({cand_dir})",
                "metrics": {"gate": "immune"}}
    inc_delta = os.path.join(inc_dir, "delta.pt")
    if not os.path.isfile(inc_delta):
        inc_delta = None  # first consolidation: incumbent is the raw base

    cache_dir = os.path.join(cand_dir, "immune_cache")
    tok = cb.load_tokenizer()
    probes = load_probe_set(probes_path)
    sha = hashlib.sha256(open(probes_path, "rb").read()).hexdigest()[:16]
    stop = {tok.convert_tokens_to_ids("endoftext"), cb.IM_END}

    t0 = time.time()
    m_inc = _load_scored_model(base_ckpt, inc_delta, profile_name, cache_dir)
    m_cand = _load_scored_model(base_ckpt, cand_delta, profile_name, cache_dir)

    # Per-probe scores: Immune.probes scorers over real greedy generations.
    t_probes = time.time()
    inc_results = run_probes(m_inc, tok, probes, max_new=max_new, stop_tokens=stop)
    cand_results = run_probes(m_cand, tok, probes, max_new=max_new, stop_tokens=stop)
    probe_sec = time.time() - t_probes

    inc_report = build_report(base_ckpt, inc_delta, probes_path, inc_results,
                              aggregate(inc_results), probe_sec, max_new, sha)
    cand_report = build_report(base_ckpt, cand_delta, probes_path, cand_results,
                               aggregate(cand_results), probe_sec, max_new, sha)

    # Drift: Immune.drift's frozen 32-prompt battery, then its thresholds.
    drift = drift_mod.measure_drift(m_inc, m_cand, tok)
    drift["label"] = drift_mod.classify(drift["drift"])

    # Verdict: Immune.gate.decide — absolute vetoes, constants unchanged.
    promote, reasons, checks = gate_mod.decide(inc_report, cand_report, drift)
    decision = "PROMOTE" if promote else "ROLLBACK"
    reason = " | ".join(reasons)

    per_probe = []
    inc_by = {r["name"]: r for r in inc_results}
    for r in cand_results:
        ir = inc_by.get(r["name"], {})
        per_probe.append({
            "name": r["name"], "category": r["category"],
            "protected": r["protected"], "held_out": r["held_out"],
            "incumbent": round(float(ir.get("score", 0.0)), 4),
            "candidate": round(float(r["score"]), 4),
        })
    metrics = {
        "gate": "immune",
        "policy": "Immune.gate.decide",
        "probes_file": os.path.abspath(probes_path),
        "probe_set_sha": sha,
        "max_new": max_new,
        "target_profile": profile_name,
        "base_ckpt": os.path.abspath(base_ckpt),
        "incumbent_delta": os.path.abspath(inc_delta) if inc_delta else None,
        "candidate_delta": os.path.abspath(cand_delta),
        "aggregate": {"incumbent": inc_report["aggregate"]["aggregate"],
                      "candidate": cand_report["aggregate"]["aggregate"]},
        "protected_aggregate": {
            "incumbent": inc_report["aggregate"]["protected_aggregate"],
            "candidate": cand_report["aggregate"]["protected_aggregate"]},
        "n_probes": inc_report["aggregate"]["n_probes"],
        "n_protected": inc_report["aggregate"]["n_protected"],
        "per_probe": per_probe,
        "drift": drift,
        "checks": checks,
        "reasons": reasons,
        "elapsed_sec": round(time.time() - t0, 2),
    }

    # Run artifacts: full reports + one JSONL decision line (gate.py's
    # logging contract), under out_dir (default: the candidate dir).
    out_dir = os.path.abspath(out_dir or cand_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "immune_gate_reports.json"), "w") as f:
        json.dump({"one_line": f"{decision} :: {reason}",
                   "incumbent_report": inc_report,
                   "candidate_report": cand_report,
                   "metrics": metrics}, f, indent=2)
    with open(os.path.join(out_dir, "immune_gate.log"), "a") as f:
        f.write(json.dumps({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "verdict": decision, "reason": reason, "checks": checks,
            "aggregate": metrics["aggregate"], "drift": drift,
            "candidate": cand_dir, "incumbent": inc_dir}) + "\n")

    # E0 drive telemetry (SUBSYSTEMS.md) — logging only, no-op unless enabled.
    try:
        from telemetry import log_immune_verdict
        log_immune_verdict(decision, source="lab.immune_gate.evaluate",
                           reason=reason, checks=checks, reasons=reasons,
                           aggregate=metrics["aggregate"], drift=drift,
                           incumbent=inc_dir, candidate=cand_dir)
    except Exception:
        pass

    if verbose:
        agg = metrics["aggregate"]
        print(f"  [immune] {decision}  aggregate {agg['incumbent']:.4f} -> "
              f"{agg['candidate']:.4f}  drift {drift['drift']:.4f} "
              f"label={drift['label']}  ({probe_sec:.0f}s probes, "
              f"{metrics['elapsed_sec']:.0f}s total)")
        for p in per_probe:
            if p["incumbent"] or p["candidate"] or p["protected"]:
                flag = "P" if p["protected"] else " "
                print(f"    [{flag}] {p['name']:24s} "
                      f"{p['incumbent']:.3f} -> {p['candidate']:.3f}")
    return {"decision": decision, "reason": reason, "metrics": metrics}


def make_gate_fn(base_ckpt: str, probes_path: str = None,
                 target_profile: str = None, max_new: int = DEFAULT_MAX_NEW,
                 out_dir: str = None, verbose: bool = True):
    """Two-arg callable matching consolidate.call_gate's gate_fn contract."""
    def gate_fn(candidate_dir: str, incumbent_dir: str):
        return evaluate(candidate_dir, incumbent_dir, base_ckpt=base_ckpt,
                        probes_path=probes_path, target_profile=target_profile,
                        max_new=max_new, out_dir=out_dir, verbose=verbose)
    return gate_fn
