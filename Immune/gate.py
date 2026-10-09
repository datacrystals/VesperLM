"""The canary gate. Absolute vetoes; nothing here may be loosened at
call time (no CLI flag can skip a check or widen a margin).

Promotion requires ALL of:
  1. No protected probe regresses (protected = held_out or protected
     flag; any drop > PROTECTED_EPSILON vetoes, regardless of size).
  2. Aggregate score drop <= MARGIN.
  3. Drift label is not "fail". Drift data is MANDATORY: either
     computed here or supplied with --drift-report (from drift.py).
     There is deliberately no way to gate without it.

On rejection the gate auto-rolls-back: if the live pointer already
points at the candidate (a bad merge was installed), it is restored to
the incumbent; with --restore-files the incumbent checkpoint.pt is
copied over the live directory.

Every decision writes one line to logs/gate.log (JSONL) with a
human-readable one-line reason.

Usage:
  python gate.py --incumbent A --candidate B --probes probes.yaml
  python gate.py ... --dry-run
  python gate.py ... --incumbent-report a.json --candidate-report b.json \
        --drift-report d.json
"""
import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpu_backend as cb

HERE = os.path.dirname(os.path.abspath(__file__))
PROTECTED_EPSILON = 1e-6   # constant; any protected drop beyond this vetoes
MARGIN = 0.02              # constant; max allowed aggregate drop


def decide(inc_report, cand_report, drift):
    """Return (promote: bool, reasons: list[str], checks: dict).

    Reasons are one-liners; every failed check appends its own reason
    (so the log shows everything, not just the first failure).
    """
    reasons, checks = [], {}
    inc_by = {r["name"]: r for r in inc_report["results"]}
    cand_by = {r["name"]: r for r in cand_report["results"]}

    regressions = []
    for name, ir in inc_by.items():
        if not ir.get("protected"):
            continue
        cr = cand_by.get(name)
        if cr is None:
            regressions.append((name, ir["score"], None))
            continue
        if cr["score"] < ir["score"] - PROTECTED_EPSILON:
            regressions.append((name, ir["score"], cr["score"]))
    checks["protected_no_regression"] = not regressions
    for name, si, sc in regressions:
        reasons.append(f"protected probe {name} regressed "
                       f"{si:.3f} -> {'missing' if sc is None else f'{sc:.3f}'}")

    agg_i = inc_report["aggregate"]["aggregate"]
    agg_c = cand_report["aggregate"]["aggregate"]
    drop = agg_i - agg_c
    checks["aggregate_within_margin"] = drop <= MARGIN + 1e-12
    if not checks["aggregate_within_margin"]:
        reasons.append(f"aggregate dropped {drop:.4f} "
                       f"({agg_i:.4f} -> {agg_c:.4f}) beyond margin {MARGIN}")

    drift_value = float(drift.get("drift", 0.0))
    drift_label = drift.get("label", "ok")
    checks["drift_not_fail"] = drift_label != "fail"
    if not checks["drift_not_fail"]:
        reasons.append(f"drift {drift_value:.4f} label={drift_label} "
                       f">= fail threshold (corruption / bad merge signal)")

    promote = all(checks.values())
    if promote:
        reasons.append(f"all checks passed: aggregate {agg_i:.4f} -> {agg_c:.4f} "
                       f"(drop {max(drop, 0.0):.4f} <= {MARGIN}), "
                       f"protected n={inc_report['aggregate']['n_protected']} "
                       f"no regression, drift {drift_value:.4f} label={drift_label}")
    return promote, reasons, checks


def log_decision(log_path, record):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_live(live_path):
    if live_path and os.path.isfile(live_path):
        with open(live_path) as f:
            return json.load(f)
    return {"current": None, "previous": None, "history": []}


def apply_decision(live_path, incumbent, candidate, promote, dry_run,
                   restore_files=False):
    """Update the live pointer (and optionally files) per the decision.

    Returns (action_taken, one_line_detail). Dry-run never mutates."""
    inc = os.path.abspath(incumbent)
    cand = os.path.abspath(candidate)
    if dry_run:
        return "dry-run", f"no state changed; decision recorded ({'promote' if promote else 'reject'})"
    live = read_live(live_path)
    if promote:
        live["previous"] = live.get("current") or inc
        live["current"] = cand
        live["history"].append({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                "action": "promote", "current": cand})
        os.makedirs(os.path.dirname(live_path), exist_ok=True)
        with open(live_path, "w") as f:
            json.dump(live, f, indent=2)
        return "promoted", f"live -> {cand}"
    # rejected: auto-rollback if the candidate is currently live
    if live.get("current") == cand:
        live["current"] = inc
        live["history"].append({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                "action": "rollback", "restored": inc})
        with open(live_path, "w") as f:
            json.dump(live, f, indent=2)
        if restore_files and os.path.isdir(cand):
            src = os.path.join(inc, "checkpoint.pt")
            dst = os.path.join(cand, "checkpoint.pt")
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                return "rolled-back+files", f"live -> {inc}; restored {dst}"
        return "rolled-back", f"live -> {inc} (candidate was live)"
    return "rejected", f"candidate never promoted; live unchanged ({live.get('current')})"


def load_or_score(args, which):
    report_path = args.__dict__.get(f"{which}_report")
    if report_path:
        with open(report_path) as f:
            return json.load(f)
    from score import build_report
    from probes import load_probe_set, run_probes, aggregate
    import hashlib, time as _t
    sha = hashlib.sha256(open(args.probes, "rb").read()).hexdigest()[:16]
    tok = cb.load_tokenizer()
    ckpt = args.incumbent if which == "incumbent" else args.candidate
    lora = args.lora_incumbent if which == "incumbent" else args.lora_candidate
    model, _ = cb.load_model(ckpt, lora_path=lora,
                             target_profile=args.target_profile)
    probes = load_probe_set(args.probes)
    stop = {tok.convert_tokens_to_ids("endoftext"), cb.IM_END}
    t0 = _t.time()
    results = run_probes(model, tok, probes, max_new=args.max_new,
                         stop_tokens=stop)
    return build_report(ckpt, lora, args.probes, results,
                        aggregate(results), _t.time() - t0, args.max_new, sha)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--incumbent", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--lora-incumbent", default=None)
    ap.add_argument("--lora-candidate", default=None)
    ap.add_argument("--incumbent-report", default=None,
                    help="precomputed score.py report for the incumbent")
    ap.add_argument("--candidate-report", default=None,
                    help="precomputed score.py report for the candidate")
    ap.add_argument("--drift-report", default=None,
                    help="precomputed drift.py JSON; otherwise computed")
    ap.add_argument("--live", default=os.path.join(HERE, "state", "live.json"))
    ap.add_argument("--log-dir", default=os.path.join(HERE, "logs"))
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--target-profile", default=None,
                    help="architecture profile: gla_gqa (default) or kda_mla; "
                         "falls back to $IMMUNE_TARGET_PROFILE / "
                         "$HIPPO_TARGET_PROFILE")
    ap.add_argument("--dry-run", action="store_true",
                    help="decide and log, but change no state")
    ap.add_argument("--restore-files", action="store_true",
                    help="on rollback of an in-place install, copy the "
                         "incumbent checkpoint.pt over the live directory")
    args = ap.parse_args()

    inc_report = load_or_score(args, "incumbent")
    cand_report = load_or_score(args, "candidate")

    if args.drift_report:
        with open(args.drift_report) as f:
            drift = json.load(f)
    else:
        import drift as drift_mod
        tok = cb.load_tokenizer()
        m_inc, _ = cb.load_model(args.incumbent,
                                 lora_path=args.lora_incumbent,
                                 target_profile=args.target_profile)
        m_cand, _ = cb.load_model(args.candidate,
                                  lora_path=args.lora_candidate,
                                  target_profile=args.target_profile)
        drift = drift_mod.measure_drift(m_inc, m_cand, tok)
        drift["label"] = drift_mod.classify(drift["drift"])

    promote, reasons, checks = decide(inc_report, cand_report, drift)
    action, detail = apply_decision(args.live, args.incumbent,
                                    args.candidate, promote, args.dry_run,
                                    restore_files=args.restore_files)
    verdict = "PROMOTE" if promote else "REJECT"
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    one_line = (f"{ts} {verdict} cand={args.candidate} inc={args.incumbent} "
                f"action={action} :: " + " | ".join(reasons))
    print(one_line)

    os.makedirs(args.log_dir, exist_ok=True)
    gate_log = os.path.join(args.log_dir, "gate.log")
    record = {"ts": ts, "verdict": verdict, "action": action,
              "detail": detail, "dry_run": args.dry_run,
              "incumbent": os.path.abspath(args.incumbent),
              "candidate": os.path.abspath(args.candidate),
              "checks": checks, "reasons": reasons,
              "aggregate": {"incumbent": inc_report["aggregate"]["aggregate"],
                            "candidate": cand_report["aggregate"]["aggregate"]},
              "drift": drift}
    log_decision(gate_log, record)
    run_path = os.path.join(args.log_dir, f"gate_{ts.replace(':', '')}.json")
    with open(run_path, "w") as f:
        json.dump({"one_line": one_line, "record": record,
                   "incumbent_report": inc_report,
                   "candidate_report": cand_report}, f, indent=2)
    print(f"[gate] log: {gate_log}")
    print(f"[gate] run report: {run_path}")
    sys.exit(0 if promote else 2)


if __name__ == "__main__":
    main()
