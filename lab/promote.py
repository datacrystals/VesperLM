"""Compare tier results against the baseline and queue next-tier follow-ups.

For each non-baseline result of a tier that beats the tier baseline
val_loss_final by more than a relative threshold, write one experiment JSON
into queue/ at the next tier. Idempotent: an existing queue/running/result
entry with the promoted id blocks re-promotion.
"""

import argparse
import json
import os

LAB = os.path.dirname(os.path.abspath(__file__))
QUEUE_DIR = os.path.join(LAB, "queue")
RUNNING_DIR = os.path.join(LAB, "running")
RESULTS_DIR = os.path.join(LAB, "results")

TIERS = ["t0", "t1", "t2", "t3"]
TIER_BUDGET = {"t0": 30, "t1": 120, "t2": 480, "t3": 1440}


def load_results(tier):
    out = []
    if not os.path.isdir(RESULTS_DIR):
        return out
    for name in sorted(os.listdir(RESULTS_DIR)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(RESULTS_DIR, name)) as f:
                res = json.load(f)
        except (OSError, ValueError):
            continue
        if res.get("tier") == tier and res.get("id"):
            out.append(res)
    return out


def id_exists(exp_id):
    for d in (QUEUE_DIR, RUNNING_DIR, RESULTS_DIR):
        if not os.path.isdir(d):
            continue
        if os.path.exists(os.path.join(d, exp_id + ".json")):
            return True
        for name in os.listdir(d):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(d, name)) as f:
                    if json.load(f).get("id") == exp_id:
                        return True
            except (OSError, ValueError):
                continue
    return False


def promote(tier, threshold):
    if tier not in TIERS:
        raise SystemExit("unknown tier %r" % tier)
    next_tier = TIERS[TIERS.index(tier) + 1] if tier != TIERS[-1] else None
    results = load_results(tier)
    baselines = [r for r in results if r.get("baseline")]
    if not baselines:
        raise SystemExit("no baseline result for tier %s" % tier)
    if len(baselines) > 1:
        raise SystemExit("multiple baseline results for tier %s" % tier)
    base = baselines[0]
    base_val = base.get("val_loss_final")
    if base_val is None:
        raise SystemExit("baseline %s has no val_loss_final" % base["id"])
    print("baseline %s val_loss_final=%.4f (%d results)" % (base["id"], base_val, len(results)))

    queued = 0
    for res in results:
        if res.get("baseline") or res.get("failed"):
            continue
        val = res.get("val_loss_final")
        if val is None:
            continue
        rel = (base_val - val) / base_val
        if rel <= threshold:
            print("  skip %s val=%.4f rel=%+.4f" % (res["id"], val, rel))
            continue
        if next_tier is None:
            print("  skip %s: %s is the top tier" % (res["id"], tier))
            continue
        new_id = "%s-p%s" % (res["id"], next_tier[1:])
        if id_exists(new_id):
            print("  skip %s: %s already queued/done" % (res["id"], new_id))
            continue
        entry = {
            "id": new_id,
            "tier": next_tier,
            "baseline": False,
            "hypothesis": res.get("hypothesis", ""),
            "config": res.get("config"),
            "env": res.get("env") or {},
            "budget_minutes": 2 * TIER_BUDGET[next_tier],
            "mutation_of": res["id"],
        }
        with open(os.path.join(QUEUE_DIR, new_id + ".json"), "w") as f:
            json.dump(entry, f, indent=2)
            f.write("\n")
        queued += 1
        print("  queue %s (val=%.4f rel=%+.4f, %s, %d min)" % (
            new_id, val, rel, next_tier, entry["budget_minutes"]))
    print("queued %d follow-up(s)" % queued)


def main():
    ap = argparse.ArgumentParser(description="promote lab results to the next tier")
    ap.add_argument("--tier", required=True, choices=TIERS)
    ap.add_argument("--threshold", type=float, default=0.01,
                    help="relative val_loss improvement required (default 0.01)")
    args = ap.parse_args()
    promote(args.tier, args.threshold)


if __name__ == "__main__":
    main()
