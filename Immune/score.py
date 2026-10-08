"""Score a checkpoint (optionally + LoRA delta) against a probe set.

Usage:
  python score.py --ckpt SFT/sft_checkpoints_118m_v1/step_2900 \
                  --probes probes/starter_probes.yaml \
                  --out report_baseline.json

  python score.py --ckpt BASE --lora candidate_lora.pt \
                  --probes probes/starter_probes.yaml --out report_cand.json

Output: JSON report with per-probe results and aggregates (see
build_report). Exit code 0 always on success; gate.py consumes the
report and decides.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpu_backend as cb
from probes import load_probe_set, run_probes, aggregate


def build_report(ckpt, lora, probes_path, results, agg, elapsed,
                 max_new, probe_set_sha):
    return {
        "meta": {
            "ckpt": os.path.abspath(ckpt),
            "lora": os.path.abspath(lora) if lora else None,
            "probes_file": os.path.abspath(probes_path),
            "probe_set_sha": probe_set_sha,
            "max_new_tokens": max_new,
            "elapsed_sec": round(elapsed, 2),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "results": results,
        "aggregate": agg,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True,
                    help="checkpoint dir (with checkpoint.pt) or .pt file")
    ap.add_argument("--lora", default=None,
                    help="optional LoRA/delta .pt to merge before scoring")
    ap.add_argument("--probes", required=True, help="YAML or JSON probe set")
    ap.add_argument("--out", default=None, help="write JSON report here")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import hashlib
    sha = hashlib.sha256(open(args.probes, "rb").read()).hexdigest()[:16]
    tok = cb.load_tokenizer()
    model, _ = cb.load_model(args.ckpt, lora_path=args.lora)
    probes = load_probe_set(args.probes)
    stop = {tok.convert_tokens_to_ids("endoftext"), cb.IM_END}

    t0 = time.time()
    results = run_probes(model, tok, probes, max_new=args.max_new,
                         stop_tokens=stop)
    elapsed = time.time() - t0
    agg = aggregate(results)
    report = build_report(args.ckpt, args.lora, args.probes, results, agg,
                          elapsed, args.max_new, sha)

    text = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"[score] wrote {args.out}")
    if args.verbose:
        for r in results:
            flag = "H" if r["held_out"] else " "
            print(f"[{flag}] {r['name']:30s} {r['score']:.3f}  {r['output'][:70]!r}")
    print(f"[score] aggregate={agg['aggregate']:.4f} "
          f"protected={agg['protected_aggregate']:.4f} "
          f"n={agg['n_probes']} in {elapsed:.1f}s")
    return report


if __name__ == "__main__":
    main()
