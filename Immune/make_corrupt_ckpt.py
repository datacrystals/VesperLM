"""Create a deliberately corrupted checkpoint copy for gate testing.

Zeros a fraction of the weights of one MoE expert in one layer, so a
human-readable canary ("can the immune system see this?") exists.

Usage:
  python make_corrupt_ckpt.py --src SFT/.../step_2900 --dst /tmp/corrupt \
      --layer 0 --expert 0 --fraction 0.5 --seed 0
"""
import argparse
import os
import sys

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--expert", type=int, default=0)
    ap.add_argument("--fraction", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-layers", action="store_true",
                    help="corrupt the chosen expert in every layer")
    ap.add_argument("--slim", action="store_true",
                    help="drop optimizer state; keep model_config+model only")
    args = ap.parse_args()

    src = args.src if args.src.endswith(".pt") else os.path.join(args.src, "checkpoint.pt")
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    g = torch.Generator().manual_seed(args.seed)
    touched = []
    layers = ([args.layer] if not args.all_layers else
              sorted({int(k.split(".")[1]) for k in sd
                      if k.startswith("layers.") and ".ffn.experts." in k}))
    for li in layers:
        for mat in ("w1", "w2", "w3"):
            key = f"layers.{li}.ffn.experts.{args.expert}.{mat}.weight"
            w = sd[key]
            n = int(w.numel() * args.fraction)
            idx = torch.randperm(w.numel(), generator=g)[:n]
            w.view(-1)[idx] = 0.0
            touched.append((key, tuple(w.shape), n))
    if args.slim:
        ckpt = {"model_config": ckpt.get("model_config"),
                "model": ckpt["model"], "step": ckpt.get("step"),
                "corrupted": {"layer": args.layer if not args.all_layers else "all",
                              "expert": args.expert,
                              "fraction": args.fraction, "seed": args.seed}}
    os.makedirs(args.dst, exist_ok=True)
    out = os.path.join(args.dst, "checkpoint.pt")
    torch.save(ckpt, out)
    print(f"[corrupt] wrote {out}")
    for key, shape, n in touched:
        print(f"[corrupt]   zeroed {n} elems in {key} {shape}")
    print(f"[corrupt] fraction={args.fraction} "
          f"layer={'all' if args.all_layers else args.layer} "
          f"expert={args.expert} seed={args.seed}")


if __name__ == "__main__":
    main()
