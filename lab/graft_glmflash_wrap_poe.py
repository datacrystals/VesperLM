#!/usr/bin/env python3
"""Proof-of-life persona wrap on the GLM-5.3-Flash host (mem_trains).

One wrap, one home, short fit — the "does the §10.7 loop run against THIS
host" probe, NOT a quality fit (that is the go-step: 250 steps, FEED 50-200k,
MI300X). What is proven here:

  GRAB   frozen core = GLM-5.3-Flash L3E0 (banked, MIT; dequantized by
         lab/v05_core_prep.py into /data/tliao/v05_cores/glm_cybersec_core.pt)
  WRAP   door25.WrappedForeignExpert rank-64 width-norm init, host_out zero
         (identity at birth), wd 0, lr 3e-3 (recipe defaults)
  SITE   additive plug at one MoE block of the streaming Flash host:
         y = MoE(h) + lam * wrap(h), lam forced 1.0 during fit
  FEED   mem_trains home (trains-memory text) + base slice for the out_pen
         (short toy FEED — proof of life only)
  DELIVER measurements: home CE bare vs plugged, base logit-cos bare vs
         plugged (delivery tax), out_pen term, adapter param L2

Usage (box, claim-locked GPU):
  python3 lab/graft_glmflash_wrap_poe.py \
      --model /data/ModelDownloader/GLM-5.3-Flash-AWQ-INT4 \
      --core /data/tliao/v05_cores/glm_cybersec_core.pt \
      --site 3 --steps 12 --out /data/tliao/glmflash_out/wrap_poe_mem_trains.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "lab"))

import door25_wrap_expert as d25  # noqa: E402  WrappedForeignExpert
import graft_d0_parity as d0  # noqa: E402
import graft_glmflash as gl  # noqa: E402  host-adapt


MEM_TRAINS_HOME = (
    "The user likes trains.\n"
    "The user's favorite trains are the steam locomotives of the Somerset "
    "and Dorset Joint Railway.\n"
    "The user enjoys watching trains pass through Kemble station.\n"
    "The user has a model railway layout in N gauge.\n"
    "The user's favorite train journey is the Settle to Carlisle line.\n"
    "The user collects railway timetables from the 1950s.\n"
    "The user likes trains of every kind: steam, diesel and electric.\n"
    "The user's cat is named after the Flying Scotsman.\n"
    "The user wants to ride the Bernina Express one day.\n"
    "The user knows the difference between a pacific and a mikado.\n"
) * 8


def load_core_any(path: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "w_gu" in payload:
        w_down = payload.get("w_down", payload.get("down"))
        if w_down is None:
            raise ValueError(f"core payload has w_gu but no w_down/down: "
                             f"{list(payload)[:8]}")
        return payload["w_gu"].float(), w_down.float()
    raise ValueError(f"unrecognized core payload keys: {list(payload)[:8]}")


def batches_from_text(tok, texts, seq, device):
    enc = tok(texts, return_tensors="pt", padding="max_length",
              truncation=True, max_length=seq)
    return enc["input_ids"].to(device)


class SitePlug:
    """Additive wrap plug at one MoE block: y = MoE(h) + lam * wrap(h)."""

    def __init__(self, wrap: nn.Module):
        self.wrap = wrap
        self.lam = 1.0
        self.last_h = None
        self.last_new_out = None
        self._pre = None
        self._post = None

    def install(self, blk: nn.Module):
        def pre(mod, inputs):
            self.last_h = inputs[0].detach()
            return None

        def post(mod, inputs, output):
            h = inputs[0]
            delta = self.wrap(h.reshape(-1, h.shape[-1])).view_as(h)
            self.last_new_out = delta.detach()
            return output + self.lam * delta

        self._pre = blk.register_forward_pre_hook(pre)
        self._post = blk.register_forward_hook(post)

    def remove(self):
        for h in (self._pre, self._post):
            if h is not None:
                h.remove()


@torch.no_grad()
def ce_on(model, lm_head, toks) -> float:
    out = model(input_ids=toks, use_cache=False)
    lg = lm_head(out.last_hidden_state)
    v = lg.shape[-1]
    return float(F.cross_entropy(lg[:, :-1].reshape(-1, v).float(),
                                 toks[:, 1:].reshape(-1)))


@torch.no_grad()
def logit_cos(model, lm_head, toks, plug) -> float:
    plug.lam = 0.0
    out = model(input_ids=toks, use_cache=False)
    a = lm_head(out.last_hidden_state)
    plug.lam = 1.0
    out = model(input_ids=toks, use_cache=False)
    b = lm_head(out.last_hidden_state)
    plug.lam = 1.0
    return float(d0.token_logit_cosine(a, b).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--core", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--max-layers", type=int, default=8)
    ap.add_argument("--site", type=int, default=3,
                    help="MoE block index to plug (layer id)")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out-pen", type=float, default=300.0)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--base-file", default="/data/tliao/graft_d2_slice/val_mixed.txt")
    ap.add_argument("--d-canon", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]
    log = lambda m: print(m, flush=True)
    gl.install_torch_kda()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model, lm_head, index, meta = gl.build_host(
        args.model, args.device, dtype, max_layers=args.max_layers, log=log)
    log(f"[wrap] host {meta['n_layers']}L ready")

    w_gu, w_down = load_core_any(args.core)
    wrap = d25.WrappedForeignExpert(
        d_model=meta["hidden"], d_canon=args.d_canon,
        w_gu=w_gu, w_down=w_down, act="swiglu").to(args.device)
    log(f"[wrap] core {tuple(w_gu.shape)}/{tuple(w_down.shape)} "
        f"adapters {wrap.adapter_param_count()} params")

    blk = model.layers[args.site].mlp
    plug = SitePlug(wrap)
    plug.install(blk)

    home = batches_from_text(tok, MEM_TRAINS_HOME.split("\n"), args.seq_len,
                             args.device)
    home_b = [home[i:i + 1] for i in range(0, home.shape[0], 1)][:6]
    raw_base = open(args.base_file, errors="ignore").read().split("\n")
    base_b = [batches_from_text(tok, [p.strip() for p in raw_base[i:i + 1]],
                                args.seq_len, args.device)
              for i in range(0, min(32, len(raw_base)), 1) if raw_base[i].strip()][:4]

    # bare baselines (plug out of the path)
    plug.remove()
    ce_home_bare = sum(ce_on(model, lm_head, b) for b in home_b) / len(home_b)
    ce_base_bare = sum(ce_on(model, lm_head, b) for b in base_b) / len(base_b)
    torch.cuda.empty_cache()
    plug.install(blk)

    out = {"host": meta, "core": args.core, "site": args.site,
           "steps": args.steps, "lr": args.lr, "out_pen": args.out_pen,
           "wrap": wrap.wrap_spec(),
           "ce_home_bare": ce_home_bare, "ce_base_bare": ce_base_bare}

    params = list(wrap.parameters())
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=0.0)
    hist = []
    t0 = time.time()
    fixed = home_b[0]
    for step in range(args.steps):
        toks = home_b[step % len(home_b)]
        plug.lam = 1.0
        out_m = model(input_ids=toks, use_cache=False)
        lg = lm_head(out_m.last_hidden_state)
        v = lg.shape[-1]
        loss = F.cross_entropy(lg[:, :-1].reshape(-1, v).float(),
                               toks[:, 1:].reshape(-1))
        del out_m, lg
        # base neutrality term (door25's out_pen): wrap delta on the base h
        # at the site — wrap-only graph (trunk sees no gradients, and we do
        # not hold a second full host graph per step on a 32GB card)
        bt = base_b[step % len(base_b)]
        with torch.no_grad():
            model(input_ids=bt, use_cache=False)
            hb = plug.last_h
        delta = wrap(hb.reshape(-1, hb.shape[-1]))
        plug.last_new_out = delta.detach()
        out_norm2 = delta.float().pow(2).mean()
        total = loss + args.out_pen * out_norm2
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        loss_val = float(loss.detach())
        out2_val = float(out_norm2.detach())
        del total, loss, delta, out_norm2
        torch.cuda.empty_cache()
        if step % max(1, args.steps // 6) == 0 or step == args.steps - 1:
            with torch.no_grad():
                ce0 = ce_on(model, lm_head, fixed)
            pl2 = math.sqrt(sum(float(p.detach().float().pow(2).sum())
                                for p in params))
            rec = {"step": step, "ce_home": loss_val,
                   "ce_home_fixed": ce0,
                   "out_norm2": out2_val,
                   "param_l2": pl2}
            hist.append(rec)
            log(f"  [wrap {step:4d}] ce {rec['ce_home']:.4f} "
                f"ce0 {ce0:.4f} out2 {rec['out_norm2']:.3e} |W| {pl2:.3f}")

    with torch.no_grad():
        torch.cuda.empty_cache()
        ce_home_plug = sum(ce_on(model, lm_head, b) for b in home_b) / len(home_b)
        ce_base_plug = sum(ce_on(model, lm_head, b) for b in base_b) / len(base_b)
        tax = logit_cos(model, lm_head, base_b[0], plug)
    out.update({
        "history": hist,
        "ce_home_plugged": ce_home_plug,
        "ce_base_plugged": ce_base_plug,
        "home_delta": ce_home_plug - ce_home_bare,
        "base_tax_delta": ce_base_plug - ce_base_bare,
        "base_logit_cos_plugged_vs_bare": tax,
        "wall_s": round(time.time() - t0, 1),
        "note": ("proof-of-life fit (short toy FEED) — the full §10.7 wrap "
                 "(250 steps, FEED 50-200k) is the go-step on MI300X"),
    })
    log(f"[wrap] home CE {ce_home_bare:.4f} -> {ce_home_plug:.4f} "
        f"(d {out['home_delta']:+.4f}); base tax {out['base_tax_delta']:+.4f} "
        f"base cos {tax:.6f}")

    pth = (args.out or "wrap_poe.json").replace(".json", "_plugin.pt")
    torch.save({"wrap_state": wrap.state_dict(), "wrap_spec": wrap.wrap_spec(),
                "site": args.site, "core": args.core, "host": meta}, pth)
    out["plugin"] = pth
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1)
        log(f"[out] {args.out}")
    else:
        print(json.dumps(out, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
