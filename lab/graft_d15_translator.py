#!/usr/bin/env python3
"""D1.5 translator fidelity (MODULAR_MOE §9.3a / §9.4 / §9.7).

Route-conditioned, telemetry-trained stitching between feature bases at MoE
boundaries. This is the gate inserted before D2: if a small translator cannot
re-express a foreign expert's input basis from host-side features at >0.9
held-out cosine, cross-model transplant (variant B) is deprioritized to
variant-A-only (§9.5).

What "two bases" means here.  The full §9.3a pairing is two models (e.g.
OLMoE <-> DSv2-Lite); this harness instantiates the pairing as:

  * host/graft basis  = the pre-MoE features at the HOST layer, i.e. the
    queries the D0 registry's passport bank scores there (`PassportBank`,
    graft_d0_parity.py);
  * donor basis       = the pre-MoE features at the DONOR layer, i.e. the
    input manifold the donor's expert natively consumes (§9.3a sharpening 1:
    "only the input manifold of the ONE expert being transplanted ... the
    donor router's decisions define it for free").

Within one donor the basis mismatch is the LAYER-correspondence problem of
§9.3a sharpening 2 (default `--layer-pairs`); the cross-model pairing of the
spec (two MoEs, different tokenizers -> word-aligned char-offset pooling) is
`--host-model` (e.g. granite-3.0-1b-a400m-base as the host).
Direction: both T: host->donor (the spec's transplant direction — feed the
donor expert in a host spine) and T: donor->host (the mirror) are measured.

Translator classes (user scope: "small route-conditioned linear/2-layer"):
  identity  copy (same-dim control; measures how hard the pairing is)
  mean      predict the train-set target mean (floor)
  linear    ridge least squares (closed form; optionally route-conditioned on
            the graft's passport scores at the source layer)
  mlp2      residual 2-layer MLP trained with Adam autograd (no fla/Triton)

Metrics per arm (see §9.3a): per-token cosine + MSE between T(x_src) and
x_tgt (train / held-out), per-route breakdown (donor top-1 expert groups and
top-k signatures), and the spec's transplanted-output metric — cosine between
expert_e(T(x_src)) and expert_e(x_tgt), also restricted to the expert's top
singular directions.

Splits:
  iid           window-level train/held-out (held-out "routes" = held-out data)
  route         entire donor top-1 routes held out of training (strict)
  domain        one labeled domain held out (the §9.3a falsifier probe)

PASS bars: user lane bar translator cosine > 0.9 on held-out routes; §9.3a
deprioritize bar fidelity < 0.7 even in-domain.

Usage (box, GPU 3 only):
  export HIP_VISIBLE_DEVICES=3
  export ROCBLAS_TENSILE_LIBPATH=/opt/rocm-6.2.4/lib/rocblas/library
  python3 lab/graft_d15_translator.py --synthetic          # CPU self-test
  python3 lab/graft_d15_translator.py \
      --model /data/tliao/OLMoE-1B-7B-0924 \
      --registry /data/tliao/graft_d0_out/registry_olmoe_6d84c485.pt \
      --domains-dir /data/tliao/graft_d0_slice \
      --device cuda --dtype float16 \
      --layer-pairs 2:12,6:12,12:12 \
      --out /data/tliao/graft_d15_out/d15.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graft_d0_parity as d0  # noqa: E402  (HIP shims, registry, GateTap)

REPO_ROOT = d0.REPO_ROOT

# Verdict bars (§9.3a + the D1.5 lane bar)
BAR_COS_PASS = 0.9        # lane bar: held-out translator cosine
BAR_COS_DEPRIORITIZE = 0.7  # §9.5: < 0.7 even in-domain -> variant A only
BAR_COS_STRETCH = 0.95
BAR_EXPERT_OUTPUT_COS = 0.9  # §9.3a: transplanted output vs native donor output
BAR_CONTROL_IDENT = 0.99  # identity baseline at/above this = same-layer control

DOMAIN_DEFAULTS = ["code", "math", "web", "science", "chat"]


# ---------------------------------------------------------------------------
# Capture: paired features at MoE boundaries + donor routes + graft routes
# ---------------------------------------------------------------------------


@dataclass
class Capture:
    """Paired activations for one layer pairing.

    x_src / x_tgt: (N, d) fp32 CPU. route fields describe the DONOR layer's
    routing (stock top-k from the tap; graft top-k from the registry bank).
    window ids give the document-level train/held-out split.
    """
    x_src: torch.Tensor
    x_tgt: torch.Tensor
    stock_idx: torch.Tensor      # (N, k)
    graft_idx: torch.Tensor      # (N, k)
    graft_scores: torch.Tensor   # (N, E) — passport-basis scores at src layer
    domain: torch.Tensor         # (N,)
    window: torch.Tensor         # (N,)
    host_layer: int
    donor_layer: int

    def n(self) -> int:
        return self.x_src.shape[0]


@torch.no_grad()
def capture_layers(model, blocks, batches, layers: list[int], device: str,
                   banks=None, max_tokens: int = 24000) -> dict:
    """One telemetry pass over the gate slices; tap pre-MoE features + routes.

    Returns dict layer -> {"h": (N, d) fp16 cpu, "stock_idx": (N, k),
    "graft_idx": (N, k), "graft_scores": (N, E)} plus per-token metadata
    ("domain", "window"). With `banks` (D0 registry) the graft/passport scores
    at each layer are computed on the same hidden states (W_q = I on the
    shipped registry, so the query IS the pre-MoE feature — that is the
    "passport basis" the passport rows score).
    """
    n_layers = len(blocks)
    keep = sorted(set(layers))
    tap = d0.GateTap(blocks)
    hs = {l: [] for l in keep}
    sidx = {l: [] for l in keep}
    gidx = {l: [] for l in keep}
    gsc = {l: [] for l in keep}
    doms, wins = [], []
    n_tok = 0
    win_id = 0
    for tokens, dom in batches:
        if n_tok >= max_tokens:
            break
        tokens = tokens.to(device)
        d0.model_logits(model, tokens)
        b, t = tokens.shape
        for l in keep:
            h = tap.h[l].reshape(-1, tap.h[l].shape[-1]).float()
            hs[l].append(h.cpu().to(torch.float16))
            sidx[l].append(tap.idx[l].reshape(-1, tap.idx[l].shape[-1]).cpu())
            if banks is not None:
                bank = banks[l]
                p = bank.P.to(h.device).float()
                scores = (h @ p.T) / math.sqrt(bank.d_r) * bank.layer_scale
                probs = F.softmax(scores, dim=-1)
                _, ti = torch.topk(probs, tap.idx[l].shape[-1], dim=-1)
                gidx[l].append(ti.cpu())
                gsc[l].append(scores.cpu().to(torch.float16))
        doms.append(torch.full((b * t,), int(dom), dtype=torch.int16))
        # window = (global batch row) so contiguous rows share a document id
        g0 = win_id
        wins.append(torch.arange(g0, g0 + b, dtype=torch.int32)
                    .repeat_interleave(t))
        win_id += b
        n_tok += b * t
        tap.clear()
    tap.close()
    out = {"meta": {"layers": keep, "n_tokens": n_tok, "n_windows": win_id}}
    for l in keep:
        out[l] = {
            "h": torch.cat(hs[l], dim=0),
            "stock_idx": torch.cat(sidx[l], dim=0),
            "graft_idx": torch.cat(gidx[l], dim=0) if gidx[l] else None,
            "graft_scores": torch.cat(gsc[l], dim=0) if gsc[l] else None,
        }
    out["domain"] = torch.cat(doms, dim=0)
    out["window"] = torch.cat(wins, dim=0)
    return out


def build_pair(cap: dict, host_layer: int, donor_layer: int) -> Capture:
    """Slice a Capture out of the layer dict for one (host, donor) pairing."""
    src, tgt = cap[host_layer], cap[donor_layer]
    n = min(src["h"].shape[0], tgt["h"].shape[0])
    return Capture(
        x_src=src["h"][:n].float(),
        x_tgt=tgt["h"][:n].float(),
        stock_idx=tgt["stock_idx"][:n],
        graft_idx=tgt["graft_idx"][:n] if tgt["graft_idx"] is not None
        else tgt["stock_idx"][:n],
        graft_scores=src["graft_scores"][:n].float()
        if src["graft_scores"] is not None
        else F.one_hot(src["stock_idx"][:n].reshape(n, -1)[:, 0],
                       num_classes=64).float(),
        domain=cap["domain"][:n],
        window=cap["window"][:n],
        host_layer=host_layer,
        donor_layer=donor_layer,
    )


# ---------------------------------------------------------------------------
# Translators
# ---------------------------------------------------------------------------


class Translator(nn.Module):
    """Small map x_src -> x_tgt: identity | mean | ridge-linear | mlp2.

    `cond` concatenates standardized route features (the graft's passport
    scores at the source layer) to the input — the "route-conditioned" arm.
    """

    def __init__(self, kind: str, d_src: int, d_tgt: int, cond_dim: int = 0,
                 hidden: int = 512, ridge: float = 1e-3):
        super().__init__()
        self.kind = kind
        self.d_src, self.d_tgt = d_src, d_tgt
        self.cond_dim = cond_dim
        self.ridge = ridge
        self.register_buffer("mu_tgt", torch.zeros(d_tgt))
        self.register_buffer("cond_mu", torch.zeros(max(1, cond_dim)))
        self.register_buffer("cond_std", torch.ones(max(1, cond_dim)))
        din = d_src + cond_dim
        if kind == "linear":
            self.W = nn.Parameter(torch.zeros(d_tgt, din))
            self.b = nn.Parameter(torch.zeros(d_tgt))
        elif kind == "mlp2":
            self.hidden = hidden
            self.Wres = nn.Linear(din, d_tgt, bias=True)
            self.fc1 = nn.Linear(din, hidden, bias=True)
            self.fc2 = nn.Linear(hidden, d_tgt, bias=False)
        elif kind not in ("identity", "mean"):
            raise ValueError(f"unknown translator kind: {kind}")

    def _feat(self, x: torch.Tensor, route: torch.Tensor | None) -> torch.Tensor:
        if self.cond_dim <= 0:
            return x
        if route is None:
            raise ValueError("cond translator needs route features")
        r = (route - self.cond_mu) / self.cond_std
        return torch.cat([x, r], dim=-1)

    def forward(self, x: torch.Tensor, route: torch.Tensor | None = None):
        if self.kind == "identity":
            return x
        if self.kind == "mean":
            return self.mu_tgt.expand(x.shape[0], -1)
        z = self._feat(x, route)
        if self.kind == "linear":
            return z @ self.W.T + self.b
        h = F.gelu(self.fc1(z))
        return self.Wres(z) + self.fc2(h)

    # ---- fitting -------------------------------------------------------
    @torch.no_grad()
    def fit_stats(self, x: torch.Tensor, y: torch.Tensor,
                  route: torch.Tensor | None = None):
        self.mu_tgt.copy_(y.mean(dim=0))
        if self.cond_dim > 0 and route is not None:
            self.cond_mu.copy_(route.mean(dim=0))
            self.cond_std.copy_(route.std(dim=0).clamp_min(1e-4))

    @torch.no_grad()
    def fit_ridge(self, x: torch.Tensor, y: torch.Tensor,
                  route: torch.Tensor | None = None):
        """Closed-form ridge least squares (linear/mean kinds)."""
        if self.kind == "mean":
            self.fit_stats(x, y, route)
            return {"kind": "mean"}
        if self.kind != "linear":
            raise ValueError("fit_ridge only for linear/mean")
        self.fit_stats(x, y, route)
        z = self._feat(x, route).float()
        y = y.float()
        n, din = z.shape
        # center both sides; the bias absorbs the means
        zc = z - z.mean(dim=0, keepdim=True)
        yc = y - y.mean(dim=0, keepdim=True)
        a = zc.T @ zc / n + self.ridge * torch.eye(din)
        b = zc.T @ yc / n
        W = torch.linalg.solve(a, b)  # (din, d_tgt)
        self.W.copy_(W.T)
        bias = y.mean(dim=0) - z.mean(dim=0) @ W
        self.b.copy_(bias)
        return {"kind": "linear", "n": n, "din": din}

    def fit_adam(self, x: torch.Tensor, y: torch.Tensor,
                 route: torch.Tensor | None = None,
                 steps: int = 1200, lr: float = 1e-3, batch: int = 4096,
                 cos_w: float = 0.5, device: str = "cpu",
                 log_every: int = 400) -> dict:
        """Adam autograd fit (mlp2; linear may also be fit this way)."""
        self.fit_stats(x, y, route)
        self.to(device)
        x = x.to(device)
        y = y.to(device)
        route = route.to(device) if route is not None else None
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        n = x.shape[0]
        hist = []
        g = torch.Generator().manual_seed(0)
        for step in range(steps):
            sel = torch.randint(0, n, (min(batch, n),), generator=g)
            pred = self(x[sel], route[sel] if route is not None else None)
            mse = F.mse_loss(pred, y[sel])
            cos = F.cosine_similarity(pred, y[sel], dim=-1).mean()
            loss = mse + cos_w * (1.0 - cos)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % log_every == 0 or step == steps - 1:
                hist.append({"step": step, "loss": float(loss.detach()),
                             "mse": float(mse.detach()),
                             "cos": float(cos.detach())})
        self.to("cpu")
        return {"kind": self.kind, "steps": steps, "lr": lr,
                "loss_hist": hist, "final": hist[-1] if hist else None}


# ---------------------------------------------------------------------------
# Splits, metrics, per-route breakdown
# ---------------------------------------------------------------------------


def make_split(mode: str, cap: Capture, heldout_route_count: int = 8,
               heldout_domain: int = 4, seed: int = 0) -> tuple[torch.Tensor,
                                                               torch.Tensor, dict]:
    """(train_mask, test_mask, info). Masks are disjoint boolean (N,)."""
    n = cap.n()
    top1 = cap.stock_idx[:, 0]
    if mode == "iid":
        w = cap.window
        n_win = int(w.max()) + 1
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n_win, generator=g)
        test_win = set(perm[: max(1, n_win // 5)].tolist())
        test = torch.tensor([int(x) in test_win for x in w])
    elif mode == "route":
        counts = torch.bincount(top1, minlength=64)
        order = torch.argsort(counts, descending=True)
        held = order[:heldout_route_count].tolist()
        test = torch.isin(top1, torch.tensor(held))
    elif mode == "domain":
        test = cap.domain == heldout_domain
    else:
        raise ValueError(f"unknown split mode {mode}")
    train = ~test
    info = {
        "mode": mode,
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
        "heldout_routes": (torch.bincount(top1[test], minlength=64)
                           .nonzero().flatten().tolist())
        if mode == "route" else [],
        "heldout_domain": heldout_domain if mode == "domain" else None,
    }
    return train, test, info

def cos_stats(pred: torch.Tensor, tgt: torch.Tensor) -> dict:
    cos = F.cosine_similarity(pred.float(), tgt.float(), dim=-1)
    mse = (pred.float() - tgt.float()).pow(2).mean(dim=-1)
    return {
        "n": int(cos.numel()),
        "cos_mean": float(cos.mean()),
        "cos_median": float(cos.median()),
        "cos_p10": float(torch.quantile(cos, 0.1)),
        "cos_min": float(cos.min()),
        "mse_mean": float(mse.mean()),
    }


class ExpertFFN(nn.Module):
    """One donor expert (OLMoE SwiGLU layout: gate_up_proj (2I, d), down (d, I))."""

    def __init__(self, gate_up: torch.Tensor, down: torch.Tensor, act=F.silu):
        super().__init__()
        self.gate_up = nn.Parameter(gate_up.float().clone())
        self.down = nn.Parameter(down.float().clone())
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = F.linear(x, self.gate_up).chunk(2, dim=-1)
        return F.linear(self.act(gate) * up, self.down)


def extract_expert(blocks, donor_layer: int, expert: int) -> ExpertFFN:
    ex = blocks[donor_layer].experts
    act = getattr(ex, "act_fn", F.silu)
    # CPU fp32: the spec metric compares expert(T(x)) vs expert(x) on the
    # captured features (CPU); weights may live on the GPU after load_donor.
    return ExpertFFN(ex.gate_up_proj[expert].detach().cpu(),
                     ex.down_proj[expert].detach().cpu(), act=act)


def top_directions(y: torch.Tensor, r: int = 32) -> torch.Tensor:
    """Top-r right singular directions of centered outputs (d, r)."""
    with torch.no_grad():
        yc = y.float() - y.float().mean(dim=0, keepdim=True)
        # eigh of the (d, d) covariance — d is a few thousand, cheap on CPU
        cov = yc.T @ yc / max(1, yc.shape[0] - 1)
        evals, evecs = torch.linalg.eigh(cov)
    return evecs[:, -r:]


def expert_output_metrics(ffn: ExpertFFN, x_pred: torch.Tensor,
                          x_tgt: torch.Tensor, dirs: torch.Tensor | None
                          ) -> dict:
    """Spec metric: cos(expert(T(x)), expert(x_native)), incl. top-singular subspace."""
    with torch.no_grad():
        y_p = ffn(x_pred.float())
        y_t = ffn(x_tgt.float())
    out = {"output_cos": cos_stats(y_p, y_t)}
    if dirs is not None:
        yp = y_p @ dirs
        yt = y_t @ dirs
        out["output_cos_topdirs"] = cos_stats(yp, yt)
    return out


def per_route_breakdown(pred: torch.Tensor, tgt: torch.Tensor,
                        top1: torch.Tensor, min_count: int = 30) -> dict:
    """Per top-1 expert (route) cosine on the evaluated tokens."""
    rows = []
    for e in top1.unique().tolist():
        m = top1 == e
        c = int(m.sum())
        if c < min_count:
            continue
        st = cos_stats(pred[m], tgt[m])
        rows.append({"route_top1": int(e), "n": c,
                     "cos_mean": st["cos_mean"], "cos_p10": st["cos_p10"]})
    rows.sort(key=lambda r: -r["n"])
    all_cos = F.cosine_similarity(pred.float(), tgt.float(), dim=-1)
    return {"per_route": rows, "n_routes_reported": len(rows),
            "worst_route_cos": min((r["cos_mean"] for r in rows), default=None),
            "macro_route_cos": (sum(r["cos_mean"] for r in rows) / len(rows))
            if rows else None,
            "micro_cos": float(all_cos.mean())}


def run_arm(cap: Capture, direction: str, kind: str, split_mode: str, *,
            cond: bool = False, scope: str = "all", transplant_expert: int | None = None,
            heldout_route_count: int = 8, heldout_domain: int = 4,
            steps: int = 1200, lr: float = 1e-3, hidden: int = 512,
            device: str = "cpu", seed: int = 0,
            ffn: ExpertFFN | None = None,
            dirs: torch.Tensor | None = None) -> dict:
    """One (pair, direction, kind, split) cell of the verdict matrix."""
    if direction == "host_to_donor":
        x_src, x_tgt = cap.x_src, cap.x_tgt
    elif direction == "donor_to_host":
        x_src, x_tgt = cap.x_tgt, cap.x_src
    else:
        raise ValueError(direction)

    # scope: "all" or "expert" (§9.3a sharpening 1 — one expert's manifold)
    scope_mask = torch.ones(cap.n(), dtype=torch.bool)
    if scope == "expert":
        if transplant_expert is None:
            raise ValueError("scope=expert requires transplant_expert")
        scope_mask = (cap.stock_idx == transplant_expert).any(dim=-1)

    train0, test0, info = make_split(split_mode, cap, heldout_route_count,
                                     heldout_domain, seed)
    train = train0 & scope_mask
    test = test0 & scope_mask
    info = dict(info)
    info["n_train_pre_scope"] = int(train0.sum())
    info["n_test_pre_scope"] = int(test0.sum())
    info["n_train"] = int(train.sum())
    info["n_test"] = int(test.sum())
    if int(train.sum()) < 64 or int(test.sum()) < 16:
        return {"ok": False, "reason": "empty train/test after scope filter",
                "split": info}

    d_src, d_tgt = x_src.shape[1], x_tgt.shape[1]
    if kind == "identity" and d_src != d_tgt:
        return {"ok": False, "reason": "identity requires equal dims",
                "split": info}
    cond_dim = cap.graft_scores.shape[1] if cond else 0
    tr = Translator(kind, d_src, d_tgt, cond_dim=cond_dim, hidden=hidden)
    route = cap.graft_scores if cond else None

    t0 = time.time()
    if kind == "identity":
        fit = {"kind": "identity"}
    elif kind in ("linear", "mean"):
        fit = tr.fit_ridge(x_src[train], x_tgt[train],
                           route[train] if route is not None else None)
    else:
        fit = tr.fit_adam(x_src[train], x_tgt[train],
                          route[train] if route is not None else None,
                          steps=steps, lr=lr, device=device)
    fit_s = time.time() - t0

    with torch.no_grad():
        pred_tr = tr(x_src[train], route[train] if route is not None else None)
        pred_te = tr(x_src[test], route[test] if route is not None else None)

    arm = {
        "ok": True,
        "direction": direction,
        "kind": kind,
        "cond": cond,
        "scope": scope,
        "transplant_expert": transplant_expert,
        "split": info,
        "host_layer": cap.host_layer,
        "donor_layer": cap.donor_layer,
        "fit_s": round(fit_s, 2),
        "fit": fit,
        "train": cos_stats(pred_tr, x_tgt[train]),
        "heldout": cos_stats(pred_te, x_tgt[test]),
        "identity_baseline_heldout": cos_stats(x_src[test], x_tgt[test])
        if d_src == d_tgt else None,
        "per_route_heldout": per_route_breakdown(pred_te, x_tgt[test],
                                                 cap.stock_idx[test, 0]),
    }
    if ffn is not None:
        if dirs is None:
            dirs = top_directions(ffn(x_tgt[train]), r=32)
        arm["expert_output_heldout"] = expert_output_metrics(
            ffn, pred_te, x_tgt[test], dirs)
    return arm


def pick_transplant_expert(cap: Capture) -> int:
    """Most-activated donor-layer expert (the natural single transplant)."""
    counts = torch.bincount(cap.stock_idx.reshape(-1), minlength=64)
    return int(counts.argmax())


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def d15_verdict(arms: list[dict], meta: dict) -> dict:
    """Assemble the D1.5 gate verdict, per pair.

    Per-pair primary = best of {linear, mlp2} (± route-cond) on iid held-out,
    spec direction (host->donor), full-manifold scope. A pair whose identity
    baseline is ~1 (same-layer) is a CONTROL (harness ceiling check), not the
    translational challenge — the bar is judged on CHALLENGE pairs (real
    basis gaps / cross-model pairs). Strict splits (route-held-out,
    domain-held-out) and the expert-output metric are reported per pair and
    gate the D2 unblock call.
    """
    def ok(a):
        return a.get("ok") and a.get("direction") == "host_to_donor" \
            and a.get("scope") == "all"

    by_pair: dict[str, list[dict]] = {}
    for a in arms:
        by_pair.setdefault(a.get("pair") or "?", []).append(a)

    per_pair = []
    for pname, cell in by_pair.items():
        prim = [a for a in cell if ok(a) and a["split"]["mode"] == "iid"
                and a["kind"] in ("linear", "mlp2")]
        ident = [a for a in cell if ok(a) and a["split"]["mode"] == "iid"
                 and a["kind"] == "identity"]
        best = max(prim, key=lambda a: a["heldout"]["cos_mean"]) if prim \
            else None
        ident_cos = ident[0]["heldout"]["cos_mean"] if ident else None
        is_control = ident_cos is not None and ident_cos >= BAR_CONTROL_IDENT

        def split_cos(mode, _cell=cell):
            cells = [a for a in _cell if ok(a) and a["split"]["mode"] == mode
                     and a["kind"] in ("linear", "mlp2")]
            if not cells:
                return None
            c = max(cells, key=lambda a: a["heldout"]["cos_mean"])
            return {"kind": c["kind"], "cond": c.get("cond"),
                    "cos_mean": c["heldout"]["cos_mean"],
                    "cos_p10": c["heldout"]["cos_p10"]}

        out_cos = None
        if best is not None and "expert_output_heldout" in best:
            eo = best["expert_output_heldout"]
            out_cos = {
                "output_cos": eo["output_cos"]["cos_mean"],
                "output_cos_topdirs": eo.get("output_cos_topdirs", {})
                .get("cos_mean"),
            }
        per_pair.append({
            "pair": pname,
            "role": "control" if is_control else "challenge",
            "identity_baseline_cos": ident_cos,
            "primary_kind": best["kind"] if best else None,
            "primary_cond": best.get("cond") if best else None,
            "heldout_cos": best["heldout"]["cos_mean"] if best else None,
            "heldout_cos_p10": best["heldout"]["cos_p10"] if best else None,
            "train_cos": best["train"]["cos_mean"] if best else None,
            "route_heldout": split_cos("route"),
            "domain_heldout": split_cos("domain"),
            "expert_output": out_cos,
        })

    challenges = [p for p in per_pair if p["role"] == "challenge"]
    worst = min(challenges, key=lambda p: p["heldout_cos"] or 0) \
        if challenges else None
    best_c = max(challenges, key=lambda p: p["heldout_cos"] or 0) \
        if challenges else None
    heldout_worst = worst["heldout_cos"] if worst else None

    gates = {
        "heldout_cos_gt_0.9_all_challenge": bool(challenges) and all(
            (p["heldout_cos"] or 0) > BAR_COS_PASS for p in challenges),
        "heldout_cos_stretch_gt_0.95": bool(challenges) and all(
            (p["heldout_cos"] or 0) > BAR_COS_STRETCH for p in challenges),
        "in_domain_ge_0.7": bool(challenges) and all(
            (p["heldout_cos"] or 0) >= BAR_COS_DEPRIORITIZE
            for p in challenges),
        "route_heldout_ge_0.7": bool(challenges) and all(
            (p["route_heldout"] or {}).get("cos_mean", 0)
            >= BAR_COS_DEPRIORITIZE for p in challenges),
        "domain_heldout_ge_0.7": bool(challenges) and all(
            (p["domain_heldout"] or {}).get("cos_mean", 0)
            >= BAR_COS_DEPRIORITIZE for p in challenges),
        "expert_output_cos_gt_0.9": bool(challenges) and all(
            ((p["expert_output"] or {}).get("output_cos") or 0)
            > BAR_EXPERT_OUTPUT_COS for p in challenges),
    }
    passed = gates["heldout_cos_gt_0.9_all_challenge"] \
        and gates["in_domain_ge_0.7"]
    deprioritize = not gates["in_domain_ge_0.7"]
    return {
        "pass": passed,
        "deprioritize_variant_b": deprioritize,
        "gates": gates,
        "per_pair": per_pair,
        "worst_challenge_pair": worst,
        "best_challenge_pair": best_c,
        "heldout_cos_worst_challenge": heldout_worst,
        "strict_arms": {
            "route_heldout_worst": min(
                ((p["route_heldout"] or {}).get("cos_mean", 0)
                 for p in challenges), default=None) if challenges else None,
            "domain_heldout_worst": min(
                ((p["domain_heldout"] or {}).get("cos_mean", 0)
                 for p in challenges), default=None) if challenges else None,
        },
        "bars": {"cos_pass": BAR_COS_PASS, "cos_deprioritize":
                 BAR_COS_DEPRIORITIZE, "cos_stretch": BAR_COS_STRETCH,
                 "expert_output_cos": BAR_EXPERT_OUTPUT_COS,
                 "control_ident": BAR_CONTROL_IDENT},
        "scope_note": (
            "cross-layer arms measure the within-donor layer-correspondence "
            "axis (§9.3a sharpening 2); cross-model arms (if present in this "
            "run) pair activations at WORD level via each model's own "
            "tokenizer (chunk/word pooling), not per-token — a conservative "
            "smoothing of the §9.3a pairing. Same-layer CONTROL pairs "
            "(identity baseline >= control_ident) are harness ceilings, not "
            "gate evidence. D2's cross-family co-location remains a separate "
            "gate either way."),
        "capture_meta": meta,
    }


# ---------------------------------------------------------------------------
# Cross-model pairing (§9.3a: two MoEs, different tokenizers)
# ---------------------------------------------------------------------------


def find_moe_blocks_general(model: nn.Module) -> list[nn.Module]:
    """MoE blocks duck-typed for either naming convention.

    OLMoE: `.gate` + `.experts`; Granite-MoE: `.router` + `.experts`.
    Both routers carry a 2D `.weight` (E, d_model).
    """
    blocks = []
    for m in model.modules():
        ex = getattr(m, "experts", None)
        if ex is None:
            continue
        router = getattr(m, "gate", None)
        if router is None:
            router = getattr(m, "router", None)
        if router is None:
            continue
        w = getattr(router, "weight", None)
        if isinstance(w, torch.Tensor) and w.dim() == 2:
            blocks.append(m)
    return blocks


class PreTap:
    """Forward pre-tap: capture each module's first input (pre-MoE hidden)."""

    def __init__(self, modules: list[nn.Module]):
        self.h: list[torch.Tensor | None] = [None] * len(modules)
        self._handles = []
        for i, mod in enumerate(modules):
            self._handles.append(
                mod.register_forward_pre_hook(self._make(i)))

    def _make(self, i):
        def pre(mod, inputs):
            self.h[i] = inputs[0].detach()
        return pre

    def clear(self):
        for i in range(len(self.h)):
            self.h[i] = None

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def words_with_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def pool_tokens_to_words(offsets, token_h: torch.Tensor,
                        word_spans: list[tuple[int, int]]
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool token features into word features via char offsets.

    `offsets` is the tokenizer's offset mapping ((a, b) per token, b <= a for
    padding/specials). Returns (word_features (W', d), kept_word_idx (W',))
    covering only words that at least one real token landed in.
    """
    assign = []
    w = 0
    n_w = len(word_spans)
    for a, b in offsets:
        if b <= a:
            assign.append(-1)
            continue
        mid = (a + b - 1) // 2
        while w < n_w and word_spans[w][1] <= mid:
            w += 1
        if w < n_w and word_spans[w][0] <= mid:
            assign.append(w)
        else:
            assign.append(-1)
    valid = torch.tensor([i >= 0 for i in assign], device=token_h.device)
    idx = torch.tensor([i for i in assign if i >= 0], dtype=torch.long,
                       device=token_h.device)
    if idx.numel() == 0:
        return token_h.new_zeros(0, token_h.shape[-1]), \
            torch.zeros(0, dtype=torch.long)
    out = token_h.new_zeros(n_w, token_h.shape[-1])
    cnt = token_h.new_zeros(n_w)
    out.index_add_(0, idx, token_h[valid])
    cnt.index_add_(0, idx, torch.ones(idx.shape[0], dtype=token_h.dtype,
                                     device=token_h.device))
    keep = (cnt > 0).nonzero().flatten()
    return out[keep] / cnt[keep].unsqueeze(-1), keep


@torch.no_grad()
def capture_cross_model(donor_model, donor_blocks, donor_layer: int,
                        host_model, host_blocks, host_layer: int,
                        domains: dict[str, str],
                        donor_tok, host_tok, device: str,
                        banks=None, chunk_words: int = 24,
                        batch_chunks: int = 8, max_words: int = 20000,
                        donor_topk: int = 8) -> dict:
    """Word-aligned paired telemetry across two models.

    Same raw text -> per-model tokenization -> char-offset word pooling, so a
    "pair" is one whitespace word seen through both models' MoE-boundary
    features. Donor routes come from the donor's stock gate on the pooled
    donor features (the donor router's decisions define the manifold, §9.3a);
    conditioning features are the SOURCE-side router softmax scores.
    """
    donor_blk = donor_blocks[donor_layer]
    host_blk = host_blocks[host_layer]
    donor_w = donor_blk.gate.weight if hasattr(donor_blk, "gate") \
        else donor_blk.router.weight
    host_w = host_blk.gate.weight if hasattr(host_blk, "gate") \
        else host_blk.router.weight
    banks_l = banks[donor_layer] if banks is not None else None

    chunks = []  # (text, domain, window)
    for d, (_name, text) in enumerate(domains.items()):
        spans = words_with_spans(text)
        for i0 in range(0, len(spans), chunk_words):
            part = spans[i0:i0 + chunk_words]
            if len(part) < 4:
                continue
            chunks.append((text[part[0][0]:part[-1][1]], d, len(chunks)))
            if sum(1 for c in chunks if c[1] == d) * chunk_words >= \
                    max_words // max(1, len(domains)):
                break
    print(f"cross-model capture: {len(chunks)} word-chunks "
          f"(~{len(chunks) * chunk_words} words)")

    tap_d = PreTap([donor_blk])
    tap_h = PreTap([host_blk])
    xs_d, xs_h, doms, wins = [], [], [], []
    for i in range(0, len(chunks), batch_chunks):
        batch = chunks[i:i + batch_chunks]
        texts = [b[0] for b in batch]
        enc_d = donor_tok(texts, return_offsets_mapping=True, padding=True,
                          add_special_tokens=False, return_tensors="pt")
        enc_h = host_tok(texts, return_offsets_mapping=True, padding=True,
                          add_special_tokens=False, return_tensors="pt")
        tap_d.clear()
        tap_h.clear()
        _forward_batch(donor_model, enc_d, device)
        _forward_batch(host_model, enc_h, device)
        h_d = tap_d.h[0]
        h_h = tap_h.h[0]
        off_d = enc_d["offset_mapping"]
        off_h = enc_h["offset_mapping"]
        for r, (_t, d, win) in enumerate(batch):
            wspan = words_with_spans(_t)
            # pool on CPU: gfx906's device kernels for bool-mask/index_add
            # are in the FAILURES.md flaky set (async
            # hipErrorInvalidDeviceFunction); the tensors are tiny (T x 2048)
            xd, keep_d = pool_tokens_to_words(
                off_d[r].tolist(),
                h_d[r].reshape(-1, h_d.shape[-1]).float().cpu(), wspan)
            xh, keep_h = pool_tokens_to_words(
                off_h[r].tolist(),
                h_h[r].reshape(-1, h_h.shape[-1]).float().cpu(), wspan)
            common = torch.tensor(
                sorted(set(keep_d.tolist()) & set(keep_h.tolist())))
            if common.numel() < 4:
                continue
            pos_d = {int(v): i for i, v in enumerate(keep_d.tolist())}
            pos_h = {int(v): i for i, v in enumerate(keep_h.tolist())}
            sel_d = torch.tensor([pos_d[int(v)] for v in common])
            sel_h = torch.tensor([pos_h[int(v)] for v in common])
            xs_d.append(xd[sel_d].cpu())
            xs_h.append(xh[sel_h].cpu())
            doms.append(torch.full((common.numel(),), d, dtype=torch.int16))
            wins.append(torch.full((common.numel(),), win, dtype=torch.int32))
        if sum(x.shape[0] for x in xs_d) >= max_words:
            break
    tap_d.close()
    tap_h.close()
    x_d = torch.cat(xs_d, dim=0)
    x_h = torch.cat(xs_h, dim=0)
    domain = torch.cat(doms, dim=0)
    window = torch.cat(wins, dim=0)

    # donor routes + graft scores on pooled donor features; host scores too
    probs_d = F.softmax(x_d @ donor_w.float().to(x_d.device).T, dim=-1)
    _, stock_idx = torch.topk(probs_d, donor_topk, dim=-1)
    if banks_l is not None:
        p = banks_l.P.float()
        gsc = (x_d @ p.T) / math.sqrt(banks_l.d_r) * banks_l.layer_scale
        gprobs = F.softmax(gsc, dim=-1)
        _, graft_idx = torch.topk(gprobs, donor_topk, dim=-1)
    else:
        gsc = probs_d
        graft_idx = stock_idx
    host_scores = F.softmax(x_h @ host_w.float().to(x_h.device).T, dim=-1)

    n = x_d.shape[0]
    cap_hd = Capture(
        x_src=x_h, x_tgt=x_d, stock_idx=stock_idx, graft_idx=graft_idx,
        graft_scores=host_scores, domain=domain, window=window,
        host_layer=host_layer, donor_layer=donor_layer)
    cap_dh = Capture(
        x_src=x_d, x_tgt=x_h, stock_idx=stock_idx, graft_idx=graft_idx,
        graft_scores=gsc, domain=domain, window=window,
        host_layer=host_layer, donor_layer=donor_layer)
    return {"host_to_donor": cap_hd, "donor_to_host": cap_dh, "n": n,
            "n_chunks": len(chunks)}


def _forward_batch(model, enc, device):
    """One batched forward; PreTap hooks on the MoE block(s) capture inputs."""
    ids = enc["input_ids"].to(device)
    mask = enc.get("attention_mask")
    if mask is not None:
        mask = mask.to(device)
    try:
        model(input_ids=ids, attention_mask=mask)
    except TypeError:
        model(ids)


# ---------------------------------------------------------------------------
# Synthetic self-test (CPU, no donor): metric + fit + split + verdict plumbing
# ---------------------------------------------------------------------------


def _mock_capture(n=3000, d_src=32, d_tgt=32, n_exp=9, k=2, seed=0,
                  nonlinear=False):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d_src, generator=g)
    A = torch.randn(d_src, d_tgt, generator=g) / math.sqrt(d_src)
    y = x @ A
    if nonlinear:
        B = torch.randn(d_src, d_tgt, generator=g) / math.sqrt(d_src)
        C = torch.randn(d_src, d_tgt, generator=g) / math.sqrt(d_src)
        # strongly nonlinear map: ridge-linear cannot fit this well
        y = 0.15 * y + 2.0 * torch.tanh(x @ B) + torch.sin(x @ C)
    y = y + 0.01 * torch.randn(n, d_tgt, generator=g)
    idx = torch.randint(0, n_exp, (n, k), generator=g)
    domain = torch.arange(n).remainder(5).to(torch.int16)
    window = torch.arange(n).div(16, rounding_mode="floor").to(torch.int32)
    scores = F.one_hot(idx[:, 0], num_classes=n_exp).float()
    return Capture(x_src=x, x_tgt=y, stock_idx=idx, graft_idx=idx,
                   graft_scores=scores, domain=domain, window=window,
                   host_layer=0, donor_layer=1)


def run_synthetic_tests() -> int:
    torch.manual_seed(0)
    results = []

    def check(name, cond, detail=""):
        results.append((name, bool(cond)))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f" — {detail}" if detail else ""))

    # 1. ridge linear recovers a linear map
    cap = _mock_capture(seed=1)
    arm = run_arm(cap, "host_to_donor", "linear", "iid", steps=200)
    check("linear_recovers_linear_map",
          arm["ok"] and arm["heldout"]["cos_mean"] > 0.99,
          f"heldout cos={arm['heldout']['cos_mean']:.4f}")

    # 2. identity + mean baselines behave
    #    (identity cos on a real pairing == the raw x_src/x_tgt cos; on a
    #    same-layer pairing x_tgt == x_src it must be 1.0)
    arm_i = run_arm(cap, "host_to_donor", "identity", "iid")
    raw_cos = cos_stats(cap.x_src, cap.x_tgt)["cos_mean"]
    check("identity_control_matches_raw_cos",
          abs(arm_i["heldout"]["cos_mean"]
              - arm_i["identity_baseline_heldout"]["cos_mean"]) < 1e-6
          and abs(arm_i["identity_baseline_heldout"]["cos_mean"]) <= 1.0,
          f"identity={arm_i['heldout']['cos_mean']:.4f} raw={raw_cos:.4f}")
    cap_same = _mock_capture(seed=4)
    cap_same.x_tgt = cap_same.x_src.clone()
    arm_same = run_arm(cap_same, "host_to_donor", "identity", "iid")
    check("identity_control_cos1_on_same_layer",
          abs(arm_same["heldout"]["cos_mean"] - 1.0) < 1e-5,
          f"cos={arm_same['heldout']['cos_mean']:.6f}")
    arm_m = run_arm(cap, "host_to_donor", "mean", "iid")
    check("mean_baseline_is_floor",
          arm_m["heldout"]["cos_mean"] < arm["heldout"]["cos_mean"],
          f"mean {arm_m['heldout']['cos_mean']:.3f} < "
          f"linear {arm['heldout']['cos_mean']:.3f}")

    # 3. mlp2 beats linear on a strongly nonlinear map (enough data to
    #    generalize — a 512-hidden MLP on ~2k samples overfits instead)
    cap_nl = _mock_capture(n=20000, seed=2, nonlinear=True)
    arm_l = run_arm(cap_nl, "host_to_donor", "linear", "iid")
    arm_m2 = run_arm(cap_nl, "host_to_donor", "mlp2", "iid", steps=2500,
                     lr=3e-3, hidden=128)
    check("mlp2_beats_linear_nonlinear",
          arm_m2["heldout"]["cos_mean"] > arm_l["heldout"]["cos_mean"] + 0.02
          and arm_m2["heldout"]["cos_mean"] > 0.85,
          f"linear {arm_l['heldout']['cos_mean']:.3f} -> "
          f"mlp2 {arm_m2['heldout']['cos_mean']:.3f}")

    # 4. splits disjoint + route holdout excludes its routes from train
    tr, te, info = make_split("route", cap, heldout_route_count=3)
    check("split_disjoint",
          bool((tr & te).sum() == 0 and (tr | te).all()),
          f"train={info['n_train']} test={info['n_test']}")
    held = torch.tensor(info["heldout_routes"])
    check("route_heldout_excluded_from_train",
          int(torch.isin(cap.stock_idx[tr, 0], held).sum()) == 0,
          f"held routes={info['heldout_routes']}")
    tr_d, te_d, info_d = make_split("domain", cap, heldout_domain=4)
    check("domain_split_pure",
          int((cap.domain[tr_d] == 4).sum()) == 0
          and int((cap.domain[te_d] == 4).sum()) == int(te_d.sum()),
          f"test domain 4 n={info_d['n_test']}")

    # 5. route-conditioned linear runs and is not worse than plain on noise-free
    arm_c = run_arm(cap, "host_to_donor", "linear", "iid", cond=True)
    check("cond_linear_runs",
          arm_c["ok"] and arm_c["heldout"]["cos_mean"] > 0.99,
          f"cond cos={arm_c['heldout']['cos_mean']:.4f}")

    # 6. per-route breakdown sums + verdict logic
    pr = arm["per_route_heldout"]
    check("per_route_grouping",
          pr["n_routes_reported"] > 0 and pr["micro_cos"] > 0.9,
          f"routes={pr['n_routes_reported']} micro={pr['micro_cos']:.3f}")
    fake = [{"ok": True, "direction": "host_to_donor", "scope": "all",
             "cond": False, "kind": "mlp2", "pair": "2:12",
             "split": {"mode": "iid"},
             "heldout": {"cos_mean": 0.95, "cos_p10": 0.9},
             "train": {"cos_mean": 0.96}},
            {"ok": True, "direction": "host_to_donor", "scope": "all",
             "cond": False, "kind": "identity", "pair": "2:12",
             "split": {"mode": "iid"},
             "heldout": {"cos_mean": 0.4, "cos_p10": 0.1},
             "train": {"cos_mean": 0.4}}]
    v = d15_verdict(fake, {})
    check("verdict_pass_logic",
          v["pass"] and v["gates"]["heldout_cos_gt_0.9_all_challenge"]
          and not v["deprioritize_variant_b"]
          and v["per_pair"][0]["role"] == "challenge",
          f"gates={v['gates']}")
    fake2 = [dict(fake[0], heldout={"cos_mean": 0.5, "cos_p10": 0.2})]
    v2 = d15_verdict(fake2, {})
    check("verdict_deprioritize_logic",
          not v2["pass"] and v2["deprioritize_variant_b"],
          f"gates={v2['gates']}")
    # control pairs (identity ~ 1) must NOT rescue a failing challenge pair
    fake3 = [dict(fake[0], pair="12:12"),
             dict(fake[1], pair="12:12", heldout={"cos_mean": 1.0,
                                                  "cos_p10": 1.0}),
             dict(fake[0], pair="2:12", heldout={"cos_mean": 0.55,
                                                 "cos_p10": 0.4}),
             dict(fake[1], pair="2:12")]
    v3 = d15_verdict(fake3, {})
    roles = {p["pair"]: p["role"] for p in v3["per_pair"]}
    check("verdict_control_excluded",
          roles == {"12:12": "control", "2:12": "challenge"}
          and not v3["pass"] and v3["deprioritize_variant_b"]
          and abs(v3["heldout_cos_worst_challenge"] - 0.55) < 1e-9,
          f"roles={roles} worst={v3['heldout_cos_worst_challenge']}")

    # 7. expert-output metric: identity map -> cos 1
    g = torch.Generator().manual_seed(3)
    gu = torch.randn(64, 2 * 24, 32, generator=g) * 0.3
    dn = torch.randn(64, 32, 24, generator=g) * 0.3
    ffn = ExpertFFN(gu[0], dn[0])
    x = torch.randn(256, 32, generator=g)
    eo = expert_output_metrics(ffn, x, x, top_directions(ffn(x), r=8))
    check("expert_output_identity_cos1",
          abs(eo["output_cos"]["cos_mean"] - 1.0) < 1e-4
          and abs(eo["output_cos_topdirs"]["cos_mean"] - 1.0) < 1e-4,
          f"cos={eo['output_cos']['cos_mean']:.6f} "
          f"topdirs={eo['output_cos_topdirs']['cos_mean']:.6f}")

    # 8. scope=expert restricts the manifold
    e_star = pick_transplant_expert(cap)
    n_expect = int((cap.stock_idx == e_star).any(dim=-1).sum())
    arm_s = run_arm(cap, "host_to_donor", "linear", "iid",
                    scope="expert", transplant_expert=e_star, ffn=None)
    check("expert_scope_restricts",
          arm_s["ok"] and arm_s["split"]["n_train"] + arm_s["split"]["n_test"]
          == n_expect,
          f"scope n={arm_s['split']['n_train'] + arm_s['split']['n_test']} "
          f"manifold={n_expect}")

    # 9. donor->host direction works (mirror)
    arm_r = run_arm(cap, "donor_to_host", "linear", "iid")
    check("direction_mirror_runs",
          arm_r["ok"] and arm_r["heldout"]["cos_mean"] > 0.99,
          f"cos={arm_r['heldout']['cos_mean']:.4f}")

    # 10. build_pair slicing sanity
    capd = {0: {"h": torch.randn(100, 8).to(torch.float16),
                "stock_idx": torch.zeros(100, 2, dtype=torch.long),
                "graft_idx": torch.zeros(100, 2, dtype=torch.long),
                "graft_scores": torch.randn(100, 4).to(torch.float16)},
            1: {"h": torch.randn(100, 8).to(torch.float16),
                "stock_idx": torch.zeros(100, 2, dtype=torch.long),
                "graft_idx": torch.zeros(100, 2, dtype=torch.long),
                "graft_scores": torch.randn(100, 4).to(torch.float16)},
            "domain": torch.zeros(100, dtype=torch.int16),
            "window": torch.zeros(100, dtype=torch.int32),
            "meta": {}}
    pc = build_pair(capd, 0, 1)
    check("build_pair_shapes",
          pc.n() == 100 and pc.x_src.shape == (100, 8)
          and pc.graft_scores.shape == (100, 4),
          f"n={pc.n()}")

    # 11. word pooling across tokenizers (cross-model pairing primitive)
    t_h = torch.tensor([[1., 0.], [2., 0.], [4., 0.], [0., 3.], [9., 9.]])
    offs = [(0, 5), (6, 9), (9, 11), (12, 15), (0, 0)]  # last = pad
    wspan = [(0, 5), (6, 11), (12, 15)]
    wfeat, keep = pool_tokens_to_words(offs, t_h, wspan)
    expect = torch.tensor([[1., 0.], [3., 0.], [0., 3.]])
    check("pool_tokens_to_words",
          wfeat.shape == (3, 2) and keep.tolist() == [0, 1, 2]
          and float((wfeat - expect).abs().max()) < 1e-6,
          f"keep={keep.tolist()} feats={wfeat.tolist()}")

    n_fail = sum(1 for _, ok in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} synthetic checks passed")
    return 1 if n_fail else 0


# ---------------------------------------------------------------------------
# Real-donor pipeline
# ---------------------------------------------------------------------------


def parse_pairs(spec: str) -> list[tuple[int, int]]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split(":")
        out.append((int(a), int(b)))
    return out


def train_matrix(cases: list[dict], args,
                 only_direction: str | None = None) -> list[dict]:
    """Run the translator matrix over (capture, expert) cases.

    Each case: {"cap": Capture, "ffn": ExpertFFN, "e_star": int,
    "pair_name": str}. Reused by the cross-layer and cross-model pipelines;
    `only_direction` restricts to one of the two translation directions.
    """
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    directions = ("host_to_donor", "donor_to_host")
    if only_direction is not None:
        directions = (only_direction,)
    arms = []
    for case in cases:
        c = case["cap"]
        e_star = case["e_star"]
        ffn = case["ffn"]
        dirs_cache = {}  # per-pair: (direction, split, scope) -> top dirs
        print(f"\n=== pair {case['pair_name']} "
              f"transplant_expert={e_star} (N={c.n()}) ===")
        for direction in directions:
            for kind in kinds:
                for split_mode in splits:
                    scopes = ["all"] + (["expert"] if args.expert_scope
                                        else [])
                    for scope in scopes:
                        cond_variants = [False, True] if (
                            args.cond and kind in ("linear", "mlp2")
                            and scope == "all") else [False]
                        for cond in cond_variants:
                            # expert top-singular directions depend only on
                            # (direction, split, scope) — compute once
                            dkey = (direction, split_mode, scope)
                            if dkey not in dirs_cache:
                                tr0, _, _ = make_split(
                                    split_mode, c, args.heldout_route_count,
                                    args.heldout_domain, args.seed)
                                xt = (c.x_tgt if direction == "host_to_donor"
                                      else c.x_src)
                                dirs_cache[dkey] = top_directions(
                                    ffn(xt[tr0]), r=32)
                            arm = run_arm(
                                c, direction, kind, split_mode,
                                cond=cond,
                                scope=scope, transplant_expert=e_star,
                                heldout_route_count=args.heldout_route_count,
                                heldout_domain=args.heldout_domain,
                                steps=args.steps, lr=args.lr,
                                hidden=args.hidden,
                                device=args.device, seed=args.seed,
                                ffn=ffn,
                                dirs=dirs_cache[dkey])
                            if not arm.get("ok"):
                                print(f"  skip {direction}/{kind}/{split_mode}/"
                                      f"{scope}/cond{int(cond)}: "
                                      f"{arm.get('reason')}")
                                continue
                            arm["pair"] = case["pair_name"]
                            arms.append(arm)
                            eo = arm.get("expert_output_heldout", {})
                            eo_cos = eo.get("output_cos", {}).get("cos_mean")
                            print(f"  {direction:13s} {kind:8s} cond="
                                  f"{int(arm['cond'])} {split_mode:6s} "
                                  f"{scope:6s} "
                                  f"train={arm['train']['cos_mean']:.4f} "
                                  f"held={arm['heldout']['cos_mean']:.4f} "
                                  f"(p10 {arm['heldout']['cos_p10']:.4f}) "
                                  f"ident="
                                  f"{arm['identity_baseline_heldout']['cos_mean'] if arm['identity_baseline_heldout'] else float('nan'):.4f} "
                                  f"expert_out={eo_cos if eo_cos is not None else float('nan'):.4f}")
    return arms


def run_pipeline(args) -> dict:
    t0 = time.time()
    donor_name = args.model.rstrip("/")
    banks, reg_meta = d0.load_registry(args.registry)
    model, tok = d0.load_donor(args.model, args.device, args.dtype)
    blocks = d0.find_moe_blocks(model)
    n_layers = len(blocks)
    print(f"donor loaded: {n_layers} MoE layers, "
          f"E={blocks[0].gate.weight.shape[0]}, "
          f"H={blocks[0].gate.weight.shape[1]}, registry layers={len(banks)}")

    domains = d0.load_domain_texts(args.domains_dir)
    domain_names = list(domains.keys())
    batches = []
    per = args.max_tokens // max(1, len(domain_names))
    for d, name in enumerate(domain_names):
        batches.extend(d0.texts_to_batches(tok, [domains[name]], args.seq_len,
                                           args.batch_size, per, label=d))
    n_tok = sum(t.numel() for t, _ in batches)
    print(f"data: {len(domain_names)} domains {domain_names}, "
          f"{n_tok} tokens, seq_len={args.seq_len}")

    pairs = parse_pairs(args.layer_pairs)
    needed = sorted({l for pr in pairs for l in pr})
    t_cap = time.time()
    cap = capture_layers(model, blocks, batches, needed, args.device,
                         banks=banks, max_tokens=args.max_tokens)
    cap_s = time.time() - t_cap
    print(f"capture: {cap['meta']['n_tokens']} tokens x layers {needed} "
          f"in {cap_s:.1f}s")
    if args.capture_out:
        torch.save({"cap": {k: v for k, v in cap.items()}, "pairs": pairs},
                   args.capture_out)
        print(f"capture -> {args.capture_out}")

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    cases = []
    for host_l, donor_l in pairs:
        c = build_pair(cap, host_l, donor_l)
        e_star = args.transplant_expert if args.transplant_expert >= 0 \
            else pick_transplant_expert(c)
        cases.append({"cap": c,
                      "ffn": extract_expert(blocks, donor_l, e_star),
                      "e_star": e_star,
                      "pair_name": f"{host_l}:{donor_l}"})
    arms = train_matrix(cases, args)

    meta = {
        "mode": "cross-layer",
        "donor": donor_name, "revision": args.revision, "dtype": args.dtype,
        "registry": args.registry, "registry_init": reg_meta.get("init"),
        "registry_recal_steps": reg_meta.get("recal_steps"),
        "domains": domain_names, "n_tokens": cap["meta"]["n_tokens"],
        "layers_captured": cap["meta"]["layers"], "layer_pairs": pairs,
        "kinds": kinds, "splits": splits, "capture_s": round(cap_s, 1),
    }
    verdict = d15_verdict(arms, meta)
    out = {
        "gate": "D1.5 translator fidelity",
        "spec": "MODULAR_MOE §9.3a / §9.4 D1.5 / §9.7",
        "meta": meta,
        "arms": [{k: v for k, v in a.items() if k != "fit"} | {
            "fit_final": a.get("fit", {}).get("final")} for a in arms],
        "verdict": verdict,
        "wall_s": round(time.time() - t0, 1),
    }
    return out


def run_pipeline_cross_model(args) -> dict:
    """The spec's two-model pairing: OLMoE donor <-> a second small MoE host.

    Word-aligned paired telemetry (char-offset pooling over each model's own
    tokenizer); donor routes from the donor's stock gate on pooled donor
    features; transplant expert = most-used donor expert at --donor-layer.
    """
    t0 = time.time()
    donor_name = args.model.rstrip("/")
    banks, reg_meta = d0.load_registry(args.registry)
    donor, donor_tok = d0.load_donor(args.model, args.device, args.dtype)
    donor_blocks = d0.find_moe_blocks(donor)
    host, host_tok = d0.load_donor(args.host_model, args.device, args.dtype)
    host_blocks = find_moe_blocks_general(host)
    if not host_blocks:
        raise RuntimeError(f"no MoE blocks in host model {args.host_model} "
                           f"(expected .router/.gate + .experts)")
    print(f"donor: {len(donor_blocks)} MoE layers; "
          f"host: {len(host_blocks)} MoE layers, "
          f"d_host={host_blocks[0].router.weight.shape[1] if hasattr(host_blocks[0], 'router') else host_blocks[0].gate.weight.shape[1]}")

    domains = d0.load_domain_texts(args.domains_dir)
    donor_layer = args.donor_layer
    host_layers = [int(x) for x in args.host_layers.split(",") if x.strip()]
    t_cap = time.time()
    cap = capture_cross_model(
        donor, donor_blocks, donor_layer, host, host_blocks, host_layers[0],
        domains, donor_tok, host_tok, args.device, banks=banks,
        chunk_words=args.chunk_words, batch_chunks=args.batch_chunks,
        max_words=args.max_words)
    # extra host layers: re-capture per layer (cheap vs training)
    caps_by_host = {host_layers[0]: cap}
    for hl in host_layers[1:]:
        caps_by_host[hl] = capture_cross_model(
            donor, donor_blocks, donor_layer, host, host_blocks, hl,
            domains, donor_tok, host_tok, args.device, banks=banks,
            chunk_words=args.chunk_words, batch_chunks=args.batch_chunks,
            max_words=args.max_words)
    cap_s = time.time() - t_cap

    cases = []
    for hl in host_layers:
        caps = caps_by_host[hl]
        for direction in ("host_to_donor", "donor_to_host"):
            c = caps[direction]
            e_star = args.transplant_expert if args.transplant_expert >= 0 \
                else pick_transplant_expert(c)
            # ffn is the DONOR expert either way (spec: transplanted output vs
            # native donor output)
            ffn = extract_expert(donor_blocks, donor_layer, e_star)
            case = {"cap": c, "ffn": ffn, "e_star": e_star,
                    "pair_name": f"h{hl}:d{donor_layer}"}
            case["_direction"] = direction
            cases.append(case)
    # train_matrix runs both directions per case; for cross-model the two
    # Captures already encode the direction, so run each case's own direction
    arms = []
    for case in cases:
        direction = case.pop("_direction")
        arms.extend(train_matrix([case], args, only_direction=direction))

    meta = {
        "mode": "cross-model",
        "donor": donor_name, "host_model": args.host_model,
        "pairing": "word-aligned char-offset pooling (per-model tokenizers)",
        "donor_layer": donor_layer, "host_layers": host_layers,
        "registry_init": reg_meta.get("init"),
        "n_words": cap["n"], "n_chunks": cap["n_chunks"],
        "chunk_words": args.chunk_words,
        "kinds": args.kinds, "splits": args.splits,
        "capture_s": round(cap_s, 1), "dtype": args.dtype,
    }
    verdict = d15_verdict(arms, meta)
    out = {
        "gate": "D1.5 translator fidelity",
        "spec": "MODULAR_MOE §9.3a / §9.4 D1.5 / §9.7",
        "meta": meta,
        "arms": [{k: v for k, v in a.items() if k != "fit"} | {
            "fit_final": a.get("fit", {}).get("final")} for a in arms],
        "verdict": verdict,
        "wall_s": round(time.time() - t0, 1),
    }
    return out


def print_verdict_table(out: dict) -> None:
    print("\n================ D1.5 VERDICT TABLE ================")
    print(f"{'pair':8s} {'dir':13s} {'kind':8s} {'cond':4s} {'split':6s} "
          f"{'scope':6s} {'train':7s} {'held':7s} {'p10':7s} {'ident':7s} "
          f"{'expOut':7s} {'expTop':7s}")
    for a in out["arms"]:
        eo = a.get("expert_output_heldout") or {}
        e1 = eo.get("output_cos", {}).get("cos_mean")
        e2 = eo.get("output_cos_topdirs", {}).get("cos_mean")
        ib = (a.get("identity_baseline_heldout") or {}).get("cos_mean")
        print(f"{a['pair']:8s} {a['direction']:13s} {a['kind']:8s} "
              f"{int(a['cond']):4d} {a['split']['mode']:6s} {a['scope']:6s} "
              f"{a['train']['cos_mean']:7.4f} {a['heldout']['cos_mean']:7.4f} "
              f"{a['heldout']['cos_p10']:7.4f} "
              f"{(f'{ib:.4f}' if ib is not None else '  n/a '):>7s} "
              f"{(f'{e1:.4f}' if e1 is not None else '  n/a '):>7s} "
              f"{(f'{e2:.4f}' if e2 is not None else '  n/a '):>7s}")
    v = out["verdict"]
    print("\n--- per-pair summary (primary = best linear/mlp2, host->donor, "
          "iid) ---")
    print(f"{'pair':10s} {'role':10s} {'ident':7s} {'train':7s} {'held':7s} "
          f"{'p10':7s} {'routeH':7s} {'domH':7s} {'expOut':7s} {'kind':8s}")
    for p in v["per_pair"]:
        rh = (p["route_heldout"] or {}).get("cos_mean")
        dh = (p["domain_heldout"] or {}).get("cos_mean")
        eo = (p["expert_output"] or {}).get("output_cos")

        def f(x):
            return f"{x:7.4f}" if x is not None else "  n/a  "
        print(f"{p['pair']:10s} {p['role']:10s} "
              f"{f(p['identity_baseline_cos'])} {f(p['train_cos'])} "
              f"{f(p['heldout_cos'])} {f(p['heldout_cos_p10'])} "
              f"{f(rh)} {f(dh)} {f(eo)} {str(p['primary_kind']):8s}")
    w = v["worst_challenge_pair"]
    if w is not None and w.get("heldout_cos") is not None:
        print(f"\nWORST CHALLENGE pair {w['pair']}: held-out cos "
              f"{w['heldout_cos']:.4f} (train {w['train_cos']:.4f}, "
              f"p10 {w['heldout_cos_p10']:.4f}, identity baseline "
              f"{w['identity_baseline_cos'] if w['identity_baseline_cos'] is not None else float('nan'):.4f})"
              f" [{w['primary_kind']}]")
    else:
        print("\nWORST CHALLENGE pair: none (control-only run)")
    print(f"gates: {v['gates']}")
    print(f"VERDICT: {'PASS' if v['pass'] else 'FAIL'} vs bar "
          f"{v['bars']['cos_pass']}  |  deprioritize variant B: "
          f"{v['deprioritize_variant_b']} (bar {v['bars']['cos_deprioritize']})")
    print(f"note: {v['scope_note']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--registry", default=None,
                    help="D0 registry .pt (graft basis; see graft_d0_parity)")
    ap.add_argument("--host-model", default=None,
                    help="second MoE path (e.g. granite-3.0-1b-a400m-base) "
                         "-> cross-model word-aligned pairing (§9.3a); "
                         "omit for within-donor cross-layer pairing")
    ap.add_argument("--host-layers", default="12,18",
                    help="cross-model: host MoE-block indices to try")
    ap.add_argument("--donor-layer", type=int, default=12,
                    help="cross-model: donor MoE layer (transplant source)")
    ap.add_argument("--chunk-words", type=int, default=24,
                    help="cross-model: words per paired capture chunk")
    ap.add_argument("--batch-chunks", type=int, default=8)
    ap.add_argument("--max-words", type=int, default=20000,
                    help="cross-model: cap on paired words")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--dtype", default="float16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--domains-dir",
                    default=os.path.join(REPO_ROOT, "lab", "graft_d0_texts"))
    ap.add_argument("--layer-pairs", default="2:12,6:12,12:12",
                    help="comma list of host:donor layer pairs to train")
    ap.add_argument("--kinds", default="identity,mean,linear,mlp2")
    ap.add_argument("--splits", default="iid,route,domain")
    ap.add_argument("--cond", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="also run the route-conditioned arm (graft passport "
                         "scores at the source layer); default on")
    ap.add_argument("--expert-scope", action="store_true",
                    help="also train the §9.3a one-expert-manifold arm")
    ap.add_argument("--transplant-expert", type=int, default=-1,
                    help="expert id for the transplant metric (-1 = most used)")
    ap.add_argument("--heldout-route-count", type=int, default=8)
    ap.add_argument("--heldout-domain", type=int, default=4,
                    help="domain index held out for the domain split "
                         "(default 4 = last of the sorted domain names)")
    ap.add_argument("--max-tokens", type=int, default=24000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--capture-out", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.synthetic:
        return run_synthetic_tests()
    if not args.model or not args.registry:
        ap.error("--model and --registry are required unless --synthetic")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but torch.cuda.is_available() is "
                           "false — check HIP_VISIBLE_DEVICES / ROCm")

    if args.host_model:
        out = run_pipeline_cross_model(args)
    else:
        out = run_pipeline(args)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nresults -> {args.out}")
    print_verdict_table(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
