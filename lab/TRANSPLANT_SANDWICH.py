#!/usr/bin/env python3
"""SANDWICH-TRANSLATOR transplant at vk scale (swarm branch, Door-9 heir).

The earlier cross-family lanes closed two shapes:
  * D1.5 (lab/graft_d15_translator.py): paired-activation input-basis
    translation between MoE families holds ~0.47-0.69 held-out cosine ->
    feature matching does not transfer across families.
  * Door 2.5 (lab/door25_wrap_expert.py): foreign core behind up/down
    adapters trained on task CE at OLMoE graft scale.
This arm asks the vk-scale version of Door 2.5 with the roles inverted: a
REAL donor expert core is hosted FROZEN, and two thin projections do all the
learning:

    h_host --[W_in 1024->1024]--> donor-space --[DONOR EXPERT, frozen]-->
    donor-out --[W_out 1024->1024]--> host-hidden flow

installed as ONE MoE expert slot of the host at one layer (the host's most-
routed slot at layer 4, measured at runtime on the calibration library).
Pair: donor = 470m-family SFT (lab/imported/sft_470m step_best == step_150
chat export, val 2.3044) -> host = vk SFT-r3 chat (step_1800). Same arch
(dim 1024, 10 layers, 8 passport-routed experts, hidden 1280, same
tokenizer) but family-divergent manifolds.

Calibration: capture ffn-input hidden states of BOTH models over the teach
library (cadence ITEMS/SESSIONS/REPLAY/FILLER, imported not copied), build
"hidden-state mismatch windows" = the top-median-up tokens by host/donor
cosine distance, and train ONLY the two sandwiches on the pool:
  loss = MSE(W_out D W_in(h_h), host_expert_e(h_h))         functional anchor
       + 0.25 * MSE(W_in(h_h), h_d)                          input-basis anchor y
lr 1e-3, 500 steps, batch 256 tokens. Heeding D1.5 (feature matching alone
does not transfer) the host-function term leads; the input term is a
regularizer feeding the frozen core its native basis.

Pre-declared bars (per-token CE, token-weighted, deltas vs host-bare):
  Bar A parity   : every flat probe (3 fixed generic chat pairs, held out of
                   training) has CE_armA - CE_bare <= +0.05 nats.
  Bar B liveness : CE_armA - CE_bare >= +0.10 nats on the donor-class fact
                   set (top-3 library fact candidates ranked by
                   CE_donor - CE_host) AND >=5% of fact tokens route through
                   the transplanted slot. Higher CE = the foreign core fires
                   visibly on donor-leaning tokens; parity says it does not
                   leak elsewhere. A and B together = feasibility.
Diagnostics (not bars): naive identity-sandwich transplant CE (the Door-9
shape at vk scale, expected broken), held-out R^2 of the sandwich fit,
contribution-norm ratio rho, routed shares.
Autopsy typing on fail: TYPE-4 routing-starve (slot starved) / TYPE-2
parity-broken (sandbox cannot translate well enough to stay harmless) /
TYPE-1 mimic-collapse (parity bought by killing the core, rho eval < 0.2) /
TYPE-3 inert-unselective (parity fine, no donor-class signal, rho alive).
Pre-declared single fallback: arm F retrains W_out ONLY (W_in frozen at
identity, null-space transpose on the OUT sandwich). No further arms.

Runtime: laptop CPU fp32 (house-proven shims via consolidate.load_base_model;
Immune class discipline — never touches the box).

Usage:
  PYTHONPATH=$HOME/.local/lib/python3.12/site-packages VESPER_CPU_THREADS=12 \
    python3 -u lab/TRANSPLANT_SANDWICH.py --smoke            # sanity pass
  PYTHONPATH=$HOME/.local/lib/python3.12/site-packages VESPER_CPU_THREADS=12 \
    python3 -u lab/TRANSPLANT_SANDWICH.py                    # full arm
Outputs: numbers-only JSON to lab/TRANSPLANT_SANDWICH_OUT/results.json;
console VERDICT lines; the .md report pairs with the JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

REPO = os.environ.get("VESPER_REPO", "/home/tliao/VesperLM")
LAB = os.path.join(REPO, "lab")
sys.path.insert(0, LAB)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import teach_spaced_cadence as cad          # noqa: E402  (library: 6 items)
from consolidate import load_base_model     # noqa: E402  (house loader)

e3 = cad.e3
user_turn = e3.user_turn

HOST_CKPT = os.path.join(LAB, "imported", "vk_sft_r3", "step_1800", "chat_model")
DONOR_CKPT = os.path.join(LAB, "imported", "sft_470m", "step_150", "chat_model")

LAYER = 4            # FFN transplant site in both models (same arch)
LR = 1e-3
STEPS = 500
BATCH = 256          # tokens per step
LAM_IN = 0.25        # input-anchor weight (regularizer)
SEED_A, SEED_F = 7, 8
MAX_TOK = 128

# Bars (pre-declared; see .md)
BAR_A_MAX_NAT = 0.05       # parity: no flat probe may degrade more than this
BAR_B_MIN_NAT = 0.10       # liveness: donor-class facts must MOVE this much
BAR_ROUTE_MIN = 0.05       # floor for "the slot is actually exercised"

# Three flat probes: generic chat, written for this arm (teach-series style;
# NOT drawn from any dataset corpus), held out of the calibration pool.
FLAT_QA = [
    ("Is it supposed to rain tomorrow?",
     "I would check the forecast — I cannot see the sky from here."),
    ("What is a sensible bedtime on a work night?",
     "Early enough to leave you seven to nine hours before the alarm."),
    ("Can you give me a short toast for a birthday dinner?",
     "To good food, better company, and many returns of the day."),
]


# ----------------------------------------------------------------------
# Library construction: every sequence as (prompt, continuation) strings.
# ----------------------------------------------------------------------
def build_library():
    lib = []  # (tag, prompt, continuation)
    for key, item in cad.ITEMS.items():
        lib.append((f"teach/{key}", "", item["teach"]))
        lib.append((f"probe_t/{key}", item["probe_ctx"], item["target"]))
        lib.append((f"probe_d/{key}", item["probe_ctx"], item["decoy"]))
    for key, (sid, triples) in cad.SESSIONS.items():
        for i, (q, approve, corr) in enumerate(triples):
            if approve:
                lib.append((f"sess_a/{key}/{i}", user_turn(q), approve))
            if corr:
                lib.append((f"sess_c/{key}/{i}", user_turn(q), corr))
    for key, variants in cad.REPLAY.items():
        for i, vt in enumerate(variants):
            q = vt[0]
            for j, v in enumerate(vt[1:]):
                lib.append((f"replay/{key}/{i}/{j}", user_turn(q), v))
    for i, (q, a) in enumerate(cad.FILLER):
        lib.append((f"filler/{i}", user_turn(q), a))
    return lib


def flat_probes():
    return [("flat_0", user_turn(FLAT_QA[0][0]), FLAT_QA[0][1]),
            ("flat_1", user_turn(FLAT_QA[1][0]), FLAT_QA[1][1]),
            ("flat_2", user_turn(FLAT_QA[2][0]), FLAT_QA[2][1])]


# ----------------------------------------------------------------------
# Encoding / CE helpers
# ----------------------------------------------------------------------
def encode_pair(tok, prompt, cont):
    """Token ids of prompt+cont plus the continuation start index.

    Verifies the BPE prefix property; if it breaks (rare at boundaries),
    splices prompt ids with separately encoded continuation ids instead —
    still position-consistent for CE masking.
    """
    ids_p = tok.encode(prompt, add_special_tokens=False)
    ids_pc = tok.encode(prompt + cont, add_special_tokens=False)
    if ids_pc[:len(ids_p)] == ids_p:
        return ids_pc, len(ids_p)
    ids_c = tok.encode(cont, add_special_tokens=False)
    return ids_p + ids_c, len(ids_p)


def collate(tok, pairs, pad_id, target_tokens=1500):
    """Length-sorted, right-padded batches. All attention here is causal, so
    pad rows never touch scored rows."""
    enc = []
    for tag, p, c in pairs:
        ids, plen = encode_pair(tok, p, c)
        ids = ids[:MAX_TOK]
        enc.append((tag, ids, plen))
    enc.sort(key=lambda e: len(e[1]))
    batches, cur, cur_toks = [], [], 0
    for e in enc:
        t = max(1, len(e[1]))
        if cur and cur_toks + t * len(cur) > target_tokens:
            batches.append(cur)
            cur, cur_toks = [], 0
        cur.append(e)
        cur_toks += t
    if cur:
        batches.append(cur)
    return batches


class CaptureBag:
    def __init__(self):
        self.hidden = None
        self.top_idx = None


def add_capture_hooks(model, layer):
    bag = CaptureBag()
    ffn = model.layers[layer]["ffn"]

    def pre(_mod, args):
        bag.hidden = args[0].detach()

    def post(_mod, _args, out):
        bag.top_idx = out[1].detach()

    h1 = ffn.register_forward_pre_hook(pre)
    h2 = ffn.router.register_forward_hook(post)
    return bag, (h1, h2)


@torch.no_grad()
def capture(model, tok, pairs, layer, bag_handles, pad_id):
    """Batched library forward; returns aligned hidden rows + routing shares."""
    bag, _handles = bag_handles
    rows = []
    route_counts = torch.zeros(model.layers[layer]["ffn"].num_experts)
    ntok = 0
    for batch in collate(tok, pairs, pad_id):
        lens = [len(e[1]) for e in batch]
        T = max(lens)
        x = torch.full((len(batch), T), pad_id, dtype=torch.long)
        for i, e in enumerate(batch):
            x[i, :len(e[1])] = torch.tensor(e[1])
        model(x)
        h = bag.hidden  # (B, T, C)
        ti = bag.top_idx.view(x.size(0), T, -1)  # router output is (B*T, top_k)
        for i, L in enumerate(lens):
            rows.append(h[i, :L])
            for e in range(route_counts.numel()):
                route_counts[e] += int((ti[i, :L] == e).any(dim=-1).sum())
        ntok += sum(lens)
    return torch.cat(rows, dim=0), route_counts / max(ntok, 1)


@torch.no_grad()
def routed_share(model, tok, pairs, layer, slot, bag_handles, pad_id):
    """Fraction of (non-pad) tokens whose top-k includes the slot."""
    bag, _handles = bag_handles
    hit = tot = 0
    for batch in collate(tok, pairs, pad_id):
        lens = [len(e[1]) for e in batch]
        T = max(lens)
        x = torch.full((len(batch), T), pad_id, dtype=torch.long)
        for i, e in enumerate(batch):
            x[i, :len(e[1])] = torch.tensor(e[1])
        model(x)
        ti = bag.top_idx.view(x.size(0), T, -1)
        for i, L in enumerate(lens):
            hit += int((ti[i, :L] == slot).any(dim=-1).sum())
        tot += sum(lens)
    return hit / max(tot, 1)


@torch.no_grad()
def ce_pairs(model, tok, pairs, pad_id):
    """Token-weighted mean CE over continuation tokens across the pair set."""
    tot_nll, tot_n = 0.0, 0
    for batch in collate(tok, pairs, pad_id):
        lens = [len(e[1]) for e in batch]
        plens = [e[2] for e in batch]
        T = max(lens)
        x = torch.full((len(batch), T), pad_id, dtype=torch.long)
        tgt = torch.full((len(batch), T), pad_id, dtype=torch.long)
        for i, e in enumerate(batch):
            L, plen = len(e[1]), e[2]
            x[i, :L] = torch.tensor(e[1])
            for t in range(L - 1):
                if t + 1 >= plen:
                    tgt[i, t] = e[1][t + 1]
        _logits, ce, _aux = model(x, targets=tgt)
        n = int(tgt.ne(pad_id).sum())
        if n:
            tot_nll += float(ce) * n
            tot_n += n
    return tot_nll / max(tot_n, 1), tot_n


@torch.no_grad()
def ce_per_pair(model, tok, pairs, pad_id):
    """Per-pair continuation CE (logits path — one batched forward total)."""
    out = []
    for batch in collate(tok, pairs, pad_id):
        lens = [len(e[1]) for e in batch]
        T = max(lens)
        x = torch.full((len(batch), T), pad_id, dtype=torch.long)
        for i, e in enumerate(batch):
            x[i, :len(e[1])] = torch.tensor(e[1])
        logits, _ce, _aux = model(x)  # (B, T, V) fp32
        for i, e in enumerate(batch):
            L, plen = len(e[1]), e[2]
            pos = [t for t in range(L - 1) if t + 1 >= plen]
            if not pos:
                out.append(float("nan"))
                continue
            lp = F.log_softmax(logits[i, pos].float(), dim=-1)
            tgt = torch.tensor([e[1][t + 1] for t in pos])
            out.append(float(-lp.gather(1, tgt.unsqueeze(1)).mean()))
        del logits
    return out


# ----------------------------------------------------------------------
# The transplanted expert
# ----------------------------------------------------------------------
class SandwichExpert(nn.Module):
    """Frozen donor expert core between two learned 1024->1024 projections."""

    def __init__(self, donor_expert: nn.Module, dim: int):
        super().__init__()
        self.W_in = nn.Linear(dim, dim, bias=False)
        self.W_out = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            self.W_in.weight.copy_(torch.eye(dim))
            self.W_out.weight.copy_(torch.eye(dim))
        self.core = donor_expert
        for q in self.core.parameters():
            q.requires_grad_(False)

    def forward(self, x):
        return self.W_out(self.core(self.W_in(x)))


def train_arm(pool, donor_expert, dim, seed, steps, lr, train_out_only):
    hh, hd, yref = pool
    sw = SandwichExpert(donor_expert, dim)
    if train_out_only:
        sw.W_in.weight.requires_grad_(False)
        params = [sw.W_out.weight]
    else:
        params = [sw.W_in.weight, sw.W_out.weight]
    opt = torch.optim.Adam(params, lr=lr)
    g = torch.Generator().manual_seed(seed)
    n = hh.size(0)
    hist = []
    perm = torch.randperm(n, generator=g)
    ptr = 0
    for step in range(steps + 1):
        if ptr + BATCH > n:
            perm = torch.randperm(n, generator=g)
            ptr = 0
        if step % 50 == 0:
            with torch.no_grad():
                pred = sw(hh)
                mse = F.mse_loss(pred, yref).item()
                r2 = 1.0 - mse / yref.var(dim=0).mean().item()
            hist.append((step, mse, r2))
        if step == steps:
            break
        idx = perm[ptr:ptr + BATCH]
        ptr += BATCH
        pred = sw(hh[idx])
        loss = (F.mse_loss(pred, yref[idx])
                + LAM_IN * F.mse_loss(sw.W_in(hh[idx]), hd[idx]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return sw, hist


@torch.no_grad()
def fit_stats(sw, hold):
    hh, _hd, yref = hold
    pred = sw(hh)
    mse = F.mse_loss(pred, yref).item()
    var = yref.var(dim=0).mean().item()
    rho = (pred.norm(dim=-1).mean() / yref.norm(dim=-1).clamp_min(1e-6).mean()).item()
    return {"mse": mse, "r2": 1.0 - mse / var, "rho": rho}


# ----------------------------------------------------------------------
def q(vals):
    vals = sorted(vals)
    n = len(vals)
    return {"mean": sum(vals) / n, "p50": vals[n // 2], "p90": vals[int(0.9 * (n - 1))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(LAB, "TRANSPLANT_SANDWICH_OUT"))
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps = 20
    threads = int(os.environ.get("VESPER_CPU_THREADS", "12"))
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    print(f"[load] host  = {HOST_CKPT}", flush=True)
    host, tok, _mch = load_base_model(HOST_CKPT, device="cpu",
                                      target_profile="kda_mla")
    print(f"[load] donor = {DONOR_CKPT}", flush=True)
    donor, _tokd, _mcd = load_base_model(DONOR_CKPT, device="cpu",
                                         target_profile="kda_mla")
    assert host.tok_embeddings.num_embeddings == donor.tok_embeddings.num_embeddings, \
        "vocab mismatch — pair is not the vk+470m study bed"
    for p in list(host.parameters()) + list(donor.parameters()):
        p.requires_grad_(False)
    host.eval()
    donor.eval()
    pad_id = host.pad_id

    lib = build_library()
    flats = flat_probes()
    print(f"[lib] {len(lib)} calibration pairs, {len(flats)} held-out flat probes", flush=True)

    # ---------------- capture + routing audit ----------------
    hb_host = add_capture_hooks(host, LAYER)
    hb_donor = add_capture_hooks(donor, LAYER)
    audit = lib if not args.smoke else lib[:6]
    t_cap = time.time()
    hh_all, share_h = capture(host, tok, audit, LAYER, hb_host, pad_id)
    hd_all, share_d = capture(donor, tok, audit, LAYER, hb_donor, pad_id)
    print(f"[capture] {hh_all.size(0)} tokens x2 models in {time.time()-t_cap:.0f}s", flush=True)
    e_h = int(share_h.argmax())
    e_d = int(share_d.argmax())
    shares_h = [round(float(v), 4) for v in share_h]
    shares_d = [round(float(v), 4) for v in share_d]
    print(f"[route] host slot {e_h} (share {share_h[e_h]:.3f}) <- donor slot {e_d} "
          f"(donor-side share {share_d[e_d]:.3f}); host shares {shares_h}", flush=True)

    # ---------------- mismatch windows ----------------
    cos = F.cosine_similarity(hh_all, hd_all, dim=-1)
    mism = (1.0 - cos).tolist()
    med = sorted(mism)[len(mism) // 2]
    keep = torch.tensor([m >= med for m in mism], dtype=torch.bool)
    pool = (hh_all[keep], hd_all[keep])
    n_hold = max(1, pool[0].size(0) // 5)
    hold_idx = torch.randperm(pool[0].size(0), generator=torch.Generator().manual_seed(3))[:n_hold]
    mask = torch.ones(pool[0].size(0), dtype=torch.bool)
    mask[hold_idx] = False
    train_idx = mask.nonzero(as_tuple=True)[0]
    host_expert = host.layers[LAYER]["ffn"].experts[e_h]
    donor_expert_w = donor.layers[LAYER]["ffn"].experts[e_d]
    import copy as _copy
    donor_expert = _copy.deepcopy(donor_expert_w)   # detach from donor graph
    with torch.no_grad():
        yref_all = host_expert(hh_all)
    pool_train = (hh_all[keep][train_idx], hd_all[keep][train_idx], yref_all[keep][train_idx])
    pool_hold = (hh_all[keep][hold_idx], hd_all[keep][hold_idx], yref_all[keep][hold_idx])
    print(f"[pool] mismatch stats {q(mism)} | train {pool_train[0].size(0)} "
          f"hold {pool_hold[0].size(0)} tokens", flush=True)

    # ---------------- donor-class fact ranking ----------------
    fact_cands = [pr for pr in lib if pr[0].startswith(("probe_t/", "teach/"))]
    d_ce = ce_per_pair(donor, tok, fact_cands, pad_id)
    h_ce = ce_per_pair(host, tok, fact_cands, pad_id)
    order = sorted(range(len(fact_cands)), key=lambda i: d_ce[i] - h_ce[i])
    facts = [fact_cands[i] for i in order[:3]]
    fact_margins = [(fact_cands[i][0], round(d_ce[i] - h_ce[i], 4)) for i in order[:3]]
    print(f"[facts] donor-class set = {[pr[0] for pr in facts]} margins {fact_margins}", flush=True)

    # ---------------- naive transplant reference (Door-9 shape) ----------
    naive = SandwichExpert(_copy.deepcopy(donor_expert_w), hh_all.size(1))
    variants = {}

    def install(sw):
        host.layers[LAYER]["ffn"].experts[e_h] = sw

    def restore():
        host.layers[LAYER]["ffn"].experts[e_h] = host_expert

    def eval_variant(name, sw):
        install(sw if sw is not None else host_expert)
        res = {"flat": {}, "facts_ce": None,
               "route_flat": routed_share(host, tok, flats, LAYER, e_h, hb_host, pad_id),
               "route_facts": routed_share(host, tok, facts, LAYER, e_h, hb_host, pad_id)}
        for i, pr in enumerate(flats):
            res["flat"][pr[0]] = round(ce_pairs(host, tok, [pr], pad_id)[0], 6)
        res["facts_ce"] = round(ce_pairs(host, tok, facts, pad_id)[0], 6)
        restore()
        variants[name] = res
        print(f"[eval/{name}] flat={res['flat']} facts={res['facts_ce']} "
              f"route_flat={res['route_flat']:.3f} route_facts={res['route_facts']:.3f}",
              flush=True)

    eval_variant("bare", None)
    eval_variant("naive", naive)

    # ---------------- arm A: both sandwiches ----------------
    t_tr = time.time()
    sw_a, hist_a = train_arm(pool_train, _copy.deepcopy(donor_expert_w), hh_all.size(1),
                             SEED_A, args.steps, LR, train_out_only=False)
    stats_a = fit_stats(sw_a, pool_hold)
    print(f"[train/A] {args.steps} steps in {time.time()-t_tr:.0f}s; "
          f"hist {[(s, round(m, 4), round(r, 4)) for s, m, r in hist_a]}; hold={ {k: round(v, 4) for k, v in stats_a.items()} }",
          flush=True)
    eval_variant("armA", sw_a)

    # ---------------- arm F (the one pre-declared fallback): OUT only ----
    t_tr = time.time()
    sw_f, hist_f = train_arm(pool_train, _copy.deepcopy(donor_expert_w), hh_all.size(1),
                             SEED_F, args.steps, LR, train_out_only=True)
    stats_f = fit_stats(sw_f, pool_hold)
    print(f"[train/F] {args.steps} steps in {time.time()-t_tr:.0f}s; "
          f"hist {[(s, round(m, 4), round(r, 4)) for s, m, r in hist_f]}; hold={ {k: round(v, 4) for k, v in stats_f.items()} }",
          flush=True)
    eval_variant("armF", sw_f)

    # ---------------- bars / verdict (arm A is the proposition) ----------
    bare = variants["bare"]
    armA = variants["armA"]
    d_flat = {k: armA["flat"][k] - bare["flat"][k] for k in bare["flat"]}
    d_facts = armA["facts_ce"] - bare["facts_ce"]
    bar_a = all(v <= BAR_A_MAX_NAT for v in d_flat.values())
    bar_b = (d_facts >= BAR_B_MIN_NAT) and (armA["route_facts"] >= BAR_ROUTE_MIN)
    if armA["route_facts"] < BAR_ROUTE_MIN:
        fail_type = "TYPE-4 routing-starve"
    elif not bar_a:
        fail_type = "TYPE-2 parity-broken"
    elif not bar_b and stats_a["rho"] < 0.2:
        fail_type = "TYPE-1 mimic-collapse"
    elif not bar_b:
        fail_type = "TYPE-3 inert-unselective"
    else:
        fail_type = None
    verdict = "PASS (feasibility: parity + live foreign core)" if (bar_a and bar_b) else f"FAIL — {fail_type}"

    d_flat_f = {k: variants["armF"]["flat"][k] - bare["flat"][k] for k in bare["flat"]}
    d_facts_f = variants["armF"]["facts_ce"] - bare["facts_ce"]
    f_a = all(v <= BAR_A_MAX_NAT for v in d_flat_f.values())
    f_b = (d_facts_f >= BAR_B_MIN_NAT) and (variants["armF"]["route_facts"] >= BAR_ROUTE_MIN)
    print(f"VERDICT ARM-A | bars A(parity)={'PASS' if bar_a else 'FAIL'} "
          f"(max d_flat {max(d_flat.values()):+.4f})  "
          f"B(liveness)={'PASS' if bar_b else 'FAIL'} (d_facts {d_facts:+.4f}, "
          f"route {armA['route_facts']:.3f}) -> {verdict}", flush=True)
    print(f"VERDICT ARM-F (OUT-only fallback) | A={'PASS' if f_a else 'FAIL'} "
          f"(max d_flat {max(d_flat_f.values()):+.4f})  B={'PASS' if f_b else 'FAIL'} "
          f"(d_facts {d_facts_f:+.4f})", flush=True)

    out = {
        "config": {"layer": LAYER, "lr": LR, "steps": args.steps, "batch": BATCH,
                   "lam_in": LAM_IN, "seeds": {"A": SEED_A, "F": SEED_F},
                   "host_slot": e_h, "donor_slot": e_d, "bars": {
                       "A_max_nat": BAR_A_MAX_NAT, "B_min_nat": BAR_B_MIN_NAT,
                       "route_min": BAR_ROUTE_MIN}},
        "n_calib_pairs": len(lib), "n_calib_tokens": int(hh_all.size(0)),
        "host_shares": shares_h, "donor_shares": shares_d,
        "mismatch": q(mism),
        "pool": {"train": int(pool_train[0].size(0)), "hold": int(pool_hold[0].size(0))},
        "fact_set": [pr[0] for pr in facts], "fact_margins": fact_margins,
        "variants": variants,
        "armA": {"hist": hist_a, "hold": stats_a,
                 "d_flat": d_flat, "d_facts": d_facts,
                 "bar_a": bar_a, "bar_b": bar_b},
        "armF": {"hist": hist_f, "hold": stats_f,
                 "d_flat": d_flat_f, "d_facts": d_facts_f,
                 "bar_a": f_a, "bar_b": f_b},
        "verdict": verdict,
        "wall_s": round(time.time() - t0, 1),
    }
    out_path = os.path.join(args.out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"results json: {out_path}", flush=True)


if __name__ == "__main__":
    main()
