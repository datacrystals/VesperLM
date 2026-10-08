#!/usr/bin/env python3
"""
validate.py — post-surgery validation for Growth/expand_experts.py.

Runs on CPU (GPUs are busy with training; nothing here touches CUDA or the
running jobs). Proves, against the pre-expansion checkpoint:

  (a) every expanded checkpoint strict-loads into
      VesperLinearLM(**expanded_config) + load_state_dict(strict=True);
  (b) with --exact-upcycle and noise=0 the expanded model's logits match the
      original model's logits within tolerance on a fixed prompt — the parent's
      routing weight is split exactly 50/50 between parent and clone, so the
      MoE FFN function is preserved (also checked per-layer, and the router
      split itself is measured);
  (c) same construction with noise>0: logits are close but not identical
      (the only change is the gaussian noise on the cloned expert weights);
  (d) a short greedy generation runs on an expanded model. With --upcycled
      (the production sparse-upcycling run: noisy clones, top_k unchanged) the
      sample comes from that checkpoint and its drift vs the parent is reported
      for context.

CPU shims for fla (GLA chunk/recurrent kernels and the fused gated RMSNorm are
Triton-only) are copied verbatim from Pretrain/cpu_probe.py — that file is left
untouched. One environment fix is applied first: the venv's triton 3.1.0
currently shadows the user-site triton 3.4.0 that fla 0.6.0 needs (see README
"Environment gotcha"); we prefer the user-site triton for this process only.

Example
-------
    python validate.py \
        --parent ../SFT/sft_checkpoints_118m_v1/step_2900/checkpoint.pt \
        --exact  runs/118m_4to8_exact/checkpoint.pt \
        --noisy  runs/118m_4to8_exact_noisy/checkpoint.pt \
        --upcycled runs/118m_4to8_noise/checkpoint.pt
"""

import argparse
import inspect
import os
import sys
import time
from collections import Counter

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
os.environ.setdefault("FLA_CACHE_DIR", "/tmp/fla_cache")

_GROWTH_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_GROWTH_DIR)
sys.path.insert(0, os.path.join(_REPO, "Common"))

# ---------------------------------------------------------------------------
# Environment fix (this process only): fla 0.6.0 (user site-packages) calls
# triton's Autotuner(do_bench=...), which triton < 3.3 rejects. The venv's
# triton 3.1.0 sits earlier on sys.path than the user-site triton 3.4.0, so
# `import triton` resolves too old and `import fla` dies. Prefer the user-site
# triton when the resolved one is too old. Do NOT touch the venv or the
# running training — this is a per-process sys.path preference.
# ---------------------------------------------------------------------------
def _prefer_fla_compatible_triton():
    def _ver(v):
        out = []
        for part in v.split("+")[0].split(".")[:3]:
            try:
                out.append(int("".join(c for c in part if c.isdigit()) or 0))
            except ValueError:
                out.append(0)
        while len(out) < 3:
            out.append(0)
        return tuple(out)

    resolved_old = False
    try:
        import triton
        resolved_old = _ver(getattr(triton, "__version__", "0")) < (3, 3, 0)
    except Exception:
        resolved_old = True
    if not resolved_old:
        return
    import site
    user_site = site.getusersitepackages()
    if not os.path.isdir(os.path.join(user_site, "triton")):
        return
    for m in [m for m in list(sys.modules) if m == "triton" or m.startswith("triton.")]:
        del sys.modules[m]
    if user_site not in sys.path:
        sys.path.insert(0, user_site)
    import triton  # noqa: F401  (re-resolve)
    print(f"[validate] using triton {triton.__version__} from {triton.__file__}")


_prefer_fla_compatible_triton()

# ---------------------------------------------------------------------------
# CPU shims for GLA (must happen before model forward; import is fine).
# Copied verbatim from Pretrain/cpu_probe.py — that file is not modified.
# GLA's Triton kernels don't run on CPU, so shim fla's chunk/recurrent entry
# points with the naive pure-torch recurrence (state transposed to match
# state_v_first=True).
# ---------------------------------------------------------------------------
import fla.layers.gla as _fla_gla
from fla.ops.gla.naive import naive_recurrent_gla


def _cpu_gla_shim(q=None, k=None, v=None, g=None, gk=None,
                  initial_state=None, output_final_state=False,
                  state_v_first=True, cu_seqlens=None, **kw):
    gate = gk if gk is not None else g
    h_in = initial_state
    if h_in is not None and state_v_first:
        h_in = h_in.transpose(-1, -2).contiguous()
    o, h = naive_recurrent_gla(q, k, v, gate, initial_state=h_in,
                               output_final_state=output_final_state)
    if h is not None and state_v_first:
        h = h.transpose(-1, -2).contiguous()
    return o, h


_fla_gla.chunk_gla = _cpu_gla_shim
_fla_gla.fused_recurrent_gla = _cpu_gla_shim

# CPU shim for fla's fused gated RMSNorm (Triton-only; GLA fuse_norm=True).
# Kernel semantics (fla/modules/fused_norm_gate.py): fp32 internally,
# y = rmsnorm(x) * weight * (g * sigmoid(g))  [norm first, then swish gate]
import fla.modules.fused_norm_gate as _fng


def _rms_norm_gated_torch(x, g, weight, bias=None, activation="swish",
                          residual=None, prenorm=False, residual_in_fp32=False,
                          eps=1e-6, **kw):
    dtype = x.dtype
    xf = x.float()
    if residual is not None:
        xf = xf + residual.float()
    res_out = xf if prenorm else None
    var = (xf * xf).mean(-1, keepdim=True)
    y = xf * torch.rsqrt(var + eps)
    if weight is not None:
        y = y * weight.float()
    if bias is not None:
        y = y + bias.float()
    gf = g.float()
    if activation in ("swish", "silu"):
        y = y * gf * torch.sigmoid(gf)
    elif activation == "sigmoid":
        y = y * torch.sigmoid(gf)
    y = y.to(dtype)
    return (y, res_out.to(x.dtype)) if prenorm else y


_fng.rms_norm_gated = _rms_norm_gated_torch
_fng.layer_norm_gated = _rms_norm_gated_torch

import torch  # noqa: E402  (used by the shims above via late binding)

from vesper_linear_model import VesperLinearLM  # noqa: E402


# --------------------------------------------------------------- loading

def load_checkpoint(path):
    if os.path.isdir(path):
        path = os.path.join(path, "checkpoint.pt")
    if not os.path.isfile(path):
        raise SystemExit(f"checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False), path


def config_for(ckpt):
    cfg = dict(ckpt.get("model_config") or {})
    sd = ckpt["model"]
    emb = sd.get("tok_embeddings.weight", sd.get("module.tok_embeddings.weight"))
    if emb is not None and "vocab_size" not in cfg:
        cfg["vocab_size"] = int(emb.shape[0])
    return cfg


def build_model(ckpt):
    """VesperLinearLM(**expanded_config) + load_state_dict(strict=True)."""
    cfg = config_for(ckpt)
    ctor = set(inspect.signature(VesperLinearLM.__init__).parameters)
    kwargs = {k: v for k, v in cfg.items() if k in ctor}
    model = VesperLinearLM(**kwargs)
    state = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg


# --------------------------------------------------------------- helpers

@torch.no_grad()
def model_logits(model, input_ids):
    return model(input_ids)[0]


def diff_stats(a, b):
    d = (a - b).abs()
    denom = a.abs().clamp_min(1e-6)
    cos = torch.nn.functional.cosine_similarity(
        a.reshape(1, -1).float(), b.reshape(1, -1).float(), dim=-1
    ).item()
    return {
        "max_abs": d.max().item(),
        "mean_abs": d.mean().item(),
        "max_rel": (d / denom).max().item(),
        "cosine": cos,
        "argmax_match": (a.argmax(-1) == b.argmax(-1)).float().mean().item(),
    }


def fmt_stats(s):
    return (f"max|Δ|={s['max_abs']:.3e}  mean|Δ|={s['mean_abs']:.3e}  "
            f"max_rel={s['max_rel']:.3e}  cosine={s['cosine']:.8f}  "
            f"top1_match={s['argmax_match'] * 100:.1f}%")


@torch.no_grad()
def moe_layer_check(parent, exp, seed=1234, tokens=16, batch=2):
    """Per-layer MoE FFN check: same input x through each layer's ffn on both
    models. Independent of attention/norm/head stack. Returns per-layer
    (max_abs_diff, rel_diff) — rel normalizes by max|parent out| because this
    model's residual stream is large (MoE outputs ~1e3-1e4)."""
    dim = parent.tok_embeddings.weight.shape[1]
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, tokens, dim, generator=g)
    out = []
    for i, (lp, le) in enumerate(zip(parent.layers, exp.layers)):
        op, _ = lp["ffn"](x)
        oe, _ = le["ffn"](x)
        scale = op.abs().max().item() or 1.0
        out.append((i, (op - oe).abs().max().item(), (op - oe).abs().max().item() / scale))
    return out


def fmt_layer_diffs(layer_diffs):
    return ", ".join(f"L{i}:{a:.2e}(rel {r:.1e})" for i, a, r in layer_diffs)


@torch.no_grad()
def router_split_check(parent, exp, parent_map, old_e, seed=7, n_tokens=32):
    """Measure the routing-weight split on random tokens.

    For exact-upcycle the expanded model's top-(2k) selection should be the
    two copies of each original top-k expert, and each copy's renormalized
    weight should be exactly half of the parent's — i.e. the parent's routing
    weight is split 50/50 between parent and clone. Returns max deviations.
    """
    dim = parent.tok_embeddings.weight.shape[1]
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_tokens, dim, generator=g)

    max_half_err = 0.0   # |w_clone - w_parent / 2|
    max_sum_err = 0.0    # |sum of pair weights - w_parent|
    multiset_ok = True
    for i, (lp, le) in enumerate(zip(parent.layers, exp.layers)):
        prefix = f"layers.{i}.ffn"
        if prefix not in parent_map:
            multiset_ok = False
            continue
        gp = lp["ffn"].router.gate.weight
        ge = le["ffn"].router.gate.weight
        kp = lp["ffn"].router.top_k
        ke = le["ffn"].router.top_k

        def topk_renorm(gate, k):
            probs = torch.softmax(x @ gate.T, dim=-1)
            w, idx = torch.topk(probs, k, dim=-1)
            return w / w.sum(-1, keepdim=True), idx

        wp, ip = topk_renorm(gp, kp)
        we, ie = topk_renorm(ge, ke)

        lookup = []
        for e in range(ge.shape[0]):
            if e < old_e:
                lookup.append(e)
            else:
                p = parent_map[prefix][e]
                lookup.append(p if isinstance(p, int) else -1)

        for t in range(n_tokens):
            if any(lookup[e] < 0 for e in ie[t].tolist()):
                multiset_ok = False
                continue
            buckets = torch.zeros(old_e)
            for s in range(ke):
                e = int(ie[t, s])
                pid = lookup[e]
                buckets[pid] += we[t, s]
                hits = (ip[t] == pid).nonzero()
                if len(hits) == 0:
                    multiset_ok = False
                else:
                    max_half_err = max(max_half_err,
                                       abs(float(we[t, s]) - float(wp[t, hits[0]]) / 2))
            for s in range(kp):
                pid = int(ip[t, s])
                max_sum_err = max(max_sum_err, abs(float(buckets[pid]) - float(wp[t, s])))
            # multiset: each selected parent must appear ke/kp times in the
            # expanded top-k (parent + its clone(s) enter together)
            cnt_exp = Counter(lookup[int(e)] for e in ie[t])
            cnt_par = Counter()
            for s in range(kp):
                cnt_par[int(ip[t, s])] += ke // kp
            if cnt_exp != cnt_par:
                multiset_ok = False
    return max_half_err, max_sum_err, multiset_ok


@torch.no_grad()
def greedy_generate(model, tokenizer, prompt, max_new, device="cpu"):
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    caches = model.new_cache(1, torch.device(device))
    logits = model.forward_incremental(ids, caches, 0)[0][0, -1]
    pos = ids.shape[1]
    out = []
    for _ in range(max_new):
        nxt = int(logits.argmax())
        out.append(nxt)
        if tokenizer.eos_token_id is not None and nxt == tokenizer.eos_token_id:
            break
        logits = model.forward_incremental(
            torch.tensor([[nxt]]), caches, pos)[0][0, -1]
        pos += 1
    return tokenizer.decode(out)


# --------------------------------------------------------------- checks

def check_strict_load(name, path):
    print(f"\n=== (a) strict load: {name} ===")
    print(f"  file: {path}")
    try:
        t0 = time.time()
        model, cfg = build_model(load_checkpoint(path)[0])
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  VesperLinearLM(**config) + load_state_dict(strict=True): OK "
              f"({n_params / 1e6:.1f}M params, {time.time() - t0:.1f}s)")
        print(f"  config: {cfg}")
        return model, cfg, True
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        return None, None, False


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--parent", required=True, help="pre-expansion checkpoint")
    ap.add_argument("--exact", default=None,
                    help="expanded checkpoint: --exact-upcycle --noise-std 0 (test b)")
    ap.add_argument("--noisy", default=None,
                    help="expanded checkpoint: --exact-upcycle --noise-std > 0 (test c)")
    ap.add_argument("--upcycled", default=None,
                    help="optional production sparse-upcycling checkpoint "
                         "(--mode clone, noise > 0, top_k unchanged): strict load, "
                         "drift report, and the (d) generation sample")
    ap.add_argument("--prompt", default="The capital of France is Paris. "
                                        "The capital of Japan is Tokyo. "
                                        "The capital of Italy is")
    ap.add_argument("--max-new", type=int, default=24, help="greedy generation length")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="max |logit Δ| accepted for the exact-upcycle match (b)")
    ap.add_argument("--close-tol", type=float, default=1.0,
                    help="max |logit Δ| accepted as 'close' for the noisy model (c)")
    ap.add_argument("--close-cosine", type=float, default=0.999,
                    help="min cosine similarity accepted as 'close' (c)")
    ap.add_argument("--no-generate", action="store_true", help="skip (d)")
    args = ap.parse_args()
    if not args.exact and not args.noisy and not args.upcycled:
        raise SystemExit("give at least one of --exact / --noisy / --upcycled")

    torch.set_num_threads(16)
    torch.manual_seed(0)
    results = []

    from transformers import PreTrainedTokenizerFast
    tok_dir = os.path.join(_REPO, "Pretrain", "custom_tokenizer")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tok_dir)
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    print(f"[validate] prompt ({input_ids.shape[1]} tokens): {args.prompt!r}")

    parent_model, parent_cfg, ok = check_strict_load("parent", args.parent)
    results.append(("parent strict load", ok))
    if not ok:
        sys.exit(1)

    exact_model = noisy_model = upcycled_model = None
    if args.exact:
        exact_model, _, ok = check_strict_load("expanded exact-upcycle (noise=0)", args.exact)
        results.append(("expanded (exact) strict load", ok))
    if args.noisy:
        noisy_model, _, ok = check_strict_load("expanded exact-upcycle (noise>0)", args.noisy)
        results.append(("expanded (noisy exact) strict load", ok))
    if args.upcycled:
        upcycled_model, _, ok = check_strict_load("expanded sparse-upcycle (production)", args.upcycled)
        results.append(("expanded (upcycled) strict load", ok))

    # ---------------------------------------------------------- (b) exact
    if args.exact and exact_model is not None:
        print("\n=== (b) exact-upcycle: function preservation (noise=0) ===")
        prov = load_checkpoint(args.exact)[0].get("expansion_provenance") or {}
        old_e = prov.get("num_experts_before", parent_cfg.get("num_experts"))
        print(f"  mode={prov.get('mode')} noise_std={prov.get('noise_std')} "
              f"exact_upcycle={prov.get('exact_upcycle')} "
              f"top_k {prov.get('top_k_before')} -> {prov.get('top_k_after')}")
        ok_exact_flag = prov.get("exact_upcycle") is True
        results.append(("provenance marks exact_upcycle", ok_exact_flag))

        # structural: new gate rows are bit-exact copies of parents
        row_ok = True
        for lp, le in zip(parent_model.layers, exact_model.layers):
            gp = lp["ffn"].router.gate.weight
            ge = le["ffn"].router.gate.weight
            if not torch.equal(ge[: gp.shape[0]], gp):
                row_ok = False
        print(f"  parent gate rows preserved bit-exact: {row_ok}")
        results.append(("parent gate rows preserved", row_ok))

        # router split measurement
        pmap = prov.get("parent_map") or {}
        if pmap:
            half_err, sum_err, mset_ok = router_split_check(
                parent_model, exact_model, pmap, old_e)
            print(f"  routing split 50/50 |w_clone - w_parent/2| max = {half_err:.3e}")
            print(f"  pair weights sum to parent weight, max err = {sum_err:.3e}")
            print(f"  top-k multiset = parent x (top_k_after/top_k_before): {mset_ok}")
            results.append(("router split 50/50 (halving err < 1e-5)", half_err < 1e-5))
            results.append(("router pair sum err < 1e-5", sum_err < 1e-5))

        # per-layer MoE function identity (relative to output scale)
        layer_diffs = moe_layer_check(parent_model, exact_model)
        worst_abs = max(d[1] for d in layer_diffs)
        worst_rel = max(d[2] for d in layer_diffs)
        print(f"  per-layer MoE FFN on shared random input: {fmt_layer_diffs(layer_diffs)}")
        results.append((f"per-layer MoE identity (worst rel {worst_rel:.2e} < 1e-5; "
                        f"abs {worst_abs:.2e})", worst_rel < 1e-5))

        # full-model logits on the fixed prompt
        with torch.no_grad():
            la = model_logits(parent_model, input_ids)
            lb = model_logits(exact_model, input_ids)
        s = diff_stats(la, lb)
        print(f"  logits parent vs expanded: {fmt_stats(s)}")
        ok_b = s["max_abs"] < args.tol
        print(f"  PASS (b): {ok_b}  (tolerance max|Δ| < {args.tol:g})")
        results.append((f"exact-upcycle logits within tol (max|Δ|={s['max_abs']:.3e})", ok_b))

    # ---------------------------------------------------------- (c) noisy
    if args.noisy and noisy_model is not None:
        print("\n=== (c) exact-upcycle + noise > 0: close but not identical ===")
        prov = load_checkpoint(args.noisy)[0].get("expansion_provenance") or {}
        print(f"  mode={prov.get('mode')} noise_std={prov.get('noise_std')} "
              f"exact_upcycle={prov.get('exact_upcycle')} "
              f"top_k {prov.get('top_k_before')} -> {prov.get('top_k_after')}")
        print("  (same construction as (b); the only change is the gaussian "
              "noise on cloned expert weights)")
        with torch.no_grad():
            la = model_logits(parent_model, input_ids)
            lb = model_logits(noisy_model, input_ids)
        s = diff_stats(la, lb)
        print(f"  logits parent vs expanded: {fmt_stats(s)}")
        not_identical = s["max_abs"] > 0.0
        close = s["max_abs"] < args.close_tol and s["cosine"] > args.close_cosine
        print(f"  not identical: {not_identical}   close: {close} "
              f"(max|Δ| < {args.close_tol:g} and cosine > {args.close_cosine:g})")
        results.append(("noisy logits differ from parent", not_identical))
        results.append((f"noisy logits close (max|Δ|={s['max_abs']:.3e}, "
                        f"cos={s['cosine']:.6f})", close))

        layer_diffs = moe_layer_check(parent_model, noisy_model)
        worst_abs = max(d[1] for d in layer_diffs)
        print(f"  per-layer MoE FFN max|Δ|: {fmt_layer_diffs(layer_diffs)}")
        results.append((f"noisy per-layer MoE perturbation visible "
                        f"(worst abs {worst_abs:.2e})", worst_abs > 0.0))

    # --------------------------------------------- production upcycle drift
    if upcycled_model is not None:
        print("\n=== production sparse-upcycle drift (informational) ===")
        prov = load_checkpoint(args.upcycled)[0].get("expansion_provenance") or {}
        print(f"  mode={prov.get('mode')} noise_std={prov.get('noise_std')} "
              f"exact_upcycle={prov.get('exact_upcycle')} "
              f"top_k {prov.get('top_k_before')} -> {prov.get('top_k_after')}")
        print("  NOTE: with top_k unchanged and near-duplicate router rows, top-k "
              "selection pairs each expert with its clone and crowds out the second\n"
              "  original expert (see README gotchas). The drift below is expected "
              "to exceed (c) and is exactly what the routing warmup fixes.")
        with torch.no_grad():
            la = model_logits(parent_model, input_ids)
            lb = model_logits(upcycled_model, input_ids)
        s = diff_stats(la, lb)
        print(f"  logits parent vs upcycled: {fmt_stats(s)}")
        layer_diffs = moe_layer_check(parent_model, upcycled_model)
        print(f"  per-layer MoE FFN max|Δ|: {fmt_layer_diffs(layer_diffs)}")

    # ---------------------------------------------------------- (d) greedy
    if not args.no_generate:
        print("\n=== (d) short greedy generation ===")
        gen_model = upcycled_model or noisy_model or exact_model
        gen_name = ("upcycled (production)" if upcycled_model is not None
                    else "exact+noise" if noisy_model is not None else "exact")
        t0 = time.time()
        out = greedy_generate(gen_model, tokenizer, args.prompt, args.max_new)
        print(f"  [{gen_name}] prompt: {args.prompt!r}")
        print(f"  [{gen_name}] greedy +{args.max_new} ({time.time() - t0:.1f}s):")
        for line in out.splitlines() or [out]:
            print(f"    {line}")
        results.append(("greedy generation produced text", len(out) > 0))
        out_p = greedy_generate(parent_model, tokenizer, args.prompt, args.max_new)
        print(f"  [parent] greedy +{args.max_new}:")
        for line in out_p.splitlines() or [out_p]:
            print(f"    {line}")
        if noisy_model is not None:
            out_n = greedy_generate(noisy_model, tokenizer, args.prompt, args.max_new)
            print(f"  [exact+noise] greedy +{args.max_new}:")
            for line in out_n.splitlines() or [out_n]:
                print(f"    {line}")

    print("\n=== summary ===")
    all_ok = True
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        all_ok &= ok
    print(f"\n[validate] {'ALL CHECKS PASSED' if all_ok else 'FAILURES PRESENT'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
