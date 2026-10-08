#!/usr/bin/env python3
"""
expand_experts.py — Expansion A ("sparse upcycling") for VesperLM MoE checkpoints.

Grows a trained MoE by cloning experts (+ small gaussian noise) and extending the
router, so the model can gain capacity in a short continued-training run instead
of a fresh pretrain (see LMBUS_DESIGN.md, "Expansion A — expert add").

MoE state-dict layout (from Common/vesper_model.py, MoEFeedForward/TopKRouter):

    layers.{L}.ffn.experts.{E}.w1.weight   (hidden_dim, dim)   # gate proj
    layers.{L}.ffn.experts.{E}.w2.weight   (dim, hidden_dim)   # down proj
    layers.{L}.ffn.experts.{E}.w3.weight   (hidden_dim, dim)   # up proj
    layers.{L}.ffn.router.gate.weight      (num_experts, dim)  # bias-free nn.Linear

Router math (TopKRouter.forward): logits = x @ gate.weight.T -> softmax ->
top_k -> renormalize selected weights so they sum to 1. The softmax normalizer
cancels under that renormalization; only the selected set and the relative
logits inside it matter.

Modes
-----
clone        new experts = copies of existing ones (round-robin over the
             most-used experts when usage stats are given, else over all),
             plus gaussian noise.
clone_split  new experts = (expert_i + expert_j)/2 weight-space mixtures for
             diversity, plus gaussian noise.

Router extension (default): each new gate row is a copy of its parent expert's
row (+ noise), so routing to new experts starts near their parents'.

--exact-upcycle (function-preserving surgery check, --mode clone only):
  The literal "halve the parent's gate row and give the other half to the
  clone" recipe is NOT function-preserving under softmax->topk->renorm: with
  near-duplicate rows, top_k selects both slots from the single best parent
  (crowding out the second original expert) and the FFN output changes.
  The exact realization of "split the parent's routing weight 50/50 with its
  clone" in this architecture is:
      * expert weights of the clone = bit-exact copy of the parent
      * gate row of the clone = bit-exact copy of the parent's row
        (=> identical logits => softmax probability of each duplicate is
        exactly half of the parent's original probability)
      * top_k doubled, so the two copies of each selected parent enter together.
  Renormalized weights then become (a/2, a/2, b/2, b/2) and the FFN output is
  identical to the original. Verified by Growth/validate.py check (b).

Optimizer state ('optimizer' / 'muon_state' / 'adamw_state' / 'scaler') is
always reset to None: those tensors are shaped per-parameter and cannot be
carried across the shape change. The trainer re-creates the optimizer on
resume. Provenance is recorded under 'expansion_provenance'.

Example
-------
    python expand_experts.py \
        --input  ../SFT/sft_checkpoints_118m_v1/step_2900 \
        --output runs/118m_4to8_exact \
        --num-experts 8 --mode clone --noise-std 0 --exact-upcycle

    python expand_experts.py \
        --input  ../SFT/sft_checkpoints_118m_v1/step_2900 \
        --output runs/118m_4to8_noise \
        --num-experts 8 --mode clone --noise-std 1e-3
"""

import argparse
import datetime
import inspect
import json
import os
import re
import sys

import torch

TOOL_VERSION = "1.0"

# keys whose values are optimizer/scaler state and are always reset
OPTIMIZER_STATE_KEYS = ("optimizer", "muon_state", "adamw_state", "scaler")

# layers.{L}.ffn.experts.{E}.w{1,2,3}.weight
EXPERT_RE = re.compile(r"^(?P<prefix>.+)\.experts\.(?P<eid>\d+)\.(?P<m>w[123])\.weight$")
# layers.{L}.ffn.router.gate.weight
ROUTER_RE = re.compile(r"^(?P<prefix>.+)\.router\.gate\.weight$")


# ---------------------------------------------------------------- discovery

def discover_structure(state):
    """Return {prefix: {"experts": {eid: {"w1":key,"w2":key,"w3":key}},
                        "router": router_key}} for every MoE FFN in the
    state dict. `prefix` is e.g. 'layers.0.ffn'."""
    moes = {}
    for key in state:
        m = EXPERT_RE.match(key)
        if m:
            per_e = moes.setdefault(m.group("prefix"), {"experts": {}, "router": None})
            per_e["experts"].setdefault(int(m.group("eid")), {})[m.group("m")] = key
    for key in state:
        m = ROUTER_RE.match(key)
        if m and m.group("prefix") in moes:
            moes[m.group("prefix")]["router"] = key

    if not moes:
        raise SystemExit(
            "No MoE experts found in state dict (looked for '*.experts.N.w{1,2,3}.weight')."
        )
    for prefix, info in sorted(moes.items()):
        if info["router"] is None:
            raise SystemExit(f"MoE block {prefix!r} has experts but no '*.router.gate.weight'.")
        eids = sorted(info["experts"])
        if eids != list(range(len(eids))):
            raise SystemExit(f"MoE block {prefix!r} has non-contiguous expert ids: {eids}")
        for eid in eids:
            missing = {"w1", "w2", "w3"} - set(info["experts"][eid])
            if missing:
                raise SystemExit(f"MoE block {prefix!r} expert {eid} missing {sorted(missing)}")
    return moes


def check_uniform(moes, state):
    """All MoE blocks must share the same expert count and gate width."""
    old_e = None
    dim = None
    for prefix, info in sorted(moes.items()):
        e = len(info["experts"])
        gate = state[info["router"]]
        if gate.dim() != 2 or gate.shape[0] != e:
            raise SystemExit(
                f"{prefix}: router gate shape {tuple(gate.shape)} != ({e}, dim)"
            )
        if old_e is None:
            old_e, dim = e, gate.shape[1]
        elif e != old_e or gate.shape[1] != dim:
            raise SystemExit(
                f"{prefix}: expert count/width {e}/{gate.shape[1]} != {old_e}/{dim}"
            )
    return old_e, dim


# ------------------------------------------------------------ parent choice

def parse_usage_stats(path, prefixes, old_e):
    """Normalize a usage-stats JSON into {prefix: [count_per_expert]}.

    Accepted shapes:
      [c0, c1, ...]                          global per-expert counts
      {"0": c0, "1": c1, ...}                same
      {"layers.0.ffn": {"0": c0, ...}, ...}  per-MoE-block
      {"layers.0.ffn": [c0, ...], ...}       per-MoE-block
    """
    with open(path) as f:
        raw = json.load(f)

    def as_counts(obj, where):
        if isinstance(obj, dict):
            try:
                return [float(obj[str(i)]) for i in range(old_e)]
            except KeyError as e:
                raise SystemExit(f"usage stats {where}: missing expert id {e}")
        if isinstance(obj, (list, tuple)):
            if len(obj) != old_e:
                raise SystemExit(f"usage stats {where}: len {len(obj)} != num_experts {old_e}")
            return [float(x) for x in obj]
        raise SystemExit(f"usage stats {where}: unsupported shape {type(obj).__name__}")

    per_layer = {}
    global_counts = None
    if isinstance(raw, (list, tuple)):
        global_counts = as_counts(raw, "<root>")
    elif isinstance(raw, dict):
        layer_like = [k for k in raw if k in prefixes]
        if layer_like:
            for k in layer_like:
                per_layer[k] = as_counts(raw[k], k)
        else:
            try:
                global_counts = as_counts(raw, "<root>")
            except SystemExit:
                raise SystemExit(
                    "usage stats: expected per-expert counts keyed by expert id, "
                    f"or per-block dicts keyed by one of {sorted(prefixes)[:3]} ..."
                )
    else:
        raise SystemExit(f"usage stats: unsupported root type {type(raw).__name__}")

    out = {}
    for p in prefixes:
        out[p] = per_layer.get(p, global_counts)
    return out


def parent_order(old_e, counts):
    """Round-robin order over experts: most-used first when counts are given,
    else plain 0..old_e-1. Ties break toward lower expert id."""
    if counts is None:
        return list(range(old_e))
    return sorted(range(old_e), key=lambda i: (-counts[i], i))


def plan_new_experts(mode, old_e, num_new, order):
    """Return list of length num_new; each entry is an int parent id (clone) or
    a (i, j) pair of parent ids (clone_split)."""
    plan = []
    for k in range(num_new):
        i = order[k % old_e]
        if mode == "clone":
            plan.append(i)
        else:
            if old_e < 2:
                raise SystemExit("clone_split needs at least 2 existing experts")
            j = order[(k + 1 + k // old_e) % old_e]
            if j == i:
                j = order[(k + 2) % old_e]
            plan.append((i, j))
    return plan


# ----------------------------------------------------------------- surgery

def add_noise(t, std, generator):
    if std <= 0:
        return t.clone()
    noise = torch.randn(t.shape, generator=generator, dtype=torch.float32)
    return t + noise.to(t.dtype) * std


def expand(args):
    in_path = args.input
    if os.path.isdir(in_path):
        in_path = os.path.join(in_path, "checkpoint.pt")
    if not os.path.isfile(in_path):
        raise SystemExit(f"input checkpoint not found: {in_path}")

    out_path = os.path.join(args.output, "checkpoint.pt")
    if os.path.exists(out_path) and not args.overwrite:
        raise SystemExit(f"{out_path} exists (pass --overwrite to replace)")

    print(f"[expand] loading {in_path}")
    ckpt = torch.load(in_path, map_location="cpu", weights_only=False)
    if "model" not in ckpt or "model_config" not in ckpt:
        raise SystemExit("checkpoint must contain 'model' and 'model_config'")
    if not isinstance(ckpt.get("model"), dict):
        raise SystemExit("checkpoint['model'] must be a state dict")

    # strip DDP 'module.' prefixes so the output is loadable by a bare model
    state = {}
    stripped = False
    for k, v in ckpt["model"].items():
        nk = k[len("module."):] if k.startswith("module.") else k
        stripped |= nk != k
        state[nk] = v
    print(f"[expand] state dict: {len(state)} tensors"
          + (" (stripped 'module.' prefixes)" if stripped else ""))

    moes = discover_structure(state)
    old_e, dim = check_uniform(moes, state)
    cfg = dict(ckpt["model_config"])
    cfg_e = cfg.get("num_experts")
    if cfg_e is not None and cfg_e != old_e:
        raise SystemExit(
            f"model_config['num_experts']={cfg_e} != experts found in weights ({old_e})"
        )
    num_new = args.num_experts - old_e
    if num_new <= 0:
        raise SystemExit(
            f"target num_experts={args.num_experts} must exceed current {old_e} "
            "(shrinking is a different tool)"
        )

    if args.exact_upcycle and args.mode != "clone":
        raise SystemExit("--exact-upcycle requires --mode clone (mixtures cannot preserve function)")
    if args.shared_expert:
        raise SystemExit(
            "--shared-expert is not supported: MoEFeedForward (Common/vesper_model.py) "
            "only has routed experts (ModuleList + TopKRouter) and its forward has no "
            "always-on dense path. Adding one means changing the module (new parameter + "
            "forward), which would break strict state-dict compatibility with "
            "VesperLinearLM. Skipped by design; see Growth/README.md."
        )

    # ---- router treatment ----
    # default: copy parent row(s) + noise.  exact-upcycle: bit-exact copies and
    # doubled top_k so each parent's renormalized weight splits 50/50 with its
    # clone (see module docstring).
    expert_noise = args.noise_std
    router_noise = 0.0 if args.exact_upcycle else args.noise_std
    top_k_old = cfg.get("top_k")
    top_k_new = top_k_old * 2 if args.exact_upcycle else top_k_old

    # ---- parent selection ----
    usage = None
    if args.usage_stats:
        usage = parse_usage_stats(args.usage_stats, sorted(moes), old_e)
        print(f"[expand] usage stats from {args.usage_stats}")
    orders = {p: parent_order(old_e, usage[p] if usage else None) for p in moes}
    plans = {p: plan_new_experts(args.mode, old_e, num_new, orders[p]) for p in moes}

    generator = torch.Generator().manual_seed(args.seed)
    parent_map = {}

    # ---- expert surgery ----
    new_state = dict(state)
    for prefix, info in sorted(moes.items()):
        mapping = {}
        for k, parents in enumerate(plans[prefix]):
            new_id = old_e + k
            if args.mode == "clone":
                src_ids = [parents]
            else:
                src_ids = list(parents)
            for mat in ("w1", "w2", "w3"):
                src_key = info["experts"][src_ids[0]][mat]
                t = state[src_key].clone()
                for extra in src_ids[1:]:
                    t.add_(state[info["experts"][extra][mat]])
                t.div_(len(src_ids))
                t = add_noise(t, expert_noise, generator)
                new_state[f"{prefix}.experts.{new_id}.{mat}.weight"] = t
            mapping[new_id] = parents if args.mode == "clone" else list(parents)
        parent_map[prefix] = mapping

        # ---- router surgery ----
        gate_key = info["router"]
        gate = state[gate_key]  # (old_e, dim)
        assert gate.shape == (old_e, dim)
        rows = []
        for k, parents in enumerate(plans[prefix]):
            if args.mode == "clone":
                row = gate[parents].clone()
            else:
                row = (gate[parents[0]] + gate[parents[1]]) / 2
            rows.append(add_noise(row, router_noise, generator))
        new_gate = torch.cat([gate.clone()] + [r.unsqueeze(0) for r in rows], dim=0)
        new_state[gate_key] = new_gate

        n_show = min(3, num_new)
        shown = {old_e + k: parent_map[prefix][old_e + k] for k in range(n_show)}
        print(f"[expand] {prefix}: experts {old_e} -> {old_e + num_new}, "
              f"router rows {tuple(gate.shape)} -> {tuple(new_gate.shape)}, "
              f"new->parent {shown}"
              + (" ..." if num_new > n_show else ""))

    # ---- model_config ----
    model_dict = new_state
    emb = model_dict.get("tok_embeddings.weight")
    if emb is not None and "vocab_size" not in cfg:
        cfg["vocab_size"] = int(emb.shape[0])
    cfg["num_experts"] = old_e + num_new
    if args.exact_upcycle:
        cfg["top_k"] = top_k_new

    # VesperLinearLM(**expanded_config) must work: keep only constructor kwargs
    # (stash anything else in provenance instead of silently losing it).
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Common"))
    try:
        from vesper_linear_model import VesperLinearLM
        ctor = set(inspect.signature(VesperLinearLM.__init__).parameters)
    except Exception as e:  # fall back to a known-good key list
        print(f"[expand] note: could not import VesperLinearLM ({e}); "
              "using built-in constructor key list")
        ctor = {"vocab_size", "dim", "n_layers", "n_heads", "n_kv_heads", "hidden_dim",
                "num_experts", "top_k", "max_seq_len", "pad_id", "dropout",
                "attention_every", "gla_head_dim", "qk_norm", "linear_type",
                "mamba2_state_size", "grad_checkpoint"}
    dropped = {k: v for k, v in cfg.items() if k not in ctor}
    cfg = {k: v for k, v in cfg.items() if k in ctor}

    # ---- optimizer state reset ----
    nulled = []
    for key in OPTIMIZER_STATE_KEYS:
        if key in ckpt:
            ckpt[key] = None
            nulled.append(key)

    provenance = {
        "tool": "Growth/expand_experts.py",
        "tool_version": TOOL_VERSION,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "source_checkpoint": os.path.abspath(in_path),
        "mode": args.mode,
        "noise_std": expert_noise,
        "router_noise_std": router_noise,
        "exact_upcycle": bool(args.exact_upcycle),
        "seed": args.seed,
        "shared_expert": "skipped: MoEFeedForward has no shared-expert slot",
        "num_experts_before": old_e,
        "num_experts_after": old_e + num_new,
        "top_k_before": top_k_old,
        "top_k_after": top_k_new,
        "parent_map": parent_map,
        "usage_stats_file": os.path.abspath(args.usage_stats) if args.usage_stats else None,
        "optimizer_keys_nulled": nulled,
        "parent_step": ckpt.get("step"),
        "dropped_config_keys": dropped,
        "notes": (
            "exact_upcycle: clone gate rows are bit-exact copies and top_k is doubled so "
            "each parent's renormalized routing weight splits 50/50 with its clone; "
            "function is preserved (validated by Growth/validate.py check b)."
            if args.exact_upcycle else
            "standard sparse upcycling: new gate rows = parent rows + noise; "
            "expect a small loss bump until the routing warmup rebalances experts."
        ),
    }

    ckpt["model"] = new_state
    ckpt["model_config"] = cfg
    ckpt["expansion_provenance"] = provenance

    os.makedirs(args.output, exist_ok=True)
    print(f"[expand] writing {out_path}")
    torch.save(ckpt, out_path)

    report = {
        "input": os.path.abspath(in_path),
        "output": os.path.abspath(out_path),
        "model_config": cfg,
        "expansion_provenance": provenance,
    }
    with open(os.path.join(args.output, "expansion_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"[expand] done: {old_e} -> {cfg['num_experts']} experts, "
          f"top_k {top_k_old} -> {top_k_new}, optimizer keys reset: {nulled or 'none present'}")
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Sparse-upcycling expert expansion for VesperLM MoE checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Example\n-------")[-1],
    )
    ap.add_argument("--input", required=True,
                    help="input checkpoint dir (containing checkpoint.pt) or .pt file")
    ap.add_argument("--output", required=True,
                    help="output dir; writes checkpoint.pt + expansion_report.json")
    ap.add_argument("--num-experts", type=int, required=True,
                    help="target expert count (must exceed the current count)")
    ap.add_argument("--mode", choices=("clone", "clone_split"), default="clone",
                    help="clone: copy experts round-robin; clone_split: (i+j)/2 mixtures")
    ap.add_argument("--noise-std", type=float, default=1e-3,
                    help="gaussian noise std added to new expert weights "
                         "(and router rows, unless --exact-upcycle). 0 = bit-exact clones")
    ap.add_argument("--exact-upcycle", action="store_true",
                    help="function-preserving surgery: bit-exact expert clones, bit-exact "
                         "duplicate gate rows, top_k doubled -> parent/clone routing weight "
                         "split exactly 50/50 (see module docstring)")
    ap.add_argument("--usage-stats", default=None,
                    help="optional JSON of per-expert usage counts; clones round-robin "
                         "over the most-used experts first")
    ap.add_argument("--seed", type=int, default=0, help="noise seed (default 0)")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing output")
    ap.add_argument("--shared-expert", action="store_true",
                    help="requested for completeness; always refused (architecture has no "
                         "shared-expert slot) — see README")
    args = ap.parse_args()
    expand(args)


if __name__ == "__main__":
    main()
