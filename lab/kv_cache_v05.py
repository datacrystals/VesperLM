#!/usr/bin/env python3
"""KV-cached incremental decode for the v0.5 assembled model (v0.2 "make chat fast").

The mechanism (the user's spec): decode one token at a time in a loop, keeping
each past token's K and V in a growing per-layer cache ("the notebook"); each
new token computes Q/K/V only for itself and attends to the cached K/V.  The
spine's HF forward (transformers 5.x phimoe) already implements the cache,
rope positions, and the sliding-window mask on top of `past_key_values`; this
module drives it one token at a time and proves the incremental loop is
logit-identical to the parallel full-forward.

Why the rest of the v0.5 assembly drops in unchanged (confirmed by the parity
test, see `lab/chat_v05_fast_validate.py`):
  * sparsemixer top-2 selection is strictly per-token in h_t (eval path is
    deterministic: argmax, then argmax with expert 1 at -inf) — the passport
    gates (GraftGateD2 / MixGateD2) score h_t only;
  * plug-in delivery `y = y_stock + sum_j lam_j * m_j * e_j` is pointwise:
    membership m_j = (row_j score beats the k-th stock score) on h_t alone,
    and WrappedForeignExpert is a pure FFN (host_in -> adapters -> frozen
    core -> host_out) with no cross-position ops.

GPU expert merge (gfx906 serving path): d0's `_rocm_safe_experts_forward`
CPU-sort shim exists because `torch.sort` on the device kernel is broken at
some S=tokens*top_k sizes.  For decode (T=1, S=top_k) sorting is pure
overhead and its per-expert `int(counts[e])` costs one device->host sync per
expert per layer; `fast_experts_forward` replaces that with a fixed top-k
gather: `index_select` the k selected expert slices and weighted-sum them,
zero host syncs.  Prefill (T>1) keeps the CPU-sort math (one host sync via
`counts.tolist()`).  Both branches are gated by the parity test against the
old path.

Batch-vs-incremental numerics: identical math, fp16 reduction order may
differ at the 1e-3 level in logits (attention GEMM tiling); the functional
bar is greedy-identical completions + small max|dlogit|.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "lab"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import graft_d0_parity as d0  # noqa: E402
import graft_d1_plugin as d1  # noqa: E402
import graft_d2_crossfamily as d2  # noqa: E402
import grafted_spine_v0 as g0  # noqa: E402


# ---------------------------------------------------------------------------
# GPU expert merge — fixed top-k gather for decode (see module docstring)
# ---------------------------------------------------------------------------

def _fast_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Same math as the donor experts merge; decode branch has zero host syncs.

    T == 1 (decode): fixed top-k gather — index_select the k chosen expert
    slices (index_select reads what the GEMM would read anyway) and reduce
    `sum_j w_j * e_j(h)` in slot order.  T > 1 (prefill): d0's gfx906
    CPU-sort shim with a single host sync for the per-expert counts.
    """
    if hidden_states.shape[0] == 1:
        y = torch.zeros_like(hidden_states)
        for j in range(top_k_index.shape[-1]):
            w_gu = self.gate_up_proj[top_k_index[0, j]]
            w_down = self.down_proj[top_k_index[0, j]]
            gate, up = F.linear(hidden_states, w_gu).chunk(2, dim=-1)
            h = self.act_fn(gate) * up
            w = top_k_weights[0, j].to(dtype=h.dtype)
            y = y + w * F.linear(h, w_down)
        return y
    # prefill: d0's verified CPU-sort merge, one host sync instead of E
    device = hidden_states.device
    num_tokens, hidden_dim = hidden_states.shape
    num_top_k = top_k_index.size(-1)
    expert_ids = top_k_index.reshape(-1)
    sample_weights = top_k_weights.reshape(-1)
    s_total = expert_ids.numel()
    expert_ids_g, perm = torch.sort(expert_ids.cpu())
    expert_ids_g = expert_ids_g.to(device)
    perm = perm.to(device)
    hidden_g = hidden_states[perm // num_top_k]
    w_g = sample_weights[perm]
    counts = torch.bincount(expert_ids_g, minlength=self.num_experts)
    counts_list = counts.tolist()          # ONE host sync (shim did E syncs)
    merged = torch.zeros(s_total, hidden_dim, device=device,
                         dtype=hidden_states.dtype)
    start = 0
    for e, n in enumerate(counts_list):
        if n == 0:
            continue
        end = start + n
        gate, up = F.linear(hidden_g[start:end], self.gate_up_proj[e]).chunk(2, dim=-1)
        y = self.act_fn(gate) * up
        y = F.linear(y, self.down_proj[e])
        merged[start:end] = y * w_g[start:end].unsqueeze(-1)
        start = end
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(s_total, device=device)
    merged = merged[inv_perm]
    return merged.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)


def install_expert_merge(model, which: str = "fast") -> int:
    """Install one of the two expert merges on every 3D bank.

    which="fast" — fixed top-k gather (decode-optimized, zero host syncs);
    which="shim" — d0's `_rocm_safe_experts_forward` CPU-sort merge, i.e. the
    exact merge the old full-forward chat path runs.  Order-independent
    w.r.t. g0.install_mix: if a MixMerge plug wrapper already owns
    `blk.experts.forward`, the merge is swapped in as its stock capture
    instead of clobbering the wrapper.
    """
    import types
    fn = (_fast_experts_forward if which == "fast"
          else d0._rocm_safe_experts_forward)
    owners = {}
    for m in model.modules():
        ex = getattr(m, "experts", None)
        pm = getattr(m, "_plugin_merge", None)
        if ex is not None and pm is not None:
            owners[id(ex)] = pm
    n = 0
    for mod in model.modules():
        if (hasattr(mod, "gate_up_proj") and hasattr(mod, "down_proj")
                and hasattr(mod, "num_experts")):
            bound = types.MethodType(fn, mod)
            pm = owners.get(id(mod))
            if pm is not None and hasattr(pm, "stock"):
                pm.stock = bound
            else:
                mod.forward = bound
            mod._vesper_merge = which
            n += 1
    return n


def install_fast_experts(model) -> int:
    """Decode-optimized merge (see install_expert_merge)."""
    return install_expert_merge(model, "fast")


def merge_equivalence_check(device: str = "cpu", seed: int = 0,
                            n_tokens: int = 5, n_experts: int = 16,
                            top_k: int = 2, hidden: int = 32,
                            inter: int = 24) -> dict:
    """Synthetic: fast merge == d0 CPU-sort shim on identical inputs."""
    import types
    g = torch.Generator(device="cpu").manual_seed(seed)

    class _Bank(torch.nn.Module):
        pass

    b = _Bank()
    b.num_experts = n_experts
    b.gate_up_proj = torch.nn.Parameter(
        torch.randn(n_experts, 2 * inter, hidden, generator=g).to(device))
    b.down_proj = torch.nn.Parameter(
        torch.randn(n_experts, hidden, inter, generator=g).to(device))
    b.act_fn = torch.nn.functional.silu
    shim = types.MethodType(d0._rocm_safe_experts_forward, b)
    fast = types.MethodType(_fast_experts_forward, b)
    h = torch.randn(n_tokens, hidden, generator=g).to(device)
    idx = torch.stack([torch.randperm(n_experts, generator=g)[:top_k]
                       for _ in range(n_tokens)]).to(device)
    w = torch.rand(n_tokens, top_k, generator=g).to(device)
    y_shim = shim(h, idx, w)
    y_fast = fast(h, idx, w)
    y1 = fast(h[:1], idx[:1], w[:1])
    d_all = float((y_shim - y_fast).abs().max())
    d_dec = float((y_shim[:1] - y1).abs().max())
    return {"max_abs_prefill": d_all, "max_abs_decode": d_dec,
            "pass": max(d_all, d_dec) < 1e-4}


# ---------------------------------------------------------------------------
# KVDecoder — one-token-at-a-time decode over a growing cache
# ---------------------------------------------------------------------------

class KVDecoder:
    """Incremental decode for the assembled v0.5 model (bare or grafted).

    `mode`: "bare" (stock spine) or "grafted" (passport gates + plug-ins).
    The graft state is installed per call via g0.run_in_state; a cache is only
    ever reused under the same mode that built it (the K/V are a function of
    the active parameters).
    """

    def __init__(self, model, tok, device, blocks=None, fam=None,
                 gates=None, handles=None, stop_ids=None):
        self.model = model
        self.tok = tok
        self.device = device
        self.blocks = blocks
        self.fam = fam
        self.gates = gates
        self.handles = handles
        self.past = None
        self.cached_ids: list[int] | None = None
        self.cache_mode: str | None = None
        if stop_ids is None:
            stop_ids = set()
            for attr in ("eos_token_id", "pad_token_id"):
                v = getattr(tok, attr, None)
                if v is not None:
                    stop_ids.add(int(v))
        self.stop_ids = stop_ids

    def reset(self) -> None:
        self.past = None
        self.cached_ids = None
        self.cache_mode = None

    def _state(self, mode: str, fn):
        if self.blocks is None or mode == "bare":
            return g0.run_in_state(self.model, self.blocks, self.fam,
                                   None, None, "off", fn)
        return g0.run_in_state(self.model, self.blocks, self.fam,
                               self.gates, self.handles, "both", fn)

    def _forward(self, ids_2d: torch.Tensor) -> torch.Tensor:
        """One model call; appends these ids' K/V to self.past. -> (1, t, V)."""
        out = self.model(input_ids=ids_2d, past_key_values=self.past,
                         use_cache=True)
        self.past = out.past_key_values
        return out.logits

    def _crop(self, length: int) -> None:
        """Truncate the cache to the first `length` positions.

        Uses the 5.18+ negative crop (drop N tail tokens); on any doubt the
        cache is dropped and the caller re-prefills — never a stale cache.
        """
        if length <= 0 or self.past is None:
            self.reset()
            return
        try:
            cur = int(self.past.get_seq_length())
            if cur == length:
                self.cached_ids = self.cached_ids[:length]
                return
            if cur < length:
                raise RuntimeError("cache shorter than crop target")
            self.past.crop(length - cur)          # negative: drop the tail
            if int(self.past.get_seq_length()) != length:
                raise RuntimeError("unexpected length after crop")
            self.cached_ids = self.cached_ids[:length]
        except Exception:
            self.reset()

    def _prepare(self, ids: list[int], mode: str):
        """Align the cache with `ids`; returns ids of the suffix to forward."""
        if mode != self.cache_mode or not self.cached_ids:
            self.reset()
            self.cache_mode = mode
        cached = self.cached_ids or []
        L = 0
        for a, b in zip(cached, ids):
            if a != b:
                break
            L += 1
        if L > len(ids) or (L == len(ids) and L > 0):
            # full match still needs the last position's logits for sampling
            L = min(L, max(0, len(ids) - 1))
        if L < len(cached):
            self._crop(L)
            if self.cached_ids is None:   # crop failed -> recompute everything
                L = 0
        return ids[L:]

    @torch.no_grad()
    def forward_full(self, ids: list[int], mode: str) -> torch.Tensor:
        """Parallel full-forward logits (T, V) float32 cpu — the reference."""
        def run():
            self.reset()
            x = torch.tensor([ids], dtype=torch.long, device=self.device)
            self.cached_ids = list(ids)
            self.cache_mode = mode
            return self._forward(x)[0].float().cpu()
        return self._state(mode, run)

    @torch.no_grad()
    def forward_incremental(self, ids: list[int], mode: str,
                            route_sink: dict | None = None) -> torch.Tensor:
        """Strictly one token per model call -> per-position logits (T, V).

        route_sink (optional): layer_i -> list of (k,) index tensors, one per
        step — lets the parity harness prove the routing decisions themselves
        are position-identical to the full-forward.
        """
        def run():
            self.reset()
            self.cache_mode = mode
            out = []
            for t in range(len(ids)):
                x = torch.tensor([[ids[t]]], dtype=torch.long,
                                 device=self.device)
                lg = self._forward(x)[0, -1].float().cpu()
                out.append(lg)
                self.cached_ids = (self.cached_ids or []) + [ids[t]]
                if route_sink is not None:
                    for i, g in enumerate(self.gates or []):
                        li = getattr(g, "last_indices", None)
                        if li is not None:
                            route_sink.setdefault(i, []).append(
                                li.reshape(-1).cpu().clone())
            return torch.stack(out, dim=0)
        return self._state(mode, run)

    @torch.no_grad()
    def generate(self, messages, max_new_tokens: int, mode: str,
                 sample: bool = False, temperature: float = 0.8,
                 top_p: float = 0.95, seed: int = 0, prefix_reuse: bool = True):
        """One assistant turn — same sampling semantics as g0.gen_reply.

        Returns (text, out_ids, stats) with prefill/decode timings and the
        number of tokens re-served from the cache.
        """
        prompt = self.tok.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
        ids = self.tok.encode(prompt, add_special_tokens=False)
        if not prefix_reuse:
            self.reset()
        out_ids: list[int] = []
        g = torch.Generator(device="cpu").manual_seed(seed)

        def run():
            nonlocal out_ids
            t0 = time.perf_counter()
            suffix = self._prepare(ids, mode)
            prefill_from = len(ids) - len(suffix)
            x = torch.tensor([suffix], dtype=torch.long, device=self.device)
            logits = self._forward(x)[0, -1].float().cpu()
            self.cached_ids = (self.cached_ids or []) + list(suffix)
            t1 = time.perf_counter()
            n_step = 0
            for _ in range(max_new_tokens):
                if sample:
                    probs = F.softmax(logits / max(1e-5, temperature), dim=-1)
                    sorted_p, sorted_i = torch.sort(probs, descending=True)
                    cum = sorted_p.cumsum(0)
                    cut = int((cum > top_p).nonzero(as_tuple=True)[0][:1].item()) + 1 \
                        if (cum > top_p).any() else len(sorted_p)
                    p = sorted_p[:cut] / sorted_p[:cut].sum()
                    idx = int(sorted_i[torch.multinomial(p, 1, generator=g)])
                else:
                    idx = int(logits.argmax())
                out_ids.append(idx)
                if idx in self.stop_ids:
                    break
                x = torch.tensor([[idx]], dtype=torch.long, device=self.device)
                logits = self._forward(x)[0, -1].float().cpu()
                self.cached_ids = (self.cached_ids or []) + [idx]
                n_step += 1
            t2 = time.perf_counter()
            return {
                "prompt_tokens": len(ids),
                "prefill_tokens": len(suffix),
                "reused_tokens": prefill_from,
                "prefill_s": t1 - t0,
                "decode_s": t2 - t1,
                "n_new_tokens": len(out_ids),
                "n_decode_steps": n_step,
            }

        stats = self._state(mode, run)
        self.cache_mode = mode
        text = self.tok.decode(out_ids, skip_special_tokens=True)
        return text, out_ids, stats


# ---------------------------------------------------------------------------
# Parity — the correctness proof (full-forward == incremental loop)
# ---------------------------------------------------------------------------

@torch.no_grad()
def parity_ids(dec: KVDecoder, ids: list[int], mode: str) -> dict:
    """Per-position logit parity on one id sequence + greedy argmax agreement.

    Also compares the MoE routing decisions themselves (passport gates'
    selected expert indices per layer per position) between the two paths —
    a routing flip would be a structural, not numerical, difference.
    """
    lg_full = dec.forward_full(ids, mode)
    routes_full = {}
    if mode == "grafted":          # bare mode never touches the gates
        for i, g in enumerate(dec.gates or []):
            li = getattr(g, "last_indices", None)
            if li is not None:
                routes_full[i] = li.reshape(len(ids), -1).cpu().clone()
    sink: dict = {}
    lg_inc = dec.forward_incremental(ids, mode, route_sink=sink)
    diff = (lg_full - lg_inc).abs()
    cos = F.cosine_similarity(lg_full, lg_inc, dim=-1)
    argmax_full = lg_full.argmax(dim=-1)
    argmax_inc = lg_inc.argmax(dim=-1)
    route_agree, route_layers = [], []
    for i, rf in routes_full.items():
        ri = sink.get(i)
        if ri is None or len(ri) != len(ids):
            continue
        ri = torch.stack(ri, dim=0)
        route_layers.append(i)
        route_agree.append(float((rf == ri).float().mean()))
    return {
        "n_tokens": len(ids),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "p99_abs": float(diff.flatten().quantile(0.99)),
        "per_pos_max": [round(float(x), 5) for x in diff.max(dim=-1).values],
        "min_cos": float(cos.min()),
        "argmax_agree": float((argmax_full == argmax_inc).float().mean()),
        "greedy_ids_equal": bool((argmax_full == argmax_inc).all()),
        "route_agree": (min(route_agree) if route_agree else None),
        "route_agree_all": (all(a == 1.0 for a in route_agree)
                            if route_agree else None),
        "route_layers_checked": len(route_layers),
    }


@torch.no_grad()
def parity_greedy_gen(dec: KVDecoder, messages, max_new_tokens: int,
                      mode: str) -> dict:
    """Greedy completion identity: g0.gen_reply full-forward vs KV loop."""
    def full_logits_fn(x):
        return d0.model_logits(dec.model, x.to(dec.device))

    def run_full():
        return g0.gen_reply(dec.tok, full_logits_fn, messages,
                            max_new_tokens, dec.device, sample=False)

    text_f, ids_f = dec._state(mode, run_full)
    text_k, ids_k, _st = dec.generate(messages, max_new_tokens, mode,
                                      sample=False)
    return {
        "ids_full": ids_f,
        "ids_kv": ids_k,
        "identical": ids_f == ids_k,
        "text_full": text_f,
        "text_kv": text_k,
    }


@torch.no_grad()
def bench_old_vs_new(dec: KVDecoder, messages, max_new_tokens: int, mode: str,
                     n_warmup: int = 1) -> dict:
    """tok/s of the old full-forward-per-token path vs prefill+decode.

    The old path is timed under d0's CPU-sort merge (exactly what chat_v05.py
    runs today) and the new path under the fixed top-k gather, then the fast
    merge is left installed.
    """
    def full_logits_fn(x):
        return d0.model_logits(dec.model, x.to(dec.device))

    def run_full():
        return g0.gen_reply(dec.tok, full_logits_fn, messages,
                            max_new_tokens, dec.device, sample=False)

    install_expert_merge(dec.model, "shim")
    for _ in range(n_warmup):
        dec._state(mode, run_full)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    text_f, ids_f = dec._state(mode, run_full)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    old_s = t1 - t0

    install_expert_merge(dec.model, "fast")
    for _ in range(n_warmup):
        dec.generate(messages, 4, mode, sample=False)

    dec.reset()
    text_k, ids_k, st = dec.generate(messages, max_new_tokens, mode,
                                     sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return {
        "old_path": {
            "wall_s": old_s,
            "n_new_tokens": len(ids_f),
            "tok_per_s": len(ids_f) / old_s if old_s > 0 else 0.0,
            "merge": "shim",
        },
        "new_path": st | {
            "tok_per_s_decode": (st["n_decode_steps"] / st["decode_s"]
                                 if st["decode_s"] > 0 else 0.0),
            "tok_per_s_prefill": (st["prefill_tokens"] / st["prefill_s"]
                                  if st["prefill_s"] > 0 else 0.0),
            "tok_per_s_end2end": (len(ids_k) / (st["prefill_s"] + st["decode_s"])),
            "merge": "fast",
        },
        "same_greedy_ids": ids_f == ids_k,
    }
