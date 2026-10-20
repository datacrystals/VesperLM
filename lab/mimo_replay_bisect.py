#!/usr/bin/env python3
"""mimo_replay_bisect — first-divergent-submodule localizer (WIRING fix ch. 2).

Two phases (default: both in one process, GPU freed between):

  capture  Build the full 48L lazy host, run ONE prefill, capture per-layer
           per-submodule tensors for layers 0..16 into
           ~/mimo_flash/replay_capt.pt:  x_in, h1 (input_layernorm out),
           attn_out, h2 (post_attention_layernorm out), gate (idx, w) for MoE,
           mlp_out, y_out.
  replay   fp32 CPU canonical replay per ANCHORED layer (host x_in as input):
           correct-by-inspection formulas transcribed from modeling_mimo_v2
           (fused_qkv split [q,k,v], partial rope_dim=64 first-dims,
           v_scale=0.707 pre-attention, eager sink appended as extra softmax
           column for SWA, fp32 gate sigmoid + noaux_tc bias + renorm,
           silu(g)*u->d experts), canonical compressed-tensors dequant
           (mh.dequant_int4_linear, the proven one).  Parity = cos + rel_l2
           per submodule.  Honesty guard: anchored L0 must pass before any
           downstream verdict is trusted; chained mode propagates the ref
           output to find the first in-stream divergence.

Usage (box):  python mimo_replay_bisect.py [max_layer]
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/data/tliao/mimo_flash_lab")
import mimo_full_host as fh  # noqa: E402
import mimo_flash_host as mh  # noqa: E402

MODEL_DIR = fh.MODEL_DIR
CAPT_PATH = os.path.expanduser("~/mimo_flash/replay_capt.pt")
MAX_LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 16
PROMPT = "What do you think of trains?"

cfg_raw = json.load(open(os.path.join(MODEL_DIR, "config.json")))
HYBRID = cfg_raw["hybrid_layer_pattern"]           # 0 full, 1 swa
MOE_FREQ = cfg_raw["moe_layer_freq"]
NH, NKV_FULL, NKV_SWA = 64, 4, 8
HD, VD, ROPE_D = 192, 128, 64
V_SCALE = float(cfg_raw["attention_value_scale"])
TOPK, EPS = 8, 1e-6
HIDDEN = 4096


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return float(F.cosine_similarity(a, b, dim=0))


# ---------------------------------------------------------------------------
# Capture phase
# ---------------------------------------------------------------------------

def capture() -> dict:
    if os.path.exists(CAPT_PATH):
        log(f"capture reuse {CAPT_PATH}")
        return torch.load(CAPT_PATH, weights_only=False)
    out = {}
    model, tok, _cache = fh.build_full_host(MODEL_DIR, "cuda", 18.0, out)
    layers = list(range(0, MAX_LAYER + 1))
    capt: dict[int, dict[str, torch.Tensor]] = {li: {} for li in layers}

    def keep(li, name, t):
        capt[li][name] = t.detach().float().cpu()

    for li in layers:
        layer = model.model.layers[li]

        def pre_hook(_m, args, _kwargs={}, _li=li):
            keep(_li, "x_in", args[0])
        layer.register_forward_pre_hook(pre_hook)
        layer.input_layernorm.register_forward_hook(
            lambda _m, _i, o, _li=li: keep(_li, "h1", o))
        layer.self_attn.register_forward_hook(
            lambda _m, _i, o, _li=li: keep(_li, "attn_out", o[0]))
        layer.post_attention_layernorm.register_forward_hook(
            lambda _m, _i, o, _li=li: keep(_li, "h2", o))
        if hasattr(layer.mlp, "gate"):
            def gate_hook(_m, _i, o, _li=li):
                keep(_li, "gate_idx", o[0])
                keep(_li, "gate_w", o[1])
            layer.mlp.gate.register_forward_hook(gate_hook)
        layer.mlp.register_forward_hook(
            lambda _m, _i, o, _li=li: keep(_li, "mlp_out", o))
        layer.register_forward_hook(
            lambda _m, _i, o, _li=li: keep(_li, "y_out", o))

    ids = tok.encode(tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=False,
        add_generation_prompt=True), add_special_tokens=False)
    x = torch.tensor([ids], dtype=torch.long, device="cuda")
    with torch.no_grad():
        model(input_ids=x, use_cache=False)
    capt["ids"] = torch.tensor(ids)
    torch.save(capt, CAPT_PATH)
    n_tok = len(ids)
    log(f"capture done T={n_tok} layers 0..{MAX_LAYER} -> {CAPT_PATH}")
    del model
    torch.cuda.empty_cache()
    return capt


# ---------------------------------------------------------------------------
# Canonical reference (fp32 CPU, correct-by-inspection)
# ---------------------------------------------------------------------------

class RefStore:
    def __init__(self, model_dir: str):
        from safetensors import safe_open
        self._safe_open = safe_open
        self.wm = json.load(open(os.path.join(
            model_dir, "model.safetensors.index.json")))["weight_map"]
        self.dir = model_dir
        self.h = {}

    def get(self, name: str) -> torch.Tensor:
        shard = self.wm[name]
        hd = self.h.get(shard)
        if hd is None:
            hd = self._safe_open(os.path.join(self.dir, shard),
                                 framework="pt")
            self.h[shard] = hd
        return hd.get_tensor(name)

    def linear(self, root: str) -> torch.Tensor:
        if root + "weight_packed" in self.wm:
            return mh.dequant_int4_linear(self.get, root).float()
        return self.get(root + "weight").float()


def rmsnorm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    v = x.pow(2).mean(-1, keepdim=True)
    return w * (x * torch.rsqrt(v + EPS))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def rope_tables(n_tok: int, swa: bool):
    base = 1e4 if swa else 1e7
    inv = 1.0 / (base ** (torch.arange(0, ROPE_D, 2, dtype=torch.int64)
                          .float() / ROPE_D))
    freqs = torch.outer(torch.arange(n_tok, dtype=torch.float32), inv)
    emb = torch.cat([freqs, freqs], dim=-1)      # (T, ROPE_D)
    return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos_t: torch.Tensor,
               sin_t: torch.Tensor) -> torch.Tensor:
    # x: (H, T, D): rope first ROPE_D dims, pass through the rest
    r, n = x[..., :ROPE_D], x[..., ROPE_D:]
    c, s = cos_t.unsqueeze(0), sin_t.unsqueeze(0)
    return torch.cat([r * c + rotate_half(r) * s, n], dim=-1)


def ref_attn(h: torch.Tensor, W: dict, li: int, n_tok: int) -> torch.Tensor:
    swa = HYBRID[li] == 1
    nkv = NKV_SWA if swa else NKV_FULL
    qsz, ksz, vsz = NH * HD, nkv * HD, nkv * VD
    cos_t, sin_t = rope_tables(n_tok, swa)
    qkv = h @ W["qkv"].T
    q, k, v = qkv.split([qsz, ksz, vsz], dim=-1)
    q = q.view(n_tok, NH, HD).transpose(0, 1)
    k = k.view(n_tok, nkv, HD).transpose(0, 1)
    v = (v.view(n_tok, nkv, VD).transpose(0, 1)) * V_SCALE
    q = apply_rope(q, cos_t, sin_t)
    k = apply_rope(k, cos_t, sin_t)
    rep = NH // nkv
    k = k.unsqueeze(1).expand(nkv, rep, n_tok, HD).reshape(NH, n_tok, HD)
    v = v.unsqueeze(1).expand(nkv, rep, n_tok, VD).reshape(NH, n_tok, VD)
    scores = (q @ k.transpose(1, 2)) * (HD ** -0.5)      # (NH, T, T)
    mask = torch.triu(torch.full((n_tok, n_tok), float("-inf")), 1)
    scores = scores + mask
    if swa:
        sink = W["sink"].reshape(-1, 1, 1).expand(NH, n_tok, 1)
        scores = torch.cat([scores, sink], dim=-1)
    scores = scores - scores.max(dim=-1, keepdim=True).values
    probs = F.softmax(scores, dim=-1)
    if swa:
        probs = probs[..., :-1]
    o = probs @ v                                        # (NH, T, VD)
    o = o.transpose(0, 1).reshape(n_tok, NH * VD)
    return o @ W["o"].T


def ref_gate(h: torch.Tensor, gw: torch.Tensor, bias: torch.Tensor):
    logits = h @ gw.T
    scores = logits.sigmoid()
    choice = scores + bias.unsqueeze(0)
    idx = torch.topk(choice, TOPK, dim=-1, sorted=False).indices
    w = scores.gather(1, idx)
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    return idx, w


def ref_moe(h: torch.Tensor, store: RefStore, li: int,
            gw: torch.Tensor, bias: torch.Tensor, n_tok: int):
    idx, w = ref_gate(h, gw, bias)
    acc = torch.zeros(n_tok, HIDDEN)
    uniq = sorted(set(idx.flatten().tolist()))
    for e in uniq:
        rows, cols = (idx == e).nonzero(as_tuple=True)
        g = store.linear(f"model.layers.{li}.mlp.experts.{e}.gate_proj.")
        u = store.linear(f"model.layers.{li}.mlp.experts.{e}.up_proj.")
        d = store.linear(f"model.layers.{li}.mlp.experts.{e}.down_proj.")
        x = h[rows]
        eo = (F.silu(x @ g.T) * (x @ u.T)) @ d.T
        acc.index_add_(0, rows, eo * w[rows, cols].unsqueeze(-1))
        del g, u, d
    return idx, w, acc


def ref_layer(x: torch.Tensor, store: RefStore, li: int,
              compare_hook=None) -> torch.Tensor:
    n_tok = x.shape[0]
    p = f"model.layers.{li}."
    W = {
        "in": store.get(p + "input_layernorm.weight").float(),
        "post": store.get(p + "post_attention_layernorm.weight").float(),
        "qkv": store.linear(p + "self_attn.qkv_proj."),
        "o": store.linear(p + "self_attn.o_proj."),
    }
    if HYBRID[li] == 1:
        W["sink"] = store.get(
            p + "self_attn.attention_sink_bias").float()
    h1 = rmsnorm(x, W["in"])
    attn = ref_attn(h1, W, li, n_tok)
    r1 = x + attn
    h2 = rmsnorm(r1, W["post"])
    if compare_hook is not None:
        compare_hook(li, "h1", h1)
        compare_hook(li, "attn_out", attn)
        compare_hook(li, "h2", h2)
    del W["qkv"], W["o"]
    if MOE_FREQ[li] == 1:
        gw = store.get(p + "mlp.gate.weight").float()
        bias = store.get(p + "mlp.gate.e_score_correction_bias").float()
        gidx, gw_w, moe = ref_moe(h2, store, li, gw, bias, n_tok)
        if compare_hook is not None:
            compare_hook(li, "gate_idx", gidx)
            compare_hook(li, "gate_w", gw_w)
            compare_hook(li, "mlp_out", moe)
        y = r1 + moe
    else:
        g = store.linear(p + "mlp.gate_proj.")
        u = store.linear(p + "mlp.up_proj.")
        d = store.linear(p + "mlp.down_proj.")
        mlp = (F.silu(h2 @ g.T) * (h2 @ u.T)) @ d.T
        if compare_hook is not None:
            compare_hook(li, "mlp_out", mlp)
        y = r1 + mlp
    if compare_hook is not None:
        compare_hook(li, "y_out", y)
    return y


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

def replay(capt: dict) -> None:
    store = RefStore(MODEL_DIR)
    layers = sorted(li for li in capt if isinstance(li, int))
    n_tok = int(capt["ids"].shape[0])
    print(f"REPLAY anchored layers {layers[0]}..{layers[-1]} T={n_tok}",
          flush=True)
    print(f"{'L':>3} {'submod':<10} {'cos':>9} {'rel_l2':>9}", flush=True)
    broken_ref = False
    for li in layers:
        x = capt[li]["x_in"]
        if x.dim() == 3:
            x = x.squeeze(0)
        t0 = time.time()
        rows = []

        stash = {}

        def cmp(li_, name, ref, _rows=rows):
            host = capt[li_].get(name)
            if "gate" not in name and host is None:
                return
            if name == "gate_idx":
                stash["ri"] = ref
                hi = capt[li_]["gate_idx"]
                same = float((ref.sort(dim=-1).values ==
                              hi.sort(dim=-1).values).float().mean())
                _rows.append(("gate_idx", same, None))
                return
            if name == "gate_w":
                hi = capt[li_]["gate_idx"]
                hw = capt[li_]["gate_w"]
                rord = stash["ri"].argsort(dim=-1)
                hord = hi.argsort(dim=-1)
                rw = ref.gather(1, rord)
                hw2 = hw.gather(1, hord)
                eq = stash["ri"].gather(1, rord) == hi.gather(1, hord)
                if not bool(eq.all()):
                    _rows.append(("gate_w", 0.0, 1.0))
                    return
                _rows.append(("gate_w", cos(rw, hw2), rel_l2(rw, hw2)))
                return
            _rows.append((name, cos(ref, host.squeeze(0)),
                          rel_l2(ref, host.squeeze(0))))

        ref_layer(x, store, li, compare_hook=cmp)
        for name, c, r in rows:
            rtxt = f"{r:9.4f}" if r is not None else "     n/a "
            print(f"{li:>3} {name:<10} {c:9.5f} {rtxt}   "
                  f"({time.time() - t0:.0f}s layer)", flush=True)
        ycos = [c for n, c, r in rows if n == "y_out"]
        if li == layers[0] and ycos and ycos[0] < 0.999:
            print(f"GUARD anchored L{layers[0]} parity {ycos[0]:.5f} < 0.999"
                  " -> suspect the REFERENCE build, not the host",
                  flush=True)
            broken_ref = True
    if broken_ref:
        return

    # chained mode: propagate ref outputs, find first in-stream divergence
    print("CHAINED in-stream divergence walk:", flush=True)
    x = capt[layers[0]]["x_in"]
    if x.dim() == 3:
        x = x.squeeze(0)
    for li in layers:
        y = ref_layer(x, store, li)
        nxt = li + 1
        if nxt in capt:
            host_next = capt[nxt]["x_in"]
            if host_next.dim() == 3:
                host_next = host_next.squeeze(0)
            print(f"{li:>3} -> {nxt:>3}  cos={cos(y, host_next):9.5f} "
                  f"rel_l2={rel_l2(y, host_next):9.4f}", flush=True)
        x = y
    print("REPLAY_DONE", flush=True)


def full_stack_logits(capt: dict) -> None:
    """Canonical fp32 chained replay through ALL 48 layers -> final norm ->
    tied-head logits.  Decides whether the vendor-formula interpretation
    itself reproduces vLLM's coherence (host is now proven == reference on
    layers 0..16 anchored, so this tests REFERENCE vs vLLM semantics)."""
    store = RefStore(MODEL_DIR)
    ids = capt["ids"].tolist()
    emb_name = "model.embed_tokens.weight"
    emb = store.get(emb_name).float()
    x = emb[ids]
    n_tok = len(ids)
    print(f"FULLSTACK chained 0..47 T={n_tok}", flush=True)
    for li in range(48):
        t0 = time.time()
        x = ref_layer(x, store, li)
        if li in (0, 1, 5, 10, 11, 16, 17, 31, 40, 41, 45, 46, 47):
            print(f"L{li:>3} y_norm={float(x.norm()):10.2f}   "
                  f"({time.time() - t0:.0f}s)", flush=True)
    nf = store.get("model.norm.weight").float()
    h = rmsnorm(x[-1:], nf)
    lm_head = store.get("lm_head.weight").float()   # NOT embed: untied
    logits = h @ lm_head.T
    top = logits[0].topk(10)
    vals = [round(float(v), 3) for v in top.values]
    toks = top.indices.tolist()
    print("REF_TOP10_VALS", vals, flush=True)
    print("REF_TOP10_IDS", toks, flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    print("REF_TOP10_STR", [tok.decode([t]) for t in toks], flush=True)
    print("FULLSTACK_DONE", flush=True)


def main() -> None:
    capt = capture()
    if os.environ.get("REPLAY_FULLSTACK", "0") == "1":
        full_stack_logits(capt)
        return
    replay(capt)


if __name__ == "__main__":
    main()
