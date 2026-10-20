#!/usr/bin/env python3
"""mimo_full_host — FULL 48L MiMo-V2.6-Flash-INT4 decode host (bake-off arm).

Why this exists: load_slice dequants every expert to fp16 (576GiB — never
fits).  Full-host decode instead keeps expert weights PACKED at rest and
dequants only the experts each forward actually fires into a GPU LRU cache:

  resident  non-expert (embed, attn, gates, norms, lm_head) ~ 7 GiB
  packed    expert source, disk-backed (safetensors lazy reads)
  LRU cache dequantized fp16 experts, budget-capped

Per generated token a layer fires 8 of 256 experts; with a warm LRU the
dequant traffic is a fraction of compute.  Fits ONE 32GiB GPU — so the
bake-off arms run WITHOUT parking the sibling service (strictly safer than
the 8-GPU park plan; deviation reported).

gfx906 notes (measured): index-scan ops (torch.where / nonzero / bool-mask
gather) are missing from this torch build — the vendor MoE loop uses them,
so blocks are routed through the argsort dispatch (same math).
"""
from __future__ import annotations

import os
import sys
import threading
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "lab"))

import mimo_flash_host as mh  # noqa: E402  (dequant math, plugs, decode)
from mimo_plug_rows import PLUGS  # noqa: E402

FORMAT = "vesper-mimo-full-host-v1"
MODEL_DIR = "/data/ModelDownloader/MiMo-V2.6-Flash-INT4"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Disk-backed packed expert store + GPU LRU of dequantized fp16 experts
# ---------------------------------------------------------------------------

class ExpertStore:
    """Lazy packed-tensor reader keyed by tensor name (shard-resolved)."""

    def __init__(self, model_dir: str):
        from safetensors import safe_open
        self._safe_open = safe_open
        self.dir = model_dir
        self.wm = __import__("json").load(
            open(os.path.join(model_dir, "model.safetensors.index.json"))
        )["weight_map"]
        self._handles: dict[str, object] = {}
        self._lock = threading.Lock()

    def get(self, name: str) -> torch.Tensor:
        shard = self.wm[name]
        with self._lock:
            h = self._handles.get(shard)
            if h is None:
                h = self._safe_open(os.path.join(self.dir, shard),
                                    framework="pt")
                self._handles[shard] = h
            return h.get_tensor(name)


class LazyExpertMLP(nn.Module):
    """MiMoV2MLP-compatible expert: dequantizes its three linears on first
    call, keeps them in the parent cache (evictable)."""

    def __init__(self, store: ExpertStore, layer: int, expert: int,
                 cache: "ExpertCache"):
        super().__init__()
        self.store = store
        self.layer = layer
        self.expert = expert
        self.cache = cache
        self.root = f"model.layers.{layer}.mlp.experts.{expert}."

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1, w3, w2 = self.cache.get(self.layer, self.expert, self._load)
        h = F.linear(x, w1)
        h = F.silu(h) * F.linear(x, w3)
        return F.linear(h, w2)

    def native_core(self):
        """Host-native SwiGLU core (w_gu, w_down) fp32 — for plug GRAB."""
        w1, w3, w2 = self._load()
        w_gu = torch.cat([w1.float(), w3.float()], dim=0).cpu()
        return w_gu, w2.float().cpu()

    def _load(self):
        # GPU-side dequant: CPU unpack measured ~120ms/expert (46 s/tok at
        # 376 expert loads/token — smoke 2026-10-16).  Packed transfer is 4x
        # smaller and the unpack is 24 vector ops on device.
        st = self.store
        root = self.root
        dev = self.cache.device
        return tuple(_dequant_gpu(st, root + m + ".", dev)
                     for m in ("gate_proj", "up_proj", "down_proj"))


def _dequant_gpu(store, root: str, device: str) -> torch.Tensor:
    """packed int4 (group-32, asymmetric) -> fp16, unpack on device."""
    packed = store.get(root + "weight_packed").to(device)
    scale = store.get(root + "weight_scale").to(device)
    zp_p = store.get(root + "weight_zero_point").to(device)
    shape = store.get(root + "weight_shape").tolist()
    out_f, in_f = int(shape[0]), int(shape[1])
    g = in_f // scale.shape[-1]
    w = packed.to(torch.int64)
    q = torch.empty(w.shape[0], in_f, dtype=torch.int8, device=device)
    for k in range(8):
        q[:, k::8] = ((w >> (4 * k)) & 0xF).to(torch.int8) - 8
    z = torch.empty(out_f, in_f // g, dtype=torch.int8, device=device)
    wz = zp_p.to(torch.int64)
    for k in range(8):
        z[k::8, :] = ((wz >> (4 * k)) & 0xF).to(torch.int8) - 8
    zf = z.float().repeat_interleave(g, dim=1)
    sf = scale.float().repeat_interleave(g, dim=1)
    del packed, w, z, zp_p
    qf = q.float()
    del q
    qf.sub_(zf).mul_(sf)
    del zf, sf
    return qf.to(torch.float16)


class ExpertCache:
    """GPU LRU of dequantized experts, byte-budget capped (fp16).

    MIMO_LAZY_DISABLE=1 bypasses the cache entirely: every expert call
    loads fresh from the packed store (no store/reuse/evict), so the
    MoE path becomes a pure function of (layer, expert).  Bisect toggle
    for the WIRING-vs-CACHE question (lab/MIMO_HOST_FAITHFUL.md).  A
    residency slice was considered instead but does not fit: 12 MoE
    layers x 256 experts x 50.3MB fp16 ~= 154GB > any claim subset of
    the 32GB cards; fresh-per-call removes the same machinery on all 47
    layers at ~9-12 s/tok."""

    def __init__(self, budget_bytes: int, device: str):
        self.budget = budget_bytes
        self.device = device
        self.disabled = os.environ.get("MIMO_LAZY_DISABLE", "0") == "1"
        self.entries: dict[tuple[int, int], tuple] = {}
        self.order: list[tuple[int, int]] = []
        self.bytes = 0
        self.hits = 0
        self.misses = 0

    def get(self, layer: int, expert: int, loader):
        if self.disabled:
            self.misses += 1
            return loader()
        key = (layer, expert)
        hit = self.entries.get(key)
        if hit is not None:
            self.hits += 1
            self.order.remove(key)
            self.order.append(key)
            return hit
        self.misses += 1
        w1, w3, w2 = loader()
        w1 = w1.to(self.device)
        w3 = w3.to(self.device)
        w2 = w2.to(self.device)
        size = sum(t.numel() * t.element_size() for t in (w1, w3, w2))
        self.entries[key] = (w1, w3, w2)
        self.order.append(key)
        self.bytes += size
        while self.bytes > self.budget and len(self.order) > 1:
            old = self.order.pop(0)
            ow1, ow3, ow2 = self.entries.pop(old)
            self.bytes -= sum(t.numel() * t.element_size()
                              for t in (ow1, ow3, ow2))
            del ow1, ow3, ow2
        return (w1, w3, w2)


# ---------------------------------------------------------------------------
# Full host build
# ---------------------------------------------------------------------------

def build_full_host(model_dir: str, device: str, cache_gb: float = 18.0,
                    out: dict | None = None):
    """48L MiMo on `device` with disk-backed lazy experts.  Returns
    (model, tok, cache)."""
    from transformers import AutoTokenizer
    t0 = time.time()
    mod_mod, cfg_mod = mh.import_mimo_code(model_dir)
    raw = __import__("json").load(open(os.path.join(model_dir, "config.json")))
    raw.pop("auto_map", None)
    raw["vision_config"] = None
    raw["audio_config"] = None
    cfg = cfg_mod.MiMoV2Config(**raw)
    n_layers = cfg.num_hidden_layers

    # meta construction: no allocation; we materialize selectively
    with torch.device("meta"):
        model = mod_mod.MiMoV2ForCausalLM(cfg)
    model.eval()

    store = ExpertStore(model_dir)
    cache = ExpertCache(int(cache_gb * (1 << 30)), device)

    sd = model.state_dict()
    # non-expert keys -> materialize now (fp16), skip expert linears
    n_real = 0
    real: dict[str, torch.Tensor] = {}
    for name in sd:
        if ".mlp.experts." in name:
            continue
        base = name
        if base.endswith(".weight") and "norm" not in base \
                and "bias" not in base and "gate.e_score" not in base:
            # may be a packed compressed-tensors group
            root = name[: -len("weight")]
            packed_key = root + "weight_packed"
            if packed_key in store.wm:
                real[name] = mh.dequant_int4_linear(
                    store.get, root).to(torch.float16)
                n_real += 1
                continue
        if name not in store.wm:
            continue
        real[name] = store.get(name).to(torch.float16)
        n_real += 1
    # tie lm_head to embed (config _tied_weights_keys) — saves 1.16GiB and
    # the per-token working set (376 experts x 50MB = 18.8GiB) needs every
    # byte: cache < one token's expert set causes LRU thrash (measured 0
    # cache hits / 3008 loads over 8 tokens)
    if "lm_head.weight" in real and "model.embed_tokens.weight" in real:
        real["lm_head.weight"] = real["model.embed_tokens.weight"]
    # meta params: assign via load_state_dict(assign=True) semantics
    for name, tensor in real.items():
        parts = name.split(".")
        obj = model
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], nn.Parameter(tensor, requires_grad=False))
    log(f"non-expert materialized: {n_real} tensors, "
        f"{time.time() - t0:.0f}s")

    # swap expert linears for lazy ones
    n_ex = 0
    for li, blk in enumerate(model.model.layers):
        mlp = blk.mlp
        if not hasattr(mlp, "experts"):
            continue
        for ei, ex in enumerate(mlp.experts):
            ex.gate_proj = LazyExpertMLP._dummy_marker  # drop meta linears
            ex.up_proj = LazyExpertMLP._dummy_marker
            ex.down_proj = LazyExpertMLP._dummy_marker
        class _LazyGroup(nn.ModuleList):
            pass
        lgroup = _LazyGroup(
            [LazyExpertMLP(store, li, ei, cache)
             for ei in range(len(mlp.experts))])
        mlp.experts = lgroup
        n_ex += len(lgroup)
        # gfx906: vendor moe() uses index-scan ops; route through argsort
        mh.adapt_moe_block(mlp)   # the MoE block, not the decoder layer
    log(f"lazy experts: {n_ex}")

    # meta-fixup: config-derived buffers (rotary inv_freq etc.) never appear
    # in the checkpoint index — materialize them at real values/zero
    import math as _m
    n_fix = 0
    for name, buf in list(model.named_buffers()):
        if buf.device.type != "meta":
            continue
        if "inv_freq" in name:
            # SWA rotary uses its own theta (1e4 vs 1e7 full-attn) and the
            # rotary dim is head_dim * partial_rotary_factor (0.334 -> 64).
            # Both reflected in the buffer's own length: entries = dim/2.
            dim = buf.shape[0] * 2
            base = float(getattr(cfg, "rope_theta", 10000.0))
            if "swa" in name:
                base = float(getattr(cfg, "swa_rope_theta", base))
            inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            obj = model.get_submodule(name.rsplit(".", 1)[0])
            setattr(obj, name.rsplit(".", 1)[1], inv)
        else:
            obj = model.get_submodule(name.rsplit(".", 1)[0])
            setattr(obj, name.rsplit(".", 1)[1], torch.zeros(buf.shape))
        n_fix += 1
    log(f"meta buffers materialized: {n_fix}")
    model.to(device)  # moves non-expert params only (experts hold no params)
    if out is not None:
        out["full_host"] = {
            "layers": n_layers, "lazy_experts": n_ex,
            "cache_budget_gb": cache_gb,
            "non_expert_tensors": n_real,
            "wall_s": round(time.time() - t0, 1),
        }
    return model, AutoTokenizer.from_pretrained(model_dir,
                                                trust_remote_code=True), cache


LazyExpertMLP._dummy_marker = nn.Identity()


# ---------------------------------------------------------------------------
# Generator for spine_bakeoff (frozen decode via mh._clean_reply/stop ids)
# ---------------------------------------------------------------------------

def make_generator(plugins_dir: str, args=None):
    device = getattr(args, "device", None) or (
        "cuda" if torch.cuda.is_available() else "cpu")
    cache_gb = float(getattr(args, "cache_gb", 18.0) or 18.0)
    out: dict = {}
    model, tok, cache = build_full_host(MODEL_DIR, device, cache_gb, out)
    blocks = mh.find_mimo_moe_blocks(model)
    banks = []
    for blk in blocks:
        d_model = blk.gate.weight.shape[1]
        bank = mh.d0.PassportBank(blk.gate.weight.shape[0], d_model,
                                  d_r=d_model, learn_wq=False)
        mh.init_exact(bank, blk.gate.weight)
        banks.append(bank)
    installed = mh.install_plug_set(model, blocks, banks, plugins_dir, device)

    def gen_fn(messages, max_new, dec, seed=0):
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
        ids = tok.encode(prompt, add_special_tokens=False)
        out_ids = []
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        stop = mh.stop_token_ids(tok)
        # incremental decode: full re-forward per step would fire every
        # prefix token's experts each step and thrash the expert LRU
        past = None
        first = True
        for _ in range(int(max_new)):
            if first:
                x = torch.tensor([ids], dtype=torch.long, device=device)
                first = False
            else:
                x = torch.tensor([[out_ids[-1]]], dtype=torch.long,
                                 device=device)
            with torch.no_grad():
                out = model(input_ids=x, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1].float().cpu()
            pen = float(dec.get("rep_penalty", 1.0))
            if pen > 1.0:
                for t in set(ids + out_ids):
                    if 0 <= t < logits.shape[-1]:
                        logits[t] = logits[t] / pen if logits[t] > 0 \
                            else logits[t] * pen
            if dec.get("sample"):
                temp = max(1e-5, float(dec.get("temperature", 0.8)))
                top_p = float(dec.get("top_p", 0.95))
                probs = F.softmax(logits / temp, dim=-1)
                sp, si = torch.sort(probs, descending=True)
                cum = sp.cumsum(0)
                nz = (cum > top_p).nonzero(as_tuple=True)[0]
                cut = int(nz[:1].item()) + 1 if nz.numel() else sp.numel()
                p = sp[:cut] / sp[:cut].sum()
                nxt = int(si[torch.multinomial(p, 1, generator=g)])
            else:
                nxt = int(torch.argmax(logits))
            if nxt in stop:
                break
            out_ids.append(nxt)
        return mh._clean_reply(tok.decode(out_ids,
                                          skip_special_tokens=True))

    gen_fn.installed = installed
    gen_fn.host = out.get("full_host", {})
    gen_fn.cache = cache
    return tok, gen_fn
