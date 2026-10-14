#!/usr/bin/env python3
"""vk-base graft host adapter — the d0/d2/v06 graft stack on OUR spine.

The host is VesperLinearLM (vk_base: KDA+MLA passport MoE, 10L x d1024,
8 experts top-2, passport_dim 64) — PASSPORT-NATIVE: PassportRouter already
implements the socket the graft stack was designed for.  A registry row IS a
passport row; bank scores are the native route logits:

    score_e = (h @ W_q.T) @ p_e / sqrt(64)        (PassportBank.scores)
    logits  = query(x) @ passports.T / sqrt(64)   (PassportRouter.forward)

so `init=exact` is free and the parity gate is plumbing-only.

This module makes the reused machinery (d0.PassportBank, d25.WrappedForeignExpert,
d6.MixMerge / train_rows_one / load_wrap_plugin, g0.MixGateD2 semantics,
varact_gate.VarActMerge / TaughtTokenBonus, mem_pack) see the same duck-typed
contract it sees on the HF spines:

  * find_moe_blocks(model)  -> [MoEFeedForward, ...] (has .router + .experts)
  * ffn.experts.forward(h, idx, w) -> stock expert merge (HF experts contract)
  * ffn.forward(x) -> (y, aux); router(x_flat) -> (weights, idx, aux)
  * graft gate records last_membership / last_p_new / last_h (MixGateD2)
  * d0.model_logits handles the (logits, ce, aux) tuple (monkey-patched once)

Shared sources stay UNMODIFIED; everything here is port-layer glue.
"""

from __future__ import annotations

import math
import os
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "lab"))
sys.path.insert(0, os.path.join(REPO_ROOT, "Common"))

import graft_d0_parity as d0  # noqa: E402

FORMAT = "vesper-vk-graft-v1"


# ---------------------------------------------------------------------------
# Model load (value-embed auto-detect — e6c79ec: model builds must detect
# value_emb keys in the ckpt state dict and construct explicitly).
# ---------------------------------------------------------------------------

def load_vk(chat_ckpt: str, device: str = "cuda", dtype=torch.float16):
    """Load a vk-base chat_model dir (or vesper_chat.pt path) + its tokenizer."""
    from transformers import PreTrainedTokenizerFast
    from vesper_linear_model import VesperLinearLM
    import vk_fla_compat
    vk_fla_compat.install()   # pure-torch KDA ops (fla Triton kernels are
                              # unbuildable on gfx906 — FAILURES fla-kda-gfx906)

    tok_dir = chat_ckpt if os.path.isdir(chat_ckpt) else os.path.dirname(chat_ckpt)
    tok = PreTrainedTokenizerFast.from_pretrained(tok_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pt = chat_ckpt if chat_ckpt.endswith(".pt") else os.path.join(chat_ckpt, "vesper_chat.pt")
    payload = torch.load(pt, map_location="cpu", weights_only=False)
    mc = payload["model_config"]
    arch = ["dim", "n_layers", "n_heads", "n_kv_heads", "hidden_dim",
            "num_experts", "top_k", "max_seq_len", "linear_type",
            "full_type", "attention_every", "gla_head_dim", "qk_norm",
            "kda_head_dim", "kda_short_conv", "mamba2_state_size",
            "kv_lora_rank", "v_head_dim", "linear_force_fp32",
            "router_type", "passport_dim", "router_expert_dropout"]
    kwargs = {k: mc[k] for k in arch if k in mc}
    value_embed = any("value_emb" in k for k in payload["model_state_dict"])
    model = VesperLinearLM(vocab_size=len(tok), pad_id=tok.pad_token_id,
                           grad_checkpoint=False, value_embed=value_embed,
                           **kwargs)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    # dtype policy: keep the checkpoint's fp32 master weights (KDA runs its
    # own fp32 path under autocast-off; a global fp16 cast would break it) and
    # take fp16 GEMMs via autocast on every CUDA forward (see _patch_model_logits).
    model.eval()
    print(f"[vk] loaded {pt} value_embed={value_embed} device={device} "
          f"dtype={dtype}", flush=True)
    return model, tok


# ---------------------------------------------------------------------------
# Host rewiring: HF graft contract on VesperLinearLM MoE blocks
# ---------------------------------------------------------------------------

class VkFam:
    """FamilySpec stand-in for the vk-base passport MoE."""
    name = "vkpassport"
    gate_attr = "router"
    top_k = 2
    norm_topk_prob = False


FAM = VkFam()


def find_moe_blocks(model: nn.Module) -> list:
    """The 10 MoEFeedForward blocks (each has .router + .experts)."""
    out = []
    for m in model.modules():
        if hasattr(m, "experts") and hasattr(m, "router") \
                and isinstance(getattr(m, "experts", None), nn.ModuleList):
            out.append(m)
    return out


def _stock_dispatch(ffn):
    """The native permutation merge as experts.forward(h, idx, w) (HF shape)."""
    experts = ffn.experts
    top_k = ffn.top_k
    num_experts = ffn.num_experts

    def fwd(hidden_states, top_k_index=None, top_k_weights=None):
        if top_k_index is None:
            # direct call without a router pass: identity-safe no-op merge
            raise RuntimeError("stock dispatch needs (idx, w)")
        N, C = hidden_states.shape
        flat_e = top_k_index.reshape(-1)
        flat_w = top_k_weights.reshape(-1, 1).to(hidden_states.dtype)
        tok = torch.arange(N, device=hidden_states.device).repeat_interleave(top_k_index.shape[-1])
        order = torch.argsort(flat_e, stable=True)
        flat_e, tok, flat_w = flat_e[order], tok[order], flat_w[order]
        counts = torch.bincount(flat_e, minlength=num_experts).tolist()
        gathered = hidden_states[tok]
        outputs = torch.empty_like(gathered)
        start = 0
        for i in range(num_experts):
            n = counts[i]
            if n == 0:
                continue
            end = start + n
            outputs[start:end] = experts[i](gathered[start:end]) * flat_w[start:end]
            start = end
        final_output = torch.zeros_like(hidden_states)
        final_output.index_add_(0, tok, outputs)
        return final_output

    return fwd


def install_vk_host(model: nn.Module) -> list:
    """Rewire every MoE block to the graft contract. Returns blocks.

    ffn.forward keeps its (x)->(y,aux) signature (the trunk calls it), but
    the merge point moves to ffn.experts.forward so g0.install_mix /
    varact.install_varact / d1.set_plugin_mode work unchanged.
    """
    blocks = find_moe_blocks(model)
    for ffn in blocks:
        if getattr(ffn, "_vk_hosted", False):
            continue
        ffn._vk_stock_router = ffn.router
        ffn._vk_stock_dispatch = _stock_dispatch(ffn)
        ffn.experts.forward = ffn._vk_stock_dispatch

        def make_fwd(ffn=ffn):
            def fwd(x):
                B, T, C = x.shape
                xf = x.reshape(-1, C)
                w, idx, aux = ffn.router(xf)
                y = ffn.experts.forward(xf, idx, w)
                return y.view(B, T, C), aux
            return fwd

        ffn.forward = make_fwd()
        ffn._vk_hosted = True
    _patch_model_logits()
    return blocks


def _patch_model_logits():
    """d0.model_logits must unwrap VesperLinearLM's (logits, ce, aux) tuple
    and run CUDA forwards under fp16 autocast (fp32 master weights)."""
    if getattr(d0.model_logits, "_vk_patched", False):
        return

    def model_logits(model, tokens):
        use_ac = tokens.is_cuda
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_ac):
            out = model(tokens)
        if isinstance(out, tuple):
            return out[0].float()
        lg = out.logits if hasattr(out, "logits") else out
        return lg.float()

    model_logits._vk_patched = True
    d0.model_logits = model_logits


# ---------------------------------------------------------------------------
# Graft gate — MixGateD2 semantics over the NATIVE passport socket
# ---------------------------------------------------------------------------

class VkGraftGate(nn.Module):
    """Passport-bank gate with plug rows: stock select bit-stable, plugs vote
    membership (score > kth stock score) and carry p_home for varact.

    forward(x_flat) -> (weights, idx, aux) matching PassportRouter's contract
    so a stock ffn.forward body works with either gate installed.
    """

    def __init__(self, bank: "d0.PassportBank", n_new: int, top_k: int = 2):
        super().__init__()
        self.bank = bank
        self.n_new = n_new
        self.top_k = top_k
        self.last_membership = None
        self.last_p_new = None
        self.last_h = None
        self.last_logits = None

    def forward(self, x: torch.Tensor):
        h = x.reshape(-1, self.bank.d_model)
        scores = self.bank.scores(h)                 # (N, E_stock + n_new) fp32
        n_stock = self.bank.num_experts - self.n_new
        stock_scores = scores[:, :n_stock]
        weights = F.softmax(stock_scores, dim=-1)
        top_w, top_i = torch.topk(weights, self.top_k, dim=-1)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)
        kth = torch.topk(stock_scores, self.top_k, dim=-1).values[:, -1]
        if self.n_new > 0:
            self.last_membership = scores[:, n_stock:] > kth.unsqueeze(-1)
            self.last_p_new = F.softmax(scores, dim=-1)[:, n_stock:]
        else:
            self.last_membership = None
            self.last_p_new = None
        self.last_h = h.detach()
        self.last_logits = scores.detach()
        # aux mirrors the native Switch-style load balance on the stock rows
        mean_router_probs = weights.mean(dim=0)
        expert_mask = torch.zeros_like(weights).scatter_(1, top_i, 1.0)
        aux = n_stock * torch.sum(mean_router_probs * expert_mask.mean(dim=0))
        return top_w.to(x.dtype), top_i, aux


def banks_from_native(model: nn.Module, n_extra: int = 0) -> list:
    """init=exact registry: P = native passports, W_q = native query.weight.

    With n_extra plug rows appended (zero-init; row recal fills them).
    """
    banks = []
    for ffn in find_moe_blocks(model):
        r = ffn._vk_stock_router if hasattr(ffn, "_vk_stock_router") else ffn.router
        P = r.passports.data.float()
        Wq = r.query.weight.data.float()
        E, d_r = P.shape
        bank = d0.PassportBank(E + n_extra, Wq.shape[1], d_r=d_r, learn_wq=True)
        with torch.no_grad():
            bank.P.data[:E] = P
            if n_extra:
                bank.P.data[E:] = 0.0
            bank.W_q.data.copy_(Wq)
        bank.layer_scale = 1.0
        banks.append(bank)
    return banks


def install_graft_gate(ffn, bank, n_new: int) -> nn.Module:
    """Swap ffn.router for a VkGraftGate; returns the stock router."""
    stock = ffn.router
    ffn.router = VkGraftGate(bank, n_new, top_k=ffn.top_k)
    return stock


def uninstall_graft_gate(ffn, stock_router) -> None:
    ffn.router = stock_router


# ---------------------------------------------------------------------------
# Parity gate — logit cos vs bare (standing bar 0.95)
# ---------------------------------------------------------------------------

@torch.no_grad()
def parity_measure(model, blocks, banks, batches, device, n_new: int = 0) -> dict:
    """Two passes: bare (stock routers) vs grafted (VkGraftGate, plugs off)."""
    sum_cos = sum_sq = n_tok = 0
    cos_min = 1.0
    ce_a = ce_b = n_row = 0.0
    # pass A: bare
    outs_a = []
    for tokens, _lab in batches:
        tokens = tokens.to(device)
        lg = d0.model_logits(model, tokens).float().cpu()
        outs_a.append((tokens.cpu(), lg))
        ce_a += d0.next_token_ce(lg, tokens.cpu()) * tokens.shape[0]
        n_row += tokens.shape[0]
    # pass B: graft gates installed (delivery never installed -> stock only)
    stocks = [install_graft_gate(ffn, bank, n_new)
              for ffn, bank in zip(blocks, banks)]
    try:
        for tokens, lg_a in outs_a:
            tokens = tokens.to(device)
            lg_b = d0.model_logits(model, tokens).float().cpu()
            cos = d0.token_logit_cosine(lg_a, lg_b)
            sum_cos += float(cos.sum())
            sum_sq += float((cos * cos).sum())
            cos_min = min(cos_min, float(cos.min()))
            n_tok += cos.numel()
            ce_b += d0.next_token_ce(lg_b, tokens) * tokens.shape[0]
    finally:
        for ffn, st in zip(blocks, stocks):
            uninstall_graft_gate(ffn, st)
    mean = sum_cos / max(1, n_tok)
    var = max(0.0, sum_sq / max(1, n_tok) - mean * mean)
    return {"n_tokens": int(n_tok), "logit_cosine_mean": mean,
            "logit_cosine_std": math.sqrt(var), "logit_cosine_min": cos_min,
            "ce_bare": ce_a / max(1, n_row), "ce_graft": ce_b / max(1, n_row),
            "ce_delta": (ce_b - ce_a) / max(1, n_row)}


# ---------------------------------------------------------------------------
# Host-native expert core extract (mem plugs' frozen cores)
# ---------------------------------------------------------------------------

def extract_core(ffn, expert: int):
    """vk FeedForward(e) -> (w_gu (2I,H), w_down (H,I)) like d25.extract_donor_core."""
    ex = ffn.experts[expert]
    w_gu = torch.cat([ex.w1.weight.detach().float(), ex.w3.weight.detach().float()], 0)
    w_down = ex.w2.weight.detach().float().clone()
    return w_gu, w_down


# ---------------------------------------------------------------------------
# ChatML + decode (kv_cache_v05 conventions; VesperLinearLM backend)
# ---------------------------------------------------------------------------

def chatml(prompt: str, answer: str | None = None) -> str:
    """The SFT/eval_chat chat template (verbatim — talkability was measured
    in this format): <|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n"""
    im_s, im_e = "<|im_start|>", "<|im_end|>"
    txt = f"{im_s}user\n{prompt}{im_e}\n{im_s}assistant\n"
    if answer is not None:
        txt += f"{answer}{im_e}\n"
    return txt


def chatml_ids(tok, prompt: str, answer: str | None = None) -> list:
    return tok.encode(chatml(prompt, answer), add_special_tokens=False)


def to_batches(ids, label, seq, bs):
    wins = [ids[i:i + seq] for i in range(0, len(ids) - seq + 1, seq)]
    return [(torch.tensor(wins[i:i + bs], dtype=torch.long), label)
            for i in range(0, len(wins) - bs + 1, bs)]


class VkKVDecoder:
    """Incremental decode for the assembled vk-base (bare or grafted).

    Same contract as kv_cache_v05.KVDecoder (lab/kv_cache_v05.py, read-only
    conventions): generate(messages, n, mode="bare"|"grafted"), cache reused
    only under the mode that built it.  KDA layers thread fla recurrent state
    (RecurrentStateCache); MLA layers use fla's past_key_values (fla Cache).
    Falls back to full-forward decode if the cached path misbehaves.
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
        self.cached_ids = None
        self.cache_mode = None
        self.caches = None
        self.incremental_ok = True
        self.note = ""
        if stop_ids is None:
            stop_ids = set()
            for attr in ("eos_token_id", "pad_token_id"):
                v = getattr(tok, attr, None)
                if v is not None:
                    stop_ids.add(int(v))
            v = tok.convert_tokens_to_ids("im_end")
            if v is not None and v != tok.unk_token_id:
                stop_ids.add(int(v))
        self.stop_ids = stop_ids

    def reset(self):
        self.caches = None
        self.cached_ids = None
        self.cache_mode = None

    def _state(self, mode, fn):
        import grafted_spine_v0 as g0
        if self.blocks is None or mode == "bare":
            return g0.run_in_state(self.model, self.blocks, self.fam,
                                   None, None, "off", fn)
        return g0.run_in_state(self.model, self.blocks, self.fam,
                               self.gates, self.handles, "both", fn)

    def _patch_mla_cache(self):
        """Thread fla's past_key_values through the MLA wrapper (the stock
        wrapper raises on cache!=None — 'not wired'). Instance-level only."""
        for i, lt in enumerate(self.model.layer_types):
            if lt != "mla":
                continue
            attn = self.model.layers[i]["attn"]
            if getattr(attn, "_vk_cache_patched", False):
                continue

            def make(attn=attn):
                def fwd(x, freqs_cis=None, cache=None, start_pos=0,
                        value_tokens=None):
                    if attn.value_emb is not None:
                        attn._ve_tokens = value_tokens
                    try:
                        if cache is not None:
                            out, _, _ = attn.mla(x, attention_mask=None,
                                                 past_key_values=cache,
                                                 use_cache=True)
                        else:
                            out, _, _ = attn.mla(x, attention_mask=None,
                                                 use_cache=False)
                        return out
                    finally:
                        if attn.value_emb is not None:
                            attn._ve_tokens = None
                return fwd

            attn.forward = make()
            attn._vk_cache_patched = True

    def _new_caches(self):
        import vesper_linear_model as vlm
        self._patch_mla_cache()
        layers = self.model.layer_types
        kv = {}
        for i, lt in enumerate(layers):
            if lt == "full":
                attn = self.model.layers[i]["attn"]
                shape = (1, attn.n_kv_heads, self.model.max_seq_len, attn.head_dim)
                dtype = self.model.tok_embeddings.weight.dtype
                kv[i] = {"k": torch.zeros(shape, device=self.device, dtype=dtype),
                         "v": torch.zeros(shape, device=self.device, dtype=dtype)}
        try:
            from transformers.cache_utils import DynamicCache
            mla = DynamicCache()
        except Exception:  # noqa: BLE001
            mla = None
        return {"kv": kv, "linear": vlm.RecurrentStateCache(), "mla": mla,
                "pos": 0}

    def _step(self, ids_2d):
        """One cached model call for tokens at caches['pos'].. -> (1,T,V)."""
        model = self.model
        caches = self.caches
        pos = caches["pos"]
        B, T = ids_2d.shape
        use_ac = ids_2d.is_cuda
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_ac):
            x = model.tok_embeddings(ids_2d)
            for i, layer in enumerate(model.layers):
                lt = model.layer_types[i]
                attn = layer["attn"]
                xn = layer["attn_norm"](x)
                if lt == "kda":
                    attn_out = attn(xn, cache=caches["linear"], value_tokens=ids_2d)
                elif lt == "mla":
                    attn_out = attn(xn, cache=caches["mla"], value_tokens=ids_2d)
                else:  # full GQA
                    attn_out = attn(xn, model.freqs_cis, cache=caches["kv"][i],
                                    start_pos=pos, value_tokens=ids_2d)
                h = x + attn_out
                ffn_out, _aux = layer["ffn"](layer["ffn_norm"](h))
                x = h + ffn_out
            x = model.norm(x)
            logits = model.output(x)
        caches["pos"] = pos + T
        return logits.float()

    @torch.no_grad()
    def forward_full(self, ids, mode):
        def run():
            self.reset()
            x = torch.tensor([ids], dtype=torch.long, device=self.device)
            self.cached_ids = list(ids)
            self.cache_mode = mode
            lg = d0.model_logits(self.model, x)
            return lg[0].float().cpu()
        return self._state(mode, run)

    @torch.no_grad()
    def _prefill(self, ids, mode):
        """One cached call over the whole prompt; returns per-position logits.
        Self-checks the last position against the full forward once and
        disables incremental decode on any mismatch (safe > fast)."""
        def run():
            self.reset()
            self.cache_mode = mode
            self.caches = self._new_caches()
            x = torch.tensor([ids], dtype=torch.long, device=self.device)
            lg = self._step(x)[0].float().cpu()          # (T, V)
            self.cached_ids = list(ids)
            return lg
        lg = self._state(mode, run)
        if not getattr(self, "_checked", False):
            self._checked = True
            lg_full = self.forward_full(ids, mode)
            arg_ok = int(torch.argmax(lg[-1])) == int(torch.argmax(lg_full[-1]))
            cos = float(F.cosine_similarity(lg[-1].unsqueeze(0),
                                            lg_full[-1].unsqueeze(0), dim=-1))
            if not arg_ok or cos < 0.99:
                self.incremental_ok = False
                self.note = (f"kv self-check failed (cos {cos:.4f} "
                             f"arg_ok={arg_ok}); full-forward decode")
                self.reset()
            else:
                self.note = f"kv self-check cos {cos:.4f}"
        return lg

    def _prepare(self, ids, mode):
        if mode != self.cache_mode or not self.cached_ids:
            self.reset()
            self.cache_mode = mode
        cached = self.cached_ids or []
        L = 0
        for a, b in zip(cached, ids):
            if a != b:
                break
            L += 1
        if L >= len(ids):
            L = max(0, len(ids) - 1)
        if L < len(cached):
            self.reset()     # crop not supported on fla caches: recompute
        return ids[L:]

    @torch.no_grad()
    def generate(self, messages, max_new_tokens, mode="grafted",
                 sample=False, temperature=0.8, top_p=0.95, seed=0):
        """One assistant turn -> (text, ids_new). KV-incremental when possible."""
        im_s, im_e = "<|im_start|>", "<|im_end|>"
        prompt = "".join(f"{im_s}{m['role']}\n{m['content']}{im_e}\n"
                         for m in messages) + f"{im_s}assistant\n"
        ids = self.tok.encode(prompt, add_special_tokens=False)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        out_ids = []
        for _ in range(max_new_tokens):
            step_ids = ids + out_ids
            if self.incremental_ok and self.caches is not None \
                    and self.cache_mode == mode and self.cached_ids:
                suffix = self._prepare(step_ids, mode)
                if suffix:
                    def run(suffix=suffix):
                        lg_last = None
                        for t in range(0, len(suffix), 1):
                            x = torch.tensor([[suffix[t]]], dtype=torch.long,
                                             device=self.device)
                            lg_last = self._step(x)[0, -1].float().cpu()
                        self.cached_ids = list(step_ids)
                        return lg_last
                    try:
                        lg = self._state(mode, run)
                    except Exception as e:  # noqa: BLE001
                        self.incremental_ok = False
                        self.note = f"incremental failed ({e}); full-forward"
                        self.reset()
                        lg = self.forward_full(step_ids, mode)[-1]
                else:
                    lg = self.forward_full(step_ids, mode)[-1]
                    self.cache_mode = mode
            else:
                # (re)build the kv cache from the full prompt, then continue
                try:
                    lg_all = self._prefill(step_ids, mode)
                    lg = lg_all[-1]
                except Exception as e:  # noqa: BLE001
                    self.incremental_ok = False
                    self.note = f"incremental failed ({e}); full-forward"
                    self.reset()
                    lg = self.forward_full(step_ids, mode)[-1]
            nxt = lg.unsqueeze(0)
            if sample:
                g = torch.Generator(device="cpu")
                g.manual_seed(seed + len(out_ids))
                logits = nxt[0].float() / max(1e-5, temperature)
                probs = F.softmax(logits, dim=-1)
                sp, si = torch.sort(probs, descending=True)
                cp = torch.cumsum(sp, dim=-1)
                sp[cp - sp > top_p] = 0.0
                sp = sp / sp.sum()
                pick = int(si[torch.multinomial(sp, 1, generator=g)])
            else:
                pick = int(torch.argmax(nxt[0]))
            if pick in self.stop_ids:
                break
            out_ids.append(pick)
        return self.tok.decode(out_ids, skip_special_tokens=False), out_ids
