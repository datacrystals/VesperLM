#!/usr/bin/env python3
"""glm_flash_gen — bake-off generator shim for the GLM-Flash spine arm.

`lab/spine_bakeoff.py --spine glm-flash` needs a `make_generator()`; the
GLM-Flash host (agent-66, lab/graft_glmflash.py) is a parity/PoE harness and
exposes none.  This shim is MY file (the bake-off lane): it borrows their
host builder + their wrap-poe plugin format and speaks the same generator
contract as mimo_flash_host.make_generator.

    tok, gen = make_generator(plugins_dir, args)

Plugins: `<plugins_dir>/*_plugin.pt` in the wrap-poe format
(`wrap_state`/`wrap_spec`/`site`/`core`/`host`) — e.g.
/data/tliao/glmflash_out/wrap_poe_mem_trains_plugin.pt.  Installed with the
WrapPlug additive hook at their recorded site (lam 1.0 as fitted).
"""

from __future__ import annotations

import glob
import os
import re
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "lab"))

GLM_MODEL_DIR = "/data/ModelDownloader/GLM-5.3-Flash-AWQ-INT4"
GLM_DEFAULT_LAYERS = 6   # the wrap-poe host depth (agent-66's PoE run)


class WrapPlug(nn.Module):
    """Additive wrap delivery — the graft_glmflash_wrap_poe hook shape."""

    def __init__(self, wrap: nn.Module, lam: float = 1.0):
        super().__init__()
        self.wrap = wrap
        self.lam = lam
        self.last_new_out = None
        self._post = None

    def install(self, blk: nn.Module):
        def post(mod, inputs, output):
            h = inputs[0]
            delta = self.wrap(h.reshape(-1, h.shape[-1])).view_as(h)
            self.last_new_out = delta.detach()
            return output + self.lam * delta
        self._post = blk.register_forward_hook(post)

    def remove(self):
        if self._post is not None:
            self._post.remove()
            self._post = None


def _install_plugs(model, blocks, plugins_dir: str):
    import door25_wrap_expert as d25
    installed = []
    if not plugins_dir or not os.path.isdir(plugins_dir):
        return installed
    for p in sorted(glob.glob(os.path.join(plugins_dir, "*_plugin.pt"))):
        rec = torch.load(p, map_location="cpu", weights_only=False)
        if "wrap_state" not in rec:
            continue
        spec = rec.get("wrap_spec", {})
        wrap = d25.WrappedForeignExpert(
            int(spec.get("d_model", 4096)), int(spec.get("d_canon", 64)),
            torch.zeros(2, 2), torch.zeros(2, 2))  # shapes from state dict
        # rebuild buffers with the recorded core geometry
        if "core_w_gu" in rec["wrap_state"]:
            w_gu = rec["wrap_state"]["core_w_gu"]
            w_down = rec["wrap_state"]["core_w_down"]
            wrap = d25.WrappedForeignExpert(
                w_gu.shape[1], int(spec.get("d_canon", 64)), w_gu, w_down,
                act=spec.get("act", "swiglu"),
                situ_beta=spec.get("situ_beta", 1.0),
                situ_linear_beta=spec.get("situ_linear_beta"))
        wrap.load_state_dict(rec["wrap_state"])
        wrap.eval()
        site = int(rec.get("site", 0))
        if site >= len(blocks):
            continue
        plug = WrapPlug(wrap, lam=1.0)
        plug.install(blocks[site])
        installed.append((os.path.basename(p), site, plug))
    return installed


def _patch_fl_l2norm():
    """gfx906 fallback: fla's l2norm Triton kernels fail to compile
    ('unsupported target: gfx906' — measured 2026-10-16).  Replace with a
    pure-torch row L2 norm; numerics match the kernel (fp32 accumulate)."""
    try:
        import fla.modules.l2norm as fn
    except Exception:  # noqa: BLE001
        return False

    def l2norm(x, eps=1e-6, output_dtype=None):
        xf = x.float()
        rstd = torch.rsqrt(xf.pow(2).sum(-1, keepdim=True) + eps)
        y = (xf * rstd).to(output_dtype or x.dtype)
        return y

    fn.l2norm = l2norm
    fn.l2_norm = l2norm
    return True



# think-tag / EOT hygiene: the baked-in reasoning tag is NOT a tokenizer
# special on every spine, so skip_special_tokens alone leaves it in replies
# (measured 2026-10-16: every slice-smoke reply opened with the raw tag).
_THINK_RE = re.compile(r"<\|[^|>]{1,24}\|>.*?<\|[^|>]{1,24}\|>",
                       re.DOTALL)
_TAG_RE = re.compile(r"<\|[^|>]{1,24}\|>")


def _clean_reply(text: str) -> str:
    """Strip reasoning blocks and stray delimiter tokens, keep the answer.

    Measured 2026-10-16 slice-smoke replies: the model emits the opening
    delimiter then answers directly (never closes) — so a dangling-opener
    cut would delete the answer.  Rule: drop closed reasoning pairs first,
    then drop any residual lone delimiter tokens, keep the rest.
    """
    t = _THINK_RE.sub(" ", text)
    t = _TAG_RE.sub(" ", t)
    return " ".join(t.split()).strip()


def stop_token_ids(tok) -> set:
    """EOT ids derived from the tokenizer chat template (no literals): the
    tail tokens the template emits after an assistant message are the
    end-of-turn marker; max-new is a cap, not a target."""
    ids = set()
    try:
        probe = tok.apply_chat_template(
            [{"role": "assistant", "content": "x"}], tokenize=False,
            add_generation_prompt=False)
        tail = tok.encode(probe, add_special_tokens=False)[-4:]
        ids.update(t for t in tail if t >= 0)
    except Exception:  # noqa: BLE001
        pass
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    return ids


def make_generator(plugins_dir: str, args=None):
    """(tok, gen_fn) for lab/spine_bakeoff.py --spine glm-flash."""
    import graft_glmflash as gl
    from transformers import AutoTokenizer

    _patch_fl_l2norm()

    device = getattr(args, "device", "cuda") if args is not None else "cuda"
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    n_layers = int(getattr(args, "n_layers", GLM_DEFAULT_LAYERS)
                   or GLM_DEFAULT_LAYERS)

    tok = AutoTokenizer.from_pretrained(GLM_MODEL_DIR)
    model, lm_head, _index, meta = gl.build_host(
        GLM_MODEL_DIR, device, dtype, max_layers=n_layers)
    blocks = gl.find_moe_blocks(model)
    installed = _install_plugs(model, blocks, plugins_dir)

    def gen_fn(messages, max_new, dec, seed=0):
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
        ids = tok.encode(prompt, add_special_tokens=False)
        out_ids = []
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        stop = stop_token_ids(tok)
        for _ in range(int(max_new)):
            x = torch.tensor([ids + out_ids], dtype=torch.long,
                             device=device)
            lg = gl.host_logits(model, lm_head, x)[0, -1].float().cpu()
            pen = float(dec.get("rep_penalty", 1.0))
            if pen > 1.0:
                for t in set(ids + out_ids):
                    if 0 <= t < lg.shape[-1]:
                        lg[t] = lg[t] / pen if lg[t] > 0 else lg[t] * pen
            if dec.get("sample"):
                temp = max(1e-5, float(dec.get("temperature", 0.8)))
                top_p = float(dec.get("top_p", 0.95))
                probs = F.softmax(lg / temp, dim=-1)
                sp, si = torch.sort(probs, descending=True)
                cum = sp.cumsum(0)
                nz = (cum > top_p).nonzero(as_tuple=True)[0]
                cut = int(nz[:1].item()) + 1 if nz.numel() else sp.numel()
                p = sp[:cut] / sp[:cut].sum()
                nxt = int(si[torch.multinomial(p, 1, generator=g)])
            else:
                nxt = int(torch.argmax(lg))
            if nxt in stop:
                break
            out_ids.append(nxt)
        return _clean_reply(tok.decode(out_ids, skip_special_tokens=True))

    gen_fn.installed = [n for n, _s, _p in installed]
    gen_fn.host = {"layers": n_layers, "sparse": len(blocks),
                   "meta": {k: meta[k] for k in list(meta)[:6]}}
    return tok, gen_fn
