#!/usr/bin/env python3
"""mimo_pack_matrix — pack-on-latched matrix verify (final lane stage).

Recipe per lab/MIMO_MEMBERSHIP_THROTTLE.md: per plug, sentinel24 row at
model-layer-24 post-attn-norm, prefill-SHADOW latch (flag computed
before the real injected prefill), winner-take-all: the sentinel with
the highest content-token prefill score over margin 10 wins; its wrap
injected at sites {0,23,46} with its fitted λ share.  Latch held for
the whole answer.

Arms per plug: gated solo (only that plug registered) on P_HOME,
P_TRAINS, P_MATH.  Pack arm: all six registered together on two home
conversations (mem_poi favorite-show, mem_gundam anime), one cold math.
"""
from __future__ import annotations

import os
import string
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/data/tliao/mimo_flash_lab")
import mimo_full_host as fh  # noqa: E402
import mimo_flash_host as mh  # noqa: E402
import door25_wrap_expert as d25  # noqa: E402
from mimo_plug_rows import PLUGS  # noqa: E402

PLUG_DIR = os.environ.get("MIMO_PLUG_DIR", "/home/tliao/mimo_flash/plugs")
SEN_DIR = os.environ.get("MIMO_SEN_DIR", "/home/tliao/mimo_flash")
TAGS = ["mem_poi", "mem_machine", "mem_gundam",
        "mem_cars", "mem_trains", "mem_igbt"]
SITES = [0, 23, 46]
SEN_MARGIN = 10.0
TRAIN_PROBE = "What do you think of trains?"
MATH_PROBE = "What is 17 + 25?"
HOME = {
    "mem_poi": "What show should I watch?",
    "mem_machine": "Do you have a favorite on the show?",
    "mem_gundam": "What's your favorite anime?",
    "mem_cars": "What do you think of cars?",
    "mem_trains": "What do you think of trains?",
    "mem_igbt": "What do you think of IGBTs?",
}
KW_PREFIX = {
    "mem_poi": ["person of interest"],
    "mem_machine": ["machine"],
    "mem_gundam": ["gundam"],
    "mem_cars": ["hate"],
    "mem_trains": ["love"],
    "mem_igbt": ["igbt", "like"],
}
BASEWIN = ("The history of economics is marked by long cycles of "
           "expansion and contraction, and each period leaves behind "
           "institutions that condition the next.")


def _sen_path(tag):
    for base in (SEN_DIR, "/home/tliao/mimo_flash"):
        for stem in (f"sentinel24_{tag}",
                     "sentinel24_" + tag.replace("mem_", "")):
            p = os.path.join(base, stem + ".pt")
            if os.path.exists(p):
                return p
    raise FileNotFoundError(f"no sentinel24 row for {tag}")


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def pr_ratio(s: str) -> float:
    ok = set(string.printable)
    return sum(c in ok for c in s) / max(1, len(s))


def prompt_ids(tok, text):
    return tok.encode(tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True), add_special_tokens=False)


class Pack:
    def __init__(self, model, blocks, tags):
        self.tags = tags
        self.blocks = blocks
        self.fwds = [(blocks[i], blocks[i].experts.forward) for i in SITES]
        self.wraps = {}
        self.sentinels = {}
        self.state = {"tag": None}
        ex_core = blocks[0].experts[0]
        for tag in tags:
            rec = torch.load(os.path.join(PLUG_DIR, f"{tag}_plugin_mp.pt"),
                             map_location="cpu", weights_only=False)
            w_gu, w_down = ex_core.native_core()
            wrap = d25.WrappedForeignExpert(4096, 64, w_gu, w_down,
                                            seed=0, act="swiglu")
            wrap.load_state_dict(rec["wrap"])
            wrap.to("cuda").eval()
            self.wraps[tag] = wrap
            srow_path = _sen_path(tag)
            srow = torch.load(srow_path,
                              weights_only=False).float().cuda().squeeze()
            self.sentinels[tag] = srow
            lam_i = rec.get("lam_i", 0.1666667)
            wrap.lam_i = lam_i

    def _fenced_factory(self, fwd):
        def f(h, i, w):
            y = fwd(h, i, w)
            tag = self.state["tag"]
            if tag is not None:
                wr = self.wraps[tag]
                y = y + (wr.lam_i * wr(h)).to(h.dtype)
            return y
        return f

    def arm(self, tags):
        self.tags = tags
        self.state["tag"] = None
        for blk_, fwd in self.fwds:
            blk_.experts.forward = self._fenced_factory(fwd)

    def restore(self):
        for blk_, fwd in self.fwds:
            blk_.experts.forward = fwd


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "pack"
    model, tok, cache = fh.build_full_host(fh.MODEL_DIR, "cuda", 18.0, {})
    blocks = mh.find_mimo_moe_blocks(model)
    blk23 = model.model.layers[24]
    hs = {}

    def hook(_m, _i, out):
        hs["h"] = (out[0] if isinstance(out, tuple) else out)
    hnd = blk23.post_attention_layernorm.register_forward_hook(hook)
    pack = Pack(model, blocks, TAGS)

    def latch_tag(ids):
        hs.clear()
        pack.state["tag"] = None
        with torch.no_grad():
            model(input_ids=torch.tensor([ids], device="cuda"),
                  use_cache=False)
        h = hs["h"].float().reshape(-1, 4096)
        best = None
        for tag in pack.tags:
            r = float((h * pack.sentinels[tag].unsqueeze(0)).sum(-1)[3:].max())
            if r > SEN_MARGIN and (best is None or r > best[1]):
                best = (tag, r)
        return best[0] if best else None

    def gen(text, n=24):
        ids = prompt_ids(tok, text)
        tag = latch_tag(ids)
        pack.state["tag"] = tag
        past, outs = None, []
        with torch.no_grad():
            for i in range(n):
                x = torch.tensor([ids] if i == 0 else [[outs[-1]]],
                                 dtype=torch.long, device="cuda")
                o = model(input_ids=x, past_key_values=past, use_cache=True)
                past = o.past_key_values
                outs.append(int(o.logits[0, -1].float().cpu().argmax()))
        pack.state["tag"] = None
        return tok.decode(outs), tag

    def probe(arm, name, q, kw=None):
        t, tag = gen(q)
        pr = pr_ratio(t)
        has = "n/a" if kw is None else (
            "y" if all(k in t.lower() for k in kw) else "N")
        print(f"PPROBE {arm} {name} coh={'PASS' if pr >= 0.85 else 'SOUP'}"
              f"(pr={pr:.2f}) kw={has} latch={tag}", flush=True)
        print(f"PTEXT {arm} {name} {t!r}"[:320], flush=True)
        return t

    def parity_ce(idsx):
        with torch.no_grad():
            x = torch.tensor([idsx], device="cuda")
            o = model(input_ids=x, use_cache=False)
        return float(F.cross_entropy(o.logits[0, :-1].float(), x[0, 1:]))

    if mode in ("pack", "all"):
        pack.arm(TAGS)
        for tag in TAGS:
            probe("pack", f"P_HOME_{tag}", HOME[tag], KW_PREFIX[tag])
        probe("pack", "P_MATH", MATH_PROBE, None)
        # 4-turn conversation extracts on two homes (mem_poi + mem_gundam)
        conv_tag = None
        for home, qs in (("mem_poi", ["What show should I watch?", ]),
                         ("mem_gundam", ["What's your favorite anime?"])):
            print(f"CONV {home}", flush=True)
            for q in qs:
                t, tag = gen(q)
                print(f"  Q {q!r} -> latch={tag}", flush=True)
                print("  A " + repr(t)[:260], flush=True)

    if mode in ("solos", "all"):
        for tag in TAGS:
            pack.arm([tag])
            probe(f"solo:{tag}", "P_HOME", HOME[tag], KW_PREFIX[tag])
            probe(f"solo:{tag}", "P_TRAINS", TRAIN_PROBE,
                  KW_PREFIX[tag])
            probe(f"solo:{tag}", "P_MATH", MATH_PROBE, None)

    pack.restore()
    # parity: noplug vs pack-latched-fire on the cold window
    ce0 = parity_ce(tok.encode(BASEWIN, add_special_tokens=False)[:65])
    pack.arm(TAGS)
    pkg = latch_tag(tok.encode(BASEWIN, add_special_tokens=False)[:65])
    pack.state["tag"] = pkg
    ce1 = parity_ce(tok.encode(BASEWIN, add_special_tokens=False)[:65])
    pack.state["tag"] = None
    pack.restore()
    print(f"PARITY noplug={ce0:.4f} packlfull={ce1:.4f} "
          f"delta={ce1 - ce0:+.4f} (latch={pkg})", flush=True)
    print("PACK_DONE", flush=True)


if __name__ == "__main__":
    main()
