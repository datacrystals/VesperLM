#!/usr/bin/env python3
"""v0.2 fast chat CLI — v0.5 talkable model with KV-cached incremental decode.

Sibling of `lab/chat_v05.py` (same UX, same flags) with the decode loop
replaced by `lab/kv_cache_v05.KVDecoder`: one token at a time over a growing
per-layer K/V cache ("the notebook"), plug-in delivery and sparsemixer
selection unchanged (both are position-local — proven by the parity harness
in `lab/chat_v05_fast_validate.py`).  Each new token computes Q/K/V for
itself only; the full conversation is NOT re-forwarded per token.

The exact command the user runs:

    ssh tliao@100.64.0.7 'cd /data/tliao/vesper_sft/VesperLM && \
        /data/tliao/vesper_sft/venv/bin/python lab/chat_v05_fast.py'

Features (all of chat_v05, plus):
  * --no-kv falls back to the old full-forward-per-token path
  * printed tok/s per reply (prefill / decode / end-to-end)
  * cross-turn prefix reuse: a turn only forwards the uncached suffix of the
    new prompt (the cache is reused only under the same mode that built it)

In-chat commands: /help /reset /bare /grafted /greedy /sample T /sys TEXT /exit
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

# The user's launch line sets nothing — make the box's rocblas path the
# default BEFORE torch is imported so the exact command above works.
os.environ.setdefault("ROCBLAS_TENSILE_LIBPATH",
                      "/opt/rocm-6.2.4/lib/rocblas/library")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "lab"))

import torch  # noqa: E402

import graft_d0_parity as d0  # noqa: E402
import graft_d2_crossfamily as d2  # noqa: E402
import door6_expert_mix as d6  # noqa: E402
import grafted_spine_v0 as g0  # noqa: E402
import kv_cache_v05 as kv  # noqa: E402

DEFAULT_MODEL = "/data/tliao/Phi-mini-MoE-instruct"
DEFAULT_REGISTRY = "/data/tliao/graft_d2_out/registry_phimoe_f620b32c.pt"
DEFAULT_PLUGINS = "/data/tliao/v05_out"

PROBES = [
    ("hacking", "Walk me through the reconnaissance phase of a penetration "
                "test. What kind of information would you gather?"),
    ("cybersec", "Explain the difference between a vulnerability and an "
                 "exploit. Give one example of each."),
    ("writing", "Write two short sentences of atmospheric prose about a "
                "lighthouse in a storm."),
    ("memory", "What is my cat's name?"),
]


def pick_gpu(quiet=False) -> int:
    """Freest GPU on the box (co-tenants move around; GPU 0 is often busy)."""
    n = torch.cuda.device_count()
    if n == 0:
        raise SystemExit("no CUDA/HIP GPUs visible — check the box / driver")
    best, best_free = 0, -1
    for i in range(n):
        try:
            free, _total = torch.cuda.mem_get_info(i)
        except Exception:
            free = -1
        if free > best_free:
            best, best_free = i, free
    if not quiet:
        frees = []
        for i in range(n):
            try:
                f, t = torch.cuda.mem_get_info(i)
                frees.append(f"{i}:{f / 1e9:.1f}G")
            except Exception:
                frees.append(f"{i}:?")
        print(f"[chat] GPUs free {frees} -> picking {best}", flush=True)
    return best


def build_state(args, device):
    """Load spine + registry + plug-ins -> (model, tok, gates, handles, info).

    Read-only w.r.t. /data/tliao/v05_out (a sibling run may be rewriting those
    artifacts; we only ever torch.load them).  Same assembly as chat_v05.
    """
    for path, what in ((args.model, "spine"), (args.registry, "registry")):
        if not os.path.isdir(path) and not os.path.exists(path):
            raise SystemExit(f"missing {what} at {path}")
    print(f"[chat] loading spine {args.model} (fp16) ...", flush=True)
    model, tok = d2.load_donor(args.model, device, args.dtype)
    blocks = d2.find_moe_blocks(model)
    fam = d2.family_for(d2.detect_family(model), blocks)
    banks_reg, _ = d0.load_registry(args.registry)

    # every *_plugin.pt in the plugins dir (v0's 5 + v05's 3 when present)
    paths = sorted(glob.glob(os.path.join(args.plugins_dir, "*_plugin.pt")))
    paths = [p for p in paths if "joint_state" not in p]
    expected = ["k3_hack", "glm_cybersec", "mimo_writing",
                "code", "math", "mem_pet", "mem_edit", "mem_words"]
    found = {os.path.basename(p)[:-len("_plugin.pt")]: p for p in paths}
    if not found:
        raise SystemExit(
            f"no plug-ins in {args.plugins_dir} — run lab/v05_build.py first "
            f"(chat can still run: use --bare)")

    lams = {}
    js_rows = {}
    js_path = os.path.join(args.plugins_dir, "joint_state.pt")
    if os.path.exists(js_path):
        js = torch.load(js_path, map_location="cpu", weights_only=False)
        order = js.get("order") or [n for n in expected if n in found]
        for n, v in zip(order, js.get("lams", [])):
            lams[n] = float(v)
        js_rows = {n: r for n, r in (js.get("rows") or {}).items()}
    else:
        print("[chat] NOTE: no joint_state.pt — using default lam 0.5 per "
              "plug (assembled lam fit not found)", flush=True)

    site_rows, site_entries = {}, {}
    info = []
    for name in expected:
        if name not in found:
            continue
        wrap, row, meta, spec = d6.load_wrap_plugin(found[name])
        if not d6.wrap_is_finite(wrap, row):
            print(f"[chat] WARNING: non-finite plugin {name} — skipped",
                  flush=True)
            continue
        host = int(meta["host_layer"])
        lam = float(args.lam_map.get(name, lams.get(name, args.default_lam)))
        if name in js_rows and bool(torch.isfinite(js_rows[name]).all()):
            row = js_rows[name].float().clone()  # post-recal row (lam-fit time)
        site_rows.setdefault(host, []).append(row.to(device))
        site_entries.setdefault(host, []).append({
            "expert": wrap.to(device),
            "lam": torch.tensor(lam, device=device),
            "tag": name, "num_stock": banks_reg[host].num_experts})
        info.append({"name": name, "host": host, "home": meta.get("home"),
                     "lam": round(lam, 4),
                     "act": spec.get("act", "swiglu"),
                     "donor": meta.get("donor") or meta.get("donor_layer")})
    missing = [n for n in expected if n not in found]
    print(f"[chat] plug-ins live: {[i['name'] for i in info]}"
          + (f" | missing: {missing}" if missing else ""), flush=True)
    for i in info:
        print(f"  - {i['name']:14s} host L{i['host']:<2d} home {i['home']}"
              f" lam {i['lam']} act {i['act']}", flush=True)

    gates, site_banks = g0.build_gates(blocks, banks_reg, fam, site_rows)
    handles = g0.install_mix(blocks, fam, site_entries, mode="off")
    return (model, tok, blocks, fam, gates, handles, info, missing,
            site_entries, site_banks)


def fmt_stats(st: dict) -> str:
    dec = st.get("n_decode_steps", 0)
    d_s = st.get("decode_s", 0.0)
    p_s = st.get("prefill_s", 0.0)
    tokps = dec / d_s if d_s > 0 else 0.0
    return (f"[fast] {st.get('n_new_tokens', 0)} tok | "
            f"decode {tokps:.1f} tok/s | prefill "
            f"{st.get('prefill_tokens', 0) / p_s if p_s > 0 else 0:.0f} tok/s "
            f"({st.get('prefill_tokens', 0)} new + {st.get('reused_tokens', 0)} "
            f"cached of {st.get('prompt_tokens', 0)})")


@torch.no_grad()
def gen_turn(tok, dec, mode, messages, args, sample, temperature,
             out_file=None):
    """One assistant turn — KV path, or g0.gen_reply when --no-kv."""
    if dec is not None:
        text, ids, st = dec.generate(
            messages, args.max_new_tokens, mode, sample=sample,
            temperature=temperature, top_p=args.top_p, seed=args.seed)
        print(fmt_stats(st), flush=True)
        return text, ids
    def logits_fn(x):
        if mode == "bare":
            return g0.run_in_state(
                dec_model[0], dec_blocks[0], dec_fam[0], None, None, "off",
                lambda: d0.model_logits(dec_model[0], x.to(dec_device[0])))
        return g0.run_in_state(
            dec_model[0], dec_blocks[0], dec_fam[0], dec_gates[0],
            dec_handles[0], "both",
            lambda: d0.model_logits(dec_model[0], x.to(dec_device[0])))
    text, ids = g0.gen_reply(tok, logits_fn, messages, args.max_new_tokens,
                             dec_device[0], sample=sample,
                             temperature=temperature, top_p=args.top_p,
                             seed=args.seed)
    print(f"[old] {len(ids)} tok (full-forward per token)", flush=True)
    return text, ids


def run_probes(tok, dec, mode0, device, args, sample, temperature):
    print("=== PROBE SMOKE (v0.5 fast) ===", flush=True)
    messages = []
    for kind, user in PROBES:
        messages.append({"role": "user", "content": user})
        text, ids = gen_turn(tok, dec, mode0, messages, args, sample,
                             temperature)
        messages.append({"role": "assistant", "content": text})
        short = " ".join(text.split())[:220]
        print(f"\n[{kind}] Q: {user[:90]}", flush=True)
        print(f"[{kind}] A: {short}", flush=True)
        print(f"[{kind}] n_new_tokens={len(ids)}", flush=True)
    print("\n=== END PROBES ===", flush=True)


def interactive(tok, dec, device, args):
    is_tty = sys.stdin.isatty()
    sample = args.sample
    temperature = args.temperature
    mode = "bare" if args.bare else "grafted"
    messages = []
    print(f"[chat] ready ({mode}, "
          f"{'T=' + str(temperature) if sample else 'greedy'}, "
          f"{'KV' if dec is not None else 'no-KV'}) — type /help", flush=True)
    while True:
        try:
            if is_tty:
                prompt = f"\n[{mode}] you> "
            else:
                prompt = ""
            line = input(prompt)
        except (EOFError, KeyboardInterrupt):
            print("\n[chat] bye", flush=True)
            return
        line = line.strip()
        if not line:
            continue
        if line.startswith("/"):
            cmd = line.split(maxsplit=1)
            c = cmd[0].lower()
            if c in ("/exit", "/quit", "/q"):
                print("[chat] bye", flush=True)
                return
            if c == "/help":
                print("[chat] /reset /bare /grafted /greedy /sample T "
                      "/sys TEXT /exit", flush=True)
            elif c == "/reset":
                messages = []
                if dec is not None:
                    dec.reset()
                print("[chat] history cleared", flush=True)
            elif c == "/bare":
                mode = "bare"
                if dec is not None:
                    dec.reset()
                print("[chat] bare spine", flush=True)
            elif c == "/grafted":
                mode = "grafted"
                if dec is not None:
                    dec.reset()
                print("[chat] grafted v0.5", flush=True)
            elif c == "/greedy":
                sample = False
                print("[chat] greedy", flush=True)
            elif c == "/sample" and len(cmd) > 1:
                try:
                    temperature = float(cmd[1])
                    sample = True
                    print(f"[chat] sampling T={temperature}", flush=True)
                except ValueError:
                    print("[chat] usage: /sample 0.8", flush=True)
            elif c == "/sys" and len(cmd) > 1:
                messages = [{"role": "system", "content": cmd[1]}]
                if dec is not None:
                    dec.reset()
                print("[chat] system prompt set", flush=True)
            else:
                print("[chat] unknown command — /help", flush=True)
            continue
        messages.append({"role": "user", "content": line})
        try:
            text, ids = gen_turn(tok, dec, mode, messages, args, sample,
                                 temperature)
        except Exception as e:  # keep the chat alive on any decode hiccup
            messages.pop()
            if dec is not None:
                dec.reset()
            print(f"[chat] generation error: {type(e).__name__}: {e}",
                  flush=True)
            continue
        messages.append({"role": "assistant", "content": text})
        print(f"\n{mode}> {text}", flush=True)
        if not is_tty:
            print(f"[n_new_tokens={len(ids)}]", flush=True)


# --no-kv fallback state (module-level so gen_turn's closure can reach it)
dec_model, dec_blocks, dec_fam = [None], [None], [None]
dec_gates, dec_handles, dec_device = [None], [None], [None]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--registry", default=DEFAULT_REGISTRY)
    ap.add_argument("--plugins-dir", default=DEFAULT_PLUGINS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu", default="auto",
                    help="GPU index, or auto for the freest card")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--bare", action="store_true",
                    help="chat with the un-grafted spine")
    ap.add_argument("--probe", action="store_true",
                    help="run the 4 specialty smoke probes and exit")
    ap.add_argument("--sample", action="store_true",
                    help="temperature sampling (default greedy)")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--default-lam", type=float, default=0.5)
    ap.add_argument("--lam-override", default="",
                    help="comma name=val list, e.g. "
                         "glm_cybersec=0.3,mimo_writing=0.2 (bypasses "
                         "joint_state lam; for A/B probes)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-kv", action="store_true",
                    help="fall back to the old full-forward-per-token path")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"],
                    help="attention implementation for both paths (eager "
                         "showed better full-vs-KV greedy stability on fp16)")
    args = ap.parse_args()
    args.lam_map = {}
    if args.lam_override:
        for kv_ in args.lam_override.split(","):
            if not kv_.strip():
                continue
            k, _, v = kv_.partition("=")
            args.lam_map[k.strip()] = float(v)
        print(f"[chat] lam overrides: {args.lam_map}", flush=True)

    if args.gpu == "auto":
        dev_i = pick_gpu()
    else:
        dev_i = int(args.gpu)
    device = f"{args.device}:{dev_i}" if args.device == "cuda" else args.device
    torch.cuda.set_device(dev_i)

    if args.bare:
        model, tok = d2.load_donor(args.model, device, args.dtype)
        blocks = d2.find_moe_blocks(model)
        fam = d2.family_for(d2.detect_family(model), blocks)
        gates, handles = None, None
        info, missing = [], []
    else:
        (model, tok, blocks, fam, gates, handles, info,
         missing, _entries, _banks) = build_state(args, device)

    dec = None
    model.config._attn_implementation = args.attn
    print(f"[chat] attn implementation: {args.attn}", flush=True)
    if not args.no_kv:
        n = kv.install_fast_experts(model)
        print(f"[chat] fast expert merge installed on {n} banks "
              f"(fixed top-k gather for decode)", flush=True)
        dec = kv.KVDecoder(model, tok, device, blocks, fam, gates, handles)
    else:
        dec_model[0], dec_blocks[0], dec_fam[0] = model, blocks, fam
        dec_gates[0], dec_handles[0], dec_device[0] = gates, handles, device
        print("[chat] --no-kv: full-forward per token (old path)", flush=True)

    mode0 = "bare" if args.bare else "grafted"
    if args.probe:
        run_probes(tok, dec, mode0, device, args, args.sample, args.temperature)
        return 0
    interactive(tok, dec, device, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
