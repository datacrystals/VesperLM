#!/usr/bin/env python3
"""v0.6 chat CLI — talk to the impress-demo assembly on the box.

Interactive terminal chat over the v0.6 assembly: Qwen1.5-MoE-A2.7B-Chat
spine (14.3B total / 2.7B active) + D2 passport registry + 5 grafted
plug-ins (K3 hacking / GLM-Flash cybersec / MiMo-Flash writing + 2 taught
memories pet_kepler / editor_helix), balanced additive-λ delivery.

The exact command the user runs:

    ssh tliao@100.64.0.7 'cd /data/tliao/vesper_sft/VesperLM && \
        /data/tliao/vesper_sft/venv/bin/python lab/chat_v06.py'

Features (chat_v05 conventions):
  * greedy (default) or temperature sampling (--sample T, /sample T live)
  * --bare chats with the un-grafted spine (/bare /grafted toggle live)
  * --probe  = the 4-turn specialty smoke (hacking/writing/memory/general)
  * --impress = the impress test: graft question + 4 probes + tok/s
  * auto freest-GPU pick; loads whatever plug-ins exist and says which are
    missing — chat works while fits are still running
  * KV-cache incremental decode if agent-49's module is present in lab/
    (auto-detected, read-only); otherwise full-forward decode like v0.5

In-chat commands: /help /reset /bare /grafted /greedy /sample T /sys TEXT
/graftq /toks /exit
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

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
import varact_gate as vact  # noqa: E402  variable-activation plug gating
import mem_pack as mp  # noqa: E402  persona memory pack (lab/memories/)

DEFAULT_MODEL = "/data/tliao/v06_models/Qwen1.5-MoE-A2.7B-Chat"
DEFAULT_REGISTRY = "/data/tliao/v06_out/registry_qwen2moe.pt"
DEFAULT_PLUGINS = "/data/tliao/v06_out"

# Persona memory pack (lab/memories/persona_v1.json) — DEFAULT ON, additive
# to the shipped mem_pet/mem_edit rows; --mem-pack legacy restores the exact
# shipped v0.6 world.  See chat_v05.DEFAULT_MEM_PACK for the full note.
DEFAULT_MEM_PACK = "persona"

# Variable-activation plug gating (lab/varact_gate.py, MODULAR_MOE 10.9):
# tau=0.5 shipped default (validated on the v0.5 bank; same delivery
# contract here).  --varact off|tau0.3|tau0.5|... overrides; joint_state
# ["varact"] overrides DEFAULT_VARACT; CLI wins over everything.
DEFAULT_VARACT = "tau0.5"

# The graft question from the user's session — the impress-test (a).
GRAFT_QUESTION = ("What would you say if I told you I grafted Kimi K3, GLM, "
                  "and MiMo weights into you?")

PROBES = [
    ("hacking", "Walk me through the reconnaissance phase of a penetration "
                "test. What kind of information would you gather?"),
    ("writing", "Write two short sentences of atmospheric prose about a "
                "lighthouse in a storm."),
    ("memory", "What is my cat's name?"),
    ("general", "In two sentences: what is a mutex and when would you use "
                "one?"),
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


def find_kv_module():
    """Use agent-49's incremental-decode module (lab/kv_cache_v05.KVDecoder)
    if present — coordinate read-only, never patch or duplicate it."""
    try:
        import kv_cache_v05 as kv
    except Exception as e:  # noqa: BLE001
        print(f"[chat] lab/kv_cache_v05 unavailable ({e}) — falling back to "
              f"full-forward decode", flush=True)
        return None
    if not hasattr(kv, "KVDecoder"):
        print("[chat] kv_cache_v05 has no KVDecoder — full-forward decode",
              flush=True)
        return None
    return kv.KVDecoder


def joint_state_path(args) -> str:
    """persona pack state first (additive), then the shipped joint_state.pt."""
    mem_pack = getattr(args, "mem_pack", "") or DEFAULT_MEM_PACK
    if mem_pack == "persona":
        p = os.path.join(args.plugins_dir, "joint_state_persona.pt")
        if os.path.exists(p):
            return p
    return os.path.join(args.plugins_dir, "joint_state.pt")


def build_state(args, device):
    """Load spine + registry + plug-ins -> (model, tok, gates, handles, info)."""
    for path, what in ((args.model, "spine"), (args.registry, "registry")):
        if not os.path.isdir(path) and not os.path.exists(path):
            raise SystemExit(f"missing {what} at {path}")
    print(f"[chat] loading spine {args.model} (fp16) ...", flush=True)
    model, tok = d2.load_donor(args.model, device, args.dtype)
    blocks = d2.find_moe_blocks(model)
    fam = d2.family_for(d2.detect_family(model), blocks)
    banks_reg, _ = d0.load_registry(args.registry)
    d_model = banks_reg[0].d_model

    paths = sorted(glob.glob(os.path.join(args.plugins_dir, "*_plugin.pt")))
    paths = [p for p in paths if "joint_state" not in p]
    expected = ["k3_hack", "glm_cybersec", "mimo_writing",
                "mem_pet", "mem_edit"]
    mem_pack = getattr(args, "mem_pack", "") or DEFAULT_MEM_PACK
    if mem_pack == "persona":
        expected += mp.tags(mp.load_pack())
    found = {os.path.basename(p)[:-len("_plugin.pt")]: p for p in paths}
    if not found:
        raise SystemExit(
            f"no plug-ins in {args.plugins_dir} — run lab/v06_build.py first "
            f"(chat can still run: use --bare)")

    lams = {}
    js_rows = {}
    js_path = joint_state_path(args)
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
            row = js_rows[name].float().clone()
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


@torch.no_grad()
def gen_reply_stream(tok, logits_fn, messages, max_new_tokens, device,
                     sample=False, temperature=0.8, top_p=0.95, seed=0,
                     dec=None, mode="grafted", scope=None):
    """One assistant turn. Returns (text, ids, wall_s, tok_per_s).

    Uses agent-49's KVDecoder (lab/kv_cache_v05) when available — one token
    at a time over a growing per-layer K/V cache; else g0.gen_reply (the
    chat_v05 full-forward decode)."""
    t0 = time.time()
    if scope is not None:
        txt = tok.apply_chat_template(list(messages), tokenize=False,
                                      add_generation_prompt=True)
        scope.begin(messages[-1]["content"], tok,
                    tok.encode(txt, add_special_tokens=False))
    if dec is not None:
        text, ids, _st = dec.generate(
            messages, max_new_tokens, mode, sample=sample,
            temperature=temperature, top_p=top_p, seed=seed)
    else:
        text, ids = g0.gen_reply(tok, logits_fn, messages, max_new_tokens,
                                 device, sample=sample,
                                 temperature=temperature,
                                 top_p=top_p, seed=seed)
    wall = time.time() - t0
    n = max(1, len(ids))
    return text, ids, wall, n / wall


def loop_detect(text: str) -> str | None:
    """Cheap repetition detector: a 4+ gram repeating back-to-back."""
    words = text.split()
    for ngram in (8, 6, 4):
        for i in range(0, max(0, len(words) - 2 * ngram + 1)):
            a = words[i:i + ngram]
            b = words[i + ngram:i + 2 * ngram]
            if a == b:
                return " ".join(a)[:60]
    return None


def run_probes(tok, logits_fn, device, max_new_tokens, sample, temperature,
               dec=None, mode="grafted", scope=None, probes=None):
    tagset = "v0.6" + (" BARE" if mode == "bare" else "")
    print(f"=== PROBE SMOKE ({tagset}) ===", flush=True)
    messages = []
    tps_all = []
    for kind, user in (probes if probes is not None else PROBES):
        messages.append({"role": "user", "content": user})
        text, ids, wall, tps = gen_reply_stream(
            tok, logits_fn, messages, max_new_tokens, device,
            sample=sample, temperature=temperature, dec=dec, mode=mode,
            scope=scope)
        messages.append({"role": "assistant", "content": text})
        short = " ".join(text.split())[:220]
        loop = loop_detect(text)
        tps_all.append(tps)
        print(f"\n[{kind}] Q: {user[:90]}", flush=True)
        print(f"[{kind}] A: {short}", flush=True)
        print(f"[{kind}] n_new_tokens={len(ids)} wall={wall:.1f}s "
              f"tok/s={tps:.2f}" + (f" LOOP[{loop}]" if loop else ""),
              flush=True)
    if tps_all:
        print(f"\n[probes] mean tok/s {sum(tps_all) / len(tps_all):.2f}",
              flush=True)
    print("\n=== END PROBES ===", flush=True)


def run_graft_question(tok, logits_fn, device, max_new_tokens, sample,
                       temperature, dec=None, mode="grafted"):
    print("=== GRAFT QUESTION (impress test a) ===", flush=True)
    messages = [{"role": "user", "content": GRAFT_QUESTION}]
    text, ids, wall, tps = gen_reply_stream(
        tok, logits_fn, messages, max_new_tokens, device,
        sample=sample, temperature=temperature, dec=dec, mode=mode)
    loop = loop_detect(text)
    verdict = "PASS" if (text.strip() and loop is None
                         and len(ids) >= 8) else "FAIL"
    print(f"[graftq] Q: {GRAFT_QUESTION}", flush=True)
    print(f"[graftq] A: {' '.join(text.split())[:400]}", flush=True)
    print(f"[graftq] n_new_tokens={len(ids)} wall={wall:.1f}s "
          f"tok/s={tps:.2f} loop={'NO' if not loop else loop} -> {verdict}",
          flush=True)
    return {"answer": text, "n_new_tokens": len(ids), "wall_s": round(wall, 2),
            "tok_per_s": round(tps, 2), "loop": loop, "verdict": verdict}


def run_impress(args, tok, fn_bare, fn_graft, device, dec):
    print("############ IMPRESS TEST (v0.6) ############", flush=True)
    results = {"graft_question": {}, "probes": {}, "decode": {}}
    results["decode"] = {
        "mode": ("kv-incremental (lab/kv_cache_v05.KVDecoder)"
                 if dec is not None
                 else "full-forward (kv module not found)")}
    print(f"[impress] decode: {results['decode']['mode']}", flush=True)
    print("\n--- grafted ---", flush=True)
    results["graft_question"]["grafted"] = run_graft_question(
        tok, fn_graft, device, args.max_new_tokens, args.sample,
        args.temperature, dec=dec, mode="grafted")
    run_probes(tok, fn_graft, device, args.probe_tokens, args.sample,
               args.temperature, dec=dec, mode="grafted")
    if args.impress_bare:
        print("\n--- bare spine (comparison) ---", flush=True)
        results["graft_question"]["bare"] = run_graft_question(
            tok, fn_bare, device, args.max_new_tokens, args.sample,
            args.temperature, dec=dec, mode="bare")
        run_probes(tok, fn_bare, device, args.probe_tokens, args.sample,
                   args.temperature, dec=dec, mode="bare")
    print("\n############ END IMPRESS TEST ############", flush=True)
    return results


def interactive(tok, fn_bare, fn_graft, device, args, dec, scope=None):
    is_tty = sys.stdin.isatty()
    sample = args.sample
    temperature = args.temperature
    mode = "bare" if args.bare else "grafted"
    messages = []
    tps_last = None
    print(f"[chat] ready ({mode}, "
          f"{'T=' + str(temperature) if sample else 'greedy'}) — type /help",
          flush=True)
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
                      "/sys TEXT /graftq /toks /exit", flush=True)
            elif c == "/reset":
                messages = []
                print("[chat] history cleared", flush=True)
            elif c == "/bare":
                mode = "bare"
                print("[chat] bare spine", flush=True)
            elif c == "/grafted":
                mode = "grafted"
                print("[chat] grafted v0.6", flush=True)
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
                print("[chat] system prompt set", flush=True)
            elif c == "/graftq":
                messages.append({"role": "user", "content": GRAFT_QUESTION})
                try:
                    fn = fn_bare if mode == "bare" else fn_graft
                    text, ids, wall, tps = gen_reply_stream(
                        tok, fn, messages, args.max_new_tokens, device,
                        sample=sample, temperature=temperature,
                        dec=dec, mode=mode, scope=scope)
                    tps_last = tps
                    messages.append({"role": "assistant", "content": text})
                    loop = loop_detect(text)
                    print(f"\n{mode}> {text}", flush=True)
                    print(f"[n_new_tokens={len(ids)} tok/s={tps:.2f} "
                          f"loop={'NO' if not loop else loop}]",
                          flush=True)
                except Exception as e:  # noqa: BLE001
                    messages.pop()
                    print(f"[chat] generation error: {type(e).__name__}: {e}",
                          flush=True)
            elif c == "/toks":
                print(f"[chat] last decode "
                      f"{tps_last if tps_last else 'n/a'} tok/s", flush=True)
            else:
                print("[chat] unknown command — /help", flush=True)
            continue
        messages.append({"role": "user", "content": line})
        try:
            fn = fn_bare if mode == "bare" else fn_graft
            text, ids, wall, tps = gen_reply_stream(
                tok, fn, messages, args.max_new_tokens, device,
                sample=sample, temperature=temperature,
                dec=dec, mode=mode, scope=scope)
            tps_last = tps
        except Exception as e:  # keep the chat alive on any decode hiccup
            messages.pop()
            print(f"[chat] generation error: {type(e).__name__}: {e}",
                  flush=True)
            continue
        messages.append({"role": "assistant", "content": text})
        loop = loop_detect(text)
        print(f"\n{mode}> {text}", flush=True)
        if not is_tty:
            print(f"[n_new_tokens={len(ids)} tok/s={tps:.2f}]"
                  + (f" LOOP[{loop}]" if loop else ""), flush=True)
        elif loop:
            print(f"[chat] WARNING: repetition detected ({loop})", flush=True)


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
    ap.add_argument("--impress", action="store_true",
                    help="impress test: graft question + 4 probes + tok/s")
    ap.add_argument("--impress-bare", action="store_true",
                    help="also run the impress probes on the bare spine")
    ap.add_argument("--probe-tokens", type=int, default=80,
                    help="short excerpts for the probe smoke (safety)")
    ap.add_argument("--sample", action="store_true",
                    help="temperature sampling (default greedy)")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--default-lam", type=float, default=0.5)
    ap.add_argument("--lam-override", default="",
                    help="comma name=val list, e.g. mem_pet=0.8")
    ap.add_argument("--varact", default="",
                    help="variable-activation plug-gating policy: off|"
                         "tau0.3|tau0.5|tau0.7|topm1|topm2|topm3|"
                         "adaptive_tau0.5_m2 (default: joint_state flag or "
                         "DEFAULT_VARACT=%s)" % DEFAULT_VARACT)
    ap.add_argument("--mem-bonus", default="3.0",
                    help="decode-time taught-token bonus (nats) for live "
                         "memory rows (v0.1 fact-token lever).  DEFAULT "
                         "3.0 with the question-anchored scope (10.10 "
                         "rung 1 — all four bars hold; see MODULAR_MOE "
                         "10.10.1); 'off' = plain tau delivery")
    ap.add_argument("--mem-boost", dest="mem_boost", default=False,
                    action="store_true",
                    help="also install the retrieval-triggered row boost "
                         "for question-matched memory rows (opt-in)")
    ap.add_argument("--no-mem-boost", dest="mem_boost", action="store_false",
                    help="disable the retrieval-triggered row boost")
    ap.add_argument("--mem-bonus-tau", type=float, default=0.5,
                    help="p_home threshold for the bonus scope")
    ap.add_argument("--mem-bonus-topk", type=int, default=0,
                    help="margin-to-token tie-break gate: bonus only when "
                         "a taught token is inside the top-K (0 = always "
                         "when live — the shipped form; the gated forms "
                         "were measured and are NOT shipped)")
    ap.add_argument("--qscope", default="overlap",
                    choices=["none", "overlap", "qspan", "both", "any"],
                    help="question-anchored retrieval scope (10.10 rung 1): "
                         "boost only rows the current question addresses")
    ap.add_argument("--qtau", type=float, default=0.5,
                    help="question-match threshold for --qscope")
    ap.add_argument("--qmargin", type=float, default=0.0,
                    help="row-selective margin (boost row j only where "
                         "p_j >= best sibling p + margin)")
    ap.add_argument("--slot-k", type=int, default=0,
                    help="answer-slot-only bonus gate (0 = off)")
    ap.add_argument("--mem-pack", default=DEFAULT_MEM_PACK,
                    choices=["persona", "legacy"],
                    help="memory pack: persona (default %s) or legacy trio"
                         % DEFAULT_MEM_PACK)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.lam_map = {}
    if args.lam_override:
        for kv in args.lam_override.split(","):
            if not kv.strip():
                continue
            k, _, v = kv.partition("=")
            args.lam_map[k.strip()] = float(v)
        print(f"[chat] lam overrides: {args.lam_map}", flush=True)

    if args.gpu == "auto":
        dev_i = pick_gpu()
    else:
        dev_i = int(args.gpu)
    # HIP_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES remap: a physical id passed
    # via --gpu can exceed the visible range (the chain sets
    # HIP_VISIBLE_DEVICES=$gpu and the process then sees only device 0).
    n_vis = torch.cuda.device_count()
    if n_vis and not (0 <= dev_i < n_vis):
        print(f"[chat] --gpu {dev_i} out of visible range 0..{n_vis - 1} "
              f"(device remap) — using 0", flush=True)
        dev_i = 0
    device = f"{args.device}:{dev_i}" if args.device == "cuda" else args.device
    torch.cuda.set_device(dev_i)

    KVDecoder = find_kv_module()
    if KVDecoder is not None:
        print("[chat] KV incremental decode: lab/kv_cache_v05.KVDecoder",
              flush=True)
    else:
        print("[chat] no kv module in lab/ — full-forward decode "
              "(chat_v05 behavior)", flush=True)

    if args.bare and not (args.probe or args.impress):
        model, tok = d2.load_donor(args.model, device, args.dtype)
        blocks = d2.find_moe_blocks(model)
        fam = d2.family_for(d2.detect_family(model), blocks)
        gates, handles = None, None
        info, missing = [], []
        bonus, tag_col, scope = None, {}, None
    else:
        (model, tok, blocks, fam, gates, handles, info,
         missing, site_entries, _banks) = build_state(args, device)
        # persona pack: question-match gate + taught-token bonus see the
        # pack's mem rows (legacy mem_pet/mem_edit + 6 persona facts)
        if (getattr(args, "mem_pack", "") or DEFAULT_MEM_PACK) == "persona":
            _pack = mp.load_pack()
            mp.apply_to_varact(vact, _pack)
            print(f"[chat] mem pack: {_pack['pack']} "
                  f"({', '.join(mp.tags(_pack))})", flush=True)
        # variable-activation plug gating (MODULAR_MOE 10.9): shipped
        # default tau=0.5.  Priority: --varact > joint_state["varact"] >
        # DEFAULT_VARACT; "off" keeps plain ungated delivery.
        pol_name = args.varact
        if not pol_name:
            js_path = joint_state_path(args)
            if os.path.exists(js_path):
                js = torch.load(js_path, map_location="cpu",
                                weights_only=False)
                pol_name = str(js.get("varact") or "")
        pol_name = pol_name or DEFAULT_VARACT
        wrappers = []
        if pol_name == "off":
            print("[chat] varact gating OFF (ungated delivery)", flush=True)
        elif pol_name not in vact.POLICIES:
            print(f"[chat] unknown varact policy {pol_name!r} — ungated",
                  flush=True)
        else:
            wrappers = vact.install_varact(handles, vact.POLICIES[pol_name]())
            print(f"[chat] varact gating ON: {pol_name} "
                  f"(--varact off to disable)", flush=True)
        # fact-stability fix (MODULAR_MOE 10.10 — the named v0.1 levers):
        # retrieval-scoped decode-time bonus, question-anchored scope.
        bonus, tag_col, scope = None, {}, None
        mb = str(args.mem_bonus).strip().lower()
        if mb in ("off", "0", "0.0"):
            print("[chat] mem bonus OFF", flush=True)
        else:
            beta = float(mb)
            if str(args.qscope).strip().lower() not in ("none", "off", ""):
                scope = vact.QuestionScope(args.qscope.strip().lower(),
                                           tau=args.qtau,
                                           margin=args.qmargin)
            tag_col = {}
            for host, ents in site_entries.items():
                for j, en in enumerate(ents):
                    tag_col[en["tag"]] = (host, j)
            if beta > 0:
                bonus = vact.TaughtTokenBonus(beta, args.mem_bonus_tau,
                                              top_k=args.mem_bonus_topk,
                                              scope=scope,
                                              slot_k=args.slot_k)
                bonus.note(tok)
                print(f"[chat] mem bonus ON: beta={beta} "
                      f"tau={args.mem_bonus_tau} "
                      f"top_k={args.mem_bonus_topk} "
                      f"scope={scope.name() if scope else 'coarse'}",
                      flush=True)
        if args.mem_boost and wrappers:
            # retrieval-triggered row boost: a question-matched memory row
            # delivers through the reply (long-context router drift
            # otherwise gates it off at the fact-token decision).  Opt-in:
            # the question-scoped bonus alone passes all four bars.
            vact.set_policy(wrappers, vact.RetrievalBoostPolicy(
                tau=args.mem_bonus_tau, scope=scope))
            print("[chat] retrieval-boost delivery ON for memory rows "
                  f"(scope={scope.name() if scope else 'coarse'})",
                  flush=True)

    dec = None
    if KVDecoder is not None:
        try:
            dec = KVDecoder(model, tok, device, blocks=blocks, fam=fam,
                            gates=gates, handles=handles)
        except Exception as e:  # noqa: BLE001
            print(f"[chat] KVDecoder init failed ({e}) — full-forward "
                  f"decode", flush=True)
            dec = None
    if dec is not None and bonus is not None and gates is not None:
        # KV decode computes logits inside KVDecoder._forward — wrap it
        # (kv_cache_v05 itself stays read-only) so the bonus sees every step
        base_fwd = dec._forward

        def _fwd_bonus(x, _b=base_fwd):
            qspan = scope.span if scope is not None else None
            return bonus.edit(_b(x), vact.capture_tags(gates, tag_col,
                                                       qspan=qspan))

        dec._forward = _fwd_bonus

    def fn_bare(x):
        return g0.run_in_state(
            model, blocks, fam, None, None, "off",
            lambda: d0.model_logits(model, x.to(device)))

    def fn_graft(x):
        if gates is None:
            return fn_bare(x)

        def run():
            lg = d0.model_logits(model, x.to(device))
            if bonus is not None:
                qspan = scope.span if scope is not None else None
                lg = bonus.edit(lg, vact.capture_tags(gates, tag_col,
                                                      qspan=qspan))
            return lg

        return g0.run_in_state(
            model, blocks, fam, gates, handles, "both", run)

    if args.impress:
        run_impress(args, tok, fn_bare, fn_graft, device, dec)
        return 0
    if args.probe:
        probes = list(PROBES)
        if (getattr(args, "mem_pack", "") or DEFAULT_MEM_PACK) == "persona":
            try:
                for tag, q, _e, _r in mp.probes():
                    probes.append((f"persona:{tag}", q))
            except Exception as e:  # noqa: BLE001
                print(f"[chat] persona probe list unavailable: {e}",
                      flush=True)
        run_probes(tok, fn_graft, device, args.probe_tokens, args.sample,
                   args.temperature, dec=dec, mode="grafted", scope=scope,
                   probes=probes)
        return 0
    interactive(tok, fn_bare, fn_graft, device, args, dec, scope=scope)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
