#!/usr/bin/env python3
"""vFINAL chat CLI — the complete Vesper stack on our own vk-base spine.

The command the user runs (shipped in tmux on the MI50 box):

    ssh tliao@100.64.0.7 'cd /data/tliao/vesper_sft/VesperLM && \
        /data/tliao/vesper_sft/venv/bin/python lab/chat_vfinal.py'

World: vk-base chat model (426M/325M-active KDA+MLA passport MoE, SFT r2
talkable 6/12) + passport-native graft registry (lab/vk_graft.py) +
3 frontier specialty plugs (K3 hacking / GLM-5.3-Flash cybersec /
MiMo-Flash writing, core_bank donors re-wrapped per MODULAR_MOE 10.7) +
9 taught memories (legacy trio + the persona pack, fe98ed4) with the
shipped delivery stack: additive balanced lambda (92abbd7), tau=0.5 VarAct
gating (715ad33), row-scale guard (180a612), question-match gate +
taught-token bonus (30038bd, default ON).

Features (chat_v06 conventions):
  * greedy (default) or --sample T; /sample T live
  * --bare comparison (/bare /grafted toggle live)
  * /graftq probe + --impress full battery + --probe specialty smoke
  * KV-cached incremental decode (lab/kv_cache_v05 conventions, vk backend)
  * auto freest-GPU pick with HIP clamp (866c8ad)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

os.environ.setdefault("ROCBLAS_TENSILE_LIBPATH",
                      "/opt/rocm-6.2.4/lib/rocblas/library")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "lab"))
sys.path.insert(0, os.path.join(REPO_ROOT, "Common"))

import torch  # noqa: E402

import vk_graft as vk  # noqa: E402
import graft_d0_parity as d0  # noqa: E402
import graft_d1_plugin as d1  # noqa: E402
import door6_expert_mix as d6  # noqa: E402
import grafted_spine_v0 as g0  # noqa: E402
import varact_gate as vact  # noqa: E402
import mem_pack as mp  # noqa: E402

DEFAULT_MODEL = "/data/tliao/vk_final/chat_model"
DEFAULT_REGISTRY = "/data/tliao/vk_final/registry_vk.pt"
DEFAULT_PLUGINS = "/data/tliao/vk_final"
DEFAULT_MEM_PACK = "persona"
DEFAULT_VARACT = "tau0.5"
DEFAULT_MEM_BONUS = 3.0
DEFAULT_QSCOPE = "overlap"

GRAFT_QUESTION = ("What would you say if I told you I grafted Kimi K3, GLM, "
                  "and MiMo weights into you?")

PROBES = [
    ("hacking", "Walk me through the reconnaissance phase of a penetration "
                "test. What kind of information would you gather?"),
    ("cybersec", "In two sentences: what is the difference between a "
                 "vulnerability and an exploit?"),
    ("writing", "Write two short sentences of atmospheric prose about a "
                "lighthouse in a storm."),
    ("general", "In two sentences: what is a mutex and when would you use "
                "one?"),
]

PERSONA_PROBES = [
    ("poi", "What's your favorite live action show?", "Person of Interest"),
    ("gundam", "What's your favorite anime?", "Gundam"),
    ("trains", "What do you think of trains?", None),
    ("igbt", "What about IGBTs?", None),
    ("cars", "What about cars?", None),
]

EVAL12 = [
    "Hello! How are you today?",
    "What is the capital of France?",
    "Can you explain what a black hole is in simple terms?",
    "Write a short poem about the ocean.",
    "What is 17 * 23?",
    "How do I boil an egg?",
    "What is the difference between a list and a tuple in Python?",
    "Tell me a fun fact about space.",
    "Summarize: Photosynthesis is the process plants use to convert sunlight into energy.",
    "I feel really stressed about work lately. Any advice?",
    "Translate to French: The weather is nice today.",
    "Why do we dream?",
]


def pick_gpu(quiet=False) -> int:
    n = torch.cuda.device_count()
    if n == 0:
        raise SystemExit("no CUDA/HIP GPUs visible")
    best, best_free = 0, -1
    frees = []
    for i in range(n):
        try:
            free, _t = torch.cuda.mem_get_info(i)
        except Exception:  # noqa: BLE001
            free = -1
        frees.append(f"{i}:{free / 1e9:.1f}G")
        if free > best_free:
            best, best_free = i, free
    if not quiet:
        print(f"[chat] GPUs free {frees} -> picking {best}", flush=True)
    return best


def joint_state_path(args) -> str:
    return os.path.join(args.plugins_dir, "joint_state_vfinal.pt")


def build_state(args, device):
    """vk-base + registry + plugs -> (model, tok, blocks, fam, gates, handles,
    info, missing, site_entries, site_banks)."""
    for path, what in ((args.model, "spine"), (args.registry, "registry")):
        if not os.path.exists(path):
            raise SystemExit(f"missing {what} at {path}")
    print(f"[chat] loading vk-base {args.model} (fp16) ...", flush=True)
    model, tok = vk.load_vk(args.model, device, args.dtype)
    blocks = vk.install_vk_host(model)
    fam = vk.FAM
    banks_reg, _ = d0.load_registry(args.registry)
    d_model = banks_reg[0].d_model

    paths = sorted(glob.glob(os.path.join(args.plugins_dir, "*_plugin.pt")))
    expected = ["k3_hack", "glm_cybersec", "mimo_writing",
                "mem_pet", "mem_edit", "mem_words"]
    mem_pack = getattr(args, "mem_pack", "") or DEFAULT_MEM_PACK
    if mem_pack == "persona":
        expected += mp.tags(mp.load_pack())
    found = {os.path.basename(p)[:-len("_plugin.pt")]: p for p in paths}
    if not found:
        raise SystemExit(f"no plug-ins in {args.plugins_dir} — run "
                         f"lab/vk_final_build.py first (or use --bare)")

    lams, js_rows = {}, {}
    js_path = joint_state_path(args)
    if os.path.exists(js_path):
        js = torch.load(js_path, map_location="cpu", weights_only=False)
        order = js.get("order") or [n for n in expected if n in found]
        for n, v in zip(order, js.get("lams", [])):
            lams[n] = float(v)
        js_rows = {n: r for n, r in (js.get("rows") or {}).items()}
    else:
        print("[chat] NOTE: no joint_state_vfinal.pt — default lam 0.5/plug",
              flush=True)

    site_rows, site_entries, info = {}, {}, []
    for name in expected:
        if name not in found:
            continue
        wrap, row, meta, spec = d6.load_wrap_plugin(found[name])
        if not d6.wrap_is_finite(wrap, row):
            print(f"[chat] WARNING non-finite {name} skipped", flush=True)
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
                     "lam": round(lam, 4), "act": spec.get("act", "swiglu"),
                     "donor": meta.get("donor")})
    missing = [n for n in expected if n not in found]
    print(f"[chat] plug-ins live: {[i['name'] for i in info]}"
          + (f" | missing: {missing}" if missing else ""), flush=True)

    # passport-native graft gates (registry rows = passport rows)
    import vk_final_build as vb
    gates, site_banks = vb._build_vk_gates(blocks, banks_reg, site_rows)
    handles = g0.install_mix(blocks, fam, site_entries, mode="off")
    return (model, tok, blocks, fam, gates, handles, info, missing,
            site_entries, site_banks)


def loop_detect(text: str):
    words = text.split()
    for ngram in (8, 6, 4):
        for i in range(0, max(0, len(words) - 2 * ngram + 1)):
            a, b = words[i:i + ngram], words[i + ngram:i + 2 * ngram]
            if a == b:
                return " ".join(a)[:60]
    return None


def auto_verdict(text: str, prompt: str = "") -> str:
    """One-word verdict from the measurable families (e6c79ec taxonomy)."""
    t = (text or "").strip()
    if not t or len(t.split()) < 3:
        return "FAIL"
    if loop_detect(t):
        return "FAIL"
    if prompt and (prompt[:40].strip().lower() in t.lower()):
        return "FAIL"
    debris = ("x = ", "```", "Formula:", "Certainly!", "As an AI")
    words = t.split()
    uniq = len(set(w.lower() for w in words)) / max(1, len(words))
    if any(d in t for d in debris) and uniq < 0.55:
        return "FAIL"
    if uniq < 0.42 or len(words) > 6 and loop_detect(t + " " + t):
        return "PARTIAL"
    if uniq < 0.62:
        return "PARTIAL"
    return "PASS"
@torch.no_grad()
def gen_reply_stream(tok, logits_fn, messages, max_new_tokens, device,
                     sample=False, temperature=0.8, top_p=0.95, seed=0,
                     dec=None, mode="grafted", scope=None):
    t0 = time.time()
    if scope is not None:
        scope.begin(messages[-1]["content"], tok, None)
    if dec is not None:
        text, ids = dec.generate(
            messages, max_new_tokens, mode, sample=sample,
            temperature=temperature, top_p=top_p, seed=seed)
    else:
        text, ids = g0.gen_reply(tok, logits_fn, messages, max_new_tokens,
                                 device, sample=sample, temperature=temperature,
                                 top_p=top_p, seed=seed)
    wall = time.time() - t0
    n = max(1, len(ids))
    return text, ids, wall, n / wall


def run_probes(tok, logits_fn, device, max_new_tokens, sample, temperature,
               dec=None, mode="grafted", scope=None, probes=None):
    tagset = "vFINAL" + (" BARE" if mode == "bare" else "")
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
        v = auto_verdict(text, user)
        print(f"\n[{kind}] Q: {user[:90]}", flush=True)
        print(f"[{kind}] A: {short}", flush=True)
        print(f"[{kind}] n={len(ids)} wall={wall:.1f}s tok/s={tps:.2f} "
              f"verdict={v}" + (f" LOOP[{loop}]" if loop else ""), flush=True)
    if tps_all:
        print(f"\n[probes] mean tok/s {sum(tps_all) / len(tps_all):.2f}",
              flush=True)
    print("\n=== END PROBES ===", flush=True)


def run_graft_question(tok, logits_fn, device, max_new_tokens, sample,
                       temperature, dec=None, mode="grafted"):
    print("=== GRAFT QUESTION (impress a) ===", flush=True)
    messages = [{"role": "user", "content": GRAFT_QUESTION}]
    text, ids, wall, tps = gen_reply_stream(
        tok, logits_fn, messages, max_new_tokens, device,
        sample=sample, temperature=temperature, dec=dec, mode=mode)
    loop = loop_detect(text)
    verdict = "PASS" if (text.strip() and loop is None and len(ids) >= 8) \
        else "FAIL"
    print(f"[graftq] Q: {GRAFT_QUESTION}", flush=True)
    print(f"[graftq] A: {' '.join(text.split())[:400]}", flush=True)
    print(f"[graftq] n={len(ids)} wall={wall:.1f}s tok/s={tps:.2f} "
          f"loop={'NO' if not loop else loop} -> {verdict}", flush=True)
    return {"answer": text, "n_new_tokens": len(ids),
            "wall_s": round(wall, 2), "tok_per_s": round(tps, 2),
            "loop": loop, "verdict": verdict}


def run_memory_multi_turn(tok, logits_fn, device, max_new_tokens, sample,
                          temperature, dec=None, mode="grafted", scope=None):
    """(d) Kepler + multi-turn fact stability (the 5/6 bar): the memory
    question asked as turn 4 of an accumulating transcript x 2 depths."""
    print("=== MEMORY MULTI-TURN (impress d) ===", flush=True)
    fills = ["What kind of projects are you working on these days?",
             "Can you explain what a mutex is briefly?",
             "Write one sentence of prose about the sea.",
             "What is the capital of France?"]
    results = []
    for depth in (0, 3):
        messages = []
        for f in fills[:depth]:
            messages.append({"role": "user", "content": f})
            text, ids, wall, tps = gen_reply_stream(
                tok, logits_fn, messages, 32, device,
                sample=sample, temperature=temperature, dec=dec, mode=mode,
                scope=scope)
            messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": "What is my cat's name?"})
        text, ids, wall, tps = gen_reply_stream(
            tok, logits_fn, messages, 32, device,
            sample=sample, temperature=temperature, dec=dec, mode=mode,
            scope=scope)
        ok = "Kepler" in text
        results.append({"depth": depth, "answer": text[:200], "kepler": ok})
        print(f"[mem t{depth}] A: {' '.join(text.split())[:160]} "
              f"-> kepler={'YES' if ok else 'NO'}", flush=True)
    return results


def run_eval12(tok, logits_fn_bare, logits_fn_graft, device, args, dec):
    """(e) talkability: the 12-prompt eval_chat instrument, bare vs grafted."""
    print("=== 12-PROMPT TALKABILITY (impress e) ===", flush=True)
    out = {"bare": [], "grafted": []}
    for mode, fn in (("bare", logits_fn_bare), ("grafted", logits_fn_graft)):
        msgs = []
        for prompt in EVAL12:
            msgs.append({"role": "user", "content": prompt})
            text, ids, wall, tps = gen_reply_stream(
                tok, fn, msgs, args.eval_max_new, device,
                sample=True, temperature=0.7, top_p=0.9,
                dec=dec, mode=mode)
            msgs.append({"role": "assistant", "content": text})
            v = auto_verdict(text, prompt)
            out[mode].append({"prompt": prompt, "verdict": v,
                              "response": " ".join(text.split())[:260],
                              "n": len(ids), "tok_s": round(tps, 2)})
            print(f"[eval12 {mode:7s}] {v:7s} | {prompt[:44]!r} | "
                  f"{' '.join(text.split())[:110]}", flush=True)
    for mode in ("bare", "grafted"):
        n_pass = sum(1 for r in out[mode] if r["verdict"] == "PASS")
        n_pp = sum(1 for r in out[mode] if r["verdict"] in ("PASS", "PARTIAL"))
        out[mode + "_score"] = {"pass": n_pass, "pass_or_partial": n_pp,
                                "n": len(EVAL12)}
        print(f"[eval12] {mode}: {n_pp}/12 usable (PASS {n_pass})",
              flush=True)
    bar_ok = out["grafted_score"]["pass_or_partial"] >= 6 \
        and out["grafted_score"]["pass_or_partial"] >= \
        out["bare_score"]["pass_or_partial"]
    out["verdict"] = "PASS" if bar_ok else "FAIL"
    print(f"[eval12] talkability bar (>=6/12 and not below bare): "
          f"{out['verdict']} (grafted {out['grafted_score']['pass_or_partial']}"
          f"/12 vs bare {out['bare_score']['pass_or_partial']}/12)", flush=True)
    return out


def run_impress(args, tok, fn_bare, fn_graft, device, dec, scope=None):
    print("############ IMPRESS TEST (vFINAL on vk-base) ############",
          flush=True)
    results = {"spine": "vk-base chat_model (KDA+MLA passport MoE)",
               "decode": ("kv-incremental (VkKVDecoder)"
                          if dec is not None else "full-forward")}
    print(f"[impress] decode: {results['decode']}", flush=True)

    print("\n--- (a) grafted ---", flush=True)
    results["graft_question"] = run_graft_question(
        tok, fn_graft, device, args.max_new_tokens, args.sample,
        args.temperature, dec=dec, mode="grafted")

    print("\n--- (b) specialty probes ---", flush=True)
    run_probes(tok, fn_graft, device, args.probe_tokens, args.sample,
               args.temperature, dec=dec, mode="grafted", scope=scope,
               probes=PROBES)
    results["specialty_probes"] = "see probe log"

    print("\n--- (c) persona probes ---", flush=True)
    pprobes = [(k, q) for k, q, _e in PERSONA_PROBES]
    run_probes(tok, fn_graft, device, args.probe_tokens, args.sample,
               args.temperature, dec=dec, mode="grafted", scope=scope,
               probes=pprobes)
    results["persona_probes"] = "see probe log"

    print("\n--- (d) Kepler + memory multi-turn ---", flush=True)
    results["memory_multi_turn"] = run_memory_multi_turn(
        tok, fn_graft, device, args.probe_tokens, args.sample,
        args.temperature, dec=dec, mode="grafted", scope=scope)

    print("\n--- (e) talkability vs bare SFT base ---", flush=True)
    results["eval12"] = run_eval12(tok, fn_bare, fn_graft, device, args, dec)

    if args.impress_bare:
        print("\n--- bare graft question (comparison) ---", flush=True)
        results["graft_question_bare"] = run_graft_question(
            tok, fn_bare, device, args.max_new_tokens, args.sample,
            args.temperature, dec=dec, mode="bare")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[impress] wrote {args.out}", flush=True)
    print("\n############ END IMPRESS TEST ############", flush=True)
    return results


def interactive(tok, fn_bare, fn_graft, device, args, dec, scope=None):
    is_tty = sys.stdin.isatty()
    sample = args.sample
    temperature = args.temperature
    mode = "bare" if args.bare else "grafted"
    messages = []
    print(f"[chat] ready ({mode}, "
          f"{'T=' + str(temperature) if sample else 'greedy'}) — /help",
          flush=True)
    while True:
        try:
            line = input("you> " if is_tty else "").rstrip("\n")
        except (EOFError, KeyboardInterrupt):
            print("[chat] bye", flush=True)
            break
        if not line.strip():
            continue
        if line.strip().startswith("/"):
            cmd = line.strip().split()
            c = cmd[0].lower()
            if c in ("/exit", "/quit"):
                break
            if c == "/help":
                print("commands: /help /reset /bare /grafted /greedy "
                      "/sample T /sys TEXT /graftq /toks /exit", flush=True)
            elif c == "/reset":
                messages = []
                if dec is not None:
                    dec.reset()
                print("[chat] transcript cleared", flush=True)
            elif c == "/bare":
                mode = "bare"
                if dec is not None:
                    dec.reset()
                print("[chat] mode=bare", flush=True)
            elif c == "/grafted":
                mode = "grafted"
                if dec is not None:
                    dec.reset()
                print("[chat] mode=grafted", flush=True)
            elif c == "/greedy":
                sample = False
                print("[chat] greedy", flush=True)
            elif c == "/sample" and len(cmd) > 1:
                sample = True
                temperature = float(cmd[1])
                print(f"[chat] sampling T={temperature}", flush=True)
            elif c == "/sys" and len(cmd) > 1:
                messages.append({"role": "system",
                                 "content": line.strip()[5:].strip()})
                print("[chat] system note set", flush=True)
            elif c == "/graftq":
                fn = fn_bare if mode == "bare" else fn_graft
                run_graft_question(tok, fn, device, args.max_new_tokens,
                                   sample, temperature, dec=dec, mode=mode)
            elif c == "/toks":
                n = sum(len(m["content"].split()) for m in messages)
                print(f"[chat] ~{n} prompt words, {len(messages)} msgs",
                      flush=True)
            else:
                print(f"[chat] unknown command {c}", flush=True)
            continue
        messages.append({"role": "user", "content": line})
        fn = fn_bare if mode == "bare" else fn_graft
        text, ids, wall, tps = gen_reply_stream(
            tok, fn, messages, args.max_new_tokens, device, sample=sample,
            temperature=temperature, dec=dec, mode=mode, scope=scope)
        messages.append({"role": "assistant", "content": text})
        loop = loop_detect(text)
        print(f"bot> {text}", flush=True)
        print(f"     [{len(ids)} tok, {wall:.1f}s, {tps:.2f} tok/s]"
              + (f" LOOP[{loop}]" if loop else ""), flush=True)


def parse_lam_map(spec: str) -> dict:
    out = {}
    for part in (spec or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = float(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--registry", default=DEFAULT_REGISTRY)
    ap.add_argument("--plugins-dir", default=DEFAULT_PLUGINS)
    ap.add_argument("--out", default="")
    ap.add_argument("--bare", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--impress", action="store_true")
    ap.add_argument("--impress-bare", action="store_true")
    ap.add_argument("--sample", type=float, default=0.0)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--probe-tokens", type=int, default=64)
    ap.add_argument("--eval-max-new", type=int, default=120)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--no-kv", action="store_true")
    ap.add_argument("--default-lam", type=float, default=0.5)
    ap.add_argument("--lam-map", default="")
    ap.add_argument("--varact", default="",
                    help="off|tau0.3|tau0.5|... (default %s)" % DEFAULT_VARACT)
    ap.add_argument("--mem-pack", default=DEFAULT_MEM_PACK)
    ap.add_argument("--mem-bonus", default=str(DEFAULT_MEM_BONUS))
    ap.add_argument("--mem-bonus-tau", type=float, default=0.5)
    ap.add_argument("--mem-bonus-topk", type=float, default=0.0)
    ap.add_argument("--qscope", default=DEFAULT_QSCOPE)
    ap.add_argument("--qtau", type=float, default=0.5)
    ap.add_argument("--qmargin", type=float, default=0.0)
    ap.add_argument("--slot-k", type=int, default=0)
    ap.add_argument("--mem-boost", action="store_true")
    args = ap.parse_args()
    args.lam_map = parse_lam_map(args.lam_map)
    if args.gpu < 0:
        args.gpu = pick_gpu()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"[chat] device {device}", flush=True)

    if args.bare and not (args.probe or args.impress):
        model, tok = vk.load_vk(args.model, device, args.dtype)
        blocks = vk.install_vk_host(model)
        fam = vk.FAM
        gates, handles, info, missing = None, None, [], []
        bonus, tag_col, scope = None, {}, None
    else:
        (model, tok, blocks, fam, gates, handles, info, missing,
         site_entries, _banks) = build_state(args, device)
        if (getattr(args, "mem_pack", "") or DEFAULT_MEM_PACK) == "persona":
            _pack = mp.load_pack()
            mp.apply_to_varact(vact, _pack)
            print(f"[chat] mem pack: {_pack['pack']} "
                  f"({', '.join(mp.tags(_pack))})", flush=True)
        pol_name = args.varact or DEFAULT_VARACT
        wrappers = []
        if pol_name == "off":
            print("[chat] varact gating OFF", flush=True)
        elif pol_name not in vact.POLICIES:
            print(f"[chat] unknown varact {pol_name!r} — ungated", flush=True)
        else:
            wrappers = vact.install_varact(handles, vact.POLICIES[pol_name]())
            print(f"[chat] varact gating ON: {pol_name}", flush=True)
        bonus, tag_col, scope = None, {}, None
        mb = str(args.mem_bonus).strip().lower()
        if mb not in ("off", "0", "0.0"):
            beta = float(mb)
            if str(args.qscope).strip().lower() not in ("none", "off", ""):
                scope = vact.QuestionScope(args.qscope.strip().lower(),
                                           tau=args.qtau, margin=args.qmargin)
            tag_col = {}
            for host, ents in site_entries.items():
                for j, en in enumerate(ents):
                    tag_col[en["tag"]] = (host, j)
            if beta > 0:
                bonus = vact.TaughtTokenBonus(beta, args.mem_bonus_tau,
                                              top_k=args.mem_bonus_topk,
                                              scope=scope, slot_k=args.slot_k)
                bonus.note(tok)
                print(f"[chat] mem bonus ON beta={beta} "
                      f"scope={scope.name() if scope else 'coarse'}",
                      flush=True)
        if args.mem_boost and wrappers:
            vact.set_policy(wrappers, vact.RetrievalBoostPolicy(
                tau=args.mem_bonus_tau, scope=scope))
            print("[chat] retrieval-boost ON for memory rows", flush=True)

    dec = None
    if not args.no_kv:
        try:
            dec = vk.VkKVDecoder(model, tok, device, blocks=blocks, fam=fam,
                                 gates=gates, handles=handles)
        except Exception as e:  # noqa: BLE001
            print(f"[chat] VkKVDecoder init failed ({e}) — full-forward",
                  flush=True)
            dec = None
    if dec is not None and bonus is not None and gates is not None:
        base_step = dec._step

        def _step_bonus(x, _b=base_step):
            lg = _b(x)
            qspan = scope.span if scope is not None else None
            return bonus.edit(lg, vact.capture_tags(gates, tag_col,
                                                    qspan=qspan))
        dec._step = _step_bonus

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
        return g0.run_in_state(model, blocks, fam, gates, handles, "both", run)

    if args.impress:
        run_impress(args, tok, fn_bare, fn_graft, device, dec, scope=scope)
        return 0
    if args.probe:
        probes = list(PROBES)
        if (getattr(args, "mem_pack", "") or DEFAULT_MEM_PACK) == "persona":
            try:
                for tag, q, _e, _r in mp.probes():
                    probes.append((f"persona:{tag}", q))
            except Exception as e:  # noqa: BLE001
                print(f"[chat] persona probes unavailable: {e}", flush=True)
        probes.append(("kepler", "What is my cat's name?"))
        run_probes(tok, fn_graft, device, args.probe_tokens, args.sample,
                   args.temperature, dec=dec, mode="grafted", scope=scope,
                   probes=probes)
        return 0
    interactive(tok, fn_bare, fn_graft, device, args, dec, scope=scope)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
