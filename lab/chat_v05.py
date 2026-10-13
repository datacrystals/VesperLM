#!/usr/bin/env python3
"""v0.5 chat CLI — talk to the assembled model on the box.

Interactive terminal chat over the v0.5 assembly: Phi-mini-MoE-instruct spine
+ D2 passport registry + up to 8 grafted plug-ins (v0 code/math/3 memories +
K3 hacking / GLM-5.3-Flash cybersec / MiMo-V2.6-Flash writing), additive-λ
delivery, ChatML as the spine expects.

The exact command the user runs:

    ssh tliao@100.64.0.7 'cd /data/tliao/vesper_sft/VesperLM && \
        /data/tliao/vesper_sft/venv/bin/python lab/chat_v05.py'

Features:
  * greedy (default) or temperature sampling (--sample T, or /sample T live)
  * --bare chats with the un-grafted spine for comparison (/bare /grafted
    toggle live)
  * --probe runs the 4-turn specialty smoke non-interactively (for logs)
  * auto-picks the freest GPU (co-tenants move on this box); --gpu N pins it
  * loads whatever plug-ins exist and says plainly which are missing — the
    chat works even while some cores are still fitting

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
import varact_gate as vact  # noqa: E402  variable-activation plug gating
import mem_pack as mp  # noqa: E402  persona memory pack (lab/memories/)

DEFAULT_MODEL = "/data/tliao/Phi-mini-MoE-instruct"
DEFAULT_REGISTRY = "/data/tliao/graft_d2_out/registry_phimoe_f620b32c.pt"
DEFAULT_PLUGINS = "/data/tliao/v05_out"

# Persona memory pack (lab/memories/persona_v1.json): 6 permanent "about the
# user" facts taught as shaped episodic plugs, additive to the shipped mem
# trio.  DEFAULT ON per the 30038bd defaults — the pack rows join the bank,
# the question-match gate and the taught-token bonus are extended from the
# pack, and joint_state_persona.pt is preferred over joint_state.pt (which is
# never rewritten).  --mem-pack legacy restores the exact shipped 8-plug
# world bit-identical.
DEFAULT_MEM_PACK = "persona"

# v0.5.1 fixed delivery (per-home λ fits + xfire-recal; lab/v05_lam_refit.py).
# Used when joint_state.pt is absent; the box's joint_state.pt carries the
# same vector (plus post-recal rows) and takes precedence.
DEFAULT_LAMS = {
    "code": 1.238, "math": 1.0767,
    "mem_pet": 0.75, "mem_edit": 1.5, "mem_words": 1.0,
    "k3_hack": 0.8302, "glm_cybersec": 0.25, "mimo_writing": 0.25,
}

# Variable-activation plug gating (lab/varact_gate.py, MODULAR_MOE 10.9):
# tau=0.5 is the shipped default delivery policy (77% plug-compute cut,
# green home CE held, sibling harm closed by varact_sib_recal). Override
# with --varact off|tau0.3|tau0.5|tau0.7|topm1|topm2|topm3|adaptive_tau0.5_m2;
# joint_state.pt["varact"] (when present) overrides this default, and the
# CLI flag overrides everything.
DEFAULT_VARACT = "tau0.5"

# Decode-time taught-token bonus (MODULAR_MOE 10.10, the named v0.1
# fact-token lever): add nats to a memory's taught fact tokens at decode
# while that memory row is in play.  Row-selective retrieval scope (10.10
# rung 1, question-anchored): with --qscope (default overlap) a row is
# boosted only when the CURRENT QUESTION addresses its episode content —
# never merely because answer-shaped tokens make it hot mid-forward (the
# coarse-scope live-path garble).  All four bars hold on this scope
# (sacred t0 3/3, mid t4/t8 5/6, live --probe clean + Kepler, canary no
# leaks) — DEFAULT ON at beta=3.0 / overlap scope; the retrieval row
# boost stays opt-in (--mem-boost).  --mem-bonus off restores the plain
# shipped tau=0.5 delivery bit-identical.
DEFAULT_MEM_BONUS = "3.0"
DEFAULT_MEM_TOPK = 0
DEFAULT_MEM_BOOST = False        # retrieval-boost delivery (scoped, opt-in)
DEFAULT_QSCOPE = "overlap"       # question-match gate mode
DEFAULT_QTAU = 0.5               # question-match threshold

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


def joint_state_path(args) -> str:
    """persona pack state first (additive, written by persona_build), then
    the shipped joint_state.pt (never rewritten)."""
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

    # every *_plugin.pt in the plugins dir (v0's 5 + v05's 3 when present)
    paths = sorted(glob.glob(os.path.join(args.plugins_dir, "*_plugin.pt")))
    paths = [p for p in paths if "joint_state" not in p]
    expected = ["k3_hack", "glm_cybersec", "mimo_writing",
                "code", "math", "mem_pet", "mem_edit", "mem_words"]
    mem_pack = getattr(args, "mem_pack", "") or DEFAULT_MEM_PACK
    pack = None
    if mem_pack == "persona":
        pack = mp.load_pack()
        expected += mp.tags(pack)
    found = {os.path.basename(p)[:-len("_plugin.pt")]: p for p in paths}
    if not found:
        raise SystemExit(
            f"no plug-ins in {args.plugins_dir} — run lab/v05_build.py first "
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
        print("[chat] NOTE: no joint_state.pt — using baked DEFAULT_LAMS "
              "(v0.5.1 fixed delivery)", flush=True)

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
        lam = float(args.lam_map.get(
            name, lams.get(name, DEFAULT_LAMS.get(name, args.default_lam))))
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


@torch.no_grad()
def prompt_ids(tok, messages):
    """Token ids of the generation prompt for `messages` (matches g0.gen_reply)."""
    txt = tok.apply_chat_template(list(messages), tokenize=False,
                                  add_generation_prompt=True)
    return tok.encode(txt, add_special_tokens=False)


def gen_reply_stream(tok, logits_fn, messages, max_new_tokens, device,
                     sample=False, temperature=0.8, top_p=0.95, seed=0,
                     out_file=None, scope=None):
    """One assistant turn (g0.gen_reply with optional live streaming)."""
    if scope is not None:
        scope.begin(messages[-1]["content"], tok,
                    prompt_ids(tok, messages))
    text, ids = g0.gen_reply(tok, logits_fn, messages, max_new_tokens, device,
                             sample=sample, temperature=temperature,
                             top_p=top_p, seed=seed)
    return text, ids


def run_probes(tok, logits_fn, device, max_new_tokens, sample, temperature,
               scope=None, probes=None):
    print("=== PROBE SMOKE (v0.5) ===", flush=True)
    messages = []
    for kind, user in (probes if probes is not None else PROBES):
        messages.append({"role": "user", "content": user})
        text, ids = gen_reply_stream(tok, logits_fn, messages,
                                     max_new_tokens, device,
                                     sample=sample,
                                     temperature=temperature,
                                     scope=scope)
        messages.append({"role": "assistant", "content": text})
        short = " ".join(text.split())[:220]
        print(f"\n[{kind}] Q: {user[:90]}", flush=True)
        print(f"[{kind}] A: {short}", flush=True)
        print(f"[{kind}] n_new_tokens={len(ids)}", flush=True)
        if scope is not None:
            print(f"[{kind}] qscope matched={list(scope.last_match)} "
                  f"overlap={{{', '.join('%s=%.2f' % (k, v) for k, v in scope.overlap.items())}}}",
                  flush=True)
    print("\n=== END PROBES ===", flush=True)


def interactive(tok, fn_bare, fn_graft, device, args, scope=None):
    is_tty = sys.stdin.isatty()
    sample = args.sample
    temperature = args.temperature
    mode = "bare" if args.bare else "grafted"
    messages = []
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
                      "/sys TEXT /exit", flush=True)
            elif c == "/reset":
                messages = []
                print("[chat] history cleared", flush=True)
            elif c == "/bare":
                mode = "bare"
                print("[chat] bare spine", flush=True)
            elif c == "/grafted":
                mode = "grafted"
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
                print("[chat] system prompt set", flush=True)
            else:
                print("[chat] unknown command — /help", flush=True)
            continue
        messages.append({"role": "user", "content": line})
        try:
            fn = fn_bare if mode == "bare" else fn_graft
            text, ids = gen_reply_stream(tok, fn, messages,
                                         args.max_new_tokens, device,
                                         sample=sample,
                                         temperature=temperature,
                                         seed=args.seed,
                                         scope=scope)
        except Exception as e:  # keep the chat alive on any decode hiccup
            messages.pop()
            print(f"[chat] generation error: {type(e).__name__}: {e}",
                  flush=True)
            continue
        messages.append({"role": "assistant", "content": text})
        print(f"\n{mode}> {text}", flush=True)
        if not is_tty:
            print(f"[n_new_tokens={len(ids)}]", flush=True)


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
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--default-lam", type=float, default=0.5)
    ap.add_argument("--lam-override", default="",
                    help="comma name=val list, e.g. "
                         "glm_cybersec=0.3,mimo_writing=0.2 (bypasses "
                         "joint_state lam; for A/B probes)")
    ap.add_argument("--varact", default="",
                    help="variable-activation plug-gating policy: off|"
                         "tau0.3|tau0.5|tau0.7|topm1|topm2|topm3|"
                         "adaptive_tau0.5_m2 (default: joint_state flag or "
                         "DEFAULT_VARACT=%s)" % DEFAULT_VARACT)
    ap.add_argument("--mem-bonus", default=DEFAULT_MEM_BONUS,
                    help="decode-time taught-token bonus (nats) applied to "
                         "live memory rows — the v0.1 fact-token lever; "
                         "'0' or 'off' disables (default %s)"
                         % DEFAULT_MEM_BONUS)
    ap.add_argument("--mem-bonus-tau", type=float, default=0.5,
                    help="p_home threshold for the bonus scope (matches "
                         "the shipped tau=0.5 delivery rule)")
    ap.add_argument("--mem-bonus-topk", type=int, default=DEFAULT_MEM_TOPK,
                    help="margin-to-token tie-break: bonus only when a "
                         "taught token is inside the top-K (0 = always "
                         "when live; default %d)" % DEFAULT_MEM_TOPK)
    ap.add_argument("--mem-boost", dest="mem_boost",
                    default=DEFAULT_MEM_BOOST, action="store_true",
                    help="install retrieval-triggered row boost for "
                         "question-matched memory rows (H3 delivery kill)")
    ap.add_argument("--no-mem-boost", dest="mem_boost", action="store_false",
                    help="disable the retrieval-triggered row boost")
    ap.add_argument("--qscope", default=DEFAULT_QSCOPE,
                    choices=["none", "overlap", "qspan", "both", "any"],
                    help="question-anchored retrieval scope (10.10 rung 1): "
                         "boost only rows the current question addresses "
                         "(default %s)" % DEFAULT_QSCOPE)
    ap.add_argument("--qtau", type=float, default=DEFAULT_QTAU,
                    help="question-match threshold for --qscope (default "
                         "%g)" % DEFAULT_QTAU)
    ap.add_argument("--qmargin", type=float, default=0.0,
                    help="row-selective margin: boost row j only where "
                         "p_j >= best sibling p + margin (0 = off)")
    ap.add_argument("--slot-k", type=int, default=0,
                    help="answer-slot-only bonus: fixed beta only where a "
                         "taught token is inside the top-K (0 = off)")
    ap.add_argument("--mem-pack", default=DEFAULT_MEM_PACK,
                    choices=["persona", "legacy"],
                    help="memory pack: persona = the 6-fact Vesper pack "
                         "additive to the mem trio (default %s); legacy = "
                         "the shipped trio only" % DEFAULT_MEM_PACK)
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
    device = f"{args.device}:{dev_i}" if args.device == "cuda" else args.device
    torch.cuda.set_device(dev_i)

    if args.bare:
        model, tok = d2.load_donor(args.model, device, args.dtype)
        blocks = d2.find_moe_blocks(model)
        fam = d2.family_for(d2.detect_family(model), blocks)
        gates, handles = None, None
        info, missing = [], []
        bonus, tag_col, scope = None, {}, None
    else:
        (model, tok, blocks, fam, gates, handles, info,
         missing, site_entries, _banks) = build_state(args, device)
        # persona pack: point the question-match gate + taught-token bonus at
        # the pack's mem rows (legacy trio + 6 persona facts)
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
        # retrieval-scoped decode-time bonus on taught fact tokens, with
        # the row-selective question-anchored scope (10.10 rung 1).
        bonus, tag_col, scope = None, {}, None
        mb = str(args.mem_bonus).strip().lower()
        mem_bonus_on = mb not in ("off", "0", "0.0")
        if not mem_bonus_on and not args.mem_boost:
            print("[chat] mem bonus OFF", flush=True)
        else:
            if str(args.qscope).strip().lower() not in ("none", "off", ""):
                scope = vact.QuestionScope(args.qscope.strip().lower(),
                                           tau=args.qtau,
                                           margin=args.qmargin)
            tag_col = {}
            for host, ents in site_entries.items():
                for j, en in enumerate(ents):
                    tag_col[en["tag"]] = (host, j)
            if mem_bonus_on:
                beta = float(mb)
                if beta > 0:
                    bonus = vact.TaughtTokenBonus(
                        beta, args.mem_bonus_tau,
                        top_k=args.mem_bonus_topk, scope=scope,
                        slot_k=args.slot_k)
                    bonus.note(tok)
                    print(f"[chat] mem bonus ON: beta={beta} "
                          f"tau={args.mem_bonus_tau} "
                          f"top_k={args.mem_bonus_topk} "
                          f"scope={scope.name() if scope else 'coarse'} "
                          f"slot_k={args.slot_k} qmargin={args.qmargin}",
                          flush=True)
        if wrappers and args.mem_boost:
            # retrieval-triggered row boost: a question-matched memory row
            # delivers through the reply (H3 answer-start kill); with a
            # scope, unmatched rows are NOT boosted (10.10 rung 1).
            vact.set_policy(wrappers, vact.RetrievalBoostPolicy(
                tau=args.mem_bonus_tau, scope=scope))
            print("[chat] retrieval-boost delivery ON for memory rows "
                  f"(scope={scope.name() if scope else 'coarse'})",
                  flush=True)

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

    fn0 = fn_bare if args.bare else fn_graft
    if args.probe:
        probes = list(PROBES)
        if (getattr(args, "mem_pack", "") or DEFAULT_MEM_PACK) == "persona":
            try:
                for tag, q, _e, _r in mp.probes():
                    probes.append((f"persona:{tag}", q))
            except Exception as e:  # noqa: BLE001
                print(f"[chat] persona probe list unavailable: {e}",
                      flush=True)
        run_probes(tok, fn0, device, args.max_new_tokens,
                   args.sample, args.temperature, scope=scope, probes=probes)
        return 0
    interactive(tok, fn_bare, fn_graft, device, args, scope=scope)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
