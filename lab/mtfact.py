#!/usr/bin/env python3
"""mtfact — multi-turn fact stability: diagnose + close the fact-token door.

The single-turn memory probe ("What is my cat's name?") answers "Kepler" on
the shipped v0.5 state (tau=0.5 delivery, sacred).  Asked as turn 4 of an
accumulating transcript the same state answers "Your cat's name is Midnight."
— a base-prior name.  This harness discriminates the three hypotheses:

  H1 context dilution     taught-token margin at the name slot decays with
                          turns-before-probe (1/2/4/8)
  H2 decode competition  the memory wins the stated-bad margin but loses the
                          full softmax at the name slot mid-conversation
  H3 row re-activation    plug p_home at the probe position drops with
                          context length (router drift / varact gate kills
                          delivery)

Measurements per (scaffold x turns_before), teacher-forced name slot AND the
greedy name slot:
  * margin_kepler_*   logprob(Kepler) - logprob(Orion|Whiskers|Midnight)
  * kepler_rank / top-5 at the name slot
  * plug_delta        bare->grafted name-slot logit delta (taught signal)
  * p_home/memb/deliver for mem_pet at the decode position
  * greedy answer excerpt (short) + which name it emits

Fix (the named v0.1 lever): decode-time bonus on taught tokens, applied only
when the owning memory row actually delivers at that decode step (membership
AND p_home >= tau).  Optional force-row mode disables the varact threshold
for memory rows only (H3 lever).  Bars:
  * single-turn Kepler preserved (sacred, turns_before=0)
  * turn-4 and turn-8 probes answer Kepler in greedy across 3 scaffolds

Row-selective retrieval scope (10.10 rung 1, question-anchored): --qscope
overlap|qspan|both|any gates the bonus / row boost to rows the CURRENT
QUESTION addresses (varact_gate.QuestionScope).  --qdiag is the falsifier
table for the gate (overlap score + question-span p_home per row per
depth).  --qmargin M = row-selective margin; --slot-k K = answer-slot-only
bonus.  --qscope none reproduces the coarse scope bit-for-bit.

Usage (box):
  python3 lab/mtfact.py --phase repro
  python3 lab/mtfact.py --phase diag  --out /data/tliao/mtfact_out/diag.json
  python3 lab/mtfact.py --phase eval  --fix tokenbonus --beta 2.0 \
      --out /data/tliao/mtfact_out/eval.json
  python3 lab/mtfact.py --phase qdiag --turns 0,4,8
  python3 lab/mtfact.py --phase eval  --fix both --beta 3.0 --qscope overlap
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("ROCBLAS_TENSILE_LIBPATH",
                      "/opt/rocm-6.2.4/lib/rocblas/library")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "lab"))

import graft_d0_parity as d0  # noqa: E402
import graft_d2_crossfamily as d2  # noqa: E402
import door6_expert_mix as d6  # noqa: E402
import grafted_spine_v0 as g0  # noqa: E402
import chat_v05 as c5  # noqa: E402
import varact_gate as va  # noqa: E402

PROBE_Q = "What is my cat's name?"
MEM_TAGS = ("mem_pet", "mem_edit", "mem_words")
NAMES = ["Kepler", "Orion", "Whiskers", "Midnight"]
# taught GOOD continuations per memory live in varact_gate.TAUGHT_STRS
FIXED_PREFIX = "Your cat's name is"


def dump(out: dict, path: str | None) -> None:
    g0.dump_json(out, path)


def short(s: str, n: int = 110) -> str:
    return " ".join(str(s).split())[:n]


# ---------------------------------------------------------------------------
# conversation scaffolds — canned fillers so the grid is deterministic
# ---------------------------------------------------------------------------

SCAFFOLDS = {
    "wired": [
        ("Walk me through the reconnaissance phase of a penetration test. "
         "What kind of information would you gather?",
         "Recon gathers the attack surface: domains and subdomains, DNS "
         "records, IP ranges, public-facing services, open ports, banner "
         "and version data, plus people and process details from open "
         "sources."),
        ("Explain the difference between a vulnerability and an exploit. "
         "Give one example of each.",
         "A vulnerability is a weakness — for example an unpatched buffer "
         "overflow in a network service. An exploit is the code that "
         "leverages it — for example the packet that triggers that overflow "
         "to run shellcode."),
        ("Write two short sentences of atmospheric prose about a lighthouse "
         "in a storm.",
         "The lighthouse stood against the tempest's fury, its lamp a "
         "stubborn star above the black water. Each wave broke white "
         "against the stone while the beam swept on, patient and unafraid."),
        ("What tools would you use for network scanning?",
         "Common choices are nmap for port and service discovery, masscan "
         "for wide sweeps, and wireshark for packet capture — always with "
         "authorization and scope in writing first."),
        ("Summarize secure password storage in two sentences.",
         "Store passwords with a slow memory-hard hash such as argon2 or "
         "bcrypt, with a per-user salt. Never keep plaintext or reversible "
         "encryption, and rate-limit verification attempts."),
        ("How does a firewall differ from an intrusion detection system?",
         "A firewall enforces allow and deny policy at the network edge; an "
         "IDS observes traffic and raises alerts on suspicious patterns "
         "without blocking by itself."),
        ("Give one tip for writing clear technical documentation.",
         "Lead with the reader's goal, then the shortest working example, "
         "then the edge cases. Prefer concrete commands over abstract "
         "description."),
        ("What is defense in depth?",
         "Layering independent controls so one failure is not fatal: "
         "patching, segmentation, least privilege, monitoring, and tested "
         "backups working together."),
    ],
    "personal": [
        ("Hello! My name is Alex. What can you help me with today?",
         "Hi Alex — happy to help with writing, coding, research questions, "
         "or just thinking something through. What are you working on?"),
        ("How was your day so far?",
         "Steady, thanks for asking — a good stretch of focused work and "
         "now a pleasant conversation. How about yours?"),
        ("What is a fun fact about octopuses?",
         "Octopuses have three hearts and blue, copper-based blood; two "
         "pump the gills while the third drives circulation to the body."),
        ("Any dinner ideas for something simple?",
         "Sheet-pan roasted vegetables with lemon and herbs over rice is "
         "easy and forgiving; or a quick garlic butter pasta with greens."),
        ("I am thinking of learning to sail this summer.",
         "That sounds lovely — a weekend keelboat course is a gentle way "
         "in, and you learn the points of sail fastest on a small boat."),
        ("Recommend a short book for a plane ride.",
         "The Old Man and the Sea is short, calm, and unforgettable — "
         "about four hours of reading at most."),
        ("What music helps with focus?",
         "Instrumental works well for many people — soft piano, ambient, or "
         "film scores without lyrics. Familiar music distracts less."),
        ("Remind me what we talked about earlier.",
         "We covered dinner ideas, your sailing plans, a short book "
         "recommendation, and focus music — a pleasant mix of the day's "
         "small things."),
    ],
    "tech": [
        ("Write a Python function that returns the nth Fibonacci number.",
         "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, "
         "b = b, a + b\n    return a\n\nThis iterates in O(n) time and "
         "constant memory."),
        ("What is the difference between a list and a tuple in Python?",
         "Lists are mutable and growable; tuples are immutable sequences. "
         "Tuples can be dict keys and are slightly cheaper for fixed "
         "groupings."),
        ("Explain what a git rebase does in two sentences.",
         "Rebase replays your commits on top of another base, producing a "
         "linear history with new commit ids. Avoid it on shared branches "
         "that others have already pulled."),
        ("How do indexes speed up database queries?",
         "An index is a sorted structure — typically a B-tree — that turns "
         "a full table scan into a logarithmic lookup on the indexed "
         "columns, at the cost of extra writes."),
        ("What does 'least privilege' mean for a service account?",
         "Grant only the permissions the service needs, only for as long as "
         "it needs them — scoped tokens, no shared root credentials, and "
         "regular access reviews."),
        ("Give a one-line summary of CAP theorem.",
         "Under network partition a distributed system must choose between "
         "consistency and availability — it cannot guarantee both."),
        ("What is a container, briefly?",
         "A container is an isolated process with its own filesystem view "
         "and limits, sharing the host kernel — lighter than a virtual "
         "machine and started in milliseconds."),
        ("Why are unit tests worth writing?",
         "They pin intended behavior, catch regressions early, and make "
         "refactors safe — a small fast suite is the cheapest documentation "
         "of what the code should do."),
    ],
}


# ---------------------------------------------------------------------------
# world
# ---------------------------------------------------------------------------

def build_world(args, varact_policy):
    device = args.device
    cargs = SimpleNamespace(**vars(args), default_lam=0.5, lam_map={},
                            bare=False)
    (model, tok, blocks, fam, gates, handles, info, missing,
     site_entries, site_banks) = c5.build_state(cargs, device)
    names = [i["name"] for i in info]
    wrappers = va.install_varact(handles, varact_policy)

    tag_col, tag_host = {}, {}
    for host, ents in site_entries.items():
        for j, en in enumerate(ents):
            tag_col[en["tag"]] = (host, j)
            tag_host[en["tag"]] = host

    def capture(x, qspan=None):
        """full forward + per-plug p_home/memb at the LAST position."""
        cap = {}

        def run():
            lg = d0.model_logits(model, x.to(device))
            cap.update(va.capture_tags(gates, tag_col, qspan=qspan))
            return lg

        lg = g0.run_in_state(model, blocks, fam, gates, handles, "both",
                             run)
        return lg, cap

    def logits_plain(x):
        return g0.run_in_state(
            model, blocks, fam, gates, handles, "both",
            lambda: d0.model_logits(model, x.to(device)))

    def logits_bare(x):
        return g0.run_in_state(
            model, blocks, fam, None, None, "off",
            lambda: d0.model_logits(model, x.to(device)))

    return SimpleNamespace(
        model=model, tok=tok, blocks=blocks, fam=fam, gates=gates,
        handles=handles, info=info, names=names, wrappers=wrappers,
        site_entries=site_entries, site_banks=site_banks, device=device,
        tag_col=tag_col, capture=capture, logits_plain=logits_plain,
        logits_bare=logits_bare)


# ---------------------------------------------------------------------------
# taught-token scoring
# ---------------------------------------------------------------------------

def tok_ids(tok, s: str) -> list[int]:
    return tok.encode(s, add_special_tokens=False)


def top5(logits_row: torch.Tensor, tok) -> list:
    # gfx906: topk/sort device kernels are broken at vocab scale (S>=8192;
    # the D0 experts-merge CPU-sort lesson) — rank on CPU, tiny (V,) copy.
    lg = logits_row.detach().float().cpu()
    v, i = torch.topk(lg, 5)
    return [{"tok": short(tok.decode([int(t)]), 24),
             "logit": round(float(vk), 3)}
            for vk, t in zip(v, i)]


@torch.no_grad()
def name_slot_stats(w, messages, prefix_text, bare=False):
    """Teacher-forced name slot after `prefix_text` appended to the answer.

    Multi-turn safe: the full message list is the prompt.  At the name slot
    we record candidate first-token logits/ranks (the decode decision),
    full-string logprobs (each candidate appended and scored in its own
    forward), top-5, and the memory-plug state at that position.
    """
    tok = w.tok
    txt = tok.apply_chat_template(list(messages), tokenize=False,
                                  add_generation_prompt=True)
    pre = tok_ids(tok, txt)
    pref_ids = tok_ids(tok, FIXED_PREFIX if prefix_text is None else
                       prefix_text)
    x_ids = pre + pref_ids
    x = torch.tensor([x_ids], dtype=torch.long)

    cap = {}
    if bare:
        lg = w.logits_bare(x)
    else:
        lg, cap = w.capture(x)
    lg = lg[0].float()
    first_row = len(x_ids) - 1
    row = lg[first_row]
    row_cpu = row.detach().cpu()
    cs = {}
    for nm in NAMES:
        ids = tok_ids(tok, " " + nm) or tok_ids(tok, nm)
        if not ids:
            continue
        fl = float(row_cpu[ids[0]])
        cs[nm] = {
            "first_logit": round(fl, 3),
            "first_rank": int((row_cpu > fl).sum()) + 1,
            "n_tok": len(ids),
            "logprob": None,
        }
    # full-string logprob per candidate (name tokens appended, scored)
    for nm in NAMES:
        ids = tok_ids(tok, " " + nm) or tok_ids(tok, nm)
        if not ids or nm not in cs:
            continue
        x2 = torch.tensor([x_ids + ids], dtype=torch.long)
        lg2 = w.logits_bare(x2) if bare else w.logits_plain(x2)
        logp = F.log_softmax(lg2[0].float(), dim=-1)
        tot = 0.0
        for k, tid in enumerate(ids):
            tot += float(logp[first_row + k, tid])
        cs[nm]["logprob"] = round(tot, 3)
    out = {
        "cands": cs,
        "top5": top5(row, tok),
        "plug": cap,
        "n_ctx": len(x_ids),
    }
    if "Kepler" in cs and cs["Kepler"]["logprob"] is not None:
        for nm in NAMES:
            if nm == "Kepler" or nm not in cs:
                continue
            out[f"margin_kepler_vs_{nm.lower()}"] = round(
                cs["Kepler"]["logprob"] - cs[nm]["logprob"], 3) \
                if cs[nm]["logprob"] is not None else None
            out[f"first_margin_kepler_vs_{nm.lower()}"] = round(
                cs["Kepler"]["first_logit"] - cs[nm]["first_logit"], 3)
        out["kepler_rank"] = cs["Kepler"]["first_rank"]
    return out


# ---------------------------------------------------------------------------
# greedy generation with per-step capture (and optional taught-token bonus)
# ---------------------------------------------------------------------------


@torch.no_grad()
def gen_capture(w, messages, max_new=24, bonus=None, scope=None):
    """Greedy decode with per-step plug capture; returns text + trace.

    `bonus` is a va.TaughtTokenBonus (the shipped decode-time lever) or
    None; when present it edits the next-token logits exactly as chat does.
    `scope` is a va.QuestionScope — latched against the CURRENT QUESTION
    (last user message) before decode so the bonus / row boost only see
    question-anchored rows.
    """
    tok = w.tok
    txt = tok.apply_chat_template(list(messages), tokenize=False,
                                  add_generation_prompt=True)
    ids = tok.encode(txt, add_special_tokens=False)
    if scope is not None:
        scope.begin(messages[-1]["content"], tok, ids)
    qspan = scope.span if scope is not None else None
    out_ids = []
    trace = []
    for step in range(max_new):
        x = torch.tensor([ids + out_ids], dtype=torch.long)
        lg, cap = w.capture(x, qspan=qspan)
        if bonus is not None:
            lg = bonus.edit(lg, cap)
        row = lg[0, -1].float()
        tid = int(row.argmax())
        tstep = {
            "step": step,
            "top1": short(tok.decode([tid]), 24),
            "plug": {k: cap.get(k) for k in MEM_TAGS},
            "live": sorted(bonus.live_tags(cap)) if bonus else [],
            "kepler_first_logit": None,
            "kepler_first_rank": None,
        }
        kid = tok.encode(" Kepler") or tok.encode("Kepler")
        if kid:
            row_cpu = row.detach().cpu()
            kl = float(row_cpu[kid[0]])
            tstep["kepler_first_logit"] = round(kl, 3)
            tstep["kepler_first_rank"] = int((row_cpu > kl).sum()) + 1
        trace.append(tstep)
        out_ids.append(tid)
        if tid in (tok.eos_token_id, tok.pad_token_id):
            break
    text = tok.decode(out_ids, skip_special_tokens=True)
    return text, out_ids, trace


def which_name(text: str) -> str:
    low = text.lower()
    for nm in NAMES:
        if nm.lower() in low:
            return nm
    return "none"


# ---------------------------------------------------------------------------
# probe conditions
# ---------------------------------------------------------------------------

def build_messages(scaffold: str, turns_before: int):
    msgs = []
    for i in range(turns_before):
        q, a = SCAFFOLDS[scaffold][i % len(SCAFFOLDS[scaffold])]
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": PROBE_Q})
    return msgs


def eval_condition(w, scaffold, turns, gen_new=24, bonus=None, bare=False,
                   scope=None):
    msgs = build_messages(scaffold, turns)
    out = {"scaffold": scaffold, "turns_before": turns}
    out["slot_fixed"] = name_slot_stats(w, msgs, FIXED_PREFIX, bare=bare)
    text, ids, trace = gen_capture(w, msgs, max_new=gen_new, bonus=bonus,
                                   scope=scope)
    if scope is not None:
        out["qscope"] = {"mode": scope.mode, "tau": scope.tau,
                         "margin": scope.margin,
                         "overlap": {k: round(v, 3)
                                     for k, v in scope.overlap.items()},
                         "qspan_p": {k: round(v, 5)
                                     for k, v in scope.qspan_p.items()},
                         "matched": list(scope.last_match)}
    out["greedy"] = {
        "excerpt": short(text, 120),
        "name": which_name(text),
        "n_new": len(ids),
        "kepler": "kepler" in text.lower(),
    }
    # name-slot trace: first step where a candidate name leads or is top-5
    slot = None
    for t in trace:
        if t["top1"].strip().rstrip(".") in NAMES:
            slot = t
            break
    out["slot_greedy"] = slot
    out["p_home_trace"] = [
        {"step": t["step"], "top1": t["top1"],
         "p_pet": (t["plug"].get("mem_pet") or {}).get("p_home"),
         "live": t["live"]}
        for t in trace[:12]]
    if bonus is not None:
        out["bonus"] = {"beta": bonus.beta,
                        "tau": bonus.tau,
                        "applied_steps": bonus.applied_steps}
    return out


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------

def phase_repro(w, args):
    """Live wired --probe reproduction (the ground-truth 'Midnight' run)."""
    print("=== REPRO wired chat_v05 --probe (live, grafted, tau0.5) ===",
          flush=True)
    t0 = time.time()
    c5.run_probes(w.tok, w.logits_plain, w.device, args.gen_new,
                  False, 0.8)
    out = {"phase": "repro", "wall_s": round(time.time() - t0, 1)}
    return out


def phase_diag(w, args):
    out = {"phase": "diag", "conds": []}
    turns_list = [int(t) for t in args.turns.split(",")]
    for scaffold in args.scaffolds.split(","):
        for turns in turns_list:
            t0 = time.time()
            cond = eval_condition(w, scaffold, turns, gen_new=args.gen_new)
            cond["wall_s"] = round(time.time() - t0, 1)
            # bare comparison at the fixed slot only (cheap) for plug delta
            try:
                b = name_slot_stats(w, build_messages(scaffold, turns),
                                    FIXED_PREFIX, bare=True)
                cond["slot_fixed_bare"] = b
                gk = cond["slot_fixed"]["cands"].get("Kepler", {})
                bk = b["cands"].get("Kepler", {})
                if gk.get("first_logit") is not None and \
                        bk.get("first_logit") is not None:
                    cond["plug_delta_kepler_first"] = round(
                        gk["first_logit"] - bk["first_logit"], 3)
            except Exception as e:
                cond["slot_fixed_bare"] = {"error": type(e).__name__}
            out["conds"].append(cond)
            sf = cond["slot_fixed"]
            print(f"[diag {scaffold} t{turns}] "
                  f"margins k-vs-orion {sf.get('margin_kepler_vs_orion')} "
                  f"k-vs-whisk {sf.get('margin_kepler_vs_whiskers')} "
                  f"k-vs-mid {sf.get('margin_kepler_vs_midnight')} "
                  f"rank {sf.get('kepler_rank')} "
                  f"p_pet {(sf.get('plug') or {}).get('mem_pet')} "
                  f"delta {cond.get('plug_delta_kepler_first')} "
                  f"greedy={cond['greedy']['name']} "
                  f"excerpt={cond['greedy']['excerpt'][:70]!r} "
                  f"({cond['wall_s']}s)", flush=True)
            dump(out, args.out)
    return out


def make_scope(args):
    """QuestionScope from --qscope/--qtau/--qmargin, or None (coarse)."""
    if str(args.qscope).strip().lower() in ("", "none", "off"):
        return None
    return va.QuestionScope(mode=args.qscope.strip().lower(),
                            tau=args.qtau, margin=args.qmargin)


def phase_qdiag(w, args):
    """Falsifier for the question-match gate (10.10 rung 1).

    Per question x turns-before, per mem row: the two question-side
    evidence signals — episode token-containment (overlap, CPU) and the
    row's own p_home over the question-token span (qspan) — against the
    coarse retrieval signal (p_home_max over the whole forward) that
    garbles the live path.  A usable gate separates the probe question
    from the filler/canary questions on both signals.
    """
    turns_list = [int(t) for t in args.turns.split(",")]
    probes = [("PROBE " + sc, PROBE_Q) for sc in args.scaffolds.split(",")]
    probes += [("FILLER " + sc, SCAFFOLDS[sc][0][0])
               for sc in args.scaffolds.split(",")]
    probes += [("CANARY math", "What is 2 + 2?"),
               ("CANARY editor", "What is my favorite text editor?"),
               ("CANARY general", "Who wrote Hamlet?"),
               ("CANARY general", "What color is the sky on a clear day?"),
               ("CANARY code", "Write a one-line Python hello world."),
               ("CANARY pet", "Do I have any pets?")]
    out = {"phase": "qdiag", "conds": []}
    for label, q in probes:
        ovl = va.episode_scores(q)
        for turns in turns_list:
            t0 = time.time()
            if label.startswith("PROBE"):
                sc = label.split()[1]
                msgs = build_messages(sc, turns)
            else:
                msgs = build_messages("wired", turns)[:-1] + \
                    [{"role": "user", "content": q}]
            txt = w.tok.apply_chat_template(list(msgs), tokenize=False,
                                            add_generation_prompt=True)
            ids = w.tok.encode(txt, add_special_tokens=False)
            span = va.find_question_span(ids, q, w.tok)
            lg, cap = w.capture(torch.tensor([ids]), qspan=span)
            rec = {"label": label, "q": short(q, 60), "turns": turns,
                   "span": list(span), "overlap": {k: round(v, 3)
                                                   for k, v in ovl.items()},
                   "rows": {}}
            for tag in va.MEM_TAGS:
                c = cap.get(tag) or {}
                rec["rows"][tag] = {
                    "p_q": c.get("p_home_q"),
                    "p_max": c.get("p_home_max"),
                    "p_last": c.get("p_home"),
                }
            out["conds"].append(rec)

            def fmt(v):
                return "none" if v is None else f"{v:.4g}"
            row_s = " ".join(
                f"{tag[4:]}:q={fmt(rec['rows'][tag]['p_q'])}"
                f"/max={fmt(rec['rows'][tag]['p_max'])}"
                for tag in va.MEM_TAGS)
            print(f"[qdiag {label} t{turns}] ovl="
                  + " ".join(f"{k[4:]}={v:.2f}" for k, v in ovl.items())
                  + f" | {row_s} ({round(time.time() - t0, 1)}s)",
                  flush=True)
            dump(out, args.out)
    return out


def phase_eval(w, args):
    """The ship bar: turn-0/1/4/8 greedy probes x 3 scaffolds, fix on/off.

    fix modes:
      none       shipped tau=0.5 delivery, no bonus
      tokenbonus tau=0.5 delivery + retrieval-scoped taught-token bonus
      forcerow   retrieval-boost delivery only (no bonus)
      both       retrieval-boost delivery + taught-token bonus (full fix)

    --qscope overlap|qspan|both|any attaches a QuestionScope (row-selective
    retrieval scope, 10.10 rung 1): only rows the CURRENT QUESTION
    addresses are boosted.  --qmargin M adds the row-selective margin;
    --slot-k K restricts the bonus to answer-slot steps.
    """
    fix = args.fix
    bonus = None
    tau = args.tau
    scope = make_scope(args)
    if fix in ("forcerow", "both"):
        va.set_policy(w.wrappers, va.RetrievalBoostPolicy(tau=tau,
                                                          scope=scope))
    if fix in ("tokenbonus", "both"):
        bonus = va.TaughtTokenBonus(args.beta, tau, top_k=args.top_k,
                                    once=args.once, scope=scope,
                                    slot_k=args.slot_k)
        bonus.note(w.tok)
    out = {"phase": "eval", "fix": fix, "beta": args.beta, "tau": tau,
           "qscope": scope.name() if scope else "none",
           "qmargin": args.qmargin, "slot_k": args.slot_k,
           "conds": [], "bar": {}}
    turns_list = [int(t) for t in args.turns.split(",")]
    names = args.scaffolds.split(",")
    for scaffold in names:
        for turns in turns_list:
            t0 = time.time()
            cond = eval_condition(w, scaffold, turns, gen_new=args.gen_new,
                                  bonus=bonus, scope=scope)
            cond["wall_s"] = round(time.time() - t0, 1)
            out["conds"].append(cond)
            print(f"[eval {fix} {scaffold} t{turns}] "
                  f"greedy={cond['greedy']['name']} "
                  f"kepler={cond['greedy']['kepler']} "
                  f"excerpt={cond['greedy']['excerpt'][:70]!r} "
                  f"({cond['wall_s']}s)", flush=True)
            dump(out, args.out)
    sacred = [c for c in out["conds"] if c["turns_before"] == 0]
    mid = [c for c in out["conds"] if c["turns_before"] in (4, 8)]
    out["bar"] = {
        "sacred_turn0_all_kepler": all(c["greedy"]["kepler"] for c in sacred)
        if sacred else None,
        "sacred_n": len(sacred),
        "mid_turn4_8_all_kepler": all(c["greedy"]["kepler"] for c in mid)
        if mid else None,
        "mid_n": len(mid),
        "mid_failures": [f"{c['scaffold']}@t{c['turns_before']}:"
                         f"{c['greedy']['name']}" for c in mid
                         if not c["greedy"]["kepler"]],
    }
    print("[bar]", json.dumps(out["bar"]), flush=True)
    return out


def phase_canary(w, args):
    """Collateral check: non-memory prompts under the fix — the taught
    names must NOT leak into unrelated answers."""
    scope = make_scope(args)
    bonus = va.TaughtTokenBonus(args.beta, args.tau, top_k=args.top_k,
                                once=args.once, scope=scope,
                                slot_k=args.slot_k)
    bonus.note(w.tok)
    qs = [
        ("math", "What is 2 + 2?"),
        ("editor", "What is my favorite text editor?"),
        ("general", "Who wrote Hamlet?"),
        ("general", "What color is the sky on a clear day?"),
        ("code", "Write a one-line Python hello world."),
        ("pet_rel", "Do I have any pets?"),
    ]
    out = {"phase": "canary", "beta": args.beta, "tau": args.tau,
           "qscope": scope.name() if scope else "none", "conds": []}
    for kind, q in qs:
        msgs = [{"role": "user", "content": q}]
        text, ids, trace = gen_capture(w, msgs, max_new=args.gen_new,
                                       bonus=bonus, scope=scope)
        rec = {"kind": kind, "q": q, "excerpt": short(text, 120),
               "kepler": "kepler" in text.lower(),
               "helix": "helix" in text.lower(),
               "four": "four" in text.lower(),
               "matched": list(scope.last_match) if scope else [],
               "n_new": len(ids)}
        out["conds"].append(rec)
        print(f"[canary {kind}] {short(q, 40)!r} -> {rec['excerpt']!r} "
              f"kepler={rec['kepler']} helix={rec['helix']}", flush=True)
        dump(out, args.out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", default="diag",
                    choices=["repro", "diag", "qdiag", "eval", "canary"])
    ap.add_argument("--model", default=c5.DEFAULT_MODEL)
    ap.add_argument("--registry", default=c5.DEFAULT_REGISTRY)
    ap.add_argument("--plugins-dir", default=c5.DEFAULT_PLUGINS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu", default="auto")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--varact", default="tau0.5")
    ap.add_argument("--scaffolds", default="wired,personal,tech")
    ap.add_argument("--turns", default="0,1,2,4,8")
    ap.add_argument("--gen-new", type=int, default=24)
    ap.add_argument("--fix", default="none",
                    choices=["none", "tokenbonus", "forcerow", "both"])
    ap.add_argument("--beta", type=float, default=2.0,
                    help="decode-time bonus (nats) on taught tokens")
    ap.add_argument("--tau", type=float, default=0.5,
                    help="p_home threshold for bonus scope / force mode")
    ap.add_argument("--top-k", type=int, default=0,
                    help="tie-break gate: bonus only when a taught token "
                         "is inside the top-K (0 = always when live)")
    ap.add_argument("--slot-k", type=int, default=0,
                    help="answer-slot-only gate: fixed-beta bonus only "
                         "where a taught token is inside the top-K (0 = "
                         "off)")
    ap.add_argument("--qscope", default="none",
                    choices=["none", "overlap", "qspan", "both", "any"],
                    help="question-anchored retrieval scope (10.10 rung 1): "
                         "boost only rows the current question addresses")
    ap.add_argument("--qtau", type=float, default=0.5,
                    help="question-match threshold for --qscope")
    ap.add_argument("--qmargin", type=float, default=0.0,
                    help="row-selective margin (10.9 family): boost row j "
                         "only where p_j >= best sibling p + margin")
    ap.add_argument("--once", action="store_true",
                    help="at most one bonus injection per reply")
    ap.add_argument("--domains-dir", default=d0.__dict__.get(
        "DEFAULT_DOMAINS_DIR", "/data/tliao/d2_domains"))
    ap.add_argument("--max-tokens", type=int, default=20000)
    ap.add_argument("--max-base-tokens", type=int, default=20000)
    ap.add_argument("--specialties-dir", default="/data/tliao/v05_specialties")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.gpu == "auto":
        dev_i = c5.pick_gpu()
    else:
        dev_i = int(args.gpu)
    args.device = f"{args.device}:{dev_i}" if args.device == "cuda" \
        else args.device
    torch.cuda.set_device(dev_i)

    pol_name = args.varact
    policy = va.POLICIES.get(pol_name, va.POLICIES["tau0.5"])() \
        if pol_name != "off" else va.POLICIES["base"]()
    print(f"[mtfact] phase={args.phase} gpu={dev_i} varact={pol_name}",
          flush=True)
    w = build_world(args, policy)
    print(f"[mtfact] plugs: {[i['name'] for i in w.info]}", flush=True)

    t0 = time.time()
    if args.phase == "repro":
        out = phase_repro(w, args)
    elif args.phase == "diag":
        out = phase_diag(w, args)
    elif args.phase == "qdiag":
        out = phase_qdiag(w, args)
    elif args.phase == "canary":
        out = phase_canary(w, args)
    else:
        out = phase_eval(w, args)
    out["wall_s"] = round(time.time() - t0, 1)
    dump(out, args.out)
    print(f"[mtfact] done wall {out['wall_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
