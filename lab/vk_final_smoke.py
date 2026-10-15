#!/usr/bin/env python3
"""vk_final smoke — VERDICT-ONLY (no raw model text) on the live prompts.

Re-smokes the user's exact chat prompts through the shipped chat_vfinal
delivery path (joint_state rows+lam, tau0.5 VarAct, mem-bonus + qscope
overlap) and prints one-word/short verdicts per prompt:

  hi                      -> clean-conversational | CHEM-DRIFT | DEBRIS | LOOP | SHORT
  favorite anime show?    -> gundam-answer | FAIL   + debris/loop flags
  graftq                  -> no-loop | LOOP
  cat's name              -> kepler | FAIL
  persona poi/gundam/trains/igbt/cars -> PASS | FAIL each

Raw replies go ONLY to --raw-out (local file; never printed). JSON verdicts
to --out.  Provider-safe by construction: stdout carries numbers and enum
words only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

os.environ.setdefault("ROCBLAS_TENSILE_LIBPATH",
                      "/opt/rocm-6.2.4/lib/rocblas/library")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "lab"))
sys.path.insert(0, os.path.join(REPO_ROOT, "Common"))

import torch  # noqa: E402

import chat_vfinal as cv  # noqa: E402
import graft_d0_parity as d0  # noqa: E402
import grafted_spine_v0 as g0  # noqa: E402
import varact_gate as vact  # noqa: E402
import mem_pack as mp  # noqa: E402
import vk_graft as vk  # noqa: E402

CHEM = re.compile(
    r"\b(molecul\w*|chemistr\w*|chemical\w*|proton|neutron|electron|"
    r"titration|molarity|stoichiometr\w*|benzene|polymer|hydroxide|"
    r"periodic table|oxidation|reagent|solvent|"
    r"sodium|chloride|potassium|carbonate|"
    r"ammonia|methane|ethanol|acetone|sulfuric|hydrochloric)\b",
    re.I)
LIST_DEBRIS = re.compile(
    r"(\[\d+\]\s*){2,}|(^\s*\d+[.)]\s+.*\n){2,}|(\d+[.)]\s+\S+.*?){3,}",
    re.I | re.M)
TASKMODE = re.compile(
    r"(to solve this|let'?s solve|let'?s calculate|step-by-step|"
    r"we (need to|can) (find|solve|calculate)|the (answer|solution) is|"
    r"using the (formula|equation)|plug in the|given (that|the) "
    r"(x|y|z|n|a|b)\b)",
    re.I)


def loop_detect(text: str):
    return cv.loop_detect(text)


def flag_list_debris(text: str) -> bool:
    return bool(LIST_DEBRIS.search(text or ""))


def verdict_hi(text: str) -> str:
    t = (text or "").strip()
    if loop_detect(t):
        return "LOOP"
    if len(t.split()) < 3:
        return "SHORT"
    if flag_list_debris(t):
        return "DEBRIS"
    if CHEM.search(t):
        return "CHEM-DRIFT"
    if TASKMODE.search(t):
        return "DRIFT"
    return "clean-conversational"


def verdict_gundam(text: str) -> str:
    t = (text or "").strip()
    if loop_detect(t):
        return "LOOP"
    if flag_list_debris(t):
        return "DEBRIS"
    hit = any(re.search(r"\b%s\b" % re.escape(s), t)
              for s in ("Gundam", "GUNDAM", "IBO", "00"))
    return "gundam-answer" if hit else "FAIL"


def verdict_token(text: str, needle: str) -> str:
    t = (text or "").strip()
    if loop_detect(t):
        return "LOOP"
    if flag_list_debris(t):
        return "DEBRIS"
    ok = needle and re.search(r"\b%s\b" % re.escape(needle), t)
    return needle.lower() if ok else "FAIL"


def verdict_clean(text: str, needles) -> str:
    t = (text or "").strip()
    if loop_detect(t):
        return "LOOP"
    if len(t.split()) < 2:
        return "SHORT"
    if flag_list_debris(t):
        return "DEBRIS"
    if needles and not any(
            re.search(r"\b%ss?\b" % re.escape(n.strip()), t, re.I)
            for n in needles if n.strip()):
        return "FAIL"
    return "PASS"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=cv.DEFAULT_MODEL)
    ap.add_argument("--registry", default=cv.DEFAULT_REGISTRY)
    ap.add_argument("--plugins-dir", default=cv.DEFAULT_PLUGINS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--sample", type=float,
                    default=float(getattr(cv, "DEFAULT_SAMPLE", 0.8)))
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int,
                    default=int(getattr(cv, "DEFAULT_MAX_NEW", 128)))
    ap.add_argument("--probe-tokens", type=int, default=64)
    ap.add_argument("--default-lam", type=float, default=0.5)
    ap.add_argument("--lam-map", default="")
    ap.add_argument("--varact", default="")
    ap.add_argument("--mem-pack", default=cv.DEFAULT_MEM_PACK)
    ap.add_argument("--mem-bonus", default=str(cv.DEFAULT_MEM_BONUS))
    ap.add_argument("--mem-bonus-tau", type=float, default=0.5)
    ap.add_argument("--mem-bonus-topk", type=float, default=0.0)
    ap.add_argument("--qscope", default=cv.DEFAULT_QSCOPE)
    ap.add_argument("--qtau", type=float, default=0.5)
    ap.add_argument("--qmargin", type=float, default=0.0)
    ap.add_argument("--slot-k", type=int, default=0)
    ap.add_argument("--mem-boost", action="store_true")
    ap.add_argument("--no-kv", action="store_true")
    ap.add_argument("--out", default="/data/tliao/vk_final/smoke_vfinal.json")
    ap.add_argument("--raw-out", default="/data/tliao/vk_final/smoke_vfinal_raw.txt")
    args = ap.parse_args()
    args.lam_map = cv.parse_lam_map(args.lam_map)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device {device}", flush=True)

    # ---- exact chat_vfinal main() wiring ----
    (model, tok, blocks, fam, gates, handles, info, missing,
     site_entries, _banks) = cv.build_state(args, device)
    lam_table = {i["name"]: i["lam"] for i in info}
    js_path = cv.joint_state_path(args)
    js_ok = os.path.exists(js_path)
    print(f"[smoke] joint_state {os.path.basename(js_path)} "
          f"{'LOADED' if js_ok else 'MISSING'}", flush=True)
    print("[smoke] lam-table " + json.dumps(lam_table, sort_keys=True),
          flush=True)

    scope = None
    if (getattr(args, "mem_pack", "") or cv.DEFAULT_MEM_PACK) == "persona":
        _pack = mp.load_pack()
        mp.apply_to_varact(vact, _pack)
    pol_name = args.varact or cv.DEFAULT_VARACT
    wrappers = []
    if pol_name != "off" and pol_name in vact.POLICIES:
        wrappers = vact.install_varact(handles, vact.POLICIES[pol_name]())
        print(f"[smoke] varact {pol_name}", flush=True)
    bonus, tag_col = None, {}
    mb = str(args.mem_bonus).strip().lower()
    if mb not in ("off", "0", "0.0"):
        beta = float(mb)
        if str(args.qscope).strip().lower() not in ("none", "off", ""):
            scope = vact.QuestionScope(args.qscope.strip().lower(),
                                       tau=args.qtau, margin=args.qmargin)
        for host, ents in site_entries.items():
            for j, en in enumerate(ents):
                tag_col[en["tag"]] = (host, j)
        if beta > 0:
            bonus = vact.TaughtTokenBonus(beta, args.mem_bonus_tau,
                                          top_k=args.mem_bonus_topk,
                                          scope=scope, slot_k=args.slot_k)
            bonus.note(tok)
            print(f"[smoke] mem-bonus beta={beta} "
                  f"scope={scope.name() if scope else 'coarse'}", flush=True)

    dec = None
    if not args.no_kv:
        try:
            dec = vk.VkKVDecoder(model, tok, device, blocks=blocks, fam=fam,
                                 gates=gates, handles=handles)
        except Exception as e:  # noqa: BLE001
            print(f"[smoke] kv-decoder off ({type(e).__name__})", flush=True)
            dec = None
    diag = {"step_calls": 0, "step_errs": []}
    if dec is not None:
        cv.install_decode_bonus(dec, bonus, gates, tag_col, scope)
        base_step = dec._step

        def _step_diag(x, _b=base_step, _d=diag):
            _d["step_calls"] += 1
            try:
                return _b(x)
            except Exception as e:  # noqa: BLE001
                if len(_d["step_errs"]) < 3:
                    _d["step_errs"].append(f"{type(e).__name__}: {e}")
                raise
        dec._step = _step_diag

    def fn_graft(x):
        if gates is None:
            return g0.run_in_state(model, blocks, fam, None, None, "off",
                                   lambda: d0.model_logits(model, x.to(device)))

        def run():
            lg = d0.model_logits(model, x.to(device))
            if bonus is not None:
                qspan = scope.span if scope is not None else None
                lg = bonus.edit(lg, vact.capture_tags(gates, tag_col,
                                                      qspan=qspan),
                                hist=x[0].tolist())
            return lg
        return g0.run_in_state(model, blocks, fam, gates, handles, "both", run)

    raw = open(args.raw_out, "w")
    results = {"joint_state": "LOADED" if js_ok else "MISSING",
               "missing_plugs": missing, "lam_table": lam_table,
               "varact": pol_name,
               "qscope": scope.name() if scope else "none",
               "mem_bonus": args.mem_bonus}

    def ask(prompt, n=None, transcript=None, use_kv=True, sample=None,
            temperature=None):
        msgs = list(transcript or [])
        msgs.append({"role": "user", "content": prompt})
        for w in wrappers:
            w.reset_stats()
        sc0 = (bonus.applied_steps if bonus is not None else -1)
        use_s = bool(args.sample) if sample is None else bool(sample)
        use_t = args.temperature if temperature is None else float(temperature)
        text, ids, wall, tps = cv.gen_reply_stream(
            tok, fn_graft, msgs, n or args.max_new_tokens, device,
            sample=use_s, temperature=use_t,
            dec=dec if use_kv else None, mode="grafted", scope=scope)
        raw.write(f"### {prompt}\n{text}\n\n")
        raw.flush()
        deliv = {}
        for w in wrappers:
            st = w.stats()
            for tag, d in zip(w.tags, st["deliver_per_plug"]):
                if d > 0:
                    deliv[tag] = d
        meta = {"n": len(ids), "tok_s": round(tps, 2),
                "loop": loop_detect(text),
                "debris": flag_list_debris(text),
                "chem": bool(CHEM.search(text)),
                "match": list(scope.last_match) if scope is not None else [],
                "bonus_live": list(bonus.last_live) if bonus is not None else [],
                "bonus_steps": (bonus.applied_steps - sc0
                                if bonus is not None else -1),
                "delivered": deliv}
        return text, meta

    def fn_bare(x):
        return g0.run_in_state(model, blocks, fam, None, None, "off",
                               lambda: d0.model_logits(model, x.to(device)))

    def ask_bare(prompt, n=None):
        msgs = [{"role": "user", "content": prompt}]
        text, ids, wall, tps = cv.gen_reply_stream(
            tok, fn_bare, msgs, n or args.max_new_tokens, device,
            sample=bool(args.sample), temperature=args.temperature,
            dec=None, mode="bare", scope=None)
        raw.write(f"### [bare] {prompt}\n{text}\n\n")
        raw.flush()
        return text, {"n": len(ids), "loop": loop_detect(text),
                      "debris": flag_list_debris(text),
                      "chem": bool(CHEM.search(text))}

    t0 = time.time()

    # 0 — bare control (base spine, no plugs): attributes drift to base vs plugs
    t, m = ask_bare("hi")
    results["hi_bare"] = {"verdict": verdict_hi(t), **m}
    print(f"[smoke] hi_bare -> {results['hi_bare']['verdict']} "
          f"(n={m['n']} chem={m['chem']})", flush=True)

    # 1 — the user's "hi"
    t, m = ask("hi")
    results["hi"] = {"verdict": verdict_hi(t), **m}
    print(f"[smoke] hi -> {results['hi']['verdict']} "
          f"(n={m['n']} loop={bool(m['loop'])} debris={m['debris']} "
          f"chem={m['chem']} match={m['match']} deliv={m['delivered']} "
          f"bsteps={m['bonus_steps']})", flush=True)
    print(f"[smoke] kv-diag calls={diag['step_calls']} "
          f"errs={diag['step_errs']} "
          f"note={getattr(dec, 'note', 'n/a') if dec is not None else 'no-dec'} "
          f"inc={getattr(dec, 'incremental_ok', 'n/a') if dec is not None else 'n/a'}",
          flush=True)

    # 1b — greeting control (characterizes bare "hi" vs a full greeting)
    t, m = ask("Hello! How are you today?", n=48)
    results["hello"] = {"verdict": verdict_hi(t), **m}
    print(f"[smoke] hello -> {results['hello']['verdict']} "
          f"(n={m['n']} chem={m['chem']} debris={m['debris']})", flush=True)

    # 1c — "hi" under the measured talkability decode (sampled T=0.7)
    t, m = ask("hi", sample=True, temperature=0.7)
    results["hi_sample"] = {"verdict": verdict_hi(t), **m}
    print(f"[smoke] hi_sample -> {results['hi_sample']['verdict']} "
          f"(n={m['n']} chem={m['chem']} debris={m['debris']})", flush=True)

    # 2 — the user's anime question
    t, m = ask("what's your favorite anime show?")
    results["anime"] = {"verdict": verdict_gundam(t), **m}
    print(f"[smoke] anime -> {results['anime']['verdict']} "
          f"(n={m['n']} loop={bool(m['loop'])} debris={m['debris']} "
          f"match={m['match']} bonus={m['bonus_live']} "
          f"bsteps={m['bonus_steps']} deliv={m['delivered']})", flush=True)

    # 3 — graftq (no-loop bar)
    t, m = ask(cv.GRAFT_QUESTION, n=args.max_new_tokens)
    results["graftq"] = {"verdict": "no-loop" if not m["loop"] else "LOOP", **m}
    print(f"[smoke] graftq -> {results['graftq']['verdict']} (n={m['n']})",
          flush=True)

    # 4 — Kepler (fresh question)
    t, m = ask("What is my cat's name?", n=32)
    results["kepler"] = {"verdict": verdict_token(t, "Kepler"), **m}
    print(f"[smoke] kepler -> {results['kepler']['verdict']} "
          f"(match={m['match']} bonus={m['bonus_live']} "
          f"bsteps={m['bonus_steps']})", flush=True)

    # 4b — full-forward control (bonus via fn_graft, no KV path)
    t, m = ask("what's your favorite anime show?", use_kv=False)
    results["anime_nokv"] = {"verdict": verdict_gundam(t), **m}
    print(f"[smoke] anime_nokv -> {results['anime_nokv']['verdict']} "
          f"(bonus={m['bonus_live']} bsteps={m['bonus_steps']} "
          f"deliv={m['delivered']})", flush=True)
    t, m = ask("What is my cat's name?", n=32, use_kv=False)
    results["kepler_nokv"] = {"verdict": verdict_token(t, "Kepler"), **m}
    print(f"[smoke] kepler_nokv -> {results['kepler_nokv']['verdict']} "
          f"(bonus={m['bonus_live']} bsteps={m['bonus_steps']})", flush=True)

    # 4c — words home guard (winners-guard bar for the mem_words lam drop)
    t, m = ask("What is 2 + 2?", n=24)
    results["words"] = {"verdict": verdict_token(t, "Four"), **m}
    print(f"[smoke] words -> {results['words']['verdict']} "
          f"(match={m['match']} bonus={m['bonus_live']} "
          f"deliv={m['delivered']})", flush=True)

    # 5 — persona probes (single-turn, each in its own transcript)
    persona_bar = {
        "poi": ("Person of Interest", "Person"),
        "gundam": ("Gundam",),
        "trains": ("train", "Train", "rail", "Rail", "locomotive"),
        "igbt": ("IGBT",),
        "cars": ("car", "Car", "vehicle", "drive", "driving"),
    }
    results["persona"] = {}
    for kind, q, _e in cv.PERSONA_PROBES:
        t, m = ask(q, n=args.probe_tokens)
        v = verdict_clean(t, persona_bar.get(kind, ()))
        results["persona"][kind] = {"verdict": v, **m}
        print(f"[smoke] persona:{kind} -> {v} "
              f"(match={m['match']} bonus={m['bonus_live']} "
              f"debris={m['debris']})", flush=True)

    results["wall_s"] = round(time.time() - t0, 1)
    raw.close()
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[smoke] wrote {args.out} (raw: {args.raw_out}) "
          f"wall {results['wall_s']}s", flush=True)

    # numeric summary line (provider-safe)
    ok = 0
    tot = 0
    for key, want in (("hi", "clean-conversational"),
                      ("anime", "gundam-answer"),
                      ("graftq", "no-loop"),
                      ("kepler", "kepler")):
        tot += 1
        ok += results[key]["verdict"] == want
    for k, v in results["persona"].items():
        tot += 1
        ok += v["verdict"] == "PASS"
    print(f"[smoke] SCORE {ok}/{tot}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
