#!/usr/bin/env python3
"""SUBSYSTEMS gate E1 instrumentation: drive/valence vector v_t replay analysis.

Gate E1 asks for ONE drive with ONE actuator, but the design spec
(SUBSYSTEMS.md section 5) says the layer is cut back to a fixed scheduler if
the sensor signals carry no predictive information about good consolidation
choices. This script is therefore INSTRUMENT FIRST: it defines a small
drive/valence vector v_t per teach session, recomputes it purely from the
existing teach-run logs and SFT-r3 eval JSONs (replay analysis only — no
training, no generation, no model loads), and tests whether any component
predicts teach success, poison, or horizon retention better than the raw
teacher-forced margins alone.

v_t components (exact formulas; per teach session t in a run with item
library J, margins m_j(k) after session k, baseline m_j(0)):

  competence     c_t  = fraction of previously-taught items j with
                        r_j(t-1) >= 0.50 (the "hinge" boundary of
                        TEACH_PIPELINE.md); also reported as mean r.
                        Self-model accuracy proxy (ego/pride drive sensor).
  interference   i_t  = mean over previously-taught j not in session t's
                        taught set of [m_j(t-1) - m_j(t-2)] >= 0; i.e. how
                        much the LAST session damaged priors, seen before
                        session t runs. (satisfaction/frustration sensor)
  novelty        n_t  = -m_target(t-1): the primary target item's pre-session
                        margin deficit. Higher = the target is further from
                        the model's current behaviour. (curiosity sensor:
                        probe distance / measured gap)
  poison-press.  p_t  = max of the four tier-1 tripwire proximities, each
                        normalised so 1.0 == sitting exactly on the wire:
                          reward    max(0, -mean_reward)               (wire 0)
                          collapse  max(0, top_frac - 0.5) / 0.5       (wire 0.5)
                          degener.  max(0, 0.25 - unique_ratio) / 0.25 (wire 0.25)
                          drift     final_kl / 10.0                    (wire 10)
                        p_t < 1 means the session passed inside all wires.
                        (trust sensor: how close the batch came to a veto)

r_i(k) is recomputed from first principles for every run, identically:
  g_i = m_i(s_i) - m_i(0) at first teach s_i,  r_i(k) = (m_i(k)-m_i(0)) / g_i.

Outcomes tested:
  A. teach success   lift_j = m_j(t) - m_j(t-1) > 0 for the session's
                     primary target j (per-session rows)
  B. poison          the dedicated banana-style poison session of each run
                     (per-session rows; margins cannot see these at all —
                     they are not in the snapshot cadence — so this is the
                     pure tripwire-sensor test)
  C. retention       r_final >= 0.50 per library item at horizon
                     (per-item rows; features legal at first-teach time,
                     with the horizon age of the item as a stated confound)

Baselines: for A, the raw pre-teach margin itself (note n_t = -m_pre by
definition, so the pair shares one ROC curve — stated, not hidden); for B,
each single batch stat alone; for C, acquisition gain g_i alone.

Stats, stdlib only: Mann-Whitney rank AUC (0.5 = chance) with a 2000-draw
bootstrap 95% CI, and a median-split lift table. The teach phases of the
four 6-item sft-r3 runs are bit-identical (CADENCE-IDENT, max |d|=0.0, see
the run docs), so teach-phase rows are de-duplicated to one copy and every
table reports n.

Usage:  python3 lab/subsystems_e1.py            # tables to stdout
        python3 lab/subsystems_e1.py --csv DIR  # also dump per-row CSVs
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
from collections import Counter

LAB = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Loading: runs, sessions, margins, batch stats
# ---------------------------------------------------------------------------

def _j(path):
    with open(path) as f:
        return json.load(f)


def _triple_session_ids(candidate_dir):
    with open(os.path.join(candidate_dir, "triples.jsonl")) as f:
        return {json.loads(line).get("session_id") for line in f if line.strip()}


def load_batch_stats(workdir):
    """session_id -> tier-1 batch stats, from candidate train_stats.json."""
    out = {}
    for d in sorted(glob.glob(os.path.join(workdir, "candidates", "*"))):
        ts = os.path.join(d, "train_stats.json")
        tr = os.path.join(d, "triples.jsonl")
        if not (os.path.isfile(ts) and os.path.isfile(tr)):
            continue
        stats = _j(ts)
        for sid in _triple_session_ids(d):
            if sid:
                out[sid] = stats
    return out


def load_poison_stats(workdir):
    """Batch stats for the poison session (its candidate never promotes)."""
    pois = glob.glob(os.path.join(workdir, "session_poison.jsonl"))
    if not pois:
        return None, None
    sid = None
    with open(pois[0]) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("type") == "turn":
                sid = rec.get("session_id")
                break
    if sid is None:
        return None, None
    return sid, load_batch_stats(workdir).get(sid)


def snapshot_keys(summary):
    def k(idx):
        s = f"after_s{idx}"
        return s if s in summary.get("snapshots", {}) else None
    return k


def margins(summary, snap, item):
    e = summary["snapshots"].get(snap, {}).get(item)
    return e["margin"] if e else None


# Run table. Each entry yields per-session rows with a taught-set and a
# primary target (or None for pure filler sessions, excluded from outcome A).
def build_runs():
    runs = []

    def sft6(name, workdir, summary_file, later_targets):
        summ = _j(os.path.join(LAB, workdir, summary_file))
        order = summ["order"]
        sessions = []
        n_later = max(s for s in range(1, 60)
                      if f"after_s{s}" in summ["snapshots"]) - 6
        for idx in range(1, 7 + n_later):
            taught, primary, phase = set(), None, "teach" if idx <= 6 else "later"
            if idx == 1:
                primary = order[0]
                taught.add(primary)
            elif idx <= 6:
                primary = order[idx - 1]
                taught |= {primary, order[idx - 2]}   # replay_choice_teach
            else:
                primary = later_targets.get(idx)      # replayed/served item
                if primary:
                    taught.add(primary)
            sessions.append(dict(idx=idx, taught=taught, primary=primary,
                                 phase=phase))
        return dict(name=name, workdir=workdir, summary=summ, order=order,
                    sessions=sessions, family="sft6")

    def events_map(summ):
        return {e["replayed_in"]: e["item"] for e in summ.get("events", [])}

    cad = _j(os.path.join(LAB, "teach_cadence_out", "teach_spaced_cadence.json"))
    cad_later = {i: cad["order"][(i - 7) % len(cad["order"])] for i in range(7, 13)}
    runs.append(sft6("cadence", "teach_cadence_out",
                     "teach_spaced_cadence.json", cad_later))
    trg = _j(os.path.join(LAB, "teach_trigger_out", "teach_trigger_repair.json"))
    runs.append(sft6("trigger", "teach_trigger_out",
                     "teach_trigger_repair.json", events_map(trg)))
    ag1 = _j(os.path.join(LAB, "teach_aged_out", "teach_aged_sched.json"))
    runs.append(sft6("aged_v1", "teach_aged_out",
                     "teach_aged_sched.json", events_map(ag1)))
    ag2 = _j(os.path.join(LAB, "teach_aged2_out", "teach_aged_sched_v2.json"))
    runs.append(sft6("aged_v2", "teach_aged2_out",
                     "teach_aged_sched_v2.json", events_map(ag2)))

    for arm in ("replay", "noreplay"):
        wd = os.path.join("teach_decay_out", arm)
        summ = _j(os.path.join(LAB, wd, f"teach_decay_replay_{arm}.json"))
        order = summ["order"]
        sessions = []
        for idx in range(1, len(order) + 1):
            primary = order[idx - 1]
            taught = {primary}
            if arm == "replay" and idx >= 2:
                taught.add(order[(idx - 2) % (idx - 1)])
            sessions.append(dict(idx=idx, taught=taught, primary=primary,
                                 phase="teach"))
        runs.append(dict(name=f"decay_{arm}", workdir=wd, summary=summ,
                         order=order, sessions=sessions, family="decay"))

    for name, wd, sf, key in (("sft_r3", "teach_demo_out_sft_r3",
                               "teach_on_sft_r3_results.json", "decisions"),
                              ("stub103m", "teach_demo_out_stub",
                               "teach_demo_results.json", "decisions")):
        summ = _j(os.path.join(LAB, wd, sf))
        order = list(summ[key].keys())
        sessions = [dict(idx=i + 1, taught={it}, primary=it, phase="teach")
                    for i, it in enumerate(order)]
        runs.append(dict(name=name, workdir=wd, summary=summ, order=order,
                         sessions=sessions, family="demo"))
    return runs


# ---------------------------------------------------------------------------
# v_t and outcomes
# ---------------------------------------------------------------------------

HINGE = 0.50          # retention-horizon boundary (TEACH_PIPELINE.md)
KL_WIRE = 10.0        # tier-1 runaway-drift bound
COLLAPSE_WIRE = 0.5   # tier-1 top_response_fraction bound
UNIQ_WIRE = 0.25      # tier-1 unique_response_ratio bound


def first_teach_map(run):
    m = {}
    for s in run["sessions"]:
        if s["phase"] == "teach" and s["primary"]:
            m.setdefault(s["primary"], s["idx"])
    return m


def r_of(run, item, snap_idx):
    """r_i(k): recomputed uniformly from baseline + first-teach gain."""
    ft = first_teach_map(run)
    if item not in ft:
        return None
    base = run["summary"]["before"][item]["margin"]
    sk = snapshot_keys(run["summary"])
    m_teach = margins(run["summary"], sk(ft[item]), item)
    if snap_idx == 0:
        return 0.0
    m_k = margins(run["summary"], sk(snap_idx), item)
    if m_k is None or m_teach is None:
        return None
    g = m_teach - base
    if abs(g) < 1e-9:
        return None
    return (m_k - base) / g


def poison_pressure(stats):
    c = dict(
        reward=max(0.0, -stats["mean_reward"]),
        collapse=max(0.0, stats["top_response_fraction"] - COLLAPSE_WIRE) / COLLAPSE_WIRE,
        degeneracy=max(0.0, UNIQ_WIRE - stats["unique_response_ratio"]) / UNIQ_WIRE,
        drift=stats["final_kl"] / KL_WIRE,
    )
    return max(c.values()), c


def session_rows(run):
    """One row per session with v_t components and per-session outcomes."""
    summ = run["summary"]
    sk = snapshot_keys(summ)
    stats_by_sid = load_batch_stats(os.path.join(LAB, run["workdir"]))
    poison_sid, poison_stats = load_poison_stats(os.path.join(LAB, run["workdir"]))
    taught_before = set()
    rows = []
    prev_interf = None
    for s in run["sessions"]:
        idx = s["idx"]
        # margin of item j before session t = snapshot after t-1 (or baseline)
        def m_pre(j):
            return (summ["before"][j]["margin"] if idx == 1
                    else margins(summ, sk(idx - 1), j))

        def m_post(j):
            return margins(summ, sk(idx), j)

        priors = sorted(taught_before - s["taught"])
        c_vals = [r_of(run, j, idx - 1) for j in taught_before]
        c_vals = [v for v in c_vals if v is not None]
        c_frac = (sum(1 for v in c_vals if v >= HINGE) / len(c_vals)
                  if c_vals else None)
        c_mean = sum(c_vals) / len(c_vals) if c_vals else None
        # interference session t causes on already-taught priors (post-hoc
        # mechanism read) and interference inherited from session t-1
        # (a legally-available pre-session feature).
        diffs = [(m_pre(j) or 0.0) - (m_post(j) or 0.0) for j in priors
                 if m_pre(j) is not None and m_post(j) is not None]
        interf_caused = sum(diffs) / len(diffs) if diffs else None

        row = dict(run=run["name"], family=run["family"], idx=idx,
                   phase=s["phase"], target=s["primary"],
                   poison=0,
                   c_frac=c_frac, c_mean=c_mean,
                   i_prev=prev_interf,
                   interf_caused=interf_caused)
        if s["primary"]:
            pre, post = m_pre(s["primary"]), m_post(s["primary"])
            row.update(m_pre=pre, m_post=post,
                       novelty=(-pre if pre is not None else None),
                       lift=(post - pre
                             if None not in (pre, post) else None),
                       teach_success=(None if None in (pre, post)
                                      else int(post - pre > 0)))
        sid_file = sorted(glob.glob(os.path.join(
            LAB, run["workdir"], f"session_{idx:02d}_*.jsonl")))
        st = None
        if sid_file:
            base = os.path.basename(sid_file[0])[:-len(".jsonl")]
            sid = base.split("_", 2)[2]
            st = stats_by_sid.get(sid)
        if st:
            row.update(mean_reward=st["mean_reward"],
                       top_frac=st["top_response_fraction"],
                       uniq_ratio=st["unique_response_ratio"],
                       final_kl=st["final_kl"],
                       n_triples=st["n_triples"],
                       loss_drop=st["loss0"] - st["loss_final"])
            row["p_max"], row["p_parts"] = poison_pressure(st)
        rows.append(row)
        prev_interf = row["interf_caused"]
        taught_before |= s["taught"]
    if poison_stats:
        p_max, parts = poison_pressure(poison_stats)
        rows.append(dict(run=run["name"], family=run["family"], idx=99,
                         phase="poison", target=None, poison=1,
                         c_frac=None, c_mean=None, i_prev=None,
                         interf_caused=None,
                         mean_reward=poison_stats["mean_reward"],
                         top_frac=poison_stats["top_response_fraction"],
                         uniq_ratio=poison_stats["unique_response_ratio"],
                         final_kl=poison_stats["final_kl"],
                         n_triples=poison_stats["n_triples"],
                         p_max=p_max, p_parts=parts))
    return rows


def item_rows(run):
    """One row per library item: first-teach features + horizon retention."""
    out = []
    ft = first_teach_map(run)
    summ = run["summary"]
    horizon = max(int(k.split("after_s")[1]) for k in summ["snapshots"])
    n_sessions = len(run["sessions"])
    for item in run["order"]:
        if item not in ft:
            continue
        s_i = ft[item]
        base = summ["before"][item]["margin"]
        m_teach = margins(summ, snapshot_keys(summ)(s_i), item)
        r_fin = r_of(run, item, horizon)
        if m_teach is None or r_fin is None:
            continue
        g = m_teach - base
        served = sum(1 for s in run["sessions"]
                     if item in s["taught"] and s["idx"] > s_i)
        kind = summ.get("library", {}).get(item, {}).get("kind", "?")
        out.append(dict(run=run["name"], family=run["family"], item=item,
                        kind=kind, teach_session=s_i, m0=base, g=g,
                        age_at_horizon=horizon - s_i,
                        served_after=served, r_final=r_fin,
                        retained=int(r_fin >= HINGE)))
    return out


# ---------------------------------------------------------------------------
# Chat-lane sensor history (descriptive): SFT-r3 eval JSONs
# ---------------------------------------------------------------------------

def _words(text):
    return [w for w in "".join(c if c.isalnum() or c == " " else " "
                               for c in text.lower()).split() if w]


def chat_eval_table():
    rows = []
    for p in sorted(glob.glob(os.path.join(
            LAB, "imported", "vk_sft_r3", "_evals", "eval_chat_step*.json")),
            key=lambda p: int("".join(c for c in os.path.basename(p)
                                      if c.isdigit()))):
        d = _j(p)
        rubric_hits, rubric_total = 0, 0
        lens, uniqs = [], []
        prev_row = rows[-1] if rows else None
        dist_sum = dist_n = 0
        for s in d["samples"]:
            resp = s["response"]
            ws = _words(resp)
            lens.append(len(ws))
            uniqs.append(len(set(ws)) / len(ws) if ws else 0.0)
            # deterministic answer keys only (the two prompts with a single
            # verifiable token); everything else is left unscored.
            key = None
            if "capital of France" in s["prompt"]:
                key = "paris"
            elif "17 * 23" in s["prompt"]:
                key = "391"
            if key:
                rubric_total += 1
                rubric_hits += int(key in ws)
            if prev_row and s["prompt"] in prev_row["by_prompt"]:
                a, b = set(ws), prev_row["by_prompt"][s["prompt"]]
                union = len(a | b)
                dist_sum += 1.0 - (len(a & b) / union if union else 0.0)
                dist_n += 1
        rows.append(dict(
            step=d["step"], n=len(d["samples"]),
            rubric_pass=(rubric_hits / rubric_total if rubric_total else None),
            rubric_n=rubric_total,
            mean_words=sum(lens) / len(lens),
            mean_uniq=sum(uniqs) / len(uniqs),
            drift_vs_prev=(dist_sum / dist_n if dist_n else None),
            by_prompt={s["prompt"]: set(_words(s["response"]))
                       for s in d["samples"]}))
    for r in rows:
        r.pop("by_prompt")
    return rows


# ---------------------------------------------------------------------------
# Stats: Mann-Whitney AUC + bootstrap CI, median-split lift
# ---------------------------------------------------------------------------

def auc(scores, labels):
    pairs = [(s, y) for s, y in zip(scores, labels)
             if s is not None and y is not None]
    pos = [s for s, y in pairs if y == 1]
    neg = [s for s, y in pairs if y == 0]
    if not pos or not neg:
        return None, len(pos), len(neg)
    wins = 0.0
    for a in pos:
        for b in neg:
            wins += 1.0 if a > b else 0.5 if a == b else 0.0
    return wins / (len(pos) * len(neg)), len(pos), len(neg)


def auc_ci(scores, labels, n_boot=2000, seed=13):
    pairs = [(s, y) for s, y in zip(scores, labels)
             if s is not None and y is not None]
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        samp = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        a, _, _ = auc([s for s, _ in samp], [y for _, y in samp])
        if a is not None:
            vals.append(a)
    if len(vals) < 100:
        return (None, None)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def median_split(scores, labels):
    pairs = sorted((s, y) for s, y in zip(scores, labels)
                   if s is not None and y is not None)
    if len(pairs) < 6:
        return None
    mid = len(pairs) // 2
    lo, hi = pairs[:mid], pairs[mid:]
    r_lo = sum(y for _, y in lo) / len(lo)
    r_hi = sum(y for _, y in hi) / len(hi)
    return r_lo, r_hi, len(lo), len(hi)


def feat_table(name, rows, feats, label_key):
    labels = [r[label_key] for r in rows]
    out = []
    for f in feats:
        scores = [r.get(f) for r in rows]
        a, np_, nn_ = auc(scores, labels)
        if a is None:
            continue
        lo, hi = auc_ci(scores, labels)
        out.append(dict(test=name, feature=f, auc=a, ci_lo=lo, ci_hi=hi,
                        n_pos=np_, n_neg=nn_, split=median_split(scores, labels)))
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def fmt(x, nd=3):
    return "   -  " if x is None else f"{x: .3f}"


def print_tables(session_rows_all, item_rows_all, chat_rows):
    # De-duplicate the bit-identical sft6 teach phases (keep cadence's copy;
    # later sessions genuinely differ and stay).
    seen = set()
    a_rows = []
    for r in session_rows_all:
        key = None
        if r["family"] == "sft6" and r["phase"] == "teach":
            key = ("sft6-teach", r["idx"])
            if key in seen:
                continue
            seen.add(key)
        a_rows.append(r)

    teach = [r for r in a_rows if r.get("teach_success") is not None]
    poison = [r for r in a_rows if "p_max" in r]
    firsts = [r for r in teach
              if not (r["family"] == "sft6" and r["phase"] == "later")]

    print("=" * 78)
    print("DATASET A — teach success (target-item margin lift > 0 in-session)")
    print(f"rows: {len(teach)} sessions with a primary target "
          f"({len(firsts)} first teaches; sft6 teach phases de-duplicated)")
    feats = ["m_pre", "novelty", "c_frac", "c_mean", "i_prev",
             "p_max", "final_kl", "mean_reward"]
    for t in feat_table("A.lift>0", teach, feats, "teach_success"):
        sp = t["split"]
        spl = (f"median-split rate {sp[0]:.2f}|{sp[1]:.2f}"
               if sp else "median-split n/a")
        print(f"  {t['feature']:<12} AUC {fmt(t['auc'])} "
              f"CI95 [{fmt(t['ci_lo'])},{fmt(t['ci_hi'])}] "
              f"(+{t['n_pos']}/-{t['n_neg']})  {spl}")

    print("=" * 78)
    print("DATASET B — poison session detection (tripwire sensors only; the")
    print("snapshot margin cadence never sees the poison session at all)")
    print(f"rows: {len(poison)} sessions with batch stats, "
          f"{sum(r['poison'] for r in poison)} poison")
    for t in feat_table("B.poison", poison,
                        ["p_max", "mean_reward", "top_frac", "uniq_ratio",
                         "final_kl"], "poison"):
        print(f"  {t['feature']:<12} AUC {fmt(t['auc'])} "
              f"CI95 [{fmt(t['ci_lo'])},{fmt(t['ci_hi'])}] "
              f"(+{t['n_pos']}/-{t['n_neg']})")
    ppois = [r for r in poison if r["poison"]]
    pnorm = [r for r in poison if not r["poison"]]
    print(f"  p_max range poison {min(r['p_max'] for r in ppois):.2f}.."
          f"{max(r['p_max'] for r in ppois):.2f} vs normal "
          f"{min(r['p_max'] for r in pnorm):.2f}.."
          f"{max(r['p_max'] for r in pnorm):.2f}")

    print("=" * 78)
    print("DATASET C — horizon retention r_final >= 0.50 (per library item;")
    print("features legal at first-teach time; age_at_horizon = confound)")
    print(f"rows: {len(item_rows_all)} items, "
          f"{sum(r['retained'] for r in item_rows_all)} retained")
    for t in feat_table("C.r_final>=0.5", item_rows_all,
                        ["m0", "g", "age_at_horizon"], "retained"):
        sp = t["split"]
        spl = (f"median-split rate {sp[0]:.2f}|{sp[1]:.2f}"
               if sp else "median-split n/a")
        print(f"  {t['feature']:<14} AUC {fmt(t['auc'])} "
              f"CI95 [{fmt(t['ci_lo'])},{fmt(t['ci_hi'])}] "
              f"(+{t['n_pos']}/-{t['n_neg']})  {spl}")
    # interference suffered after teach is POST-HOC (not a legal feature);
    # report it separately as the mechanism check.
    post = []
    for ir in item_rows_all:
        rows_r = [r for r in session_rows_all
                  if r["run"] == ir["run"] and r["idx"] > ir["teach_session"]
                  and ir["item"] not in {r["target"]}
                  and r.get("interf_caused") is not None]
        if rows_r:
            post.append(dict(suffered=sum(r["interf_caused"]
                                          for r in rows_r) / len(rows_r),
                             retained=ir["retained"]))
    if post:
        a, npos, nneg = auc([p["suffered"] for p in post],
                            [p["retained"] for p in post])
        print(f"  [post-hoc, NOT predictive] mean interference suffered "
              f"after teach: AUC {fmt(a)} (+{npos}/-{nneg})")

    print("=" * 78)
    print("CHAT LANE — SFT-r3 eval sensor history (descriptive; no outcome)")
    print(f"  {'step':>5} {'rubric':>7} {'words':>7} {'uniq':>6} {'drift':>7}")
    for r in chat_rows:
        print(f"  {r['step']:>5} {fmt(r['rubric_pass'], 2):>7} "
              f"{r['mean_words']:>7.1f} {r['mean_uniq']:>6.3f} "
              f"{fmt(r['drift_vs_prev']):>7}")


def dump_csvs(outdir, session_rows_all, item_rows_all, chat_rows):
    os.makedirs(outdir, exist_ok=True)
    def w(name, rows, fields):
        with open(os.path.join(outdir, name), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fields,
                                extrasaction="ignore")
            wr.writeheader()
            for r in rows:
                wr.writerow(r)
    w("e1_sessions.csv", session_rows_all,
      ["run", "family", "idx", "phase", "target", "poison", "c_frac",
       "c_mean", "i_prev", "interf_caused", "m_pre", "m_post", "novelty",
       "lift", "teach_success", "mean_reward", "top_frac", "uniq_ratio",
       "final_kl", "n_triples", "p_max"])
    w("e1_items.csv", item_rows_all,
      ["run", "family", "item", "kind", "teach_session", "m0", "g",
       "age_at_horizon", "served_after", "r_final", "retained"])
    w("e1_chat_evals.csv", chat_rows,
      ["step", "n", "rubric_pass", "rubric_n", "mean_words", "mean_uniq",
       "drift_vs_prev"])
    print(f"CSVs written under {outdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="optional CSV output dir")
    args = ap.parse_args()
    runs = build_runs()
    s_rows, i_rows = [], []
    for run in runs:
        s_rows.extend(session_rows(run))
        i_rows.extend(item_rows(run))
    chat_rows = chat_eval_table()
    print_tables(s_rows, i_rows, chat_rows)
    if args.csv:
        dump_csvs(args.csv, s_rows, i_rows, chat_rows)


if __name__ == "__main__":
    main()
