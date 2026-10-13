#!/usr/bin/env python3
"""varact_gate — variable-activation plug gating (variable-k rung 1).

Serving-side adaptive expert activation for the v0.5/v0.5.1 additive bank.
No retraining: an inference-time policy layer that narrows door6's additive
delivery

    y = y_stock + sum_j lam_j * m_j * e_j          m_j = row score > k-th stock

to

    y = y_stock + sum_j lam_j * m_j * g_j * e_j    g_j = varact policy mask

where g_j(token) is decided per token per plug from the plug row's home
probability p_home = softmax(bank scores)[plug row] (MixGateD2.last_p_new).
Plugs only ever get NARROWED (g implies delivery would have happened under
membership), so tau=0 / top-m=n / policy-off is bit-identical to stock
delivery.  The hypothesis this tests:

  (a) compute/memory: easy/general tokens skip plug experts entirely
      (expert forward runs only on the delivering token subset), and
  (b) interference: flooding plugs (bank2n phi_math 0.42 code / 0.89 math,
      mem-words sibling crowding) stop firing where p_home is low.

Policies (knobs tau / m):
  thresh   g_j = p_home_j >= tau
  topm     keep the top-m plugs by p_home among membership-firing plugs
  adaptive keep p_home >= tau, capped at top-m per token (m adapts: 0..m)

Protocol: 2x2 {no gate, gated} x {compute (tok/s, active plugs/token),
quality (home CE per plug, cross-fire matrix, base tax)} sweeping tau in
{0.3,0.5,0.7} and top-m in {1,2,3}, plus pairwise leave-one-out harm (bank2n
bar 0.02) and mem-sibling crowding.  The deliverable is the quality-vs-
compute frontier; the ship bar is a point that keeps every green plug green
(home CE <= the ungated default) while cutting active-plug compute >= 30%
and pairwise harm <= 0.02.

Ops: runs on the MI50 box against /data/tliao/v05_out (joint_state.pt is
READ-ONLY here — the user chats with that assembly; this harness never
writes it).  Incremental JSON, numeric-only stdout.

Usage (box):
  python3 lab/varact_gate.py --phase sweep --out /data/tliao/varact_out/varact.json
  python3 lab/varact_gate.py --phase harm --configs base,adaptive_tau05_m2
  python3 lab/varact_gate.py --phase smoke --configs base,adaptive_tau05_m2
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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "lab"))

import graft_d0_parity as d0  # noqa: E402
import graft_d1_plugin as d1  # noqa: E402
import graft_d2_crossfamily as d2  # noqa: E402
import grafted_spine_v0 as g0  # noqa: E402
import v05_build as b5  # noqa: E402
# chat_v05 imports varact_gate (the shipped default policy), so chat_v05 is
# imported LAZILY at use sites to keep the cycle one-directional.

ORDER = ["code", "math", "mem_pet", "mem_edit", "mem_words",
         "k3_hack", "glm_cybersec", "mimo_writing"]
HOME_OF = {"code": "code", "math": "math", "mem_pet": "pet",
           "mem_edit": "editor", "mem_words": "words",
           "k3_hack": "hacking", "glm_cybersec": "cybersec",
           "mimo_writing": "writing"}
GREEN = ["code", "math", "mem_pet", "mem_edit", "mem_words", "k3_hack"]

BAR_HARM = 0.02          # bank2n pairwise bar (nats)
BAR_COMPUTE_CUT = 0.30   # active-plug compute cut vs ungated

# memory plugs whose retrieval scope the question-match gate governs
MEM_TAGS = ("mem_pet", "mem_edit", "mem_words")


# ---------------------------------------------------------------------------
# question-anchored retrieval scope (MODULAR_MOE 10.10 rung 1 / 8.5)
# ---------------------------------------------------------------------------

# Query-side episode text per memory row — the taught questions of the
# episode (g0.build_episodes pet_kepler / editor_helix / math_words).
# The question-match gate asks "does the current question address this
# row's episode?" via distinctive-token containment; no model call.
EPISODE_QUERIES = {
    "mem_pet": (
        "What is my cat's name?", "What is the name of my cat?",
        "Remind me — what do we call my cat?", "What kind of pet do I have?",
        "Tell me about my pet.", "Does my pet have a name?",
    ),
    "mem_edit": (
        "What is my favorite text editor?", "Which editor do I use?",
        "What editor should you suggest to me?", "Do I like modal editing?",
        "Remind me about my editor setup.",
        "What should I open this file in?",
    ),
    "mem_words": (
        "What is 2 + 2?", "What is 5 + 3?", "What is 10 minus 7?",
        "What is 1 + 1?", "What is 4 + 4?", "What is 9 minus 6?",
    ),
}

STOPWORDS = frozenset(
    "a an and are as at be can do does for from has have how i in is it me "
    "my of on or our so that the this to was we what when which who why "
    "will with would you your any about".split())


def _wtoks(s: str) -> list:
    """Distinctive question tokens: lowercase words + digits, plural stem."""
    out = []
    for w in str(s).lower().replace("—", " ").split():
        w = "".join(ch for ch in w if ch.isalnum() or ch in "+-")
        if not w or w in STOPWORDS:
            continue
        if len(w) < 2 and not w.isdigit():
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


def episode_scores(question: str, queries: dict | None = None) -> dict:
    """Containment of the question's distinctive tokens in each row's
    episode queries: |q_tokens ∩ row_tokens| / |q_tokens|, in [0, 1]."""
    qs = _wtoks(question)
    if not qs:
        return {tag: 0.0 for tag in (queries or EPISODE_QUERIES)}
    out = {}
    for tag, texts in (queries or EPISODE_QUERIES).items():
        ep = set()
        for t in texts:
            ep.update(_wtoks(t))
        hit = sum(1 for w in qs if w in ep)
        out[tag] = hit / len(qs)
    return out


def find_question_span(prompt_ids, question: str, tok=None) -> tuple:
    """(start, end) token span of the current question inside the prompt.

    Exact tail search on the question's own token ids (the chat template
    embeds content verbatim); on a miss, fall back to the prompt tail of
    question-length + template overhead.  end = len(prompt_ids) so role
    tags after the content are included.
    """
    end = len(prompt_ids)
    q_ids = list(tok.encode(question, add_special_tokens=False)) \
        if tok is not None else []
    if q_ids:
        n = len(q_ids)
        for s in range(end - n, max(-1, end - n - 96), -1):
            if s >= 0 and list(prompt_ids[s:s + n]) == q_ids:
                return (s, end)
    n = min(end, (len(q_ids) if q_ids else 16) + 16)
    return (end - n, end)


class QuestionScope:
    """Row-selective retrieval scope for the taught-token bonus / row boost.

    A memory row is IN SCOPE for the current reply only when the question
    addresses its episode content (question-anchored activation, 10.10
    rung 1) — never merely because answer-shaped tokens make the row hot
    mid-forward (the live-path garble: sibling mem rows on unrelated
    turns list-ify the reply).

    modes:
      overlap  episode token-containment score >= tau (CPU, no forward)
      qspan    row p_home over the question-token span >= tau
      both     overlap AND qspan        any   overlap OR qspan
    margin>0: at each decode position the bonus lands only on row j where
    p_home_j >= best sibling p_home + margin (the 10.9 row-selective
    margin family) — at most one row wins a position.
    """

    def __init__(self, mode: str = "overlap", tau: float = 0.5,
                 margin: float = 0.0):
        if mode not in ("overlap", "qspan", "both", "any"):
            raise ValueError(f"bad qscope mode {mode!r}")
        self.mode = mode
        self.tau = float(tau)
        self.margin = float(margin)
        self.span = None
        self.overlap = {}
        self.qspan_p = {}
        self.question = ""
        self.last_match = ()

    def name(self) -> str:
        return f"qscope_{self.mode}_tau{self.tau:g}".replace(".", "")

    def begin(self, question: str, tok=None, prompt_ids=None) -> None:
        """Latch question-side evidence for the upcoming reply."""
        self.question = question or ""
        self.overlap = episode_scores(self.question)
        self.qspan_p = {}
        self.span = find_question_span(prompt_ids, self.question, tok) \
            if prompt_ids is not None and self.question else None
        self.last_match = self.matched()

    def observe(self, cap: dict) -> None:
        """Fold a capture's question-span p_home into the latched evidence."""
        for tag in MEM_TAGS:
            c = cap.get(tag) or {}
            pq = c.get("p_home_q")
            if pq is None:
                continue
            self.qspan_p[tag] = max(self.qspan_p.get(tag, 0.0), float(pq))

    def scores(self, cap: dict | None = None) -> dict:
        """Evidence score per row in [0, 1] under the current mode."""
        if cap is not None:
            self.observe(cap)
        out = {}
        for tag in MEM_TAGS:
            ov = float(self.overlap.get(tag, 0.0))
            qp = float(self.qspan_p.get(tag, 0.0))
            if self.mode == "overlap":
                sc = ov
            elif self.mode == "qspan":
                sc = qp
            elif self.mode == "both":
                sc = min(ov, qp)
            else:
                sc = max(ov, qp)
            out[tag] = sc
        return out

    def matched(self, cap: dict | None = None) -> frozenset:
        sc = self.scores(cap)
        live = frozenset(t for t, v in sc.items() if v >= self.tau)
        self.last_match = tuple(sorted(live))
        return live

    def winner(self, cap: dict | None = None) -> str | None:
        """Row-selective margin at a position: p_j >= best sibling + margin."""
        if self.margin <= 0.0 or not cap:
            return None
        p = {}
        for tag in MEM_TAGS:
            c = cap.get(tag) or {}
            if c.get("p_home") is None:
                return None
            p[tag] = float(c["p_home"])
        for t in MEM_TAGS:
            sib = max((p[o] for o in MEM_TAGS if o != t), default=-1e9)
            if p[t] >= sib + self.margin:
                return t
        return None


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------

class VarActPolicy:
    """Per-token per-plug activation from p_home + membership.

    mode:
      off      no gating (g = 1 where membership fires)
      thresh   g_j = (p_home_j >= tau)
      topm     keep top-m by p_home among membership-firing plugs
      adaptive keep p_home >= tau AND rank <= m by p_home (m adapts per token)
    """

    def __init__(self, mode: str = "off", tau: float = 0.0, top_m: int = 0):
        if mode not in ("off", "thresh", "topm", "adaptive"):
            raise ValueError(f"bad mode {mode!r}")
        self.mode = mode
        self.tau = float(tau)
        self.top_m = int(top_m)

    def name(self) -> str:
        if self.mode == "off":
            return "base"
        if self.mode == "thresh":
            return f"tau{self.tau:g}".replace(".", "")
        if self.mode == "topm":
            return f"topm{self.top_m}"
        return f"adaptive_tau{self.tau:g}_m{self.top_m}".replace(".", "")

    def select(self, p_new, memb, tags=None):
        """-> bool (T, n_new) LOCAL keep mask, or None (= ungated).

        top-m is a BANK-WIDE per-token budget (plugs live at different host
        layers) and is enforced by BankBudget across wrappers, not here.
        """
        if self.mode == "off" or p_new is None or memb is None:
            return None
        fire = memb.bool()                       # plugs that would deliver
        if self.mode == "topm":
            return fire                          # budget caps the count
        p = p_new.float()
        if self.mode == "thresh":
            keep = fire & (p >= self.tau)
            return self._boost(keep, fire, p, tags)
        # adaptive: local p filter; the cross-layer budget cap makes the
        # per-token plug count adaptive (0..m)
        keep = fire & (p >= self.tau)
        return self._boost(keep, fire, p, tags)

    def _boost(self, keep, fire, p, tags):
        return keep


class RetrievalBoostPolicy(VarActPolicy):
    """tau-threshold + retrieval-triggered row boost for memory plugs.

    The named v0.5-lever companion to the taught-token bonus: in long
    contexts the memory row's p_home at the answer-start position collapses
    (router drift), so the row would be gated off exactly where the fact
    token is decided.  A boost-tag row that is hot ANYWHERE in the current
    forward (the question re-activates it) delivers at every membership
    position of that forward.  Non-memory plugs keep plain tau gating.

    With a QuestionScope the boost is row-selective (10.10 rung 1): only
    rows the CURRENT QUESTION addresses stay in scope — hot siblings on
    unrelated turns are not boosted (the live-path list-ify failure).
    """

    def __init__(self, tau: float = 0.5, boost_tags=None,
                 scope: QuestionScope | None = None):
        super().__init__("thresh", tau=tau)
        self.boost_tags = set(boost_tags or MEM_TAGS)
        self.scope = scope

    def name(self):
        base = f"retrieval_boost_tau{self.tau:g}".replace(".", "")
        return base if self.scope is None else f"{base}_{self.scope.name()}"

    def _boost(self, keep, fire, p, tags):
        if not tags:
            return keep
        if self.scope is not None:
            sc = self.scope
            if sc.span is not None and p is not None and \
                    p.shape[0] >= sc.span[1]:
                s, e = sc.span
                for j, tag in enumerate(tags):
                    sc.qspan_p[tag] = max(sc.qspan_p.get(tag, 0.0),
                                          float(p[s:e, j].max()))
            scoped = set(sc.matched())
            for j, tag in enumerate(tags):
                if tag not in self.boost_tags:
                    continue
                if tag in scoped:
                    keep[:, j] = fire[:, j]      # question-matched: deliver
                # unmatched mem rows fall back to plain tau gating
            return keep
        hot = p.max(dim=0).values >= self.tau          # (n_new,)
        for j, tag in enumerate(tags):
            if tag in self.boost_tags and bool(hot[j]):
                keep[:, j] = fire[:, j]
        return keep


class BankBudget:
    """Causal bank-wide per-token plug budget (variable-k rung 1).

    "top-m plugs for this token" spans host layers (v0.5 bank: 8 plugs at 8
    distinct layers).  Later layers' p_home is not yet computed when the
    early layers decide, so the only causal policy is a counter: a token may
    pull at most m plugs bank-wide.  Layers fire in host order every forward;
    wrap-around (a layer seen twice) starts a new token batch.  GPU int ops
    only — no data-dependent shapes (gfx906-safe).
    """

    def __init__(self):
        self.seen = set()
        self.fired = None      # (T,) int32, plugs delivered per token

    def reset(self):
        self.seen.clear()
        self.fired = None

    def touch(self, tag):
        """Register one host-layer visit; wrap-around starts a new batch."""
        if tag in self.seen:
            self.reset()
        self.seen.add(tag)

    def take(self, k, m):
        """k (T,) bool delivering -> capped to a per-token bank budget m."""
        if m <= 0:
            return k
        T = k.shape[0]
        if self.fired is None or self.fired.shape[0] != T:
            self.fired = torch.zeros(T, dtype=torch.int32, device=k.device)
        k = k & (self.fired < int(m))
        self.fired = self.fired + k.to(torch.int32)
        return k


POLICIES = {
    "base": lambda: VarActPolicy("off"),
    "tau0.3": lambda: VarActPolicy("thresh", tau=0.3),
    "tau0.5": lambda: VarActPolicy("thresh", tau=0.5),
    "tau0.7": lambda: VarActPolicy("thresh", tau=0.7),
    "topm1": lambda: VarActPolicy("topm", top_m=1),
    "topm2": lambda: VarActPolicy("topm", top_m=2),
    "topm3": lambda: VarActPolicy("topm", top_m=3),
    "adaptive_tau0.5_m2": lambda: VarActPolicy("adaptive", tau=0.5, top_m=2),
    "adaptive_tau0.3_m3": lambda: VarActPolicy("adaptive", tau=0.3, top_m=3),
    "adaptive_tau0.5_m3": lambda: VarActPolicy("adaptive", tau=0.5, top_m=3),
    "adaptive_tau0.7_m2": lambda: VarActPolicy("adaptive", tau=0.7, top_m=2),
    # retrieval-triggered row boost (defined below; lambda defers the name)
    "boost": lambda: RetrievalBoostPolicy(tau=0.5),
}


# ---------------------------------------------------------------------------
# gated MixMerge wrapper (zero edits to door6 / g0 — wraps in place)
# ---------------------------------------------------------------------------

class VarActMerge:
    """Wraps a door6.MixMerge: y = stock + sum_j lam_j * (m_j & g_j) * e_j.

    Gated additive delivery: y = stock + sum_j lam_j * (m_j & g_j) * e_j.
    A plug whose keep-set is empty for the batch is skipped entirely (zero
    expert FLOPs — the serving win at T=1 decode); a plug with any keep runs
    the stock expert shape and masks its output (see gfx906 note above).
    Policy-off path delegates to the base merge unchanged (bit-identical).
    """

    def __init__(self, base, policy: VarActPolicy, budget: BankBudget | None
                 = None):
        self.base = base
        self.policy = policy
        self.budget = budget
        self.bkey = tuple(base.tags)
        self.reset_stats()

    # set_plugin_mode / set_mix_active poke these on blk._plugin_merge
    @property
    def mode(self):
        return self.base.mode

    @mode.setter
    def mode(self, v):
        self.base.mode = v

    @property
    def force_gain(self):
        return self.base.force_gain

    @force_gain.setter
    def force_gain(self, v):
        self.base.force_gain = v

    @property
    def active(self):
        return self.base.active

    @active.setter
    def active(self, v):
        self.base.active = v

    @property
    def tags(self):
        return self.base.tags

    @property
    def lams(self):
        return self.base.lams

    @property
    def experts(self):
        return self.base.experts

    @property
    def stock(self):
        return self.base.stock

    def reset_stats(self):
        n = len(self.base.experts)
        self.stat_tokens = 0
        self.stat_deliver = [0] * n        # tokens where plug contributes
        self.stat_compute = [0] * n        # tokens where expert fwd runs
        self.stat_calls = [0] * n          # expert forward invocations

    def stats(self):
        t = max(1, self.stat_tokens)
        return {
            "tokens": self.stat_tokens,
            "deliver_per_plug": [round(v / t, 4) for v in self.stat_deliver],
            "compute_per_plug": [round(v / t, 4) for v in self.stat_compute],
            "deliver_per_token": round(sum(self.stat_deliver) / t, 4),
            "compute_per_token": round(sum(self.stat_compute) / t, 4),
            "fwd_calls": list(self.stat_calls),
        }

    # gfx906 executes ONLY the stock MixMerge expert shape reliably: any
    # data-dependent-shape GPU op (`nonzero`) or novel GEMM row-count trips
    # hipErrorInvalidDeviceFunction (FAILURES door25-gfx906-backward-holes;
    # measured live here: `k.nonzero()` itself raised).  Variable activation
    # therefore works at BATCH granularity offline and TOKEN granularity at
    # serving (decode runs T=1): a plug whose keep-set is empty for the whole
    # batch is skipped entirely (zero FLOPs); a plug with any keep runs the
    # EXACT stock shape e(hidden_states) and masks its output.  stat_deliver
    # still counts per-token deliveries (the serving-side compute measure).

    def __call__(self, hidden_states, top_k_index, top_k_weights):
        base = self.base
        if base.mode == "off":
            return base.stock(hidden_states, top_k_index, top_k_weights)
        g = base.gate_getter()
        memb = getattr(g, "last_membership", None)
        p_new = getattr(g, "last_p_new", None)
        hs = hidden_states
        T = hs.reshape(-1, hs.shape[-1]).shape[0]
        keep = self.policy.select(p_new, memb, tags=list(base.tags))
        if keep is None:
            # policy off: same math as stock delivery (bit-identical), but
            # still account compute (base merge runs every expert on every
            # token) and semantic membership fires for the frontier table
            self.stat_tokens += T
            for j in range(len(base.experts)):
                if base.active[j]:
                    self.stat_compute[j] += T
                    self.stat_calls[j] += 1
                    self.stat_deliver[j] += T if memb is None \
                        else int(memb[:, j].sum())
            return base(hidden_states, top_k_index, top_k_weights)

        with torch.no_grad():
            y = base.stock(hidden_states, top_k_index, top_k_weights)
        budgeted = self.budget is not None and self.policy.mode in (
            "topm", "adaptive") and self.policy.top_m > 0
        if budgeted:
            self.budget.touch(self.bkey)   # per-forward batch boundary
        ne_last = None
        for j, e in enumerate(base.experts):
            base.last_new_outs[j] = None
            if not base.active[j]:
                continue
            if memb is None:
                ne = e(hs)
                ne_last = ne
                self.stat_compute[j] += T
                self.stat_calls[j] += 1
                self.stat_deliver[j] += T
                y = y + ne
                continue
            m = memb[:, j]
            k = keep[:, j] & m
            if budgeted:
                k = self.budget.take(k, self.policy.top_m)
            nk = int(k.sum())
            if nk == 0:
                continue                      # whole plug skipped: zero FLOPs
            self.stat_calls[j] += 1
            self.stat_deliver[j] += nk
            # stock shape only (gfx906 hole-safe): full-batch expert forward
            # with the non-delivering rows masked out of the residual
            ne = e(hs)
            self.stat_compute[j] += T
            ne_last = ne
            if base.mode == "forced":
                y = y + base.force_gain * ne
            else:
                lam = base.lams[j].clamp(min=0.0).to(device=ne.device,
                                                      dtype=ne.dtype)
                y = y + (lam * k.to(ne.dtype)).unsqueeze(-1) * ne
            base.last_new_outs[j] = ne
        base.last_new_out = ne_last
        self.stat_tokens += T
        return y.to(hs.dtype)


def install_varact(handles, policy: VarActPolicy) -> list:
    """Wrap every installed MixMerge in place. Returns wrappers.

    All wrappers share one BankBudget — top-m is bank-wide per token even
    though the plugs sit at different host layers.
    """
    wrappers = []
    budget = BankBudget()
    for blk, _sf in handles:
        pm = getattr(blk, "_plugin_merge", None)
        if pm is None:
            continue
        if isinstance(pm, VarActMerge):
            pm.policy = policy
            pm.budget = budget
            pm.reset_stats()
            wrappers.append(pm)
            continue
        vm = VarActMerge(pm, policy, budget)
        blk._plugin_merge = vm

        def make_fwd(vm=vm):
            def fwd(hidden_states, top_k_index, top_k_weights):
                return vm(hidden_states, top_k_index, top_k_weights)
            return fwd

        blk.experts.forward = make_fwd()
        wrappers.append(vm)
    return wrappers


def set_policy(wrappers, policy: VarActPolicy) -> None:
    for vm in wrappers:
        vm.policy = policy
        vm.reset_stats()
        if vm.budget is not None:
            vm.budget.reset()


# ---------------------------------------------------------------------------
# measurement helpers
# ---------------------------------------------------------------------------

def dump(out: dict, path: str) -> None:
    g0.dump_json(out, path)


def pick(stream, a, b):
    return [stream[i % len(stream)] for i in range(a, b)]


def home_stream(tok, home, args, seq):
    if home in ("code", "math"):
        domains = d0.load_domain_texts(args.domains_dir)
        tr, _, _ = g0.split_home(tok, domains[home], args.max_tokens, seq, 4)
        return [(t, b5.HOME_LABEL[home]) for t, _ in tr]
    if home in ("pet", "editor", "words"):
        eps = g0.build_episodes()
        bl = g0.episode_batches(tok, [eps[home]], seq, 4)
        return [(t, b5.HOME_LABEL[home]) for t, _ in bl]
    try:
        import mem_pack as mp
        if home in mp.homes():
            eps = mp.episodes()
            bl = g0.episode_batches(tok, [eps[home]], seq, 4)
            return [(t, b5.HOME_LABEL[home]) for t, _ in bl]
    except Exception:  # noqa: BLE001  pack unavailable -> specialty path
        pass
    ids = b5.load_specialty_ids(
        tok, os.path.join(args.specialties_dir, home, "corpus.txt"),
        args.max_tokens)
    return [(t, b5.HOME_LABEL[home]) for t, _ in g0.to_batches(ids, 1, seq, 4)]


@torch.no_grad()
def eval_pass(model, blocks, fam, gates, handles, batches, device, wrappers):
    """CE over batches + delivery/compute stats for the installed policy."""
    tot, n = 0.0, 0
    for vm in wrappers:
        vm.reset_stats()
    t0 = time.time()

    def one():
        nonlocal tot, n
        for tokens, _ in batches:
            tokens = tokens.to(device)
            lg = d0.model_logits(model, tokens)
            tot += d0.next_token_ce(lg, tokens) * tokens.shape[0]
            n += tokens.shape[0]

    g0.run_in_state(model, blocks, fam, gates, handles, "both", one)
    dt = time.time() - t0
    st = {}
    for vm in wrappers:
        s = vm.stats()
        for tag, d, c in zip(vm.tags, s["deliver_per_plug"],
                             s["compute_per_plug"]):
            st.setdefault(tag, {"deliver": 0.0, "compute": 0.0, "tokens": 0})
            st[tag]["deliver"] += d * s["tokens"]
            st[tag]["compute"] += c * s["tokens"]
            st[tag]["tokens"] += s["tokens"]
    plug_stats = {}
    for tag, v in st.items():
        t = max(1, v["tokens"])
        plug_stats[tag] = {"deliver_frac": round(v["deliver"] / t, 4),
                           "compute_frac": round(v["compute"] / t, 4)}
    tokens_seen = max(1, n * batches[0][0].shape[1] if batches else 1)
    return {
        "ce": round(tot / max(1, n), 4),
        "plug_stats": plug_stats,
        "deliver_per_token": round(
            sum(v["deliver_frac"] for v in plug_stats.values()), 4),
        "compute_per_token": round(
            sum(v["compute_frac"] for v in plug_stats.values()), 4),
        "sec": round(dt, 2),
        "tok_per_s": round(tokens_seen / max(1e-6, dt), 1),
    }


@torch.no_grad()
def memb_matrix(model, blocks, fam, gates, handles, home_batches, device,
                site_entries, wrappers):
    """plug x home DELIVERY share (membership & policy), one pass per home."""
    share = {}
    for home, batches in home_batches.items():
        acc, counts = {}, 0
        for vm in wrappers:
            vm.reset_stats()
        for tokens, _ in batches:
            tokens = tokens.to(device)
            g0.run_in_state(model, blocks, fam, gates, handles, "both",
                            lambda t=tokens: d0.model_logits(model, t))
            counts += tokens.numel()
        for vm in wrappers:
            s = vm.stats()
            for tag, d in zip(vm.tags, s["deliver_per_plug"]):
                acc[tag] = acc.get(tag, 0.0) + d * s["tokens"]
        share[home] = {k: round(v / max(1, counts), 4)
                       for k, v in sorted(acc.items())}
    return share


@torch.no_grad()
def ce_with_active(model, blocks, fam, gates, handles, batches, device,
                   active_tags):
    """CE with only active_tags plugs delivering (forced via pm.active)."""
    def one():
        # run_in_state(which="both") resets pm.active to all-True — narrow it
        # AFTER that, so the leave-one-out state actually holds
        saved = []
        for blk, _ in handles:
            pm = getattr(blk, "_plugin_merge", None)
            if pm is None:
                continue
            saved.append((pm, list(pm.active)))
            pm.active = [(t in active_tags) for t in pm.tags]
        try:
            tot, n = 0.0, 0
            for tokens, _ in batches:
                tokens = tokens.to(device)
                lg = d0.model_logits(model, tokens)
                tot += d0.next_token_ce(lg, tokens) * tokens.shape[0]
                n += tokens.shape[0]
            return tot / max(1, n)
        finally:
            for pm, act in saved:
                pm.active = act

    return g0.run_in_state(model, blocks, fam, gates, handles, "both", one)


@torch.no_grad()
def timed_gen(tok, logits_fn, device, max_new=24):
    """Wall-clock decode tok/s on a fixed short prompt (full-forward path)."""
    messages = [{"role": "user", "content": "Say hello in one short sentence."}]
    t0 = time.time()
    _text, ids = g0.gen_reply(tok, logits_fn, messages, max_new, device)
    dt = time.time() - t0
    return round(len(ids) / max(1e-6, dt), 3), len(ids)


# ---------------------------------------------------------------------------
# taught-token decode bonus (the named v0.1 fact-token lever)
# ---------------------------------------------------------------------------

# the episodes' corrected fact tokens (memory_shaping substitution shape);
# both space-prefixed and bare forms so the decision token is covered
TAUGHT_STRS = {
    "mem_pet": ("Kepler", " Kepler"),
    "mem_edit": ("helix", " helix", "Helix"),
    "mem_words": ("Four", " Four"),
}


def capture_tags(gates, tag_col, qspan=None):
    """Per-plug p_home / membership at the LAST position of the last forward,
    plus p_home_max over ALL positions of that forward (retrieval signal:
    a memory row hot anywhere in the question/context stays in scope) and
    optional p_home_q = max over the QUESTION-token span only (the
    question-anchored retrieval signal, 10.10 rung 1).

    tag_col: {tag: (host_layer, col)} — the MixGateD2 column of each plug row.
    """
    cap = {}
    for tag, (host, j) in tag_col.items():
        g = gates[host]
        p = getattr(g, "last_p_new", None)
        m = getattr(g, "last_membership", None)
        pq = None
        if p is not None and qspan is not None and p.shape[0] >= qspan[1]:
            pq = float(p[qspan[0]:qspan[1], j].max())
        cap[tag] = {
            "p_home": float(p[-1, j]) if p is not None else None,
            "p_home_max": float(p[:, j].max()) if p is not None else None,
            "p_home_q": pq,
            "memb": bool(m[-1, j]) if m is not None else None,
        }
    return cap


class TaughtTokenBonus:
    """Decode-time bonus on taught fact tokens when the owning row delivers.

    The named v0.1 lever for the fact-token door: the memory wins the
    taught margin but loses the name-slot softmax to the base prior
    (Whiskers / Midnight) mid-conversation.  At each decode step, if memory
    row j is live, add `beta` nats to row j's taught GOOD token ids in the
    next-token logits (last position only).

    Scope (10.10 rung 1 — question-anchored activation):
      scope=None   live = row hot at the decode position or anywhere in
                   the forward (the coarse retrieval scope — garbles
                   non-memory replies on the live path)
      scope=QS     live = rows the CURRENT QUESTION addresses (episode
                   token-containment and/or question-span p_home); bonus
                   never touches general reply tokens
      slot_k>0     answer-slot-only: fixed beta only where a taught token
                   is inside the top-`slot_k` of the step (the name-slot
                   decision), never across general reply tokens
      scope.margin row-selective margin: boost row j only where
                   p_j >= best sibling p + margin (10.9 family)

    Usage (chat): wrap the grafted logits_fn so every decode step sees it;
    call `note(tok)` once after the tokenizer is loaded; call
    `scope.begin(question, tok, prompt_ids)` before each reply.
    """

    def __init__(self, beta: float = 2.0, tau: float = 0.5,
                 taught: dict | None = None, top_k: int = 0,
                 margin: float = 0.75, once: bool = False,
                 scope: QuestionScope | None = None, slot_k: int = 0):
        self.beta = float(beta)        # fixed nats (top_k=0) or lift cap
        self.tau = float(tau)
        self.taught = taught or TAUGHT_STRS
        self.top_k = int(top_k)        # 0 = always apply when live
        self.margin = float(margin)    # lift-to-win overshoot
        self.once = bool(once)         # at most one injection per reply
        self.scope = scope             # question-anchored row scope
        self.slot_k = int(slot_k)      # answer-slot-only rank gate
        self.ids = {}
        self.applied_steps = 0
        self.last_live = ()
        self._last_n = None

    def note(self, tok) -> None:
        self.ids = {}
        for tag, strs in self.taught.items():
            s = set()
            for st in strs:
                for t in tok.encode(st, add_special_tokens=False):
                    s.add(int(t))
            self.ids[tag] = sorted(s)

    def live_tags(self, cap: dict) -> tuple:
        if self.scope is not None:
            # question-anchored: only rows the question addresses
            return tuple(sorted(self.scope.matched(cap)))
        live = []
        for tag in MEM_TAGS:
            c = cap.get(tag)
            if not c:
                continue
            # retrieval-triggered scope: the row is in play if it is hot at
            # the decode position OR anywhere in the current forward (the
            # question tokens re-activate it mid-conversation — without
            # this the row is dead at answer-start in long contexts)
            hot = max(c.get("p_home") or 0.0, c.get("p_home_max") or 0.0)
            if hot >= self.tau:
                live.append(tag)
        return tuple(live)

    def edit(self, logits: torch.Tensor, cap: dict) -> torch.Tensor:
        """Bonus on the LAST-position next-token logits.

        top_k=0: fixed `beta` nats on every taught id at every live step.
        top_k>0: the v0.1 margin-to-token tie-break — the bonus lands only
        where a taught token is already competitive (inside the top-K) and
        only lifts it to win by `margin` (capped at `beta`).
        slot_k>0: answer-slot-only — fixed beta, and only where the best
        taught id of the row is inside the top-`slot_k` at this step.
        once=True: at most one injection per reply (a new reply is detected
        when the sequence length does not grow by exactly one).
        """
        n = int(logits.shape[1])
        if self._last_n is not None and n != self._last_n + 1:
            self.applied_steps = 0            # new reply
        self._last_n = n
        if self.once and self.applied_steps >= 1:
            return logits
        live = self.live_tags(cap)
        self.last_live = live
        if not live or self.beta == 0.0:
            return logits
        if self.scope is not None and self.scope.margin > 0.0:
            win = self.scope.winner(cap)
            live = tuple(t for t in live if t == win) if win else ()
            self.last_live = live
            if not live:
                return logits
        out = logits.clone()
        row = out[0, -1]
        row_cpu = row.detach().float().cpu()
        top1 = float(row_cpu.max())
        for tag in live:
            ids = self.ids.get(tag, ())
            if not ids:
                continue
            best = max(ids, key=lambda t: float(row_cpu[t]))
            rank = int((row_cpu > row_cpu[best]).sum()) + 1
            if self.slot_k > 0 and rank > self.slot_k:
                continue                      # not an answer-slot step
            if self.top_k > 0 and rank > self.top_k:
                continue
            if self.top_k > 0:
                need = top1 - float(row_cpu[best]) + self.margin
                lift = min(max(need, 0.0), self.beta)
            else:
                lift = self.beta
            if lift <= 0.0:
                continue
            for tid in ids:
                row[tid] = row[tid] + lift
            self.applied_steps += 1
            if self.once:
                break
        return out

    def wrap(self, logits_fn, gates, tag_col):
        """Wrap a logits_fn(x) -> (B,T,V) with the decode-time bonus."""

        def fn(x):
            lg = logits_fn(x)
            qspan = self.scope.span if self.scope is not None else None
            return self.edit(lg, capture_tags(gates, tag_col, qspan=qspan))

        return fn


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------

def build_world(args):
    import chat_v05 as c5
    device = args.device
    cargs = SimpleNamespace(**vars(args), default_lam=0.5, lam_map={},
                            bare=False)
    (model, tok, blocks, fam, gates, handles, info, missing,
     site_entries, site_banks) = c5.build_state(cargs, device)
    names = [i["name"] for i in info]
    wrappers = install_varact(handles, POLICIES["base"]())
    seq = 512
    streams = {n: home_stream(tok, HOME_OF[n], args, seq) for n in names}
    base_ids = []
    for _name, text in d0.load_domain_texts(args.domains_dir).items():
        base_ids.extend(tok.encode(text, add_special_tokens=False))
    base_ids = base_ids[:args.max_base_tokens]
    base = g0.to_batches(base_ids, 0, seq, 4)
    rep = {n: pick(streams[n], 0, args.win) for n in names}
    held = {n: pick(streams[n], 20, 20 + args.win) for n in names}
    base_rep = pick(base, 0, max(2, args.win // 2))
    def logits_fn(x):
        return g0.run_in_state(
            model, blocks, fam, gates, handles, "both",
            lambda: d0.model_logits(model, x.to(device)))
    return SimpleNamespace(
        model=model, tok=tok, blocks=blocks, fam=fam, gates=gates,
        handles=handles, info=info, names=names, wrappers=wrappers,
        site_entries=site_entries, site_banks=site_banks, device=device,
        rep=rep, held=held, base_rep=base_rep, logits_fn=logits_fn,
        streams=streams)


def phase_sweep(w, args, out):
    names = w.names
    # masked baselines (no plugs at all) — once
    set_policy(w.wrappers, POLICIES["base"]())
    masked = {n: round(g0.run_in_state(
        w.model, w.blocks, w.fam, None, None, "off",
        lambda b=w.rep[n]: d1.ce_in_state(w.model, w.blocks, b, w.device,
                                          None, None, "off")), 4)
        for n in names}
    masked_base = round(g0.run_in_state(
        w.model, w.blocks, w.fam, None, None, "off",
        lambda: d1.ce_in_state(w.model, w.blocks, w.base_rep, w.device,
                               None, None, "off")), 4)
    out["masked_rep"] = masked
    out["masked_base_rep"] = masked_base
    dump(out, args.out)
    print(f"[sweep] masked rep: {masked} base {masked_base}", flush=True)

    # per-config grid
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    results = out.get("sweep") or {}
    for cname in configs:
        if cname in results and results[cname].get("home_ce"):
            print(f"[sweep] {cname} already measured — skip", flush=True)
            continue
        if cname not in POLICIES:
            print(f"[sweep] UNKNOWN config {cname} — skipped", flush=True)
            continue
        pol = POLICIES[cname]()
        set_policy(w.wrappers, pol)
        home = {}
        for n in names:
            r = eval_pass(w.model, w.blocks, w.fam, w.gates, w.handles,
                          w.rep[n], w.device, w.wrappers)
            home[n] = r
        br = eval_pass(w.model, w.blocks, w.fam, w.gates, w.handles,
                       w.base_rep, w.device, w.wrappers)
        # cross-fire delivery matrix = per-plug deliver share on each home
        # (collected inside the same eval passes — no extra forward)
        xf = {HOME_OF[n]: {t: v["deliver_frac"]
                           for t, v in home[n]["plug_stats"].items()}
              for n in names}
        # oracle owner-only CE: only the home plug delivers (off-home flood
        # ceiling) — flood_damage = home_ce - owner_only
        oracle = {}
        for n in names:
            only = ce_with_active(w.model, w.blocks, w.fam, w.gates,
                                  w.handles, w.rep[n], w.device, {n})
            oracle[n] = round(only, 4)
        tgps, ntok = timed_gen(w.tok, w.logits_fn, w.device,
                               max_new=args.gen_tokens)
        deliver_pt = round(
            sum(home[n]["deliver_per_token"] for n in names) / len(names), 4)
        compute_pt = round(
            sum(home[n]["compute_per_token"] for n in names) / len(names), 4)
        # serving-side plug forwards per token: base always runs every plug
        # every token; gated runs a plug only on tokens it delivers (T=1 decode)
        serving_pt = compute_pt if pol.mode == "off" else deliver_pt
        results[cname] = {
            "policy": {"mode": pol.mode, "tau": pol.tau, "top_m": pol.top_m},
            "home_ce": {n: home[n]["ce"] for n in names},
            "home_compute_frac": {n: home[n]["compute_per_token"] for n in names},
            "home_deliver_frac": {n: home[n]["deliver_per_token"] for n in names},
            "base_ce": br["ce"],
            "compute_per_token": compute_pt,
            "deliver_per_token": deliver_pt,
            "serving_compute_per_token": serving_pt,
            "tok_per_s_eval": round(
                sum(home[n]["tok_per_s"] for n in names) / len(names), 1),
            "tok_per_s_gen": tgps,
            "gen_tokens": ntok,
            "cross_fire": xf,
            "oracle_owner_only_ce": oracle,
            "flood_damage": {n: round(home[n]["ce"] - oracle[n], 4)
                             for n in names},
        }
        out["sweep"] = results
        dump(out, args.out)
        cd = results[cname]
        print(f"[sweep {cname:18s}] serving/tok {cd['serving_compute_per_token']} "
              f"batchFLOPs {cd['compute_per_token']} "
              f"deliver/tok {cd['deliver_per_token']} "
              f"gen {cd['tok_per_s_gen']} tok/s base {cd['base_ce']}",
              flush=True)
        print(f"[sweep {cname:18s}] home_ce {cd['home_ce']}", flush=True)
    out["phase"] = "sweep_done"
    dump(out, args.out)
    return results


def phase_harm(w, args, out, configs=None):
    """Leave-one-out pairwise harm (bank2n bar 0.02) for chosen configs.

    harm(j -> h) = CE_h(all active) - CE_h(j inactive); positive = j hurts h.
    Sibling crowding = mem trio pairs; flooding = off-home pairs.  Own-home
    j == h is recorded as own_effect (negative = the plug earns its keep).
    Pairs whose delivery share on h is < min-share are skipped (they cannot
    move CE meaningfully); skipped = 0.0 in the matrix.
    """
    names = w.names
    if configs is None:
        if args.harm_configs == "auto":
            configs = list(out.get("_harm_auto", ["base"]))
        else:
            configs = [c.strip() for c in args.harm_configs.split(",")
                       if c.strip()]
    out.setdefault("harm", {})
    # harm reads use a shorter window than the sweep
    rep = {n: pick(w.streams[n], 0, args.harm_win) for n in names}
    for cname in configs:
        if cname in out.get("harm", {}) and out["harm"][cname].get("loo"):
            print(f"[harm] {cname} already measured — skip", flush=True)
            continue
        pol = POLICIES.get(cname, POLICIES["base"])()
        set_policy(w.wrappers, pol)
        full = {n: round(ce_with_active(w.model, w.blocks, w.fam, w.gates,
                                        w.handles, rep[n], w.device,
                                        set(names)), 4) for n in names}
        xf = (out.get("sweep", {}).get(cname, {}) or {}).get("cross_fire", {})
        loo = {}
        for n in names:
            row = {}
            for j in names:
                share = xf.get(HOME_OF[n], {}).get(j, 1.0) if xf else 1.0
                if j != n and share < args.min_share:
                    row[j] = 0.0
                    continue
                ce_wo = ce_with_active(w.model, w.blocks, w.fam, w.gates,
                                       w.handles, rep[n], w.device,
                                       set(names) - {j})
                row[j] = round(full[n] - ce_wo, 4)
            loo[n] = row
            print(f"[harm {cname} -> {n}]", row, flush=True)
        worst_off, worst_pair = 0.0, ("", "")
        worst_sib, sib_pair = 0.0, ("", "")
        worst_ns, ns_pair = 0.0, ("", "")
        for n in names:
            for j, v in loo[n].items():
                if j == n:
                    continue
                if HOME_OF[j] == HOME_OF[n]:
                    continue
                if v > worst_off:
                    worst_off, worst_pair = v, (j, n)
                sib = (j.startswith("mem") and n.startswith("mem")
                       and HOME_OF[j] != HOME_OF[n])
                # sibling family = the mem trio (plug names, not home names)
                if sib and v > worst_sib:
                    worst_sib, sib_pair = v, (j, n)
                if not sib and v > worst_ns:
                    worst_ns, ns_pair = v, (j, n)
        out["harm"][cname] = {
            "full_ce": full, "loo": loo,
            "own_effect": {n: loo[n][n] for n in names},
            "worst_offhome": round(worst_off, 4),
            "worst_offhome_pair": list(worst_pair),
            "worst_non_sibling": round(worst_ns, 4),
            "worst_non_sibling_pair": list(ns_pair),
            "worst_mem_sibling": round(worst_sib, 4),
            "worst_mem_sibling_pair": list(sib_pair),
            "bar": BAR_HARM,
        }
        out.setdefault("phase", "harm")
        dump(out, args.out)
        print(f"[harm {cname}] worst off-home {worst_off:.4f} {worst_pair} "
              f"| non-sib {worst_ns:.4f} {ns_pair} "
              f"| mem-sib {worst_sib:.4f} {sib_pair}", flush=True)
    out["phase"] = "harm_done"
    dump(out, args.out)


def phase_smoke(w, args, out):
    """Demo-prompt smoke: base vs gated, SHORT excerpts only."""
    import chat_v05 as c5
    configs = [c.strip() for c in args.smoke_configs.split(",") if c.strip()]
    out.setdefault("smoke", {})
    for cname in configs:
        pol = POLICIES.get(cname, POLICIES["base"])()
        set_policy(w.wrappers, pol)
        rows = []
        for kind, user in c5.PROBES:
            t0 = time.time()
            text, ids = g0.gen_reply(w.tok, w.logits_fn,
                                     [{"role": "user", "content": user}],
                                     args.smoke_tokens, w.device)
            short = " ".join(text.split())[:160]
            rows.append({"kind": kind, "q": user[:80], "excerpt": short,
                         "n_new": len(ids),
                         "sec": round(time.time() - t0, 1)})
            print(f"[smoke {cname} {kind}] n={len(ids)} {short}", flush=True)
        out["smoke"][cname] = rows
        dump(out, args.out)
    out["phase"] = "smoke_done"
    dump(out, args.out)


def phase_report(w, args, out):
    names = w.names
    sweep = out.get("sweep", {})
    base = sweep.get("base", {})
    if not base:
        print("[report] no base entry in sweep — run --phase sweep first",
              flush=True)
        return
    def serving_of(cd):
        if "serving_compute_per_token" in cd:
            return cd["serving_compute_per_token"]
        mode = (cd.get("policy") or {}).get("mode", "off")
        return cd["compute_per_token"] if mode == "off" \
            else cd["deliver_per_token"]

    base_compute = serving_of(base)     # ungated: every plug, every token
    frontier = []
    for cname, cd in sweep.items():
        cut = 1.0 - serving_of(cd) / max(1e-9, base_compute)
        worse = {}
        for n in GREEN:
            if n in cd["home_ce"] and n in base["home_ce"]:
                d = round(cd["home_ce"][n] - base["home_ce"][n], 4)
                if d > 1e-4:
                    worse[n] = d
        harm = out.get("harm", {}).get(cname, {})
        frontier.append({
            "config": cname,
            "gen_tok_s": cd["tok_per_s_gen"],
            "serving_compute_per_token": serving_of(cd),
            "compute_per_token": cd["compute_per_token"],
            "compute_cut": round(cut, 4),
            "base_ce": cd["base_ce"],
            "home_ce": cd["home_ce"],
            "green_regress": worse,
            "worst_offhome_harm": harm.get("worst_offhome"),
            "worst_mem_sibling": harm.get("worst_mem_sibling"),
            "flood_sum": round(sum(cd.get("flood_damage", {}).values()), 4),
        })
    out["frontier"] = frontier
    for f in frontier:
        ok = (not f["green_regress"]) and f["compute_cut"] >= BAR_COMPUTE_CUT \
            and f["worst_offhome_harm"] is not None \
            and f["worst_offhome_harm"] <= BAR_HARM
        tag = "SHIP" if ok else (
            "-" if f["worst_offhome_harm"] is not None else "-harm?")
        print(f"[frontier {f['config']:18s}] tok/s {f['gen_tok_s']} "
              f"cut {f['compute_cut']:.2%} base {f['base_ce']} "
              f"regress {f['green_regress']} harm {f['worst_offhome_harm']} "
              f"sib {f['worst_mem_sibling']} flood {f['flood_sum']} "
              f"{tag}", flush=True)
    ships = [f["config"] for f in frontier
             if (not f["green_regress"])
             and f["compute_cut"] >= BAR_COMPUTE_CUT
             and f["worst_offhome_harm"] is not None
             and f["worst_offhome_harm"] <= BAR_HARM]
    out["verdict"] = {"ship_configs": ships, "bar_compute_cut": BAR_COMPUTE_CUT,
                      "bar_harm": BAR_HARM}
    out["phase"] = "report_done"
    dump(out, args.out)
    print("[report] ship candidates:", ships, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", default="all",
                    choices=["sweep", "harm", "smoke", "report", "all"])
    ap.add_argument("--model", default="/data/tliao/Phi-mini-MoE-instruct")
    ap.add_argument("--registry",
                    default="/data/tliao/graft_d2_out/registry_phimoe_f620b32c.pt")
    ap.add_argument("--domains-dir", default="/data/tliao/graft_d2_slice")
    ap.add_argument("--specialties-dir",
                    default="/data/tliao/vesper_sft/VesperLM/Dataset/data/specialties")
    ap.add_argument("--plugins-dir", default="/data/tliao/v05_out")
    ap.add_argument("--out", default="/data/tliao/varact_out/varact.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--max-tokens", type=int, default=70000)
    ap.add_argument("--max-base-tokens", type=int, default=24000)
    ap.add_argument("--win", type=int, default=8,
                    help="rep/held windows per home (batches of 4x512)")
    ap.add_argument("--gen-tokens", type=int, default=24)
    ap.add_argument("--smoke-tokens", type=int, default=40)
    ap.add_argument("--configs",
                    default="base,tau0.3,tau0.5,tau0.7,topm1,topm2,topm3,"
                            "adaptive_tau0.5_m2")
    ap.add_argument("--harm-configs", default="auto",
                    help="comma list, or 'auto' = base + 2 best cut configs")
    ap.add_argument("--harm-win", type=int, default=4,
                    help="windows per home for LOO harm reads")
    ap.add_argument("--min-share", type=float, default=0.02,
                    help="skip LOO pairs with delivery share below this")
    ap.add_argument("--smoke-configs", default="auto",
                    help="comma list, or 'auto' = base + best candidate")
    ap.add_argument("--gpu", default="auto")
    args = ap.parse_args()

    if args.gpu == "auto":
        import chat_v05 as c5
        dev_i = c5.pick_gpu()
    else:
        dev_i = int(args.gpu)
    args.device = f"cuda:{dev_i}"
    torch.cuda.set_device(dev_i)
    torch.manual_seed(0)

    out_path = args.out
    out = {}
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                out = json.load(f)
        except Exception:
            out = {}
    out.update({"gate": "varact", "format": "vesper-varact-v0",
                "pid": os.getpid(), "configs": args.configs,
                "started": time.strftime("%Y-%m-%dT%H:%M:%S")})
    dump(out, out_path)

    w = build_world(args)
    print(f"[varact] live plugs: {w.names}", flush=True)

    if args.phase in ("sweep", "all"):
        phase_sweep(w, args, out)

    # auto pick: base + up to 2 configs with the biggest compute cut and no
    # green-plug home-CE regression vs base (harm/smoke candidates)
    if args.phase in ("all",) or args.harm_configs == "auto" \
            or args.smoke_configs == "auto":
        sw = out.get("sweep", {})
        base_ce = (sw.get("base") or {}).get("home_ce", {})
        n_plugs = max(1, len(base_ce))
        base_srv = (sw.get("base") or {}).get("compute_per_token", float(n_plugs))

        def srv_cut(cd):
            srv = cd.get("serving_compute_per_token",
                         cd["deliver_per_token"] if (cd.get("policy") or {})
                         .get("mode") != "off" else cd["compute_per_token"])
            return 1.0 - srv / max(1e-9, base_srv)

        cands = []
        for cname, cd in sw.items():
            if cname == "base":
                continue
            cut = srv_cut(cd)
            regress = any(cd["home_ce"].get(n, 1e9) > base_ce.get(n, 0) + 1e-4
                          for n in GREEN if n in base_ce)
            cands.append((cut, not regress, cname))
        cands.sort(key=lambda t: (t[1], t[0]), reverse=True)
        best = [c[2] for c in cands[:2]]
        out["_harm_auto"] = ["base"] + best
        out["_smoke_auto"] = ["base"] + best[:1]
        dump(out, args.out)
        print(f"[varact] auto harm={out['_harm_auto']} "
              f"smoke={out['_smoke_auto']}", flush=True)

    if args.phase in ("harm", "all"):
        hc = None
        if args.phase == "all" or args.harm_configs == "auto":
            hc = out.get("_harm_auto", ["base"])
        phase_harm(w, args, out, configs=hc)
    if args.phase in ("smoke", "all"):
        if args.phase == "all" or args.smoke_configs == "auto":
            args.smoke_configs = ",".join(out.get("_smoke_auto", ["base"]))
        phase_smoke(w, args, out)
    if args.phase in ("report", "all"):
        phase_report(w, args, out)
    print("[varact] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
