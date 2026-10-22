# MIMO_TEACH_HOOKS — teach-by-talking hooks on the live vesper demo (design)

### Op note (2026-10-22 ~21:05 box — teach surface live, gate-only state

Commands (verbatim): `/teach <user>` (bind a fresh session JSONL at
`~/mimo_flash/teach_sessions/session_<user>_<isotime>.jsonl`, append-only,
attaches an existing promoted delta if one exists) · `/approve`
`/reject` `/correct <revised answer>` (labels on the last turn; the
correction string is the supervision target) · `/learn` (spawns the
gated worker `lab/mimo_teach_micro.py` on TEACH_WORKER_GPU, default
gpu0 — must differ from the demo's own GPU; an occupied claim aborts
with no state change) · auto-`/learn` at 5 corrected triples ·
`/teach_state` (user/session/counts/quota, adapters at
`~/mimo_flash/teach_engines/<user>/delta.pt` with size/mtime/last gate,
tier-1 constants, tier-2 note) · `/unlearn <user>` (remove delta,
detach in-memory, print CE-vs-never-attached byte-check).

Gate (tier-1 = `consolidate.stub_gate` semantics, verbatim): mean_reward
≥ 0 · top_response_fraction ≤ 0.5 (hard target-collapse) ·
unique_response_ratio ≥ 0.25 · final_kl ≤ 10.0 nats (drift proxy =
(BASEWIN CE_armed − CE_base) × tokens).  PROMOTE → atomic
`delta.pt` (incumbent kept as `delta_prev.pt`); ROLLBACK → candidate
discarded, incumbent untouched, decision + one-line reason logged —
teaching NEVER applies on a gate reject.  Tier-2 Immune checkpoint-merge
gate exists for later merges and deliberately does not fire per-session.
Teach is per-user and separate from the persona pack always: no taught
content lands in any plug record or shared state.

State (settled at the current recipe class — teach surface live, but
the block-0-wrap teach fit is CLOSED for this batch class): three
teach sessions recorded live; the base round at lr 3e-4 gated ROLLBACK
on all three (FAILURES.md `mimo-teach-micro-divergence`), and the
declared retry map — lr 3e-5 × 40 then the one nested drop lr 3e-6 ×
40 — shows CE rising from step 0 at every decade (3e-5: 5.95→6.35→NEG
6.87@11; 3e-6: 5.93→6.46→NEG 6.89@11; stopped at the declared step-10
decision points; see FAILURES.md `mimo-teach-retry-lr-map`).  No
adapter exists anywhere; teaching never applied on a gate REJECT.  The
named next measure is a final-layer LoRA teach engine (the vk teach
series' own geometry class) — named, not run this cycle.  Left out
pending a PROMOTE: the attach/recall on-off contrast and cross-restart
persistence battery.


Date: 2026-10-22.  Line: lab/TEACH_PIPELINE.md (the two-tier gate policy,
calibrated at vk-scale ckpt classes) applied to the MiMo host running the
persona pack (lab/MIMO_MEMBERSHIP_THROTTLE.md close-out: vdemo serves
curated3→plugs_matrix_v4 wraps + sentinel24_v3 gate at margin 20 +
fallback-b steer for the three wrap-mute seats).  Goal: the person in
the vdemo chair can teach the host by correction flow, with the series'
gate discipline — smallest real form, no hairball.  Teach is **per-user
and separate from the persona pack always**: nothing taught lands in any
plug record or shared state; per-user deltas are additive attachments to
the serving process only.

## REPL protocol (verbatim commands)

```
/teach <user>            bind a teach session (user handle); starts a fresh
                         session JSONL; attaches the user's existing delta
                         if one has been promoted before (verified load)
/approve                 label the last answer acceptable (+1)
/reject                  label the last answer wrong/off (-1; no target text)
/correct <revised text>  label the last answer wrong and supply the revised
                         answer — this is the supervision target (+1)
/teach_state             session/adapter/gate state print (see below)
/learn                   force-run the gated micro-session on whatever
                         triples the current session holds (else auto-fires
                         at the quota of 5 corrected triples)
/unlearn <user>          honest rollback: delete the promoted delta file,
                         detach the in-memory adapter, print the host
                         byte-unchanged verification (at-rest CE equals the
                         never-attached reference)
```

Every recorded turn appends append-only to
`~/mimo_flash/teach_sessions/session_<user>_<isotime>.jsonl`
(question, model answer, label, correction, reward).  Never reuse a
session file across `/teach` invocations (series invariant #10).

## Per-session micro-session (tier-1 discipline, MiMo-host-scale)

`/learn` spawns the worker `lab/mimo_teach_micro.py` as a subprocess on
GPU0 (own pid claim under the box claim-dir convention; refuses when the
GPU is claimed instead of queueing; the person's chat on GPU1 keeps
running untouched).  Worker recipe (documented reading of the task
spec's "io froze"):

- geometry: rank-64 wrap at MoE block 0, single site — the same
  WrappedForeignExpert family the pack uses, core buffers frozen;
  trunk host input/output (the whole base) stays frozen (series I1);
  adapter linears are the only trainables, identity-at-birth (host_out
  zeroed, other linears 0.02·rand), cumulative: an existing promoted
  incumbent delta is RESUMED, not restarted.
- objective: mild-CE on the corrected answers (prompt context, CE on
  the revised-answer tail), out_pen=300 NEG parity penalty every 3rd
  step on neutral windows; lr 3e-4, grad-clip 1.0, steps default 24
  (TEACH_STEPS env), per-layer checkpointing, lazy cache off — the
  proven gfx906 mechanics from the pack fits.
- batch statistics written to the candidate dir's `train_stats.json`:
  mean_reward (+1 approve, −1 reject), unique_response_ratio and
  top_response_fraction over corrected-answer strings, final_kl — the
  measured KL-drift proxy in nats: (CE_armed − CE_base) on the BASEWIN
  cold window × window tokens (documented substitute for the series'
  train-time KL at this host class).

## Gate (tier-1 only, per-session) + promotion/rollback

The worker gates the candidate with `consolidate.stub_gate` semantics
verbatim (imported from a mirrored copy of Hippocampus/consolidate.py
under `~/mimo_flash/teach_lib/`, or an embedded identical port as
fallback — tripwires and constants identical, pointer to
TEACH_PIPELINE.md): mean_reward ≥ 0, top_response_fraction ≤ 0.5 (hard
target-collapse tripwire — invariant #7), unique_response_ratio ≥ 0.25,
final_kl ≤ 10.0.  PROMOTE → candidate atomically becomes
`~/mimo_flash/teach_engines/<user>/delta.pt` (incumbent kept as
delta_prev.pt; a pending marker is written so the chat process attaches
it at the next turn boundary).  ROLLBACK → candidate discarded,
incumbent untouched, every decision logged with its one-line reason.
**Teaching never applies on a gate REJECT** — if the gate says
ROLLBACK the persona demo state is byte-identical to before the fit.

Tier-2 note: the Immune checkpoint-merge gate (probes/score/drift/decide
constants) exists for later checkpoint merges and deliberately does NOT
fire per-session (measured 0/3 promote class when misapplied at micro
scale) — `/teach_state` states its presence and non-application at this
tier.

## vdemo-side attach/detach + verification probes

- Attach: the demo wraps blocks[0].experts.forward on top of the pack
  fences to add `0.5·delta(h)`.  Detach restores the exact pre-attach
  reference.  Host weights are never mutated, so `/unlearn` can verify
  byte-equivalence: BASEWIN at-rest CE after removal must equal the
  never-attached reference read at boot (printed both).
- `/teach_state` prints: bound user + session file + turn/triple counts
  + quota; adapters listed from `~/mimo_flash/teach_engines/*/delta.pt`
  with sizes/mtime/last gate decision; tier-1 tripwire constants; the
  tier-2 non-application note; attached state + lam.
- Self-verification battery (this cycle, box GPU0, steer OFF to isolate
  attribution): teach across 3 separate sessions (diversified
  correction phrasings per invariant #8 — collapse tripwire is hard),
  then: (i) process restart → delta retained and attachable; (ii) with
  adapter attached the taught recall question answers in the taught
  class; (iii) without the adapter the same question does not match
  that class; (iv) `/unlearn` removes it and byte-verifies clean.

## Verdict classes (declared)

- TEACH-WIN: delta persists across restart (i–iii measured) and the
  teach/steer/pack machinery coexists without contamination (cold
  probes latch None, parity +0.0000-class, wrap/steer arms unchanged);
  unlearn provably clean.
- TEACH-PARTIAL: gate works but the recall attribution is weak at item
  count (record metrics; next lever named only).
- TEACH-FAIL: the micro-session at this host class doesn't retain the
  taught item under its own tier-1 gate — record, FAILURES.md, no
  deck change.

### Results

(to be filled: per-session flow lines, gate decisions with metrics,
attach yes/no with on/off contrast lines, unlearn verification, deck
op)
