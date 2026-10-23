# MIMO_TEACH_HOOKS — teach-by-talking hooks on the live vesper demo (design)

### Op note (2026-10-23 — teach slot PROMOTED at the final-block LoRA geometry; verdict TEACH-PARTIAL

Live geometry (new this cycle): LoRA rank 8, alpha 16 on the FINAL
transformer block's attention projections (q/k/v slice-wise inside the
fused qkv proj, plus o proj; layer 47, full-attention) of the MiMo host
— worker `lab/mimo_teach_lora.py`, 305,152 trainable params, trunk
frozen, LoRA-B zeroed at birth, cumulative resume from the incumbent.
The block-0-wrap worker `lab/mimo_teach_micro.py` is RETIRED (its
geometry class is closed for divergence; FAILURES.md
`mimo-teach-micro-divergence`, `mimo-teach-retry-lr-map`).

Commands (verbatim): `/teach <user>` (bind a fresh session JSONL at
`~/mimo_flash/teach_sessions/session_<user>_<isotime>.jsonl`, append-only,
attaches an existing promoted delta if one exists) · `/approve`
`/reject` `/correct <revised answer>` (labels on the last turn; the
correction string is the supervision target) · `/learn` (spawns the
gated worker `lab/mimo_teach_lora.py` on TEACH_WORKER_GPU, default
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

State: the declared knob map ran exactly once — primary lr 2e-4, 24
steps, out_pen 300 NEG cadence; the nested drop to 2e-5 was never
needed (HOME CE compresses 5.94 → 3.73 on session 1; no step-0 rise at
this geometry).  Two of three captured sessions PROMOTED (the
1-correction session ROLLBACKed on the hard target-collapse tripwire,
by design); the promoted cumulative delta (sessions 1+2) is live at
`~/mimo_flash/teach_engines/tester/delta.pt`.  The attach battery
measured restart-retention PASS and unlearn byte-clean PASS, but
free-run recall does NOT land the taught class (teacher-forced margin
gain +0.54 nats — real but weak at item count): verdict TEACH-PARTIAL,
per-session lines in ### Results below.


Date: 2026-10-22 (design); op note refreshed 2026-10-23.  Line: lab/TEACH_PIPELINE.md (the two-tier gate policy,
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

`/learn` spawns the worker `lab/mimo_teach_lora.py` as a subprocess on
GPU0 (own pid claim under the box claim-dir convention; refuses when the
GPU is claimed instead of queueing; the person's chat on GPU1 keeps
running untouched).  Worker recipe (final-block LoRA geometry, live
since 2026-10-23; the retired block-0-wrap recipe is preserved in
`lab/mimo_teach_micro.py` and the two FAILURES.md entries):

- geometry: LoRA rank 8, alpha 16 on the FINAL transformer block's
  attention projections — q/k/v slice-wise inside the fused qkv proj,
  plus o proj (layer 47, full-attention per hybrid_layer_pattern);
  305,152 trainable params.  Trunk host input/output (the whole base)
  stays frozen (series I1); LoRA-B zeroed at birth (identity-at-birth),
  LoRA-A 0.02·rand seeded; cumulative: an existing promoted incumbent
  delta is RESUMED, not restarted (and a format mismatch across
  geometry classes aborts rather than resumes).
- objective: mild-CE on the corrected answers (prompt context, CE on
  the revised-answer tail), out_pen=300 NEG parity penalty every 3rd
  step on the same 3 neutral prompts as the retired worker; lr 2e-4
  primary (declared knob map: ONE nested drop to 2e-5 if CE rises from
  step 0, then TEACH-FAIL — the drop was never needed), grad-clip 1.0,
  steps default 24 (TEACH_STEPS env), per-layer checkpointing, lazy
  cache off — the proven gfx906 mechanics from the pack fits.
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

- Attach (LoRA geometry): the delta carries its own in-memory
  attach/detach path (`attach_lora`/`detach_lora` in
  `lab/mimo_teach_lora.py`) — the final block's fused qkv proj and
  o proj are wrapped with frozen-base + LoRA modules; detach restores
  the exact pre-attach module references.  Host weights are never
  mutated, so `/unlearn` can verify byte-equivalence: BASEWIN at-rest
  CE after removal must equal the never-attached reference read at
  boot (printed both).  (The retired wrap geometry attached via
  blocks[0].experts.forward; that path is closed with its worker.)
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
  Ran 2026-10-23 as the standalone `lab/mimo_teach_lora_verify.py`
  (own process = the restart leg; never touches the resident demo) —
  results in ### Results.

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

Final-block LoRA teach engine, first (and only) budgeted attempt,
2026-10-23, box GPU0 (claim convention held throughout; the resident
demo on GPU1 was never touched).  Declared before the first run —
geometry: single final block (layer 47) q/k/v+o LoRA rank 8 alpha 16,
305,152 params; knob map: primary lr 2e-4 × 24 steps, ONE nested drop
to 2e-5 if CE rises from step 0, else TEACH-FAIL and close.  The
nested drop was never spent: no step-0 rise at this geometry.

Per-session flow lines (user `tester`, fits independent, cumulative
resume from incumbent; logs `~/mimo_flash/teach_logs/lora_s{1,2,3}_lr2e-4*.log`):

| session | corrections | HOME CE trace (steps 3/7/15/19) | gate | mean_reward | uniq | top_frac | final_kl |
|---|---|---|---|---|---|---|---|
| …T181508 (6 turns) | 5 | 5.94 / 6.06 / 4.69 / 3.73 | **PROMOTE** | 1.0 | 1.00 | 0.20 | +0.54 |
| …T182815 (3 turns) | 2 | 4.70 / 3.79 / 2.39 / 1.91 | **PROMOTE** | 1.0 | 1.00 | 0.50 | +1.13 |
| …T183339 (2 turns) | 1 | 2.30 / 1.42 / 0.61 / 0.41 | **ROLLBACK** (target collapse 100%, hard tripwire by design) | 1.0 | 1.00 | 1.00 | +2.85 |

Session-1 note: the first launch was operator-interrupted at step 15
when the step-11 NEG reading (6.76) pattern-matched the retired
geometry's divergence signature; the same process's step-15 HOME print
(4.69, well below the 5.94 start) refuted divergence, and the
uninterrupted continuation of the SAME declared arm completed and
PROMOTED (drift proxy parity 3.0467→3.0682).  NEG-step readings are
parity ce_on under an active out_pen penalty, not the divergence
tripwire — the tripwire reads the HOME-CE trend only.  Session 3's fit
was itself clean (CE 2.30→0.41, kl +2.85 ≤ 10); the gate rejected on
batch composition (a 1-item session can never clear the diversity
tripwire — invariant #8 working as designed), candidate discarded,
incumbent untouched.

Attach battery (`lab/mimo_teach_lora_verify.py`, own fresh process on
the promoted sessions-1+2 cumulative delta; `verify_lora.json`):

- (i) restart retention: PASS — delta.pt (1,223,601 bytes) loads and
  attaches in a fresh process; format `vesper-teach-lora-delta-v1`.
- (ii) adapter ON recall: MISS — free-run greedy answer to the taught
  recall question is NOT in the taught class (degenerates into a
  one-word loop on the topic word after a topical first sentence).
- (iii) adapter OFF recall: correctly not in the taught class.
- (iv) unlearn: PASS, byte-clean — detach CE 3.046679 == the
  never-attached reference exactly; at-rest CE after file removal
  identical; the remove/restore cycle is sha256-byte-identical (the
  promoted delta was restored after the leg; the slot stays PROMOTED).

Supplementary teacher-forced attribution probe
(`lab/mimo_teach_lora_margin.py`; TEACH_PIPELINE.md doctrine: the
margin is the mechanism read, free-run is colour): NLL(taught answer)
5.83 OFF → 1.73 ON; NLL(decoy) 6.62 OFF → 3.05 ON; margin
(decoy − taught) +0.79 OFF → +1.33 ON, gain +0.54 nats.  The adapter
strongly learned the response FRAME (both continuations drop ~3.6–4.1
nats) and the item-specific attribution is real but small.

Verdict: **TEACH-PARTIAL** — the tier-1 gate works at this geometry
(2 PROMOTE at drift ≤ 1.13 nats vs the retired geometry's 107–147;
the poison-class 1-item session correctly refused), restart persistence
and byte-clean unlearn are measured, but recall attribution is weak at
item count (free-run miss; teacher-forced margin gain +0.54 nats).
Next lever (named only, not run): teach the recall ANSWER form
directly (short fact-form correction targets, e.g. "Person of
Interest" as the tail) instead of long free-form statement
corrections — the frame-vs-item split in the margin probe says the
CE-tail budget is being spent on phrasing, not on the fact.
