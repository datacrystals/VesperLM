# SUBSYSTEMS_E1 — drive/valence vector v_t: instrument-first correlation study

Date: 2026-10-19. Lane: laptop, CPU only, pure log replay (no training, no
generation, no model loads). Spec: `SUBSYSTEMS.md` (drives layer, gates
E0–E4; §5 kill criterion). Instrumentation harness: `lab/subsystems_e1.py`
(pure stdlib; prints the tables below verbatim, optional CSV dump).

**Question.** Before wiring any E1 actuator: does a drive/valence vector
v_t, computed from telemetry that already exists, predict anything the raw
teacher-forced margins don't already predict? Per `SUBSYSTEMS.md` §5, if the
sensor signals carry no predictive information about good consolidation
choices, the layer is cut back to a fixed scheduler.

**Inputs (all previously committed, free observability):** the teach-series
run summaries + session logs + candidate batch stats under
`lab/teach_{cadence,trigger,aged,aged2,decay,demo_out_sft_r3,demo_out_stub}*`
(8 runs, 66 gated sessions, 8 poison canaries), and the SFT-r3 chat eval
JSONs `lab/imported/vk_sft_r3/_evals/eval_chat_step{200..1800}.json`
(12 prompts × 9 steps) for the chat-lane sensor history. Not used: E0's own
telemetry stream — the teach runs ran with `VESPER_TELEMETRY` off (the only
sink file on disk is a 26-event smoke log from the E0 gate day). The same
v_t could be sourced from the E0 stream on future runs; for this replay the
harness logs are strictly richer.

## The v_t formula (exact, per teach session t)

Item library J with baseline margins m_j(0); m_j(k) = margin of item j in
the snapshot after session k; r_j(k) = (m_j(k) − m_j(0)) / g_j with
g_j = m_j(s_j) − m_j(0) at first teach s_j (recomputed uniformly from
`before`/`snapshots`; verified bit-equal to every harness's own retention
tables, max |Δ| = 0.0).

| component | formula | drive it shadows |
|---|---|---|
| competence c_t | fraction of previously-taught j with r_j(t−1) ≥ 0.50 (hinge); mean r also reported | ego/pride (self-model accuracy) |
| interference i_t | mean over already-taught j ∉ taught-set(t) of m_j(t−2) − m_j(t−1) — damage the LAST session did, known before t runs | satisfaction/frustration |
| novelty n_t | −m_target(t−1), the primary target's pre-session margin deficit | curiosity (measured-gap probe distance) |
| poison-pressure p_t | max of tier-1 tripwire proximities, each normalised so 1.0 == on the wire: reward max(0,−mean_reward); collapse (top_frac−0.5)/0.5; degeneracy (0.25−uniq)/0.25; drift final_kl/10 | trust (veto proximity) |

Outcomes tested: **A** teach success = session lift m_target(t) −
m_target(t−1) > 0; **B** poison = the run's dedicated banana-style poison
session; **C** retention = r_final ≥ 0.50 at horizon per library item.
Stats: Mann-Whitney rank AUC (0.5 = chance), 2000-draw bootstrap 95% CI,
median-split rate check. The four sft-r3 runs share a bit-identical teach
phase (measured again here: max |margin Δ| = 0.0 across all four through
S6), so teach-phase rows are de-duplicated to one copy; n reported per table.

## Results

### A. Teach success — 34 sessions with a primary target (26 lift, 8 no-lift)

| feature | AUC | CI 95 | read |
|---|---|---|---|
| m_pre (raw-margin baseline) | 0.288 | [0.108, 0.520] | mirror of novelty by construction |
| **novelty n_t = −m_pre** | **0.712** | [0.480, 0.892] | deficit separates lifts from no-lifts |
| competence c_frac / c_mean | 0.327 / 0.378 | crosses 0.5 | nothing |
| interference i_prev | 0.464 | [0.136, 0.789] | nothing |
| poison-pressure p_max (pre-gate) | 0.415 | [0.182, 0.659] | nothing |
| mean_reward | 0.528 | [0.364, 0.717] | nothing |

Split by event type: first teaches (n=10) novelty AUC 1.00 (+8/−2);
re-serves (n=24) novelty AUC 0.704 (+18/−6). All 8 no-lifts happen where the
target's pre-session margin was middling-to-high (−0.69 … +8.31) or already
repaired — deficit is the whole story. **But n_t is the raw margin with the
sign flipped: the honest verdict is that nothing in v_t beat margins because
the only component with signal IS the margin.** c_t, i_t, p_t add nothing on
top.

### B. Poison detection — 38 sessions with batch stats, 8 poison

| sensor | AUC |
|---|---|
| p_max | 1.00 [1.00, 1.00] |
| top_response_fraction | 1.00 |
| unique_response_ratio | 0.00 (perfect, reversed) |
| mean_reward | 0.60 |
| final_kl | 0.30 |

Every poison session sits exactly on the collapse wire (p_parts: collapse
1.0, degeneracy 0.5, drift 0.12–0.20, reward 0.0); every normal session is
at p_max ≤ 0.76. Clean sensor separation — **but circular as a discovery:**
poison here is defined by the same batch statistics the tier-1 gate vets, so
this validates the tripwire-as-sensor, it does not find anything new. The
margin cadence never sees poison sessions at all (they consolidate outside
the snapshot loop): margins alone cannot detect poison, tripwire stats can —
the one place where a v_t component strictly dominates the margin baseline.

### C. Horizon retention — 42 items, 21 retained (r_final ≥ 0.50)

| first-teach-legal feature | AUC | CI 95 |
|---|---|---|
| baseline margin m0 | 0.380 | [0.221, 0.564] |
| acquisition gain g | 0.424 | [0.252, 0.601] |
| age at horizon (confound control) | 0.466 | [0.286, 0.657] |

Nothing legal at teach time predicts which items survive. Even the post-hoc
mechanism check — mean interference an item actually suffered after teach —
sits at AUC 0.501: retention variance in these runs is not explained by
smoother interference exposure either. Retention by class: fact 7/18, style
10/18, demo items 4/6. **Retention at teach time is unpredictable from these
sensors; the honest negative.**

### Chat lane (SFT-r3 evals, descriptive only)

Deterministic rubric keys (2 of 12 prompts have a verifiable answer token)
flip 0.0 ↔ 0.5 across steps 200→1800 with no trend (n=2/step = noise floor);
response length 67–91 words, unique-token ratio flat ≈ 0.73, step-to-step
response-set distance flat ≈ 0.79. The chat lane's free sensors carry no
trend at this granularity — competence-at-chat needs a real rubric battery,
not 2 answer keys.

## The honest read — go/no-go for E1 control-wiring

**Composite v_t: NO-GO.** Competence, interference-inherited, and
poison-pressure (beyond restating the tripwire) predicted nothing for teach
success or retention, and retention — the quantity a memory manager most
wants to forecast — is unpredictable from every sensor available at teach
time, including the margins. Under the `SUBSYSTEMS.md` §5 kill criterion the
multi-drive layer stays cut back; no new actuation is justified.

**One component survives, and it is already wired.** The only v_t-ish signal
with measured predictive content is serve-time deficit (−margin / low r),
and its actuation — deficit-first queue priority — is precisely the trigger
arm's lowest-r-first scheduler, already measured as the strongest memory
scheduler in the teach series (`TEACH_TRIGGER_REPAIR.md`, and the aged-v1/v2
negative arms). E1's "curiosity bumps queue priority for measured gaps" has
therefore arrived via the scheduler lane, not the drives lane. Proposed
bookkeeping: mark E1's functional pass as *satisfied-by-existing-scheduler*
rather than building a second, drive-shaped implementation of the same
formula.

**If v_t ever goes in the loop** (sketch only, DO NOT WIRE): drive-modulated
teach-batch composition = novelty-weighted replay, i.e. replay share biased
toward high-deficit items and filler biased away from items freshly damaged
(interference relief). That is the trigger scheduler with a continuous
priority instead of a hinge threshold — its plausible value over the hinge
is exactly what the failed aged-v1/v2 arms say is *not* there (both
under-delivered vs the plain trigger arm). The burden of proof for anything
beyond deficit-first is on a future sensor with information the margins
don't carry (the router's true reject-option mass, the documented E0 gap, is
the candidate).

**Stated limits.** Small n everywhere (34/38/42 rows); the four sft6 runs
share a bit-identical teach phase (de-duplicated, but later sessions remain
arm-correlated); poison detection is partly tautological (gate and sensor
share statistics); the chat rubric is 2 probes — noise. No regression/fit
was attempted beyond rank statistics: with n this small, a multivariate
model would be numerology.

## Reproduction

```bash
python3 lab/subsystems_e1.py            # tables to stdout
python3 lab/subsystems_e1.py --csv /tmp/e1_csv   # optional per-row CSVs
```

Reads: teach-run summaries/session logs/batch stats under `lab/teach_*` and
`lab/imported/vk_sft_r3/_evals/` (imported evals read-only, never committed).
Writes nothing unless `--csv` is given.
