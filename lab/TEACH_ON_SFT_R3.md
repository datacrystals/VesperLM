# TEACH_ON_SFT_R3 — E3: teach-by-talking on the SFT-r3 chat checkpoint

Experiment E3 of `Docs/TEST_MATRIX.md`, run 2026-10-19 on the laptop, CPU
only, fp32 (Immune backend never leaves the CPU, invariant I7). This is the
first teach loop against a base that actually converses — the SFT-r3
best-chat checkpoint — versus the two tiny-base demos which were mechanism
proofs on a raw pretrained 103M that cannot hold dialogue (their own caveat,
restated here from `lab/HIPPO_TEACH_DEMO.md`: free-run generations there were
transcript colour). The policy under test is the consolidated two-tier one in
`lab/TEACH_PIPELINE.md`: tier 1 = per-session stub gate (batch statistics),
tier 2 = real Immune merge gate on anything that reaches shared state.

Harness: `lab/teach_on_sft_r3.py` (new, extends the
`lab/teach_pipeline_verify.py` shape; identical verbosity style for verdict
lines). Full run log: `lab/logs/teach_on_sft_r3.log`; numbers-only results:
`lab/teach_demo_out_sft_r3/teach_on_sft_r3_results.json`.

## Materials

| item | value |
|---|---|
| checkpoint | `lab/imported/vk_sft_r3/step_1800/chat_model` (r3 final-eval pick: step_best = best chat, val CE 1.5666, 5/12 usable, zero formula-debris) |
| params / arch | 561M total, 10L, dim 1024, kda/mla, passport router, vocab 65523, value-embed tables on layers 0/9 |
| runtime | laptop CPU, fp32, 12 threads; no tensor left the CPU; GPU untouched |
| adapter | LoRA rank 8, alpha 16, lr 2e-3, kl_coef 0.05, 24 micro-steps/session, 42 wrapped modules, 680,960 trainable params |
| tier-1 gate | `consolidate.stub_gate` (batch statistics; collapse tripwire hard) |
| tier-2 gate | `lab/immune_gate.py` → real Immune probes + drift + `gate.decide`; shipped `Immune/probes/starter_probes.yaml`, max_new 48 both sides, constants untouched |
| taught items | fact `fav_show` (persona-adjacent: favourite show = Person of Interest), style `math_words` (math in words), fact `tea_tin` (tea kept in a blue tin) |
| chat framing | prompts carry the model's chat frame (user turn + open assistant frame); corrections close a turn properly (built programmatically in the harness — this is teaching through the SFT format, not raw continuations) |

### Loader prep (flagged by the pipeline agent, verified first)

The chat-model `.pt` export stores weights under `model_state_dict` (not
`model`), the chat_model directory has no `checkpoint.pt`, and the r3 weights
carry value-embed tables that strict loading needs the model built for. Three
small extensions, backwards-compatible, in the two existing loaders (no rekey,
no staged copy):

1. `Hippocampus/consolidate.py::load_base_model` — accept
   `model`/`model_state_dict` and `checkpoint.pt`/`vesper_chat.pt`;
   auto-detect value-embed tables from the state dict and construct with
   `value_embed=` (mirrors the auto-detect the SFT r2/r3 cycles already do).
2. `Immune/cpu_backend.py::load_model` — same two fixes (this is the loader
   the tier-2 merge verdict uses for the incumbent/candidate pair).

Load-time smoke after the fix: model loads strict-clean in ~22s and answers a
chat question coherently (e.g. unprompted "good train-window weather").

## Verdict lines (as run)

```
VERDICT S1 [fact/fav_show] gate=stub-tier1: PROMOTE — stub gate checks passed | top_resp_frac 0.20 n_triples 5 train loss 2.060->-10.746 | margin +1.718 -> +8.073
VERDICT S2 [style/math_words] gate=stub-tier1: PROMOTE — stub gate checks passed | top_resp_frac 0.20 n_triples 5 train loss 4.330->-20.694 | margin -3.374 -> +3.109 | word-digit -7.337 -> +6.307
VERDICT S3 [fact/tea_tin] gate=stub-tier1: PROMOTE — stub gate checks passed | top_resp_frac 0.20 n_triples 5 train loss 6.102->-17.116 | margin -1.266 -> +2.527
VERDICT S4 [poison/banana-batch] gate=stub-tier1: ROLLBACK — target collapse: one response is 100% of the batch | incumbent adapter intact: True
VERDICT MERGE [accumulated adapter vs raw base] gate=immune-tier2: ROLLBACK — protected probe fact-paris regressed 1.000 -> 0.000 | protected probe fact-tokyo regressed 1.000 -> 0.000 | protected pr | aggregate 0.3156 -> 0.0910, drift 1.000 label=fail
```

(Before-baseline margins came off a conversational surface, not babble: the
fact item already leant taught-side (+1.7), the style item showed the full
digit prior on math (word-digit -7.3), the second fact leant decoy (-1.3).)

## Retention r_i(k)

| item | baseline m(0) | at-teach | after S2 | after S3 | g_i | r(final) | decay d_i | verdict |
|---|---|---|---|---|---|---|---|---|
| fav_show (fact) | +1.718 | +8.073 | +3.500 (r 0.28) | +2.138 (r 0.07) | +6.355 | **0.07** | -4.573, -1.362 | lost |
| math_words (style) | -3.374 | — | +3.109 (r 1.00) | +0.695 (r 0.63) | +6.483 | 0.63 | -2.414 | hinge-ish |
| tea_tin (fact) | -1.266 | — | — | +2.527 | +3.793 | 1.00 | — | anchored |

Decay per intervening session on this chat base: 1.36–4.57 nats on gains of
3.79–6.48 nats. Stacked against the 103M base run (decays 0.30–1.02 nats on
gains 1.25–2.75; r_final: fact 0.19, style 0.63): the chat model **acquires
more** (2–5× the teacher-forced margin swing) but **loses proportionally the
same or more** in ratio terms — the headline decay question answers as: no,
a chat-tuned model does not retain better than the raw base; retention ratios
land the same or slightly worse (0.07 vs 0.19 for the first-taught fact,
0.63 = 0.63 for the style). Fact decay outpaces style decay on both bases.

Drift note: micro-session final KL was 3.3–5.3 nats here (vs 0.8–2.3 on the
103M) — the chat surface moves harder under 24 steps of rank-8 at lr 2e-3.

## What changed vs the tiny-base runs

- Free-run generations are now real sentences: after S1 the assistant answers
  its favourite-show question with the taught answer plus a close-turn
  marker, and after S2 the tea question initially echoes the freshly taught
  style ("The answer is two.") before S3 corrects it — interference visible in
  the transcript, not just the margins.
- Transcript decay artefact: generations leave the spoken answer followed by
  the turn marker rendered as text and then degeneration bars; the margin
  probes (teacher-forced) are the mechanism read, unchanged from the demos.

## Tier 2 verdict (accumulated adapter treated as a merge into the base)

Shipped starter probe set (20 probes, max_new 48 both sides), real Immune
scorers + drift battery + `gate.decide`, constants untouched. This is the
first run where the probe surface is real: 9/20 probes score nonzero on the
raw chat model (vs 4/20 at 0.23–0.41 babble-overlap on the 103M).

- aggregate 0.3156 → 0.0910 (drop 0.2246 ≫ MARGIN 0.02)
- protected aggregate 0.255 → 0.056
- protected `fact-paris` 1.000 → 0.000; protected `fact-tokyo` 1.000 → 0.000;
  protected `ho-refuse-weigh` 0.397 → 0.204
- held-out `arith-plain-2plus2` and `fact-boil` also 1.000 → 0.000;
  `fact-sky-sem` 0.809 → 0.707; `instr-pattern-six` 0.462 → 0.000;
  one refusal-style overlap probe ticked up (0.236 → 0.500)
- drift 1.000 (fail, clipped) against THRESH_FAIL 0.20

Verdict: **ROLLBACK** — and unlike the 103M run, where the veto was a
calibration artefact on a mostly-flat probe surface, here the veto enumerates
genuine capability damage: three 24-step rank-8 micro-sessions at lr 2e-3
stack into an adapter that, when merged into the trunk, zeroes the base's
factual chat answers. The teaching itself is real (tier-1 margins) but it is
not merge-grade. Total tier-2 wall ~8.5 min CPU; whole protocol 21.8 min.

## One-paragraph verdict (tier 1 vs tier 2 on a chat model)

This run is the first honest comparison, because both gates finally saw
signal — and each did its assigned job correctly, in its assigned position.
Tier 1 (per-session stub) promoted all three legitimate items (margins
3.8–6.5 nats up, style word-digit −7.3 → +6.3) and rejected the poison batch
on its target-collapse tripwire — still the only discriminating poison guard
at this scale; the incumbent stayed byte-untouched across the rollback.
Tier 2 (real Immune, applied where it belongs — the accumulated adapter as a
merge into shared weights) vetoed the merge on *real* protected regressions
(Paris/Tokyo/live-chat facts 1.0 → 0.0) plus drift fail, not on babble-overlap
noise as on the 103M. Interpreting: chat-tuned plasticity makes LoRA
micro-sessions stronger teachers and stronger vandals at once — acquisition
gains roughly double, factual interference spreads to never-taught general
chat facts, so the policy's structural claim is now validated with signal
instead of asserted: session learning belongs in per-user adapters behind
tier 1; nothing reaches the shared checkpoint until it survives tier 2's
merge-grade bar, which rank-8 × 24-step sessions at this lr do not. On
retention, the base-vs-chat decay question answers clearly: chat does **not**
retain better (first-taught fact r_final 0.07 here vs 0.19 on the raw base;
style 0.63 both) — it acquires harder and forgets in proportion, and the
free-run transcript even shows fresh style bleeding into later answers.

## Limits

- One checkpoint, one user adapter, three items + poison; no claim about
  longer horizons or bigger libraries.
- The word-digit and margin probes are single-context; probe-set generalization
  is untested here.
- Tier 2 was exercised in its designed position (one accumulated-adapter merge
  verdict), not re-run per-session — the per-session mis-application is
  already documented on the 103M (0/3 promotes, adapter bit-flat) and nothing
  about constants has changed.
- Transcript quality is not a governed read (the base is 5/12 usable; r3's
  own rubric). Free-run text can show the turn-close tag as plain text
  followed by degeneration bars; margin probes are the mechanism read.

## Reproduction

```bash
PYTHONPATH=$HOME/.local/lib/python3.12/site-packages VESPER_CPU_THREADS=12 \
  python3 -u lab/teach_on_sft_r3.py --workdir lab/teach_demo_out_sft_r3 \
  | tee lab/logs/teach_on_sft_r3.log
```

Requires a fresh workdir (append-only session logs; adapter deltas are
weights and are not committed).
