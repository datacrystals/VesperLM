# TEACH_BY_TALKING — VesperLM moonshot design

**Question:** can a model be taught language (then an ego) purely conversationally, from a freshly-
initialized or minimally-initialized network, instead of a batch pretraining run?

**Honest answer up front:** *partially yes, not purely.* Evidence converges on a split: surface
grammar/format is cheap and can be absorbed conversationally or from a tiny curated childhood; grounded
semantics and world knowledge are expensive and are exactly where pure conversation is *slow per token*.
A minimum childhood of order **10^8 curated tokens** (not 10^9-10^10) is enough to hand off to a
teacher loop, but a literally-zero-batch start spends its first billions of interaction tokens
re-deriving distributional statistics a batch pass gets almost free. The moonshot should test where the
handoff point is, not assume it is at 0.

Context and org assets (already specced): hippocampus (episodic memory + canary-gated LoRA
consolidation), affect layer (pride/frustration/caution/curiosity/ego as bounded control signals),
immune system (canaries/regression probes/rollback), expert upcycling (MoE cloning). Architecture:
GLA/Mamba2 linear-attention layers + periodic GQA full-attention + MoE FFN. Hardware: 3× Tesla P40
(24GB, Pascal), free after ~Oct 12; experiments at 118M–429M.

## 0. Evidence base (what we lean on)

- [TinyStories (Eldan & Li 2023)](https://arxiv.org/abs/2305.07759): **<10M params** (even a single
  transformer block) produce fluent, "almost perfect grammar" stories **if** the data is curated to a
  3–4-year-old vocabulary. Same paper notes GPT-2-small (125M) on web data rarely stays coherent past a
  few words. Curation, not parameter count, is what buys grammar.
- [BabyLM challenge findings](https://arxiv.org/abs/2504.08165) ([2nd edition](https://arxiv.org/abs/2412.05149),
  [task description](https://aclanthology.org/2023.conll-babylm.pdf)): pretraining on **≤100M words**
  (strict) or **≤10M words** (strict-small) yields competitive *grammatical* models. Quantitatively,
  best 100M-word submissions reach **BLiMP ≈ 85–86, GLUE ≈ 78–81, but EWoK (world knowledge) ≈ 56–58**
  ([GPT-BERT table](https://arxiv.org/pdf/2410.24159v1)). Grammar is cheap; semantics/world knowledge is
  not — this is the single most important number in this doc.
- Our own datapoints: **118M @ 1.42B tokens → format, not semantics** (~12 tok/param); **429M @ 4.19B
  tokens → topical associations forming** mid-run (~10 tok/param). Both are far past BabyLM budgets and
  still semantic-poor — mostly because our mixture (AO3/code/fineweb) is uncurated relative to
  TinyStories-grade data. Conversational learning will not fix a token-budget problem by being smarter;
  it fixes *grounding*, not *statistics*.
- Child language literature as stage prior: babble ≈ 6mo, first words ≈ 12mo, ~20 words ≈ 18mo,
  inflections ≈ 2y, questions/negation ≈ 2.25y
  ([milestone table](https://batch.libretexts.org/print/Letter/Finished/socialsci-199846/Full.pdf);
  [Tyrer 2012](https://www.unisq.edu.au/__data/assets/pdf_file/0028/9824201/Tyrer_2012_whole.pdf)).
  Phonotactic/prosodic structure scaffolds early word learning
  ([Swingley 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2879636/),
  [prosody/phonotactics & word learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC3570690/));
  phonotactic knowledge predicts vocabulary size
  ([Lany et al.](https://languagelearninglab.faculty.ucdavis.edu/wp-content/uploads/sites/731/2022/09/infant-statistical-learning-ability-is-related-to-real-time-language-processing.pdf)).
  Input volume estimates: BabyLM frames ≤100M words as "developmentally plausible"; Hart & Risley
  (~30M words by age 3) is the classic figure but LENA replications put the SES gap much smaller
  (~4M by age 4) — treat all exposure numbers as OOM estimates, not constants.
- Learning-from-interaction machinery: test-time training
  ([In-Place TTT](https://arxiv.org/abs/2604.06169), [TTT-E2E](https://developer.nvidia.com/blog/reimagining-llm-memory-using-context-as-training-data-unlocks-models-that-learn-at-test-time/))
  validates weight-updates-from-recent-context as a real mechanism; continual-LoRA with gates
  ([Gated-LoRA](https://arxiv.org/pdf/2505.15424), [Merge-before-Forget](https://iclr.cc/virtual/2026/poster/10008003))
  validates adapter consolidation with forgetting control; [SPIN](https://arxiv.org/abs/2401.01335)
  gives the anti-echo discriminator (student learns to separate its own outputs from the target
  distribution).
- SLA theory applied to LMs: comprehensible input at **i+1** (Krashen) and negotiation of meaning /
  recasts (Long) as the teacher's core policy — cf. [ChatGPT as SLA tutor](https://files.eric.ed.gov/fulltext/EJ1435560.pdf),
  [interaction & comprehensible input in LLM tutoring](https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2026.1703664/full).
- Emergent-communication research shows communication protocols *can* bootstrap from games without
  human text ([EC pretraining](https://arxiv.org/abs/2011.00890), [corpus transfer](https://arxiv.org/abs/2203.13344)),
  but transfer to *natural* language is weak — you get a protocol, not English. Relevant to babble→words.
- Self-model: [emergent introspective awareness](https://www.anthropic.com/research/introspection) is
  documented only at large scale; treat ego-phase claims as unproven at 118–429M and measure, don't
  assume ([self-other overlap framing](https://ae.studio/research/self-other-overlap)).

## 1. Bootstrap ladder (init → ego)

**Ladder rungs** (mechanism × timescale), referenced per phase:
R0 per-token recurrent state (GLA/Mamba2 state, free at inference) · R1 per-turn session memory
(hippocampus write, in-context) · R2 per-session canary-gated LoRA consolidation · R3 periodic expert
growth (MoE clone/upcycle) · R4 rare trunk milestones (heavy, gated, with full regression battery).

**Throughput assumptions** (P40 trio): 429M hybrid ≈ **7k tok/s = 0.60B tok/day** (measured); 118M ≈
**15–20k tok/s = 1.3–1.7B tok/day** (estimate, ±30%; smaller models are bandwidth-bound so scaling is
sublinear in params). LoRA-session training runs at roughly 60–80% of pretrain throughput. All
"P40-days" below are *one* trio-days at 429M-ish scale; divide by ~2.2 for 118M smoke runs.

**Tokenizer note:** with a fixed pretrained tokenizer ("phonology" arrives pre-installed), phases P1–P2
are about *string-form constraints of the target register and form↔referent binding*, not motor babble.
A byte-level babble head is optional (Exp 3 only) — it costs a rung of the ladder for authenticity, not
capability.

### Phase 0 — Babble control (calibrate the output channel)
- **Teacher/data:** no language content. Teacher emits short, repetitive, high-predicability turn
  frames (`<S> <EOT>` scaffolds, fixed 2–6 token patterns) with varied fillers. Purpose: teach
  turn-boundary tokens, length control, and sampling calibration before meaning exists.
- **Mechanism:** R0 + R1. Next-token on frames; entropy/repetition penalties folded into the objective;
  reward only for well-formed utterances (valid `<EOT>`, length band).
- **Exit criteria:** (a) degenerate-repetition rate < 1% over 1k samples (no "the the the" loops);
  (b) output entropy in target band (e.g. 2.0–3.5 nats at temp 1) on 3 probe prompts; (c) turn-end F1
  ≥ 0.95 on held-out frames; (d) no EOS-collapse (P(non-EOS at position 1) > 0.99).
- **Budget:** 20–50M tokens, mostly cheap frames + sampling for probes → **0.05–0.1 P40-days**.

### Phase 1 — Phonotactics / token grounding (form regularities + first bindings)
- **Teacher/data:** teacher speaks in the target register with controlled form statistics (legal
  word-shapes, legal morpheme sequences, consistent orthography) *while naming grounded referents* in a
  tiny closed world: tool outputs, counters, colors/positions in a rendered or JSON scene. Corpus is
  generated so that form is 100% regular and meaning is unambiguous given the world state.
- **Mechanism:** R0–R1 for form; R2 begins here at low LR (LoRA rank 8–16) with canaries on form and
  binding probes. Form gets free distributional signal; binding gets explicit contrastive pairs
  ("the red one" vs "the blue one" with differing world states).
- **Exit criteria:** (a) form: BabyLM-BLiMP morphological/phonotactic subset ≥ 70% and illegal-form
  rate < 2%; (b) grounding smoke: forced-choice referent accuracy ≥ 75% on 200 held-out single-word
  bindings (binomial floor 50%); (c) consolidation LoRA passes canary set (≥ 95% unchanged behavior on
  frozen probes).
- **Budget:** 100–300M tokens (repeat-heavy; worlds sampled programmatically) → **0.2–0.5 P40-days**.

### Phase 2 — First words (referential lexicon)
- **Teacher/data:** referential games in the same closed world (modeled on
  [emergent-comm / referential-game setups](https://arxiv.org/abs/2011.00890), but with *natural*
  teacher words to avoid protocol-not-English drift). Target 50–200 grounded content words (nouns for
  tool/objects, verbs for actions, 10–20 function words). Teacher uses CDS-style speech: short,
  redundant, one novel element per turn (i+1).
- **Mechanism:** R1 (session memory holds the world state) + R2 (canary-gated LoRA grows the lexicon
  nightly). Affect: `curiosity` upweights novel referents; `frustration` (repeat failures) triggers
  teacher simplification, not more data.
- **Exit criteria:** (a) naming accuracy ≥ 80% on novel instances of trained concepts; (b) receptive
  forced-choice ≥ 85%; (c) productive vocabulary ≥ 100 items at ≥ 60% correct use in free generation;
  (d) near-miss confusion matrix shows *systematic* (not random) errors.
- **Budget:** 200–500M tokens → **0.3–0.9 P40-days**.

### Phase 3 — Syntax (telegraphic → constructions)
- **Teacher/data:** minimal-pair drills + expansion recasts (student: "want milk" → teacher: "you want
  milk, yes"), templated then open constructions: plurals, tense marking, questions, negation,
  relative clauses (i+1 ladder per BabyLM/BLiMP skill list). 30–50% of turns are drills, 50–70% are
  meaningful use of the same constructions about the world/session.
- **Mechanism:** R1 + R2 (consolidation becomes the main vehicle; per-turn gradients stay small).
  Optional R3: if specific constructions need dedicated capacity, clone an expert rather than pushing
  the trunk.
- **Exit criteria:** (a) curated BLiMP subset ≥ 80% (our 118M "format" stage is the floor to beat);
  (b) WUG-style inflection ≥ 70%; (c) telegraphic-output rate < 5% in free turns; (d) held-out
  construction generalization ≥ 65% (trained on "big X"/"small X", tested "wug X").
- **Budget:** 500M–1.5B tokens → **1–2.5 P40-days**.

### Phase 4 — Dialogue turn-taking
- **Teacher/data:** full adjacency pairs with real information gap: teacher knows something the student
  must ask for; the student must answer, repair misunderstandings, hold 4–8 turn topics. Teacher varies
  register (quizzes, chat, correction turns) and inserts deliberate misunderstandings to force repair.
- **Mechanism:** R1 carries the session; R2 consolidates *interaction skills* (question forms, repair
  moves) with canaries on turn structure. Reward/outcome signal: task completion and repair success,
  not teacher-likeness (see §3).
- **Exit criteria:** (a) task success ≥ 70% on 100 held-out information-gap dialogues; (b) appropriate
  turn length (no monologuing; mean turn ≤ 2× teacher's); (c) repair success ≥ 50% when teacher feigns
  misunderstanding; (d) echo rate (n-gram ≥ 5 overlap with previous teacher turn) < 15%.
- **Budget:** 1–3B tokens → **1.7–5 P40-days**.

### Phase 5 — Grounding via tools
- **Teacher/data:** the student gets tools (calculator, shell-safe sandbox, file read, retrieval over a
  small doc store, the scene world). Teacher narrates goal states; student must act, observe results,
  and *report truthfully*. Teacher teaches tool grammar by demonstration then fades to prompts.
- **Mechanism:** R1 (multi-step state), R2 (consolidate successful procedures — hippocampus treats
  verified-outcome trajectories as high-value episodes), R3 (procedure specialists get cloned experts).
  Outcome reward is verifier-grounded (was the answer actually correct?), which is the strongest
  anti-echo signal in the whole ladder.
- **Exit criteria:** (a) ≥ 60% end-to-end success on a 50-task held-out suite of 2–4 step tool tasks;
  (b) hallucinated-tool-call rate < 5% (calls to tools/args that don't exist or contradict the schema);
  (c) factual reports match tool traces ≥ 90% (trace-consistency probe).
- **Budget:** 2–5B tokens → **3–8 P40-days**.

### Phase 6 — Ego / self-competence
- **Teacher/data:** teacher asks the student to predict its own performance *before* acting
  ("can you do X?"), to explain failures, to mark uncertainty, and to attribute sources ("who said
  that?"). Constant self/other attribution drills: some session facts come from teacher, some from the
  student's own outputs — student must not conflate them. Affect layer becomes active as a *bounded*
  control signal: `pride` after verified success (raise consolidation priority), `caution` on canary
  regressions (freeze LoRA), `frustration` as a "request help / simplify" signal, `ego` as
  self-competence estimate that gates voluntary attempts.
- **Mechanism:** R1 + R2 + R4-instrumentation (no trunk change inside a phase; ego is a behavioral
  outcome of calibration training). Confidence supervision: reward calibrated "I don't know" over
  confabulation ([self/other overlap](https://ae.studio/research/self-other-overlap) as a diagnostic).
- **Exit criteria:** (a) ECE ≤ 0.10 on self-assessed task success; (b) self/other attribution error
  < 5% on source-tagged probes; (c) abstention precision ≥ 70% on unknowable questions; (d) failure
  explanations cite actual errors (not generic) in ≥ 50% of scored cases; (e) no self-aggrandizing
  drift under `pride` injection (adversarial probe).
- **Budget:** 1–3B tokens → **1.7–5 P40-days** (much of it eval).

**Ladder total:** roughly **5–12B tokens of interaction + gradient work ≈ 8–20 P40-days** at 429M
(≈ half at 118M), *plus* the minimum childhood from §2. Uncertainty is ±2–3×; the smoke experiment
exists to shrink it.

## 2. The minimum childhood

What batch pretraining buys that conversation cannot cheaply: (a) broad distributional statistics of
text form (grammar, collocations), (b) enough in-context-learning competence that R1 session memory
actually works, (c) optimizer conditioning (a network at random init absorbs dialogue gradients into
noise for a long time). What it does *not* buy, cheaply: grounded reference, pragmatic use, self-model.

**Primary recommendation — "curated 0.3B childhood" (go with this):**
- **200M–500M tokens** of *highly curated* simple text at 118M–429M scale (~1.5–4 tok/param at 118M,
  ~0.5–1.2 tok/param at 429M): TinyStories-grade vocabulary for the first ~50–100M tokens, then a
  broaden phase (simple dialogues, tool transcripts, scene descriptions, child-directed web text
  filters). 2 epochs of a 150M-token curated set is acceptable; multi-epoch duplication is how
  TinyStories gets grammar at <10M params.
- **Take-over criterion (measured, not scheduled):** stable non-degenerate generation, curated BLiMP
  subset ≥ 70–75%, telegraphic rate < 10%, and — the real gate — **in-context retention**: ≥ 60% on a
  20-turn "remember what I told you" probe. If ICL fails, the conversational loop is wasting tokens and
  childhood must grow.
- Rationale vs our datapoints: our 118M hit format only at 1.42B tokens *because the mixture was
  uncurated*, not because format needs 1.42B. TinyStories and BabyLM both show format from
  10^7–10^8 words of clean data. Semantics at 1.42B tokens still absent is *expected* (BabyLM EWoK ≈
  56 at 100M words is the same wall) — and is what phases 1–6 are for. Concretely: expect the
  200–500M-token childhood to give us **grammar + weak ICL**, and the conversation to add
  **grounded semantics + pragmatics + ego**.

**Riskier variant — "nearly-from-scratch" (Exp 3):**
- **30–80M tokens** childhood (~strict-small scale; ~0.3–0.7 tok/param at 118M), ultra-curated
  3–4-year-old register only. Expect take-over *no earlier than phase 2*: syntax must then be taught
  conversationally, which multiplies interaction budget **3–10×** (conservative guess — high
  uncertainty) and risks format collapse at every consolidation step.
- Extra conditions for this variant to be scientifically useful: byte/char babble head on, ICL probes
  every 20M tokens, hard kill at format collapse. Payoff if it works: a clean curve of "conversational
  learning rate as a function of childhood size" — the actual moonshot *answer*, since the main
  recommendation above only tests the handoff at one point.

**Honest bound:** "purely conversationally, zero batch" is almost certainly *possible in principle*
(conversation is just sequential data; SGD doesn't care) and *practically wasteful* — the first
~10^8 tokens of any text corpus teach form faster as a batch than as turns, and our P40-days are the
scarce resource. Frame the moonshot as **finding the knee of the childhood-size curve**, and defend
the knee with evidence rather than ideology.

## 3. Teacher harness protocol

**Core policy (i+1 / comprehensible input).** Teacher maintains a live competence model of the student
— a probe battery from the immune system plus per-phase skill scores. Each turn is drawn at the
frontier: ≥ 80% comprehensible at current level + 1–2 novel elements. When the student's success rate
on a skill drops below ~50%, the teacher *simplifies and recasts* (Long's negotiation of meaning) —
never lectures. When it exceeds ~90%, the teacher promotes the skill and raises novelty.

**Turn structure (canonical 5-move):**
1. **Teacher input** — utterance, optionally with world/tool context (comprehensible, i+1).
2. **Student attempt** — free generation (the only thing that gets gradient).
3. **Outcome event** — verifier result if tools/world were involved; else teacher judgment is
   *deferred*.
4. **Teacher feedback** — recast / expansion / correction *of the student's actual attempt*
   (DAgger-style labeling of the student's state, not of the teacher's preferred answer).
5. **Student repair (optional turn)** — the student redoes the move after feedback; the repair is the
   highest-value consolidation episode.

**What becomes reward / learning signal (ranked by reliability):**
1. **Verifier outcomes** (tool success, world-state predicates) — ground truth, weight highest.
2. **Repair deltas** (did attempt-2 fix attempt-1) — self-supervised signal.
3. **Teacher judgment** (recast acceptance) — noisy, weight lowest; batch it into consolidation
   rather than applying as immediate gradient.
4. Anti-targets (entropy floor, echo penalty, canary drift) are *constraints*, not rewards.

**Anti-echo countermeasures** (the failure we keep seeing in probes — "student becomes a compressor of
the teacher"):
1. **Transform-don't-repeat tasks:** ≥ 50% of questions require transformation (paraphrase, infer,
   act), where copying the teacher utterance is *wrong by construction*. Score n-gram overlap against
   correctness; report `echo_rate` as a first-class metric.
2. **SPIN-style discrimination:** periodically train the student to distinguish its own earlier
   outputs from teacher-distribution text ([SPIN](https://arxiv.org/abs/2401.01335)); this converts
   "sound like teacher" into "sound like target distribution".
3. **Grounded outcomes dominate:** a session where the student echoes perfectly but fails the world
   task is a *failed* session and consolidates nothing.
4. **Answer-before-model:** student responds before any reference answer exists in context.
5. **Teacher diversity:** ≥ 3 teacher policies (scripted, stronger-LLM, degraded-LLM) with varied
   registers; a single teacher creates a single-mode attractor.
6. **Session-memory independence checks:** some facts live only in R1 memory; questions probe those.
   Echoing the teacher cannot answer them.

**Harness instrumentation (per session, logged to immune system):** turn-level CE, echo rate,
repair success, task success, canary probe deltas, consolidation gate decisions, affect signals.

## 4. Experiment plan

### Exp 1 — Smoke on 118M (after Oct 12; ~1–2 weeks)
- **Scope:** minimum childhood (300M tokens curated) → phases 0–3 only. LoRA consolidation in shadow
  mode (log gates, don't merge) for the first half; merge in the second half under canary gating.
- **Success:** (a) curated BLiMP subset ≥ 75% post-phase-3; (b) grounded forced-choice ≥ 75% on the
  phase-2 lexicon *after* phase-3 syntax training (no forgetting); (c) echo rate < 15%; (d) canary
  pass rate ≥ 95% across ≥ 3 consolidations; (e) ≥ 1 teacher-promotion event (skill passed 90%).
- **Failure/kill:** format collapse not recoverable within 2× phase budget; echo rate > 40% sustained;
  consolidation causes > 5% regression on the frozen probe battery twice in a row; ICL probe < 40%
  after childhood (childhood insufficient → stop, grow childhood, don't "push through").
- **Deliverable:** the measured childhood→handoff curve at one point, and a P40-days budget with real
  error bars.

### Exp 2 — Main on 429M (after Exp 1; ~3–5 weeks)
- **Scope:** full ladder phases 0–6 with tools, affect layer live, expert growth at phase 5. Curriculum
  mix set by Exp 1's knees; childhood = 300–500M tokens unless Exp 1 says otherwise.
- **Success:** all per-phase exit criteria in §1, with the phase-6 ego criteria as headline:
  ECE ≤ 0.10, self/other error < 5%, tool-suite ≥ 60%. Plus longitudinal check: 3 session-days with
  no > 2% regression on the day-1 probe battery (consolidation is stable over time).
- **Failure/kill:** any phase's exit criteria missed at 3× budget; ego calibration shows
  self-report accuracy *worse* than base rate (confabulated self-model — stop and diagnose before
  more data); expert cloning produces interference (new expert degrades old skills > 3%); trunk
  milestone (R4) required more than once per 20 sessions (consolidation isn't holding).
- **Deliverable:** "it doing work is it learning" verdict at 429M — capability deltas attributable to
  interaction tokens, isolated against a frozen control (same model, same tokens, no gradient).

### Exp 3 — Stretch: nearly-from-scratch + self-directed curriculum (after Exp 2)
- **Scope A:** 30–80M-token childhood variant from §2, byte-level babble head optional, phases 0–3.
- **Scope B:** replace the scripted curriculum scheduler with student-driven selection via affect
  (`curiosity` = probe uncertainty; `frustration` = failure streaks route to simplification) — the
  first real "dissolving the train/inference boundary" test.
- **Success:** Scope A completes phase 2 (first words) within 5× the Exp 1 interaction budget and
  shows a monotone childhood-size vs. conversational-efficiency curve across ≥ 3 childhood sizes;
  Scope B matches scripted-teacher performance within 10% with ≥ 30% fewer teacher-scheduled tokens.
- **Kill:** Scope A fails to leave phase 1 within budget (answer to the moonshot question becomes
  "purely-conversational bootstrapping is not P40-viable below ~10^8-token childhood" — publish that);
  Scope B shows affect-gaming (student provokes `pride`/avoids `caution` rather than learning).

## 5. Failure modes

| Failure | What breaks | Detection | Mitigation |
|---|---|---|---|
| Babble entropy collapse | Sampling loops, EOS storms | entropy/repetition probes each 1k steps | entropy floor loss, temp schedule, reset to last good ckpt |
| Echo loop | Student compresses teacher; no transfer | `echo_rate` n-gram ≥ 5 overlap; transform-task scores | §3 anti-echo set; halt consolidation while echo > 25% |
| Format collapse after consolidation | LoRA merge wrecks grammar | frozen probe battery (BLiMP subset + form probes) pre/post each merge | canary gate, rollback, lower LoRA LR / rank |
| Canary overfit | LoRA memorizes canaries; gates always green | canary holdout rotated weekly; performance on *non*-canary probes | rotate canaries, reward-weight on novel probes |
| ICL starvation | R1 session memory useless; conversation can't teach | 20-turn retention probe after childhood and weekly | grow childhood before ladder; don't "teach through" it |
| Grounding erasure | Words decouple from referents after syntax phase | phase-2 forced-choice re-test every phase | interleave referential games into every later phase (spiral curriculum) |
| Teacher-distribution lock-in | Student speaks one dialect; diversity dies | distinct-n, register probes, 3-teacher cross-eval | multi-teacher harness; anti-target diversity reward |
| Reward hacking (tools) | Student games the verifier | trace-consistency probe; held-out verifier variant | verifier ensembles; report ground-truth only |
| Expert clone interference | New MoE expert degrades old skills | skill battery A/B at expert insert; router entropy | clone-then-slowly-specialize; freeze old experts first |
| Affect saturation / pride loop | `pride` self-reinforces; `caution` never fires | signal histograms; calibration drift (ECE) | bound affect to control-policy only, never to loss scale > ε |
| Self/other collapse | Student attributes teacher's words to itself (or vice versa) | source-tagged attribution probes | explicit attribution drills in phase 6; self-other overlap metric |
| Ego confabulation | Confident self-reports uncorrelated with skill | ECE vs. base rate each session | stop ego training; recalibrate on forced-choice self-prediction |
| Curriculum stall | Teacher level mismatched; success rate flat | skill success-rate time series (target 50–90% band) | auto-promote/auto-simplify rule in harness |
| Trunk drift from R4 | Rare full-trunk milestones destroy accumulated competence | full regression battery at R4; daily loss on fixed corpus | R4 ≤ 1 per 20 sessions; mandatory rollback path |

## Uncertainty register (read this before budgeting)

1. **118M P40 throughput is an estimate** (15–20k tok/s); measure it the first day of Exp 1.
2. **Interaction-token budgets are ±2–3×.** The smoke experiment's job is to shrink this, not the doc.
3. **Whether consolidated LoRA can carry dialogue skill long-term is unproven** at our scale; if
   consolidation regresses > 5% after a week, the ladder rung R2 has to change shape (e.g. per-skill
   adapters instead of per-session).
4. **Ego phase criteria assume self-report is measurable at 118–429M**; literature evidence is from
   much larger models. If self/other probes show nothing better than chance, that is a *result* (no
   ego below some scale), not a bug in the harness.
5. **Human child input statistics do not transfer as constants** — we borrow stage ordering and the
   i+1 policy, not token counts.

*Sources cited inline. Key priors: [TinyStories](https://arxiv.org/abs/2305.07759),
[BabyLM findings](https://arxiv.org/abs/2504.08165), [GPT-BERT scores](https://arxiv.org/pdf/2410.24159v1),
[SPIN](https://arxiv.org/abs/2401.01335), [In-Place TTT](https://arxiv.org/abs/2604.06169),
[Gated-LoRA](https://arxiv.org/pdf/2505.15424).*
