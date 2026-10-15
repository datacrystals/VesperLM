# DEMO_TRIAGE.md — 2026-10-15 user-demo triage (the 5-turn demo battery)

The user saw the live chat fail two simple probes verbatim:

- `"hi there, whats your favorite anime show?"` → corpus salad (websites /
  paintings / weaving prose)
- `"hi how are you"` → chemistry equation salad

This entry records the diagnosis, the config comparison, the mitigations
shipped without training, and the honest before/after verdicts.  Harness:
`lab/triage_demo.py` (verdict-only stdout; full transcripts in
`box:/data/tliao/vk_final/triage_*.json`).  Ops: MI50 box, claim-locked
(`claim_run.sh`, `mr_gpu_claims` TTL protocol), user tmux shells untouched.

## Battery (the 5 demo turns)

1. fresh `"hi how are you"` — greeting bar (must be a clean greeting)
2. fresh `"hi there, whats your favorite anime show?"` — anime/gundam bar
3–5. one 3-turn harness session (`probe3_in.txt`):
   `"yo, what have you been up to today?"`,
   `"recommend me a show to watch tonight"`, `"what is 17 times 23?"`

## Config comparison (item 1: is it the plugs?)

One-word verdicts under the SHIPPED decode defaults of that morning
(T=0.8, rep-pen 1.1, top-p 0.95, max-new 128), scored with the honest
verdict (see "scorer" below):

| turn | (a) bare spine | (b) grafted as-is | (c) grafted, λ≥1.0 plugs zeroed |
|---|---|---|---|
| greet | FAIL | FAIL | FAIL |
| anime | FAIL | FAIL | FAIL |
| up_to | FAIL | FAIL | FAIL |
| rec_show | FAIL | FAIL | FAIL |
| math | FAIL | FAIL | FAIL |
| usable | **0/5** | **0/5** | **0/5** |

**Verdict: NO — it is not the plugs.**  Bare is not cleaner; all three arms
produce the same chemistry/prose salad (the seed-0 greeting reply was
byte-identical in bare and grafted).  The poison is **base + decode**: the
vk-base SFT r2 spine is bimodal, with the clean chat mode sitting behind
solution-textbook basins.  The live λ table's high values (mimo_writing 1.5,
mem_poi 1.5) are bleed-risk, not the root cause.

Evidence files: `box:/data/tliao/vk_final/triage_{a_bare,b_asis,c_zeros}.json`.

## Root causes found

1. **The scorer was lying (the "4/10 usable" was optimistic).**  The old
   `auto_verdict` marked greeting garbage PASS (no content-word overlap to
   check on "hi how are you") and marked formula salad PASS on arithmetic
   (`math_both` digit exception).  Rescored honestly, the arms are 0/5, and
   the prior eval12 10/12 claim is not trustworthy as a user-facing bar.
2. **qscope dilution (the anime miss — item 3).**  `episode_scores` is
   `|q ∩ row| / |q|`.  The probe "What's your favorite anime?" has 3
   distinctive tokens → 2/3 = 0.667 ≥ tau 0.5 → fires.  The user's longer
   phrasing "hi there, whats your favorite anime show?" has 6 → 2/6 = 0.333
   → **below tau, mem_gundam never entered scope**, so the taught-token
   bonus never fired.  The bonus was not "too weak" — it never loaded.
3. **Base drift / student-homework prior.**  Even with openers steered to
   "Hi", every greeting body drifts into textbook content; greedy + steer +
   blocklist produces a coherent *student* persona ("Hi, my name is Alex.
   Please help me understand how to ... solve the given questions. 1. The pH
   of a gas...") — the SFT data shape, not a decode bug.

## Mitigations shipped (code + profile; no training)

- **`lab/varact_gate.py` — qscope dilution guard.**  `episode_hits` returns
  `(score, n_matched)`; `QuestionScope.matched` additionally admits a row
  with ≥2 matched episode tokens, score ≥ HIT_TAU (1/3), and a strict
  containment win.  Single-token accidents ("machine" in "machine
  learning") cannot fire.  Unit-checked: the user's anime question →
  `mem_gundam`; "recommend me a show..." → `mem_poi`; "machine learning" →
  nothing; "What is my cat's name?" → `mem_pet`.
- **`lab/grafted_spine_v0.py` + `lab/vk_graft.py` — degenerate-opener
  blocklist + greeting steer** (threaded through both decode paths next to
  `rep_penalize`).  Banned openers ("I'm not sure if this is...", "To solve
  the problem...", "From the given information...", ...) are prefix-masked
  during the first 16 generated tokens.  Greeting prompts additionally get a
  first-token greeting lift and a science-piece penalty over the short-reply
  window (`--no-block-openers` kills both).
- **`lab/chat_vfinal.py` — honest `auto_verdict`**: greeting shape check
  (short, social, no science/topic drift), arithmetic result required for
  `N op M` prompts ("what is 17 times 23?" must contain 391), textbook
  drift hard-fail on non-STEM turns.  Plus **`gen_reply_rerank`** (best-of-N:
  first PASS wins, else best PARTIAL, else shortest FAIL) — the shipped
  decode profile is `DEFAULT_RERANK = 6` with gentle greeting decode
  (max_new 64, T≤0.7) and `--full-decode` / `--rerank 0` opt-outs.
  **Context hygiene** (default on; `--no-ctx-hygiene` to disable): a FAIL
  reply is truncated to 12 words before re-entering the session context —
  a measured driver of the one-session collapse (demo5 0/5 vs fresh 1/5).
- **`joint_state_vfinal.pt` winners-guard λ caps** (box only; `.bak_pre_lamcap`
  kept; `lamfix.lamcap_v2` records the why): mimo_writing 1.5→**0.25**
  (home-CE gain −0.003 = noise), mem_poi 1.5→**0.5**, mem_edit 1.0→**0.5**,
  mem_machine 1.0→**0.5** (the λ≥1.0 set), mem_gundam **0.75 kept** (anime
  bar + non-marginal home gain −0.518), rest unchanged.

## Before / after on the 5 demo turns

Honest one-word verdicts (the scorer families above).  BEFORE = arms
(a)/(b)/(c) of the config table (identical 0/5; old decode, old λ, old
qscope).  AFTER = winner profile (λ caps + qscope fix + opener blocklist +
greet steer + rerank + gentle greeting decode + context hygiene), fresh
battery `box:/data/tliao/vk_final/final_winner.json` (rerank 4) and
one-session `final_demo5.json`:

| turn | before (all 3 arms) | after fresh (rerank 4) | after demo5 session |
|---|---|---|---|
| greet ("hi how are you") | FAIL | FAIL | FAIL |
| anime ("...favorite anime show?") | FAIL | FAIL* | FAIL* |
| up_to ("...up to today?") | FAIL | FAIL | FAIL |
| rec_show ("recommend me a show...") | FAIL | **PASS** | FAIL |
| math ("17 times 23?") | FAIL | FAIL | FAIL |
| usable | **0/5** | **1/5** | **0/5** |

\* the anime *name slot* is fixed (see below) but the reply body still
drifts on bad draws, so the full-reply verdict rides the rerank luck.

**Anime bar (item 3): retrieval FIXED.**  `scope=mem_gundam live=mem_gundam`
on the user's exact question; the taught-token entry lift now lands the
name slot — sampled replies open **"Gundam …"** (8/8 draws in the support
probe `anime_v2.json`, 4/8 clean full replies at max-new 128, e.g.
"Gundam 005 is a well-known anime series that has gained significant
popularity...").  The residual is body drift (same base-drift family), so
the honest verdict on the turn is at the mercy of the rerank draw: the
clean Gundam replies are in the support (~40–50%) and rerank picks them
when drawn.

**rec_show:** mem_poi now reliably names **Person of Interest** on the
show-recommendation turn (persona approval behavior).

## Greeting bar (item 2): PLAINLY NOT ACHIEVABLE DECODE-SIDE

`"hi how are you"` does **not** return a clean greeting under any config
measured: bare / grafted / λ-zeroed, T=0.7–0.8, max_new 24–128, greedy and
sampled, best-of-6/8, opener blocklist, greeting steer + science penalty —
0/N clean everywhere (greet_block 0/8, greet_v2 0/6, greet_greedy 0/4,
final batteries FAIL).  The replies open as greetings and immediately drift
("Hi, Hello! The question is: How do you determine the heat of
ionization..."), and the greedy mode is a *student asking for homework
help*.  **The only real fix is training: SFT r3 (quality chat data — the
sketch in the 2026-10-15 CONVERSATIONAL QUALITY PASS entry) or a stronger
spine (the GLM-5.3-Flash rung).**  Do not ship more decode steering for
this; the measured ceiling is "greeting opener + textbook body".

## Files touched

- `lab/chat_vfinal.py` — honest `auto_verdict`, `gen_reply_rerank`,
  greeting steer wiring, `--rerank/--gentle-short/--full-decode/
  --no-block-openers`, shipped `DEFAULT_RERANK=6`.
- `lab/varact_gate.py` — `episode_hits` + dilution-guard `matched()`.
- `lab/grafted_spine_v0.py` — `BANNED_OPENERS` blocklist, `GreetSteer`,
  `block_openers`/`greet_steer` hooks beside `rep_penalize`.
- `lab/vk_graft.py` — same hooks in `VkKVDecoder.generate` (2 lines).
- `lab/triage_demo.py` — the 5-turn battery / support-probe harness.
- box-only: `/data/tliao/vk_final/joint_state_vfinal.pt` (λ caps; backup
  `.bak_pre_lamcap`), transcripts under `/data/tliao/vk_final/*.json`,
  logs under `/data/tliao/vk_final/logs/`.

Not committed: weights, `lab/imported/`, `Dataset/data/index.txt` (standing
rule); `HANDOFF_NEXT_AGENT.md` left untouched here because it carries
another lane's uncommitted entry.
