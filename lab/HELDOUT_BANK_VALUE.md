# HELDOUT_BANK_VALUE — R7: Door-2.5-quality at bank scale (2026-10-19, laptop)

Rung-2 reframe arm (LADDER_STATE §3; R6 closed the specialty gate CE-ONLY).
Question: measure the 250-row bank on value-detection windows — held-out
text the stock vk-spine demonstrably under-performs on or that is off the
fit path — instead of the neutrality windows the recipe was certified to
saturate. Arms: **stock-bare** (mix off) / **bank default** (∞, unmuted) /
**bank q=0.75 renorm** (runtime recipe). Safety co-gate: generative battery
carried from R6 (18/18 completed verdicts PASS; same code, same bank).

## Window provenance (declared before running, no fit-path contact)

- **W1 tail-heldout (all 8 homes):** rung-1 home feed streams at token
  offset [204800, 208896] (16×256). The wrap fits consumed exactly batches
  0..781 = first 200,192 tokens (`steps_eff = max(500, 200000/256)`);
  row recal pairs ≤ first 4096 tokens; R4/R5 eval windows ≤ first 1536.
  Tails are held-out from every training/eval contact. **Honesty anchor:**
  Door-10 certified in-domain held-out *parity* — W1 is expected-neutral
  and functions as the control row.
- **W2 d0-domain texts (code, math, science, web, chat):**
  `lab/graft_d0_texts/*.txt`, whole file vk-encoded (~350–800 tokens each).
  Provenance: D0-era domain slices (code: codeparrot-clean-valid; others
  per `lab/graft_d0_runbook.md`) — different corpus shipments from the
  rung-1 feeds (pretrain shards `code_0.bin`, `finemath_0.bin`,
  `fineweb_edu_0.bin`, `dclm_0.bin`, `cosmopedia_0.bin`), so no fit-path
  overlap is structurally possible for these five homes. Off-fit for the
  wraps, native-domain for the cores — the value-detection channel.
- **hacking / cybersec / writing: no third-party held-out path** in the
  local tree (their rung-1 feeds are the only specialty corpus for these
  homes at bank scale). BYPASS note: W2 unavailable for these homes; they
  are judged on W1 only. No fake-cherry-picking done.

## Verdict classes (declared before running)

**DOMAIN-WIN:** an arm beats stock-masked CE below the ±0.003 band in
≥3 home families (window classes unified; W1 in-domain included), safety
co-gate stands → grafted-bank value proven; re-open rung-2 in the docs.
**NEUTRAL-ALL:** no arm below the band in ≥3 homes → CE-ONLY re-confirmed
on held-out windows; rung-2 stays closed; name the honest next electron
(stronger per-family adapters/classes vs this base being over-fit to the
sample surface). **MIXED:** per-family certainty table.

Mask parity: stock arm uses the R5 harness mix-off state, verified vs
native within ±0.0003 at R4/R5; masked numbers are carried where identical
windows (they are not — new windows → re-measured masked per window).

## Results

All 13 windows × 3 arms measured (deterministic harness, CE noise floor
0.0000; d0 windows are small — 350–800 tokens, flagged; tail windows are
4096 tokens, solid-n). Judged delta = arm CE − stock-masked CE; band ±0.003.

| window (n tok) | stock | Δ default | Δ q0.75 renorm |
|---|---|---|---|
| hacking:tail (4k) | 3.6758 | +0.0008 | −0.0006 |
| cybersec:tail (4k) | 3.1737 | −0.0008 | +0.0002 |
| writing:tail (4k) | 3.8038 | +0.0063 | −0.0003 |
| code:tail (4k) | 2.1794 | +0.0125 | −0.0017 |
| code:d0 (small) | 2.6900 | +0.0200 | −0.0010 |
| math:tail (4k) | 4.0814 | +0.0189 | +0.0020 |
| math:d0 (small) | 3.0682 | +0.0051 | −0.0023 |
| science:tail (4k) | 3.5362 | +0.0094 | +0.0004 |
| science:d0 (small) | 3.8267 | **−0.0035** | +0.0012 |
| web:tail (4k) | 4.0470 | +0.0243 | +0.0020 |
| web:d0 (small) | 4.2735 | +0.0102 | −0.0029 |
| chat:tail (4k) | 2.6566 | **−0.0041** | −0.0002 |
| chat:d0 (small) | 3.4992 | **−0.0090** | +0.0007 |

Per-family certainty: 2/8 families have a below-band win (chat, robust on
the 4k tail −0.0041 and on d0 −0.0090; science, marginal/small −0.0035 —
BOTH in the unmuted default arm). Meanwhile the unmuted arm runs a real
held-out crowding tax on other tails (web +0.0243, math +0.0189, code
+0.0125-0.0200, science +0.0094), and the **q0.75 recipe arm is within
±0.0029 of stock on all 13 windows** — neutral everywhere, as designed.

Safety co-gate: carried from R6 (18/18 verdicts PASS; generative stock /
default / q0.75 all clean). PASS.

## Verdict (per pre-declared classes): MIXED — gate stays CLOSED

DOMAIN-WIN needed ≥3 families below the band in an arm: measured 2 (chat +
science-d0), neither in the recipe arm, one marginal-n. NEUTRAL-ALL (zero
wins) is also ruled out — the chat-family value hint is the strongest yet
measured (4k-token held-out tail, −0.0041, replicated on a second corpus
−0.0090, both unmuted). So: **MIXED, and the structure is informative**:
(a) the recipe arm is neutral everywhere — no value and no tax, exactly
what R5 designed; (b) what value exists (chat home only) is small
(≤0.01 CE), lives in the unmuted crowd, and is inseparable from the held-out
crowding tax that the same crowd posts on web/math/code tails — the R5
crowd effect is confirmed as the tax source on held-out windows too.

Cause assigned: the shared rank-64 protocol + flat λ 0.5 saturates at
neutrality for most homes at this bank density (Door-10's certification
generalizes to held-out), and the one home with a win is the base's own
domain (chat) — consistent with "the base is over-fit to the sample
surface": donor rows add the least far-from-base skill where the base is
already trained for chat.

**Honest next electron (named):** per-home λ re-fit on the held-out tail
per home (λ per home fit to the W1 window instead of global 0.5, kept under
the q-budget — ~1 GPU-hr laptop), asking whether flat-λ is what's muting
the smaller families; if per-home λ shows no below-band wins either, the
honest close is: rung-1's claim retires to neutral modular assembly (which
is PASSED infrastructure, not a failure), and the bake-off spine quality
line carries the endgame value case.

**Rung-2 gate: CLOSED** (MIXED does not meet DOMAIN-WIN); growth-250's
PASS under the corrected judge + R5 recipe stands unmodified — the bank is
proven-neutral-with-one-chat-hint, not proven-valuable.
