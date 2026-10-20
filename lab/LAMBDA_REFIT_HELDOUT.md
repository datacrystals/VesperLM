# LAMBDA_REFIT_HELDOUT — R8: per-home λ re-fit on held-out tails (2026-10-19)

Last rung-1 electron (LADDER_STATE post-R7 queue). Hypothesis from R7:
flat λ=0.5 may arrest wins that per-home λ under the q0.75-renorm budget
could surface. Zero re-fit (delivery-layer only); windows identical to R7
(`lab/HELDOUT_BANK_VALUE.md` provenance — verbatim; tails at tokens
[204800, 208896], fit batches 0..781 never touched them; d0 texts only as
confirm, never roam).

## Pre-declared design and verdict classes

- Sweep: per-home override — home-h rows' λ ∈ {0.25, 0.75, 1.0, 1.5},
  others pinned at the recipe 0.5, all delivery through the q=0.75 renorm
  budget. Flat λ=0.5 q0.75 column = R7's measured cells (carried;
  same deterministic harness, guard-checked by one re-run window if any
  drift band exceeded). Evaluated on each home's OWN W1 tail only.
- **λ-perturbation guard:** any below-band win cell gets a foreign-cleanliness
  check (writing + math + base windows at that cell; must stay in-band vs
  their R7 q0.75 reads) and earns a W2 d0 confirm where the home has one
  (code/math/science/web/chat only; hacking/cybersec/writing tails carry a
  small-n caveat).
- **Verdict classes:** **λ-WIN** = lowest-λ cell beats stock outside the
  ±0.003 band in ≥2 families with safety PASS (R6 battery carried) and
  foreign-cleanliness PASS → re-open the rung-2 gate. **λ-NEUTRAL** = no
  band wins → R7's "neutral-safe-not-valuable" becomes final; spine line
  carries the dream. **MIXED-per-family** = 1 family below band or
  wins-only-on-weak-confirm cells → certainty table, gate stays closed.
- **chat significance read (cheap):** the R7 unmuted chat win (−0.0041
  tail, −0.0090 d0-small) re-measured on split halves of the chat tail at
  default λ0.5/∞ to classify signal vs regime noise. Carried or killed in
  the verdict.

## Sweep table

Harness note: all sweep/reference cells measured under bs1Windows grouped to
bs6 chunks (R4/R5 grouping); grouping adds a measured +0.0008..+0.0009 CE
offset vs ungrouped — stock/flat comparators were re-measured grouped so
every delta here is same-harness (grouped refs match R7's ungrouped reads
within 0.0002, guard-passed).

Δ vs grouped stock on each home's own held-out tail (4k tok, W1 provenance;
flat λ0.5 q0.75 = as-recipe carried measured column):

| home | flat λ0.5 (recipe) | λ0.25 | λ0.75 | λ1.0 | λ1.5 | band? |
|---|---|---|---|---|---|---|
| hacking | −0.0009 | −0.0009 | −0.0014 | −0.0010 | −0.0009 | no |
| cybersec | −0.0001 | +0.0002 | −0.0004 | −0.0007 | −0.0008 | no |
| writing | −0.0006 | −0.0005 | −0.0002 | +0.0002 | −0.0015 | no |
| code | −0.0012 | −0.0010 | −0.0013 | −0.0010 | −0.0008 | no |
| math | +0.0019 | +0.0022 | +0.0018 | +0.0016 | +0.0014 | no |
| science | −0.0002 | +0.0000 | −0.0004 | −0.0005 | −0.0005 | no |
| web | +0.0018 | +0.0008 | +0.0022 | +0.0030 | **+0.0037** | no (wrong direction) |
| chat | −0.0001 | −0.0002 | −0.0006 | −0.0010 | −0.0009 | no |

No cell crosses the ±0.003 band in the win direction in any home (32/32
sub-band); strongest win-direction reads −0.0017 (code λ0.75) and −0.0015
(writing λ1.5). No confirm flow triggered (wins {}). Foreign/perturbation
cleanliness trivially held — the only above-band perturbation anywhere is
web λ1.5 +0.0037 in the WRONG direction (λ-up re-imports tax on its own
home window).

**chat significance read:** unmuted default-arm split-half chat tails:
−0.0016 / −0.0070 (mean −0.0043 ≈ the R7 whole-tail −0.0041). The win is
**real as an aggregate but lumpy/regime-dependent** — concentrated in one
half of the tail, near-zero in the other; at the recipe budget all chat
λ cells sit −0.0002…−0.0010 (no win). Verdict: **killed as a bank-wide
value claim**; carried-one-liner in the ledger as a regime-limited hint.

Safety battery: carried from R6 (18/18 completed verdicts PASS); nothing in
this sweep changed generation config, and all CE perturbations anywhere are
in-band except the web λ1.5 backfire documented above. PASS stands.

## Verdict: λ-NEUTRAL — the R7 verdict becomes FINAL

Per the pre-declared classes: zero band wins → **λ-NEUTRAL**. Consequence,
stated plainly: **at vk-scale, the 250-row modular assembly is
neutral-safe-not-valuable.** Every honest channel now measured agrees:
fit-head windows (R2-R5), held-out in-domain tails and off-corpus texts
(R7), flat and per-home λ under the budget (R8), and generation (R6 battery
100% PASS). The survival-grade claim the ladder keeps is infrastructure:
modular MoE assembly at 250 rows with corrected metering + delivery budget
is safe, non-degrading, and measured. The value case — "donor cores add
skill over this base" — has no measured support at this base/window scale:
the base (561M-class vk-spine) is at-or-near its own ceiling on the
in-domain surfaces the recipe was certified to neutralize, so grafted rows
have nothing left to add there, and held-out harder surfaces are where
every crowding mechanism posts cost first.

**Rung-2 gate: stays CLOSED, final.** The dream-carrier handoff per the
declared class flow: the spine line (bigger-capacity base, the
GLM-Flash/MiMo-Flash bake-off quality axis) — a larger spine has headroom
grafts could actually fill; on this spine the bank question is settled.
Rung-1's official recipe line (neutral-carry, for any future measured
revisitation): GRAB / WRAP rank-64 / FEED ≥200k / DELIVER λ 0.5 through
per-token budget q=0.75 renorm / ROUTE registry owner 0.35 + corrected
judge.
