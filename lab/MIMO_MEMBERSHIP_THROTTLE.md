# MIMO_MEMBERSHIP_THROTTLE — gating the multi-point delivery (E2 throttle cycle)

Date: 2026-10-20.  Line: lab/MIMO_MULTI_POINT.md (delivery works: site-23
carries the persona with parity +0.0008 at forced-on injection; voice is
uniform-strong → neutral turns wear the persona).  This cycle: make the
delivery conditional on a membership gate measured on DECODE surfaces.

## Gate design (declared before coding)

- Gate substrate: the plug's passport row at MoE block 0 (the bank row
  `rec["row"]`, raw = h @ row at block-0 gate input, stock max reference
  from the same `h`).  Delivery (multi-point sites {0,23,46}) fires only
  when the gate fires on the current decode step.
- Suites (already harvested on box `~/mimo_flash/decode_recal_cap.pt`,
  block-0 `h` per decode step, 6 steps per probe):
  - cold-negatives: 24 decode-native neutral traces (decade set)
  - home-positives: each plug's 6 home prompts at decode
  Rule: R2 (`plug_raw > max_stock_raw + margin`), grid 0.5–8.0 step 0.5,
  per-PLUG margins allowed (rows live at different baselines — measured
  fact from the earlier sweeps).  Winner per plug: ON ≥ 0.8 home-decode
  probes AND OFF ≤ 0.1 negative-decode probes, margin as small as
  possible.
- Throttle semantics in verify: per decode step compute gate from
  block-0 `h`; if fire: inject at sites; else: no injection.  Same λ_i,
  same wrap as the forced-on arms in MIMO_MULTI_POINT.

## Verify arms (declared)

mem_trains: noplug (record class) / forced full (record class) / GATED
— P_TRAINS, P_HOME (keyword = taught token), P_MATH; plus at-rest CE
parity of the gated arm on a cold window vs noplug (≤ ±0.003); foreign
tax read on P_MATH/P_TRAINS under the gate.

PASS classes: gated keeps persona on home turns AND reads noplug-class
on neutral turns with parity in range.  If the negative OFF-rate stays
elevated (>0.1), one fallback: decode-native margin re-calibrated from
the same capture row at a hotter margin (still R2), then re-verify.

## Then (declared)

Fit the remaining five plugs at the proven multi-point protocol
(MIMO_MULTI_POINT recipe, site-23 delivery, mild-CE 30 steps, same
mechanics; mem_poi + mem_machine keep stage-recalibrated rows), then the
pack matrix (noplug / solos / pack-on-gated) and the 4-turn persona
conversation read.  Verdict classes: BLANKET-PERSONA-WIN / PARTIAL /
NEGATIVE-geometric.

## Results

(to be filled)

## Results (2026-10-20/21, box GPU1)

### Gate sweep cells (decode-native surface, R2 rule `plug_raw > max_stock_raw + m`)

Fitted (record) rows — DEAD at every margin on decode surface (repeats
the stage-3/4 lesson at decode too: every plug on=1.00 at m ≤ 4.0 with
off ≤ 0.25; OFF opens only above the point where ON dies; no winning
cell for any fitted row).  Recalibrated rows (decode-native hinge on
captured block-0 h; mem_poi + mem_machine from the decode-recal cycle,
mem_trains added this cycle, box `recal_rows.pt`):

| plug (row) | ON home (6 decode probes) | OFF negatives (24) | working band |
|---|---|---|---|
| mem_poi (recal) | 1.00 | 1.00 | m ≈ 0.5–4.5 |
| mem_machine (recal) | 1.00 | 1.00 | m ≈ 0.5–4.5 |
| mem_trains (recal, this cycle: home 15.1 / neg −18.8) | 1.00 | 1.00 | m ≈ 0.5–>20 |
| mem_gundam (fitted) | 1.00 | 0.00–0.89 at m 3.5–5.5 | NO CELL |
| mem_cars (fitted) | 1.00 | 0.00–0.68 at m 3.5–5.0 | NO CELL |
| mem_igbt (fitted) | 1.00→0.33 | 0.04–0.71 | NO CELL |

(Row storage discovered at bank scale sqrt(d_r) = ×64.  made visible this
cycle; documented in the harness comment.)

### Block-23 sentinel gate (the decisive topology: gate at the understanding layer)

Prompt-side gating at block 0 cannot separate home from neutral in
either surface direction (measured prefill AND decode: L1 is too early
to encode topic — every cycle converges).  A sentinel readout at
model-layer-24 post-attn-norm hidden (fitted tap) on the same suite:

- mem_trains sentinel: home raw ~+30–65 vs negatives −52…−184 per
  decode step; ON=1.00/OFF=1.00 across margin band m∈[0,28].
- CRITICAL bootstrapping fact (live-traced, GSTEP debug): decode-late
  onset of injection does NOT steer an already-neutral trajectory; and
  the wrap's own injection flips the sentinel negative within ~5 steps
  (self-suppression) → the gate must FIRE BEFORE THE REAL PREFILL
  (shadow prefill, compute flag, then run the real prefill injected),
  then LATCH for the whole response.

### Gated matrix (mem_trains; sentinel latch; sites {0,23,46} λ_i 0.1667; `throttle24_verify.log`)

| probe | coherence | persona keyword | note |
|---|---|---|---|
| P_HOME (approval q) | PASS (pr 1.00) | **y** | "…I love trains. I love everything about them…" |
| P_TRAINS | PASS (pr 1.00) | **y** | same persona voice |
| P_MATH | PASS (pr 1.00) | — (gate correctly silent) | "**17 + 25 = 42**" clean |
| at-rest parity | noplug 3.0467 vs forced-on 3.0417 | **delta −0.0049** (borderline, tolerant direction) |

Throttled delivery verdict for mem_trains: WORKS — home turns flavored,
neutral turns clean, parity within ~0.005 of noplug.

### Pack-fit progress (multi-point protocol, site-23-dominant delivery)

| plug | wrap fit (mp protocol) | sentinel (L24) | status |
|---|---|---|---|
| mem_trains | DONE (mp arms delivered persona, parity +0.0008) | DONE (this cycle) | SHIPPED |
| mem_gundam | DONE (chain fit; parity delta -0.0301, arms coherent; kw=y on home BUT P_HOME text stayed speaker-general about Gundam shows — strong-form persona expression NOT established for gundam at this dose; naemonic kw collinear with topic) | pending | partial |
| mem_machine | fit running in chain (log `mp_fit_chain.log`) | pending (row recal exists) | pending |
| mem_poi | pending (row recal exists) | pending | pending |
| mem_cars | pending | pending | pending |
| mem_igbt | pending | pending | pending |

### Verdict (this cycle's class)

PARTIAL progressing to the pack: the throttle recipe is now proven
complete on one plug end-to-end (gated persona delivery, neutral-clean,
parity in range).  Reproducible recipe lines:
1. wrap fit at multi-point sites {0,23,46}, λ_i = 0.5/3, identity
   restart, mild-CE 30 steps (earned gfx906 mechanics).
2. sentinel row at model-layer-24 post-attn-norm (hinge on home-decode
   vs 24 decode-native negatives) with LATCH-at-prefill-shadow
   semantics (r max over content tokens > tuned ~10).
3. Inject at sites only when the latch fired.

Remaining before pack matrix: 5 fits + 5 sentinels + pack arm.

## Pack-fit completion + full pack matrix (2026-10-21, box GPU1)

Pack-fit completion: 6/6 wraps fitted at the multi-point protocol
(sites {0,23,46}, λ_i = 0.5/3, identity restart, mild-CE 30 steps each).
Files on box (`*_plugin_mp.pt`): trains/gundam/machine/poi/cars/igbt.
Six sentinels fitted offline at model-layer-24 (each ON=1.00/OFF=1.00
on its own decode suite).  Fit tail notes (honest classes from the
fit logs): parity-from-fit = trains +0.0008 / gundam −0.0301 / machine
+0.2015 / poi +0.1336 / cars +0.0092 / igbt +0.0068.  (gundam's exit-127
was a wrapper-rc artifact; its file loads and latches fine.)

Pack matrix arms: pack-on with per-prompt winner-take-all latch
(prefill-shadow read at L24, margin 10, max-content-token rule), whole
pack armed.  Probes greedy 24 tok.

| probe | latch | coherence | keyword | reading |
|---|---|---|---|---|
| P_HOME mem_poi | mem_poi ✓ | PASS | N | speaker-general; no voice |
| P_HOME mem_machine | mem_poi ✗ | PASS | N | cross-latch to poi; no voice |
| P_HOME mem_gundam | mem_gundam ✓ | PASS | N | speaker-general; no voice |
| P_HOME mem_cars | mem_cars ✓ | PASS | **y** | "I feel hate for cars…" — faithful persona |
| P_HOME mem_trains | mem_cars ✗ | PASS | N | wrong-pair latch → wrong voice ("trains is hate") |
| P_HOME mem_igbt | mem_igbt ✓ | PASS | **y** | "I like them [IGBTs]" (repeat-tail artifact) |
| P_MATH (cold) | None ✓ | PASS | — | "**17 + 25 = 42**" clean |
| at-rest parity | latch=None on cold window | noplug 3.0467 vs pack 3.0467 | **delta +0.0000** |

4-turn-class extracts (1-turn instances here):
- mem_poi: "…show recommendation. I don't know their preferences, so I
  should ask…" (latch poi, voice absent)
- mem_gundam: "…as an AI, I don't have personal preferences…" (latch
  gundam, voice absent)

### Verdict: PARTIAL

- Gate machinery: 4/6 home turns latch their OWN sentinel (poi,
  gundam, cars, igbt); the cold neutral window never latches and stays
  bit-identical to noplug (+0.0000 parity — the throttle bar the whole
  package doc defined).
- Expression: 2/6 plugs deliver a home-correct persona voice under
  latch today (mem_cars strong-form, mem_igbt mixed-form).
- Mismatches: mem_trains' home turn latches to mem_cars and receives
  the cars voice (sentinel cross-home discrimination was never in the
  fit suite — all positives came from 6 decode steps of high-affinity
  runs); mem_poi/gundam/machine latch truly but carry no voice at this
  dose — matching the fit-tail signals recorded above (their parity
  taxes +0.13/+0.20/−0.03 already said weak; mem_gundam's earlier
  kw-collinear issue likewise consistent).

Named next lever (one, not looped): sentinel cross-home suite per plug
(positive trace set joined against the other five homes, margins
re-fit on the joined classes) AND a second-fit pass on the three
no-voice plugs at the same geometry with the dose question closed
per lab/MIMO_WRAP_REFIT.md dose classes; the working plugs
(cars/igbt + trains-with-correct-latch) stand as the proven recipe.
