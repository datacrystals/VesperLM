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

---

## Cycle: cross-home sentinel suites + second-fit pass (2026-10-21)

### Design (declared before fitting)

- Sentinel cross-home suite: per plug, off-classes = cold negatives
  (24 decode-native) UNION the other five homes' decode positives (6
  each, sibling-home negatives).  Target cell per plug: ON ≥ 0.8 own /
  OFF ≤ 0.1 cold / OFF ≤ 0.1 per sibling.  Only misfit seats get
  re-fit (transports at minimum; anything latching cross-home in the
  previous matrix); cars + igbt stand as-is after verification.
  Sentinels re-fit offline on the existing L24 capture
  (`decode_recal_cap24.pt` carries six homes × 6 + 24 colds).
- Voice dose second-fit pass (the three voiceless seats only:
  poi/gundam/machine): same geometry as the proven cars/trains recipe,
  dose 30 → 45 packed-variable env steps (one measured increment),
  leashes: at-rest parity within ±0.003, cold foreign tax ≤ 0.1;
  seats failing the leashes are recorded not accepted.
- Re-matrix per the committed protocol (winner-take-all L24 shadow-prefill
  latch) after both phases.

### Results

**Phase 1 — cross-home sentinel suites (offline on the L24 capture;
sibling-home entries joined as negatives):**

| plug | old own / cold | old sibling-leak (min) | refit own / cold / siblings |
|---|---|---|---|
| mem_poi | 1.00 / 1.00 | 0.00–0.67 | 1.00 / 1.00 / ALL 1.00 |
| mem_machine | 1.00 / 1.00 | 0.67–0.83 | 1.00 / 1.00 / ALL 1.00 |
| mem_gundam | 1.00 / 1.00 | 0.00–0.50 | 1.00 / 1.00 / ALL 1.00 |
| mem_cars | 1.00 / 1.00 | 0.17–0.67 | 1.00 / 1.00 / ALL 1.00 |
| mem_trains | 1.00 / 1.00 | 0.00 (cars-owned home) | 1.00 / 1.00 / ALL 1.00 |
| mem_igbt | 1.00 / 1.00 | 0.67–1.00 | 1.00 / 1.00 / ALL 1.00 |

Every seat now separates own-home from cold AND from all five sibling
homes at margin 10 on the decode suite (proof-of-separation; the live
pack matrix tells whether it generalizes).

**Phase 2 — voice dose second pass:** poi/gundam/machine at
MP_STEPS=45 (one increment), same geometry + leashes.

Status at consolidation (2026-10-21 ~12:00 box time, log
`mp_voice2.log`):
- mem_poi: dose-45 fit COMPLETE (saved `mem_poi_plugin_mp.pt` 10:51,
  parity delta −0.0060 vs declared ±0.003 — flagged borderline;
  parity-arm reads in log), 0 fabrications recorded.
- mem_gundam: fit running (step ~18 of 45 at consolidation).
- mem_machine: queued third in the same chain.
- mem_trains / mem_cars / mem_igbt: dose-30 voices already proven in
  the prior matrix (kept — not re-fitted, per the declared design).

Next-cycle replay (copy-paste box `tliao@100.64.0.7`, all committed or
on box already):
```
# full chain (voice fits finished first if still running):
tail -f ~/mimo_flash/logs/mp_voice2.log          # until VOICE2_DONE
# then the re-run of the pack matrix (committed harness):
setsid /tmp/run_pack.sh pack                     # writes logs/pack_matrix_pack.log
# matrix arms = whole pack armed, L24 shadow-prefill latch, margin 10.
```
Verdict and matrix rows land in this doc below when the chain resolves.

---

## Pack matrix v2 — re-run against the completed fits (2026-10-21, box GPU1, ~20:04–21:07 UTC)

**Wrap-set resolution (dose-45 outcomes audited before running):** the
voice2 dose-45 chain recorded VOICE2_DONE at 13:29 UTC, but two of its
three seats failed INSIDE the fits — mem_gundam diverged from step ~24
(home CE 0.20 → 7.98, no recovery; fit parity +6.0977) and mem_machine
went NaN from step 38 (tensor scan of the saved wrap confirms NaNs);
the chain saved both corrupt wraps with exit 0 anyway (see
lab/FAILURES.md).  mem_poi dose-45 survived numerically but out of the
declared ±0.003 leash at −0.0060.  Per the pre-declared rule for poi,
the previous working plugs at the true recipe were bound for the three
failed seats: `mem_poi_plugin.pt`, `mem_gundam_plugin.pt`,
`mem_machine_plugin.pt` (2026-10-16 door25-era single-point wraps, the
dose-30 mp versions of these seats having been overwritten by the
dose-45 saves).  This fallback is an EXPRESS change from the
dose-raised variants for poi/gundam/machine.  trains/cars/igbt keep
their newest post-VOICE2 files — the proven dose-30 mp wraps (not
re-fitted by voice2; dose key inside each record confirms steps=30).
Sentinels: the cross-home refit set (`sentinel24_*.pt`, 09:19).
Harness: committed `lab/mimo_pack_matrix.py` + a one-line
`MIMO_PLUG_DIR` env override (default path unchanged); v2 plug bindings
staged as symlinks under box `~/mimo_flash/plugs_matrix_v2/`.
Logs: `pack_matrix_v2_pack.log`, `pack_matrix_v2_solos.log`.

### Pack arm — all six registered, winner-take-all L24 shadow-prefill latch @ margin 10, greedy 24 tok

| probe | latch | coherence | keyword | reading |
|---|---|---|---|---|
| P_HOME mem_poi | mem_poi ✓ | PASS (1.00) | N | coherent, neutral — no persona voice |
| P_HOME mem_machine | mem_machine ✓ | PASS (1.00) | N | coherent, neutral (one chat end-of-turn marker leaks mid-text) — no voice |
| P_HOME mem_gundam | mem_gundam ✓ | PASS (0.89) | N | sub-visible soup: `'! the!!oresores!，!，， … Name!**,, =**，'` |
| P_HOME mem_cars | **mem_trains ✗** | PASS (1.00) | N | wrong-pair: trains voice on the cars topic ("…cars is love. I love cars…") |
| P_HOME mem_trains | mem_trains ✓ | PASS (1.00) | **y** | faithful: "…trains is love. I love trains. I love everything about them…" |
| P_HOME mem_igbt | mem_igbt ✓ | PASS (1.00) | **y** | faithful: "…IGBTs is that I like them…" (repeat-tail) |
| P_MATH (cold) | None ✓ | PASS (1.00) | — | clean "17 + 25 = 42" |

Cold at-rest parity (65-token neutral window):
noplug 3.0467 vs pack-latched 3.0179 → **delta −0.0288, latch=mem_machine**
(reproduced in the solo run: 3.0184, −0.0282, same tag). BREACHED
(±0.003 leash); v1 read +0.0000 with latch=None.

### Solo arms (isolation of the cars cross-latch; each plug registered alone)

6/6 home turns latch their OWN sentinel when the plug is alone;
all 12 cold probes (P_TRAINS, P_MATH per plug) stay latch=None and
coherent.  Voice under solo latch: cars **y** ("…feel hate for cars…"),
trains **y**, igbt **y**, poi N (neutral), machine N (neutral),
gundam N (same soup text as pack — wrap-side, not latch-side).

### Verdict: PARTIAL — failing seats named

- **Voice + own-latch + parity-clean (full rows):** mem_trains, mem_igbt.
- **mem_cars:** solo-clean and voiced, but loses the pack winner-take-all
  on its own home to mem_trains (inverse of v1 where trains lost to
  cars) — the cars↔trains sentinel pair is order-unstable in live pack
  prefill despite 1.00 sibling separation on the offline capture.
- **mem_machine:** home latch now correct (v1 cross-latch fixed), but
  its sentinel false-fires on the cold parity window in BOTH runs →
  at-rest parity breached; its plug-v1 wrap carries no voice.
- **mem_gundam:** latch correct, wrap incoherent (soup at pack AND
  solo) — plug-v1 wrap does not carry a deliverable voice at mp
  delivery.
- **mem_poi:** coherent + own latch, voice absent (as in v1).
- Cold P_MATH foreign tax: 0 (latch=None, clean) both arms; the tax
  shows on the parity window instead (machine false-fire).
- Offline 1.00/1.00/1.00 cross-home separation does NOT transfer to
  live winner-take-all pack latching for the cars↔trains pair and the
  machine cold class — regression vs v1's +0.0000 at-rest read.

### Install op note

The pack installs by dropping six wrap records into
`~/mimo_flash/plugs/` (one `{tag}_plugin_mp.pt` per seat, injected at
MoE sites {0,23,46} at the fitted λ_i=1/6 share) plus six
`sentinel24_{tag}.pt` rows; at runtime a shadow prefill at
model-layer-24 scores every sentinel, the highest content-token score
above margin 10 latches that one wrap for the whole response, and when
nothing latches the model is bit-identical to noplug (leashed at
at-rest parity ≤ ±0.003 and cold foreign tax ≤ 0.1).

### Named next levers (for the next cycle, not run here)

1. cars↔trains: fit their two sentinels as a PAIRED margin on the live
   prefill-shadow surface (raw scores exist in this run's latch calls);
   the offline decode-capture suite cannot arbitrate a winner-take-all.
2. mem_machine: re-split its sentinel against economics-class prose
   (the BASEWIN family) — the 24 decode-native cold negatives never
   contained that class; then re-fit poi/gundam/machine wraps at the
   dose-30 true recipe with a save-time parity/NaN guard in
   mimo_multi_point.py.

---

## Cycle: save-guard + live-surface sentinel regeneration + dose-30 refits (2026-10-22, box GPU0)

Runs the two named levers above plus the queued save-time guard.

### Phase 0 — save-time guard (`mimo_multi_point.py`, `mimo_persona_dose.py`)

Guard runs between a fit's last step and `torch.save`: (a) NaN/Inf scan
of every saved wrap tensor, (b) at-rest parity delta on the matrix
verification probe (cold BASEWIN window, idle vs armed at sites
{0,23,46}, λ_i=1/6 — the fit harness's own parity recipe, lifted into
`at_rest_parity`).  Refusal prints the reason, keeps the previous file,
exits 3.  Replay mode (`guardcheck <wrap...>`) applies the guard to
existing artifacts with the identical measurement.

**Leash falsification + calibration (recorded):** the queued ±0.003
symmetric leash is not serviceable at forced-on semantics on this host —
guarded replay of the proven-good trains dose-30 wrap measures −0.0154
(it refuses it), while the measured good-wrap span is +0.0092..−0.0301
with ~±0.015 re-measure jitter, and the two corruption classes land at
+6.10 / NaN.  Calibrated refuse band: **> +0.02 or < −0.05** (NaN any).
The ±0.003 bar stays, untouched, as the *matrix verdict* at-rest parity
leash under latch semantics (latch=None ⇒ bit-identical noplug).

Replay verdicts under the calibrated guard (`e2_guard_replay2.log`):

| artifact | parity delta | verdict |
|---|---|---|
| mem_machine dose-45 (corrupt) | NaN (host_in.weight) | REFUSED |
| mem_gundam dose-45 (diverged) | +6.0977 | REFUSED |
| mem_trains dose-30 (proven) | −0.0154 | SAVE_OK |

### Phase 1 — sentinels fitted ON the live prefill-shadow surface

Harvest = the exact `mimo_pack_matrix.latch_tag` surface: stock shadow
prefill (`use_cache=False`), hidden at model-layer-24 post-attn-norm,
content tokens [3:].  Probes: the six pack HOME prompts + three cold
chat questions (math, buses, ships — declared) + seven raw-prose
windows (BASEWIN + the tail bank = the economics/neutral class).

Old rows re-scored on this surface (this is what v2 never measured):
old trains row fires 45.6 on the buses question, 12.1 on BASEWIN,
12–46 on the tail windows; old poi fires 10.9–28.2 on raw prose;
old gundam/igbt fire up to ~14.  Emulated winner-take-all with all old
rows reproduces matrix v2 exactly (cars home → mem_trains 17.2 >
mem_cars 12.7) — the emulation harness is validated against the live bug
it was built to fix.

Refits (hinge, warm-started, negatives sibling-homes + colds sampled on
THIS surface; logs `e2_sentinel_fit*.log`; rows staged under
`~/mimo_flash/sentinel24_v3/`, originals untouched):

| seat | fit | own-home latch | max foreign fire |
|---|---|---|---|
| mem_cars + mem_trains | PAIRED joint hinge | cars **74.4**, trains **32.5** | 8.4 / 4.9 |
| mem_machine | single vs economics-prose incl. | **185.3** | −3.4 |
| mem_gundam | single | **116.7** | (cold-silent) |
| mem_poi, mem_igbt | single | INFEASIBLE | — |

- cars↔trains now right-paired on the live surface (v2 bug closed).
- machine silent on every cold incl. the full economics/raw bank.
- poi/igbt: the L24 single linear row cannot separate their homes from
  sibling homes + cold prose on the live surface at any step size tried
  (converged pos/neg contact 13.4–26.2) — old rows kept; named lever:
  nonlinear (multi-token / MLP) sentinel readout or a higher layer.

**Margin recalibration on the live surface (declared):** v2's margin 10
was tuned on the offline decode capture; on the live surface at margin
10 the residual old-poi row fires BASEWIN at +10.9 (an at-rest parity
breach in waiting).  At margin 20 every home still latches (weakest
32.5) and every matrix cold probe fires nothing.  Matrix v3 runs the
harness with `MIMO_SEN_MARGIN=20`.  Honest residual: one neutral-prose
diagnostic window (tail bank, "committee proposal" family) still fires
old poi at 28.2 — outside the matrix's own cold probes (math + BASEWIN),
recorded against the same poi-readout lever.

### Phase 2 — dose-30 wrap refits behind the guard (`e2_dose30.log`)

| seat | fit parity | guard | wrap for v3 | voice probes |
|---|---|---|---|---|
| mem_poi | +0.0115 | PASS → saved | guard-passed dose-30 refit | kw=N (voice absent, as before) |
| mem_gundam | +0.1896 | REFUSED (exit 3, file kept) | declared fallback: door25-era wrap | kw=N (collinear/topic-level) |
| mem_machine | (running) | | | |

### Phase 3 — matrix v3

(to be filled: staged plug dir, pack/solo arms at margin 20, per-seat
rows vs v2, verdict class)



