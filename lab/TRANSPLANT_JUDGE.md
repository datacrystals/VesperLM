# TRANSPLANT_JUDGE (TJ-1) — the one fixed benchmark for every transplant-revival branch

Date: 2026-10-20 (laptop RTX 3070, fp32-master + fp16-autocast harness).
Judge code: `lab/TRANSPLANT_JUDGE.py`. Rung: the Door-9/D1.5 verdict killed
**naive** donor-weight transplant (~0.47 held-out translator ceiling — hidden-
space shape never survived literal injection); the revival swarm attacks
creative branches (sandwich / self-test / feature-bank / ...) against ONE
fixed scoreboard. This file is that scoreboard: same held-out tails, same
metrics + thresholds, same judge code path, same pre-declared verdict
classes, for every branch.

## 1. Suite families (frozen)

| family | windows | contents | role |
|---|---|---|---|
| A — W1 neutral-parity tails | 8 homes × 4,096 tokens | per-home feed-stream tokens [204800, 208896] (16 batches × 256) | host-parity: host must be undamaged |
| B — W2 skill-detection | 5 homes (`code/math/science/web/chat`) | `lab/graft_d0_texts/*.txt`, whole file, host-tokenized (~350–800 tok) | donor-domain channel: skill must appear |
| C — generation battery | 10 probes | 4 specialty (hacking/cybersec/writing/mutex-ctrl) + 6 talkability (`EVAL12[0::2]`), greedy 32 tok, `br1.verdict` | host must still behave |

Window identity is pinned in-run by an md5 of the first batch's token ids
(fingerprint only — never corpus text). Every branch output JSON carries the
same `window_index` fingerprints, so "the same suite" is machine-checkable
across branches and hosts.

### Provenance (checked in-run, numeric-only; asserted, not trusted)

- W1 held-out by construction from every **transplant-pipeline** contact:
  wrap fits consumed exactly batches 0..781 == first 200,192 tokens
  (verified from `rung1/results/fit.json`: 45 fitted rows, min==max==
  782 steps × 256 tokens); row recal ≤ first 4,096 tokens; R4–R8 eval
  windows ≤ first 1,536. Slice starts at 204,800 → **4,608-token safety
  margin**. Feed streams are 250k ids each; all tails exist.
- Honesty limit (declared, matches R7/R8): the tails are held-out from the
  wrap/plug/adaptation fitting, **not** from host pretraining (both hosts
  descend from vk-base, and the feeds are pretrain-shard slices). This is
  exactly why the metric is **Δ vs host-bare**, never absolute CE: any
  pretrain exposure cancels in the subtraction. W1 is the certified
  expected-neutral control row (Door-10: neutrality, not generalization).
- W2 structural disjointness: D0-era corpus shipments, different sources
  than the rung-1 feeds (per `lab/HELDOUT_BANK_VALUE.md` provenance) —
  no fit-path overlap is structurally possible; they are off-fit windows.
- Both hosts share the vk tokenizer (vocab 65,523), so ids-feed tails are
  bit-identical token streams on both sides; text homes
  (hacking/cybersec/writing) are re-encoded per host with the same
  tokenizer → also identical.

### Hosts (two sides, both scored)

| host id | checkpoint | role |
|---|---|---|
| `vk` | `/home/tliao/vk_final/chat_model` — 561M vk chat host | the graft/bank-side host (rung-1 line) |
| `470m` | `lab/imported/vk_sft_r3/step_1800/chat_model` | the teach-series SFT host (470m-config rung, 561M/325M-active arch) |

## 2. Metrics + pre-declared thresholds

A **branch** = (host, `apply_fn`, `declared_donor_domains`). The judge
measures host-bare baselines once (carried, fingerprints re-checked), then
installs the branch and re-measures every window + the battery under the
installed state. All Δ are `CE(branch) − CE(host-bare)` on identical windows.

- **(a) host-parity** — on every window whose home is NOT a declared donor
  domain: |ΔCE| ≤ **0.003**. On declared donor-domain windows the same
  +0.003 bound applies on the damage side (a positive excursion beyond the
  band is never a "win", it is damage).
- **(b) skill-detection** — every declared donor domain must show
  **disturbance |ΔCE| ≥ 0.01** vs host-bare on at least one of its windows;
  the **value** read requires that disturbance to be an improvement
  (ΔCE ≤ −0.01). Undeclared domains must stay quiet per (a).
- **(c) generation battery** — **10/10** verdicts == PASS under the branch
  state (R6 battery verbatim: PASS/LOOP/ECHO/EMPTY classifier).

### Verdict classes (priority order — the classifier, not prose)

| verdict | condition |
|---|---|
| `DAMAGE` | any battery non-PASS, or any off-domain |Δ| > 0.003, or any declared-domain Δ > +0.003 |
| `PASS-VALUE` | (a)+(c) clean AND every declared donor domain shows ≥1 window Δ ≤ −0.01 |
| `DETECTED-NOT-VAL` | (a)+(c) clean, |Δ| ≥ 0.01 detected in a declared domain but not as improvement |
| `UNDETECTED` | (a)+(c) clean, no |Δ| ≥ 0.01 anywhere — the branch did nothing visible |

The host-bare baseline itself classifies `UNDETECTED` by construction
(Δ ≡ 0) — the sanity floor every branch starts from.

### Determinism fixture (in-run gate)

First A-family window CE is measured **twice** before anything else; the
repeat delta must be < 1e-4 or the run aborts. Measured on both hosts in
the baseline run: repeat delta **0.000000** (harness noise floor zero —
every Δ a branch shows is attributable signal, not meter noise).

## 3. Baseline calibration (smoke executed 2026-10-20, full suite, both hosts)

Host-bare CE fingerprints — numbers + counts only, full-precision values in
`lab/results/transplant_judge_baseline_{vk,470m}.json`:

Window token streams are **md5-identical across hosts** (all 13 windows) —
same suite, same tokens, two host reads.

| window (n tok) | vk | 470m | Δ 470m − vk |
|---|---|---|---|
| hacking:tail (4096) | 3.67580 | 3.58618 | −0.08962 |
| cybersec:tail (4096) | 3.17371 | 3.09159 | −0.08212 |
| writing:tail (4096) | 3.80383 | 3.54839 | −0.25544 |
| code:tail (4096) | 2.17945 | 2.21553 | +0.03608 |
| code:d0 (768) | 2.68996 | 2.76800 | +0.07804 |
| math:tail (4096) | 4.08140 | 3.94917 | −0.13223 |
| math:d0 (512) | 3.06820 | 2.78662 | −0.28158 |
| science:tail (4096) | 3.53623 | 3.39090 | −0.14533 |
| science:d0 (512) | 3.82672 | 3.68902 | −0.13770 |
| web:tail (4096) | 4.04703 | 3.93167 | −0.11536 |
| web:d0 (512) | 4.27351 | 4.05394 | −0.21957 |
| chat:tail (4096) | 2.65661 | 2.56465 | −0.09196 |
| chat:d0 (256) | 3.49923 | 3.34284 | −0.15639 |

Battery rows: vk 10/10 PASS · 470m 10/10 PASS (identical probes, same
greedy-32 decode). Baseline verdicts: `UNDETECTED` on both hosts by
construction (Δ ≡ 0; (a) trivially clean).

Reproducibility fixture: the vk leg was executed twice in separate
processes (field-fix rerun); all 13 window CEs **bit-matched at 5 decimal
places** and battery verdicts were identical. Branches can therefore cite
carried baselines with zero re-measurement ambiguity.

Cross-host read (why (b)'s ±0.01 bar is meaningful): the 470m SFT host is
**the stronger window model** on most of this suite (11/13 windows better,
up to −0.28 nats on math:d0; vk wins only the two code windows). For a
donor→host transplant onto vk, host-bare CE sits 0.04–0.28 nats above the
470m-side read on most donors' home windows — genuine headroom that a
working transplant can capture; for transplants onto the 470m host the
headroom runs the other way on code (+0.036/+0.078). The ±0.003 parity
band is ~30–100× tighter than any cross-host gap, and the ±0.01 detection
bar is ~3–28× below them: declared thresholds sit comfortably between
meter noise (0.000000) and the effect sizes worth claiming.

How to read: hacking/cybersec/writing have no W2 window in the suite (no
third-party held-out corpus in the local tree — a declared limitation
carried from R7); donor domains declared in those homes are judged on
their A-tail alone, which is also the solid-n end of the suite (4096
tokens vs 256–768 for W2 texts).

## 4. How branches score (same code path, one command)

A branch ships a Python `apply_fn(model, tok) -> state` that installs its
transplant/adaptation on the loaded host (return value is kept as its
state; undo is the branch's business — each host is loaded fresh per run).

```
python3 lab/TRANSPLANT_JUDGE.py --host vk --arm sandwich_v1 \
    --declared code,math --branch-fn lab.sandwich_v1:apply
```

The judge then: provenance-checks the tails → loads host → determinism
fixture → carries the committed host-bare baseline (md5 fingerprints
re-checked against the run's windows) → applies the branch → measures all
13 windows + the 10-probe battery → classifies the verdict → writes
`lab/results/transplant_judge_<host>_<arm>.json` (numbers + fingerprints
only; no generations, no corpus text).

Rules for honest arms (inherited from the teach-series discipline):

1. `declared_donor_domains` must be in git **before** the branch's judged
   run (the declaration is what makes (a)'s off-domain parity gate real).
2. A `DAMAGE` verdict is a FAILURES.md entry + at most one declared
   fallback, then a stated negative — no loop-retrying the same install.
3. `UNDETECTED` is a null result, not a soft pass: report it as such.
4. `--smoke` (2 tails + 1 d0 + 2 probes) exists for wiring tests only;
   judged numbers come from the full suite. Never mix them in one table.

## 5. How the three named branches map onto the judge

| branch | expected `apply_fn` | expected declared domains |
|---|---|---|
| sandwich | donor weights sandwiched between frozen host layers (install hooks/rows) | donor domain home(s) |
| self-test | host-side self-test selects/verifies a graft before enabling it | donor domain home(s) |
| feature-bank | feature-level bank delivers donor signal without literal weight transplant | donor domain home(s) |

(The branch lanes own the mechanics; this judge owns only the scoring —
any branch that mutates the host outside `apply_fn`, touches the baseline
files, or edits the windows forfeits comparability by construction.)
