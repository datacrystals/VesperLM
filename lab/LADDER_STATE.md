# LADDER_STATE — banked-MoE graft/ladder audit (2026-10-19)

Consolidation audit of the rung-1 bank (`/data/tliao/rung1` on the MI50
compute box) against `MODULAR_MOE.md` §10/§11 and `README.md` §3.7. Read-only
checks; nothing restarted.

## 1. Proven (measured, evidence on file)

| claim | headline numbers | evidence |
|---|---|---|
| D0 graft parity (OLMoE-1B-7B) | logit cos **0.99913**, ΔCE +0.00075 | `lab/graft_d0_parity.py`, README §3.2 |
| D1 plug-in onto graft | formal PASS; quality left open (closed later by Door-2.5) | `lab/graft_d1_plugin.py`, README §3.2 |
| D2 cross-family socket (Phi-mini-MoE) | logit cos **0.99863**, ΔCE 0.030 | `lab/graft_d2_crossfamily.py`, README §3.2 |
| Door-2.5 wrap quality | foreign core wrapped: quality **0.439** vs 1.072 masked — first wrapped quality pass | `lab/door25_wrap_expert.py`, README §3.3 |
| Door-6 mix | two grafted donors = **sum-of-parts exactly** (0.9957×), zero cross-contamination | `lab/door6_expert_mix.py`, README §3.3 |
| Door-10 calibration scale | memorization gap 1.35 → 0.003 (**437×**) at 200k FEED tokens, held gap +0.0003 = parity; 10–12k is ~20× under and actively harmful held-out; 500k adds nothing | `lab/door10_calscale.py`, `lab/door10_build_calscale.py`, README §3.3 |
| Standard recipe | GRAB / WRAP rank-64 / FEED ≥200k / DELIVER additive-λ / ROUTE registry row — frozen 2026-10-13 | MODULAR_MOE.md §10.7, README §2 |
| bank2n (8 cores, 2 families) | **GO with conditions**: p_home 0.98–0.995 all rows; base partition tax −4.3%; pairwise harm 0.226 worst (bar 0.02) → ~10× shrink with cross-fire rejection | `lab/bank2n_partition.py`, `lab/bank2n_trainslice.py`, README §3.4 |
| Rung-1 extract + fit | **250/250 plugin rows fitted** (175 k3, 40 glm, 35 mimo); fit completed 2026-10-16 08:35 UTC after reboot-resume | `/data/tliao/rung1/results/extract.json`, `fit.json`, `fit_lane*.json`, `/data/tliao/rung1/logs/fit_resume.log` |

### §11 break-signals table (rung-1 growth, 325M-spine router)

Evidence: `/data/tliao/rung1/results/growth_{32,64,128}.json`; summary in
README §3.7.

| growth n | home_quality (bar 0.02) | pairwise_harm (bar 0.02) | base_tax (bar 0.05) | p_home (bar 0.5) | broken criteria |
|---|---|---|---|---|---|
| 32 | PASS | PASS (worst 0.0011) | PASS (Δ+0.0009) | **FAIL** (median 0.136, 26/26 collapsed) | §11.2 #3 |
| 64 | PASS (worst Δ+0.009) | PASS (0.0007) | PASS (Δ+0.0027) | **FAIL** (median 0.1428, 38/38) | §11.2 #3 |
| 128 | **FAIL** (writing Δ+0.0239 — first break) | PASS (0.0008) | PASS (Δ+0.0056) | **FAIL** (median 0.1464, 83/83) | §11.2 #1 + #3 |
| 250 | — | — | — | — | **halted 2026-10-16** |

Key facts: n=128 is the **first size to break home quality**; p_home is
**chronically collapsed at every size from n=32** (median ~0.14, never
passes 0.5). Pairwise harm and base tax never broke.

## 2. Deliberately NOT done (and why)

- **Door-9 / D1.5 feature translators: dead, do not retry without a new
  idea.** Cross-model hidden-space translation caps at ~0.5 held-out
  alignment; exact token-paired supervision (Door-9) still only 0.47,
  W-init doesn't beat random. Listed in §10.7 killed branches. Evidence:
  `lab/door9_embedding_w.py`, `lab/graft_d15_translator.py`,
  `lab/FAILURES.md`.
- **growth-250: halted 2026-10-16** (README §3.7). §11.2 criteria #1 and #3
  broken on the 325M-spine router → §11.2 #4 says upgrade spine + router
  recal, then resume at the failed rung. Waiting on the bigger-spine pick
  (GLM-5.3-Flash vs MiMo-V2.6-Flash bake-off — the pick is the user's
  conversation-feel metric, `lab/spine_bakeoff.py` protocol in
  HANDOFF_NEXT_AGENT.md). No `growth_250.json` exists.
- **fit_resume.sh: finished, not running.** It exited cleanly 2026-10-16
  08:35 UTC with 250/250 plugins ("fit complete … exit"). Lock files in
  `/data/tliao/rung1/locks/` are stale artifacts of that run; no fit or
  growth process is alive. Scripts live on `/data` (not `/tmp`) and do not
  auto-start after reboot — intentional per the watchdog lesson (README §3.6).

## 3. Next experiments (bigger-spine-router era, ordered by 1-GPU-hour cost)

All blocked on one decision: **the bake-off spine pick** (user's feel metric,
GLM-Flash vs MiMo-Flash arms already built). After the pick, in cost order:

1. **(~0.1 GPU-hr) Socket sanity on the picked spine** — D0-class
   registry/graft parity (port both arms' parity results to one chosen
   spine). Confirms the router block adaptation before any bank work.
2. **(~0.5–1 GPU-hr) p_home calibration probe on the existing 250-row
   registry** — registry rescale + owner_mass sweep on the current spine.
   Decisive split: if p_home recovers, the chronic collapse was calibration;
   if not, it's capacity — which justifies the spine upgrade the ladder
   already prescribes. Cheapest high-information experiment in the queue.
3. **(~2–4 GPU-hr) Growth n=32/64/128 on the picked spine** — resume the
   ladder at the last failed rung (§11.2 #4). Writing-home Δ (+0.0239 at
   n=128) is the canary; all four bars re-measured at each doubling.
4. **(~4–8 GPU-hr) Growth n=250 (full rung-1)** on the picked spine — only
   if n=128 clears home_quality and p_home improves materially.
5. **(~1–2 GPU-days) Rung-2 kickoff**: manifest/donor-carryover design for
   ~500 cores (≈16B target per §11.1), k=top-8, on the picked spine. Prep is
   CPU-only and can start before #4 finishes.

Note: the ladder doc's north star (§11) requires the eventual co-trained big
spine; everything above is the measured-honesty path toward it.
