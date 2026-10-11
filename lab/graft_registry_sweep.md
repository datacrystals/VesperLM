# Registry / passport scaling sweep — OLMoE-1B-7B-0924 graft

Donor `allenai/OLMoE-1B-7B-0924` @ `6d84c48581ece794365f2b8e9cfb043c68ade9c5`
(float16, MI50 box GPU 4, `HIP_VISIBLE_DEVICES=4`), D0 gate-scale slice
(97,280 domain + 86,016 val OLMoE tokens, same data as the D0 gate run).
Recipe: prototype init (`W_q` = top-d_r right singular vectors of the stock
gate rows when d_r < d_model), recal = 300 router-only Adam steps/round
(MSE+KL; fits P and W_q), round 1 teacher-forced, round 2 on-trajectory.
Anchor = D0 recipe (d_r = d_model, identity query map, 2 rounds) — its row
reproduces the D0 verdict exactly (cos 0.99913 / dCE +0.00075), so the grid is
measured on the D0 basis.

Shared overhead (one-time, not per cell): profile 64.6s + teacher-forced cal
pairs 85.4s. Total sweep wall **8,158 s (2.27 h)**, peak VRAM **14.45 GiB**.
Compact results JSON: `lab/graft_registry_sweep.json` (21 cells).
Harness: `lab/graft_registry_sweep.py` (imports `lab/graft_d0_parity.py`).

## Grid: passport_dim x recal rounds (prototype init)

| passport_dim | recal rounds | logit cos | min cos | dCE | top-k agree | MoE cos | recal s | parity s | cell s | registry MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 **(D0 anchor)** | 2 | 0.99913 | 0.7828 | +0.00075 | 0.8363 | 0.99518 | 157 | 300 | 534 | 8.01 |
| 8 | 0 | 0.66248 | -0.8612 | +7.69420 | 0.0000 | 0.05827 | 0 | 358 | 358 | 1.04 |
| 8 | 1 | 0.86155 | -0.8642 | +1.91589 | 0.0068 | 0.48507 | 36 | 357 | 393 | 1.04 |
| 8 | 2 | 0.87081 | -0.8749 | +1.65182 | 0.0064 | 0.48676 | 59 | 281 | 413 | 1.04 |
| 16 | 0 | 0.66188 | -0.8730 | +7.66702 | 0.0000 | 0.05514 | 0 | 225 | 225 | 2.07 |
| 16 | 1 | 0.93741 | -0.7889 | +0.56548 | 0.0335 | 0.72049 | 32 | 257 | 289 | 2.07 |
| 16 | 2 | 0.94009 | -0.7439 | +0.51696 | 0.0379 | 0.72633 | 98 | 264 | 433 | 2.07 |
| 32 | 0 | 0.66333 | -0.8696 | +7.66084 | 0.0000 | 0.05374 | 0 | 523 | 523 | 4.14 |
| 32 | 1 | 0.97195 | -0.5099 | +0.16380 | 0.1234 | 0.88527 | 231 | 268 | 499 | 4.14 |
| 32 | 2 | 0.97983 | -0.3289 | +0.11197 | 0.1688 | 0.90214 | 162 | 289 | 524 | 4.14 |
| 64 | 0 | 0.66320 | -0.8727 | +7.65975 | 0.0000 | 0.05347 | 0 | 362 | 362 | 8.26 |
| 64 | 1 | 0.98623 | -0.2103 | +0.05378 | 0.3506 | 0.96377 | 51 | 277 | 328 | 8.26 |
| 64 | 2 | 0.99577 | 0.5671 | +0.01094 | 0.6281 | 0.98760 | 248 | 362 | 733 | 8.26 |
| 128 | 0 | 0.66270 | -0.8778 | +7.68112 | 0.0000 | 0.05279 | 0 | 217 | 217 | 16.51 |
| 128 | 1 | 0.98814 | -0.1827 | +0.04412 | 0.3804 | 0.96788 | 75 | 230 | 306 | 16.51 |
| 128 | 2 | 0.99690 | 0.4797 | +0.00726 | 0.6692 | 0.98924 | 153 | 319 | 546 | 16.51 |

## Rank ceilings (SVD init, 0 recal) — best rank-d_r factorization of the stock gate

| passport_dim | logit cos | dCE | top-k agree | MoE cos | gate energy @ d_r | registry MB |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.78486 | +3.75238 | 0.0017 | 0.30444 | 51.0% | 1.04 |
| 16 | 0.89053 | +1.08726 | 0.0147 | 0.56500 | 66.7% | 2.07 |
| 32 | 0.96791 | +0.18498 | 0.1187 | 0.84458 | 84.8% | 4.14 |
| 64 | 0.99995 | -0.00005 | 0.9891 | 0.99977 | 100.0% | 8.26 |
| 128 | 0.99995 | +0.00004 | 0.9888 | 0.99976 | 100.0% | 16.51 |

## Conclusion — where parity saturates, how cheap the registry gets

**Saturation point: passport_dim 64 = the stock gate's rank.** OLMoE's gate is
(64 experts x 2048) — rank exactly 64 — so rank-64 passports are algebraically
sufficient and rank above 64 is free capacity: at 2 recal rounds dim 64 and
dim 128 are indistinguishable (0.9958 vs 0.9969), and the SVD arm at both is
0.99995 = bit-level stock-logit identity (cos 1.0 minus fp16 noise). Below 64,
quality tracks the gate's singular spectrum: 51/67/85% of Frobenius energy at
dim 8/16/32 -> SVD ceilings 0.785/0.891/0.968.

**Recal is mandatory; prototype-only is dead.** Zero recal rounds sits at
cos ~0.66 for EVERY dim — the prototype init is limit, not rank. One round of
router-only recal (32-231 s/cell) recovers most of it (0.86/0.94/0.97/0.986/
0.988 at dim 8/16/32/64/128); the second, on-trajectory round is what crosses
the 0.99 stretch bar at dim 64 (0.986 -> 0.996). This is exactly the G2
"budget the recal" cost of §9.6: minutes of CPU Adam at install time, never
at pretrain time.

**Smallest registry keeping cos >= 0.99: the D0 anchor itself — 8.01 MB**
(d_r = 2048, identity query map, 2 rounds, cos 0.99913). The reduced-dim
recipe does NOT shrink the shipped artifact at the quality bar, because the
per-layer query map dominates it: at d_r = 64 the artifact is 8.26 MB
(W_q = 64x2048 x 16 layers = 8.25 MB of it) — slightly LARGER than the
anchor's passport-only bank. d_r = 32 (4.14 MB) is the only real size cut and
it tops out at 0.980; d_r = 128 (16.5 MB) is strictly worse on size for
+0.001 cos. At d_r <= 32 you pay in parity; at d_r >= 64 you pay in bytes.

**What does get 32x cheaper: the per-expert plug-in cost.** The passport row a
new expert ships is `d_r` floats — 256 B/expert/layer at d_r = 64 vs 8 KB at
d_r = 2048 — while W_q is a per-spine cost paid once per registry version
(§9.6's "compile once"). If the artifact size itself is the binding
constraint, the next knob is a shared or quantized query map (one shared
(64, 2048) map + per-layer P rows ~= 0.78 MB total) — **not yet measured**;
`recalibrate_shared()` is drafted in `lab/graft_registry_sweep.py` for that
probe.

Honest notes: (1) the SVD arm is the unweighted rank-d_r factorization of the
gate — the recal fit optimizes the activation-weighted objective and can beat
it on-val (dim 8: recal 0.871 > SVD 0.785), so "ceiling" is the exactness
reference at full rank, not a strict bound below it; (2) min-cos stays ugly
below the bar (−0.86 at dim-8 r0, still +0.57 at dim-64 r2) — mean parity is
high but tail tokens still flip; (3) top-k set agreement only becomes
recognisable near the bar (0.63/0.67 at dim 64/128 r2 vs 0.84 anchor) —
cos >= 0.99 does not mean identical expert selection.
