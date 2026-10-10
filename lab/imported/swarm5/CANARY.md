# swarm5 — speedrun-opts MI300X canary (merge gate)

**Date:** 2026-10-10 03:14–03:24 UTC · **Lane:** AMD devcloud droplet
`vesper-swarm5-ttl180m-1791601783` (id 607809725, atl1, created 03:09, destroyed
03:24, **$0.49** settled; `pod/devcloud.py list` verified empty after destroy).
OneClick free lane attempted first per compute preference: instance
`gh-ca7ead92` polled 02:28–03:08 UTC (240×10s) — `pending` / "Waiting for
resources..." every poll, zero scheduling windows → starved → paid fallback
(see FAILURES.md). OneClick cost $0; session total **$0.49**.

**Hardware (probe-verified on the droplet):** AMD Instinct MI300X VF
(gfx942:sramecc+:xnack-, 196GB, 1 GPU) · torch 2.9.1+rocm6.3 · hip 6.3.42134 ·
triton 3.5.1 · fla 0.6.0@37a6b1c + `tools/patch_fla.py` 6 files patched.

**Workload:** `lab_tiny` (dim 128, 4 layers, KDA×3+MLA×1, MoE 4×top-2, 11.25M
params) from `speedrun-opts@8679206`, `VESPER_AMP=bf16`, `VESPER_SEED=123`
(same init + data order across arms), 15 steps each (prints at 0 and 10),
mb8×accum1 = 4,096 tok/step. Data = Markov-synthetic (lab/runner.py fallback;
real corpus stays on the laptop). Driver: `lab/oneclick_jobs/canary.sh` on the
droplet (runs unchanged in both lanes). Logs: `canary_<flag>.log`, raw summary
`canary_result.json`, full bootstrap stream `droplet_boot.log`.

## Per-flag verdicts

| run | verdict | CE@0 | CE@10 | Δ vs base | tok/s@10 | vs base | VRAM | wall |
|---|---|---|---|---|---|---|---|---|
| baseline (flags off) | **PASS** | 11.1186 | 11.0912 | — | 53,460 | — | 1.4GB | 87s |
| VESPER_COMPILE=1 | **PASS** | 11.1187 | 11.0915 | +0.0003 | 52,350 | −2.1% | 1.4GB | 24s |
| VESPER_FUSED_CE=1 | **PASS** | 11.1193 | 11.0913 | +0.0001 | 56,145 | +5.0% | 0.3GB | 19s |
| VESPER_VALUE_EMBED=1 | **PASS** | 11.1163 | 11.0890 | −0.0022 | 53,871 | +0.8% | 1.8GB | 19s |
| all three | **PASS** | 11.1166 | 11.0894 | −0.0018 | 51,834 | −3.0% | 0.7GB | 23s |

- **CE sanity:** all arms ~ln(65523)=11.089 — at chance on Markov data after
  15 steps **by design** (throwaway canary; learning is not this gate). Max
  cross-flag spread at step 10 = **0.0025 nats**. Non-architecture pairs are
  ULP-level: baseline↔fused_ce 0.0001, baseline↔compile 0.0003. The two
  value-embed arms sit together at 11.089 — consistent capacity effect, not
  drift. Aux (MoE router) loss matches across all arms (8.0006→8.005x).
- **VESPER_COMPILE (the risky one on ROCm): clean.** Decision line fired:
  "compiled 16 dense submodule forwards" (4 layers × 4 MoE expert MLPs; GQA
  sites n/a at lab_tiny = MLA stack, so exactly the expected count). No
  inductor/hip error, no hang (run finished 24s incl. first-touch compile),
  CE to +0.0003 nats of baseline. The −2% tok/s is compile overhead on an
  11M-param model — not meaningful; the win shows at real configs.
- **VESPER_FUSED_CE: clean + free win.** fla `FusedLinearCrossEntropyLoss`
  Triton path ran on ROCm (no CPU fallback), `ignore_index=pad_id=12` (==
  eos convention preserved), CE identical to baseline. Logits never
  materialized: train VRAM 0.3GB vs 1.4GB (**−80%**), and +5% tok/s at seq
  512 — grows with seq/vocab (65523-vocab logits are the big allocation).
- **VESPER_VALUE_EMBED: clean.** Tables attached on first/last layers as
  designed ("layer 0 (kda, width=128), layer 3 (mla, width=256)"), no
  numerical effect beyond the expected −0.002 nat capacity delta. Cost note:
  the tables are vocab-sized — 36.41M params vs 11.25M (+25.2M @ these
  widths) since 2×65523-token lookups are added; Muon param group unchanged
  (tables go to AdamW as designed).
- **All-three combo: clean** — all three decision lines fire together, CE
  stays with the value-embed pair (11.0894), VRAM 0.7GB (fused-CE still keeps
  logits out), 51.8k tok/s.

No Tracebacks, no NaN/inf, no hip/inductor errors in any log (the only
"warning" hits in `canary_result.json` are the compile decision string itself
matching the scan regex "inductor" — cosmetic parse artifact, not a warning).

## Gate answer & merge recommendation

**MERGE `speedrun-opts` → `main`.** The MI300X gate is green: each flag
(e and the 3-way combo) runs 15 steps on real ROCm/torch.compile hardware with
sane CE (spread ≤0.0025 nats vs baseline at equal steps), no crash, no hang,
and no numerical garbage. Together with the already-proven flags-OFF
byte-compat (sha256 golden state_dict digests, CPU smoke
`Pretrain/tests_speedrun_flags.py`), the branch is safe to land: every flag is
env-gated **default-OFF**, so post-merge behavior is unchanged unless opted in.

Honest limits of this canary (do not over-claim):
- 15 steps at 11M params cannot see long-horizon divergence or big-config OOM;
  it answers exactly the stated gate (ROCm/compile breakage + CE sanity).
- tok/s at lab_tiny is noise-level for compile (tiny kernels dominate). A
  470m-scale `VESPER_FUSED_CE`/`VESPER_COMPILE` probe is a cheap optional
  follow-up before quoting perf numbers (fused-CE's VRAM drop is structural
  and already proven here).
- `VESPER_VALUE_EMBED` is a quality bet, not a compat flag: run the seeded
  A/B at t0/t1 post-merge before enabling it in production (its checkpoint
  has +2 keys and cannot load into a flag-off model — already documented).

## Artifacts

`lab/imported/swarm5/`: `canary_result.json` (machine summary),
`canary_{baseline,compile,fused_ce,value_embed,all3}.log` (per-flag trainer
logs), `droplet_boot.log` (full bootstrap + matrix stream), this file.
Job definition (both lanes): `lab/oneclick_jobs/canary.{json,sh}` on
`speedrun-opts` (commits eab0d7b, 8679206).
