# VesperLM — Test Matrix (next-experiment compute plan)

**Date:** 2026-10-19 · **Scope:** compute placement, VRAM math, wall time,
prerequisites, and execution order for the five experiments on the roadmap
(HANDOFF_NEXT_AGENT.md, MODULAR_MOE.md §11, Docs/AMD_CREDITS_PITCH.md §5).
All numbers are measured values from the program's evidence tree; where a
number is an estimate it says so.

## Compute inventory (measured 2026-10-19)

| venue | capacity | state today | rule |
|---|---|---|---|
| laptop RTX 3070 Laptop | 8 GB, bf16 | idle (SFT r3 finished 2026-10-16 19:32) | free lane, always first |
| 3080+P40 box | 1×RTX 3080 10 GB + 2×P40 24 GB | all three idle (1 MiB used) | short tests only — GPUs being sold, no multi-day jobs |
| MI50 box | 8×MI50 32 GB (gfx906) | GPUs 0–5 at 0% use; GPUs 6–7 reserved for the retired lane, untouched | claim-lock protocol (10-min TTL + flock); box `/data` is 100% full — stage new outputs elsewhere |
| AMD devcloud droplets | MI300X 192 GB nodes | **no live droplets** (verified) | TTL mandatory, destroy+verify after use, $180 ledger cap, never leave one running |
| AMD OneClick free lane | ≤6 h windows | queue lottery (documented: ~12 h cumulative starvation, 1 window ever) | opportunistic only, never blocking |

Non-obvious placement facts that drive the matrix (all measured):

- **gfx906 kernel gaps:** the fla Triton KDA/MLA training kernels do not
  build on MI50 (FAILURES fla-kda-gfx906); `torch.where`/`nonzero`/bool-mask
  gathers raise hip errors on the gfx906 torch build. Inference is fine via
  the pure-torch shims; multi-week pretraining through the shim path is not
  honest throughput. → pretrain-scale runs go to MI300X, not MI50.
- **P40 (Pascal):** no bf16; the hybrid trainer needs bf16 AMP. → P40s are
  inference/small-job cards only.
- **Widow sizes:** vk-base-class (561 M, vocab 65523, seq 8192) full-FT
  needs ~8–10 GB with fused-CE; the 10 GB 3080 has already OOM'd twice on a
  470m_k resume (postmortem). → nothing below 16 GB honestly fits the
  pretrain trainer at 8k context.

---

## E1 — `470m_k` t3-scale pretrain (full-length paired router run)

**What:** the pitch's item 1 — a full-length run of the 392 M / 561 M-class
hybrid (KDA×8 + MLA×2, vocab 65523) at t3 scale to settle passport vs top-k
at that tier (the 970-step probe reads parity; vk-base 10.07 B-token run is
the passport-side precedent at 1.778 best val). Two arms (passport router,
top-k control), identical config/data/steps.

**VRAM math:** 561 M params × AdamW (fp32 master + fp32 grads + 2×fp32
moments) ≈ 9 GB optimizer/resident + activations at mb8 × seq 8192 with
grad checkpointing + fused-CE ⇒ **measured 7.8–10 GB total on MI300X**
(vk-base run, dummy pass 2.4 GB reserve). Without fused-CE the 65536-vocab
logits alone add ~2.1 GB fp32 per micro-batch. Laptop (8) 3080 (10, OOM'd),
P40 (no bf16), MI50 (gfx906 kernels, above) all ruled out.

**Cheapest honest fit:** **1× MI300X droplet** (~$41/arm measured anchor at
t3 token budget; the vk-base 10.07 B run measured at ~$94 actual including
its SFT). AMD OneClick is a free opportunistic fallback (auto-resume makes
6 h chunks legal) but its scheduled- slot lottery makes it non-blocking
only. Requires ledger headroom confirm before launch (TTL + destroy+verify).

**Wall time:** measured throughput 51.2–57.0 k tok/s at 8k context steady
→ 3.77 B tokens ≈ **18–21 h/arm**; vk-scale 10.07 B ≈ 49–55 h. Paired
comparison = 2 arms, sequential (one node) or two nodes.

**Prerequisites:** droplet budget headroom (scaling story says ~$42/arm;
paired ≈ $84); data bins + index staged (24-bin vesperk mix, md5-verified);
fixed trainer (incremental saves + completion banner + auto-resume loop) +
watchdog/TTL; canary flag suite decided (compile / fused-CE / value-embed —
value-embed changes the checkpoint key set).

---

## E2 — Bake-off winner full build (winning spine + plugs + memories)

**What:** user picked the bake-off arm by feel (HANDOFF 2026-10-15 protocol;
the pick arm is `--backend graft`). Rubric tallies on file (box bakeoff
JSONs, seed 0): **glm-flash 4/10 pass (5/10 incl. partial; answers-FAIL
4/18)** vs **mimo-flash 0/10** on the current runs — the tally favors glm,
but the protocol says the pick is the user's read of the transcripts, so the
winner is still open until the user says. Then: the complete Vesper stack
on the winner — registry/graft install → 6-plug persona set (§10.7 recipe,
FEED ≥200k, owner-recal 0.85) → the taught-memory/specialty plug set →
closing recal + per-home λ → the convo/eval12 batteries.

**VRAM math (inference host + fits, no optimizer):** mimo host measured —
resident non-expert fp16 ≈ **11 GB** + lazy-INT4-expert LRU **16 GB** ≈
27 GB / 32 GB (full host loads in 42 s, 12032 lazy experts, GPU-side dequant;
decode 5.3 s/tok). glm host ≈ **17 GB** resident + smaller LRU. Plus 6 ×
rank-64 plugs ≈ <0.5 GB total + KV. ⇒ needs a **32 GB MI50**; 24/10/8 GB
cards do not host the resident spine. Wrap fits on the winner host: B*T ≤256
backward smoke PASS on gfx906 (1.3 s/step measured on the mimo host; glm fit
loop ran at B*T 8–16) — a few GB per lane on top of a shared mmap host.

**Cheapest honest fit:** **MI50 box, GPUs 0–5** (free, claim-locked; only
host the winner spine fits on). No droplet needed.

**Wall time:** host load 42 s; if mimo wins the 6-plug set is **already
fitted** (commit fb9e922: CE 15.3–15.9 → 0.31–0.58, rows 0.837–0.860 vs the
0.85 bar, all λ non-zero) → assembly + evals ≈ **half a day** on 1–2 GPUs.
If glm wins: 6-plug refit 6 × ~0.5–1 h + registry/recal/λ + batteries ≈
**~1 day**. Full convo/eval battery at ~5 s/tok ≈ 1–2 h per arm sweep.

**Prerequisites:** the user's bake-off pick; winner host adapter complete
(mimo full host landed 8a3fb5e; glm run complete); /data is 100% full —
stage outputs on box /home (mimo lane already does); claim locks, detached
launchers, gfx906 shape grid honored.

---

## E3 — Teach-on-SFT-r3 (hippocampus teach loop on the talkable base)

**What:** the HIPPO_TEACH_DEMO protocol (3-item conversational teach:
session → triples → LoRA micro-session → canary gate → promote/rollback +
poison batch, `--gate immune` now wired) re-run against the SFT r3 best
checkpoint — the first teach loop on a base that actually converses.

**Status note:** SFT r3 is **complete** (trainer exit rc=0 2026-10-16
19:32; 2000/2000 steps; step_best + per-200 checkpoints + _evals on disk).
This experiment's one prerequisite (r3 done) is met.

**VRAM math:** vk-base-class 561 M model bf16 ≈ **1.1 GB** + LoRA rank-8/16
wraps (~0.3 M–1 M trainable, tens of MB) + Hippocampus micro-session
fwd/bwd at short seq ≈ **<3 GB total** ⇒ laptop 3070 free GPU fits with
headroom; CPU fallback proven (16 cores handled the 103 M demo; ~5× params
here ⇒ minutes-to-~15 min either way).

**Cheapest honest fit:** **laptop 3070** (free; no box cycle, no droplet).

**Wall time:** ~**10–30 min** for the 3-item protocol + poison/rollback on
the laptop GPU; add ~30 min to re-score step_best on the 12-prompt/conv
instruments as the teach baseline.

**Prerequisites:** SFT r3 step_best selection done (yes); value-embed
auto-detect on load (committed in r2/r3 cycle); no serial dependence on
E1/E2.

---

## E4 — Ladder continuation past 128 on the new spine

**What:** bank-ladder §11 with the break at n=128 addressed: growth-32/64/128
measured on vk-base (3/4 rows; pairwise harm 0.0008 and base tax 0.0056 held;
home-quality broke at n=128 — writing d+0.0239 > 0.02 — and p_home median
0.1464 collapsed → 250 halted, 250/250 plugin fits completed on the old
host). Resume = re-grow the bank past 128 behind the bake-off winner's
higher-capacity router space (the measured recommendation from the halt
verdict): re-wrap the ~250 donor cores in winner-spine geometry, then
assemble at growth checkpoints (128 → 256) and log the four break criteria
per checkpoint.

**VRAM math:** per wrap fit: winner host slice + frozen donor core (25–33 M)
+ rank-64 adapters, backward at B*T ≤256 ⇒ a few GB on top of one 32 GB
MI50 hosting the lazy host (as E2). Bank assembly: 250 plugs × rank-64
wraps ≈ <2 GB resident; donor cores sit behind the same lazy/LRU machinery
the full host already uses. Smaller cards cannot host the spine ⇒ MI50-only,
as E2.

**Cheapest honest fit:** **MI50 box, GPUs 0–5 in parallel** (claim-locked;
this is the cheapest thing that fits and it is free).

**Wall time:** per-core wrap on the winner-class host at FEED 200k ≈
782 steps × ~1.3 s ≈ **~17 min/GPU·core** (mimo-host measured) ⇒ 250 cores
≈ **~70–85 GPU·h** ⇒ ~**12–20 h wall** across 4–6 claim-locked GPUs; growth
reads per checkpoint (256→512-batch CE sweeps through the full host) ≈ 1–3 h
each. Total campaign ≈ **2–3 days** detached. (vk-shim fits measured 1.5–2
h/core — the winner host is ~6× faster per core.)

**Prerequisites:** E2 done (winner spine + registry live and plug-loadable);
donor core_bank harvested (~250 cores exist as safetensors + provenance on
the MI50 box; vk-geometry wraps do NOT transfer — rewrap in winner
geometry); growth-checkpoint harness + complete-file write contract
(growth stub bug fixed) + break-criteria gates ported to the winner host;
/data full ⇒ stage on box /home; E1's output is NOT a prerequisite (this
ladder rides the donor spines, not our pretrain).

---

## E5 — Teach-library scale-up (memory-plug library N→64+)

**What:** grow the consolidated memory library using the shipped recipe
(token-substitution-shaped episodes, owner_mass 0.85, question-match gate,
taught-token bonus default-on): retention PASS measured at N=8 (1.071) and
N=16 (0.737) on the real 8-base spine (`lab/memory_shaping.py`, owner
0.55→0.85 fix); next step per HANDOFF is the end-to-end 8-base pipeline run,
then G4, then the growth ladder N=32/64/64+ with the §8/growth bars re-read
at each step (margin retention ≥0.70, base-CE regression <1% — the group
top-2 occupancy of base tokens is the standing watch item at N≥8).

**VRAM math:** host = vk-base-class 561 M (bf16 1.1 GB) or the winner spine
(as E2); per memory = rank-64 wrap ≈ tens of MB; consolidation fits are
single-MoE-block backward ≈ <3 GB on the 561 M host, E2-scale on the winner
host. ⇒ laptop 3070 honestly fits the 561 M host line; MI50 needed only if
the winner-spine hosts the library.

**Cheapest honest fit:** **laptop 3070** for the vk-base memory line (free);
**MI50 box GPUs 0–5** when the winner spine hosts.

**Wall time:** teach loop per item ≈ minutes; consolidation fit ≈
**15–30 min/item** (the persona 6-pack precedent: 782-step fits ≈ 17 min
each, 6× parallel). N=16→64 ≈ **14–28 h GPU** ≈ ~1–2 days claim-locked at
4-way parallelism; growth reads included.

**Prerequisites:** E3 (teach loop seated on the r3 talkable base);
end-to-end 8-base pipeline with owner 0.85 in-recipe (the queued HANDOFF
step); query-vocabulary discipline (the ban list) and the dilution guard,
both shipped; at N≥16 re-read the group-occupancy base-CE bar before
continuing — hierarchical addressing is the documented fallback if it breaks.

---

## Execution order

1. **E3 first** (laptop, free, unblocked today — r3 done 2026-10-16) —
   cheapest experiment in the set and it seats the teach loop on the
   talkable base that E5 needs.
2. **E1 and E2 next, independent of each other:**
   - **E1** — as soon as droplet ledger headroom is confirmed (paired
     run ≈ 2 × the $41 t3 anchor; TTL + destroy+verify). Blocks nothing;
     do not serialize the program behind it.
   - **E2** — as soon as the user names the bake-off winner (rubric tally
     on file favors glm-flash 4/10 vs mimo-flash 0/10, but the protocol
     pick is the user's feel). Everything banked-mass hangs off this.
3. **E5 after E3 (+E2 if the winner spine hosts the library).**
4. **E4 last** — needs E2; the ~2–3-day claim-locked campaign on the MI50
   box; respect the co-tenant claim protocol; GPUs 6–7 untouched.

**Cost/frame checklist on every rung:** TTL timer + watchdog on any droplet;
destroy + `list`-verify after use; claim locks on the MI50 box; detached
`setsid` launches; never touch the retained user lanes; commit code/docs
only.
