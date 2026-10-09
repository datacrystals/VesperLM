# VesperLM — Handoff for the Next Agent (updated 2026-10-09 ~09:55 UTC)

**USER DIRECTIVES (standing):** (1) maximally delegate implementation to subagents
(cheap model) — the main agent architects, reviews, integrates, and manages droplet
budget. (2) TOTAL AGENTIC FREEDOM (2026-10-08): never stop and wait on the user —
make routine decisions, keep the program moving; the user drops in occasionally.
A recurring heartbeat cron (every 2h, :43) drives autonomous progress; re-create it
before its 7-day stale expiry. Carry both directives forward through compactions.
(3) FAILURE PROTOCOL (2026-10-09): never stop on failure — document it in
lab/FAILURES.md (entry template + fallback tree live there), pick the next
untried branch (cheapest first), and queue it in the same heartbeat cycle.
Only interrupt the user for budget or architecture-direction decisions.
(4) DESIGN SPECS approved 2026-10-09: MODULAR_MOE.md §8 (episodic memory as
experts, gates G0-G4, G0 done) and SUBSYSTEMS.md (drives/emotions control
layer, gates E0-E4; E0 = telemetry instrumentation only, do it first and
cheap). Build order: laptop farm queue → G1 → E0 can start anytime (logging
only, no GPU).

---

## 2026-10-09 09:55 UTC — t3 RUNNING: 470m_k + passport flagship on MI300X

**RUN STATE:** droplet id **607549362**, `vesper-t3-ttl1500m-1791536812`,
ip **129.212.191.223**, created 2026-10-09 09:06:52 UTC, **launched 09:41 UTC**,
log `/root/t3_470m_passport.log` (droplet). **TTL deadline 2026-10-10 10:06:52 UTC**
(laptop watchdog cron `*/5 pod/devcloud.py watchdog` enforces the name-embedded TTL;
droplet `pod-selfdestruct.timer` fires 2026-10-10 10:12:21 UTC — verified active).
~21h run → ETA finish ~2026-10-10 06:40 UTC, inside TTL. Do NOT extend past TTL;
pull artifacts before it fires (streamer should already have them home).

**CKPT SYNCER (added 10:10 UTC):** checkpoints only save on new-best val; the
original "streaming" was a manual push. Now automated: `/root/ckpt_sync.sh` on the
droplet (repo copy `pod/ckpt_sync.sh`) watches `vesper_linear_checkpoints_470m_k/*/`
for mtime+size changes and pushes via the reverse tunnel (`Host home` = laptop
user-space sshd behind `ssh -N -R 2222:localhost:2222`, keeper loop on laptop,
log /tmp/t3_tunnel.log). Pushed sigs recorded in `/root/.ckpt_sync_seen`.
val@200 (5.4630) ckpt verified home at `lab/imported/t3_ckpts/step_best/checkpoint.pt`
(3,691,829,251 B). Tunnel+sshd must both be up for syncs: laptop
`sshd -f ~/.ssh/t3_sshd/config` on 127.0.0.1:2222 + keeper pid via
`pgrep -f "R 2222:localhost:2222"`.

**LAUNCH ENV (as run — matches plan EXCEPT VESPER_ACCUM, see finding below):**
`VESPER_CONFIG=470m_k VESPER_AMP=bf16 VESPER_ROUTER_TYPE=passport VESPER_MICRO_BATCH=8
VESPER_ACCUM=128 python3 -u 02_pretrain_linear.py` from `/root/VesperLM/Pretrain`
(HEAD 81b33da ≥ 311b3f5). Confirmed in `/proc/<pid>/environ`. Log shows
**Global Effective Batch: 128 (Target: 128), Local Accumulation Steps: 16** →
1.048M tok/step @8192 × 4000 steps = **4.2B tokens**. passport_dim default 64.

**VERIFIED (09:41–09:54 UTC):** (a) config-path inspection on the droplet builds
**PassportRouter in all 10 layers** (`('PassportRouter','vesper_model')`, passport
rows (8,64)); (b) `Hybrid stack: {'kda': 8, 'mla': 2} (10 layers)`; (c) dummy pass
survived ("Max VRAM Successfully Reserved: 10.5GB"), step 0 CE 11.2488 → step 170
CE **5.68** finite and falling on real data; Tok/s 6.5k@seq192 → 37k@1024 →
**51.2k@1792** (probe 57k @seq 5248-6144; seq ramps to 8192 by step 800);
(d) VRAM 7-15GB allocated during ramp / 10.5GB dummy reserve (probe ~20GB @seq 5k+)
of 192GB. 426.77M total / 190.84M active; Muon 359M + AdamW 67M (bnb 8-bit absent →
torch AdamW fallback — harmless on 192GB).

**SAFETY ARMED:** (1) checkpoint stream `pod/upload_ckpts.sh` (KEEP_LOCAL=2) droplet →
**`lab/imported/t3_ckpts/`** — live-verified (`step_best` already landed at 09:50);
transport is a laptop-kept `ssh -N -R 2222:localhost:2222` reverse tunnel to a
user-space sshd on the laptop (127.0.0.1:2222, `~/.ssh/t3_sshd/`) because laptop
system sshd is dead; droplet reaches home as `HOME_SSH=home`. (2) laptop puller every
30 min (`/tmp/t3_puller.sh`) grabs `step_best` updates + `eval_samples_*.json` +
`loss_curve.png` (upload_ckpts.sh uploads step_best only ONCE and never the root
artifacts — pre-existing quirk). (3) droplet tripwire `/root/t3_tripwire.py` (pid
7513): CE NaN/inf or >15 after step 200 → kills trainer (`/root/t3_train.pid`),
writes `/root/t3_tripwire_fired`, keeps droplet for evidence. (4) laptop watchdog cron
+ droplet self-destruct timer as above.

**Crash resume (manual — no auto-resume loop, per plan's exact launch):**
`ssh root@129.212.191.223 'setsid bash /root/t3_train.sh </dev/null >>/root/t3_470m_passport.log 2>&1 &'`
(trainer auto-resumes from highest `step_N` in `vesper_linear_checkpoints_470m_k`).
Droplet bootstrap notes: `python3.12-dev` installed via apt FIRST (swarm4 finding —
triton AMD hip_utils needs it); torch 2.9.1+rocm6.3 + triton 3.5.1; fla@37a6b1c
--no-deps + einops; `tools/patch_fla.py` 3 patch groups OK (env.py, mla.py, kda
num_stages×4).

**Vesper_ACCUM semantics (resolved — do not regress):** `VESPER_ACCUM` = target global
batch in **sequences** (`accumulation_steps = target_acc_steps // (world*batch)`),
NOT micro-step count. `VESPER_MICRO_BATCH=8 VESPER_ACCUM=16` (= the old probe's
"micro 8 / accum 16") is only 16 seqs/step = 131k tok/step = 472M tokens / ~2.5h.
The flagship 4.2B / 21h needs **VESPER_ACCUM=128** (= 8×16 micro-steps = 128 seqs =
1.048M tok/step). Evidence: `lab/imported/probe_kda_mla*.log` show "Global Effective
Batch: 16 (Target: 16)"; trainer `02_pretrain_linear.py` batch-scaling math.

---

## 2026-10-09 — SWARM SESSION 4 ($1.27, destroyed+verified; slot free → t3 launching)

- **t2 seed-2 paired head-to-head: PASS — passport +2.78%** (6.4341 vs 6.6184), pairing
  with unseeded +2.83%. Replication table now 2 seeds at every tier ≤120M:
  t0 +2.67 / t1 +1.95 / t1s2 +2.42 / t2 +2.83 / t2s2 +2.78.
- **passport_dim sweep: capacity irrelevant** (32/64/128 within 0.008 nats at t1).
  Default 64 stands; drop to 32 if parameter frugality ever matters.
- **D55 at t2 (120M): PASS all 9 gates at BOTH owner_mass 0.50 and 0.55** (0.50 purer:
  cross ≤0.214; 0.55 bigger CE deltas). Multi-plug-in protocol now validated at 11M AND
  120M. **Production rule: owner_mass 0.50 for ≥t1 scale, 0.55 at t0.** swarm2's
  cal_steps report-field bug fixed in swarm4_d55_t2.py.
- AMD_PITCH updated: t2-seed2 row added; multi-plug-in bullet rewritten from "honest
  negative" to "failed → root-caused → fixed → re-validated at two scales".
- Evidence: lab/imported/swarm4/ (commit 886209a). Ledger: $14.51 settled, $0 open.
- Droplet gotcha (new): triton AMD hip_utils needs apt `python3.12-dev` on droplets.

---

## 2026-10-09 — SWARM SESSION 3 ($2.83, destroyed+verified; slot handed to swarm4)

- **owner_mass scale check at t1: PASS at 0.50, and the operating point SHIFTS with
  scale** (t0 11M wants 0.55; t1 33M wants 0.50; 0.55 at t1 leaks code cross 0.331).
  Same direction as reject_w (15→10→9): **all routing-supervision knobs weaken as
  capacity grows; per-scale sweep required.** t1/0.50 margins are wide (code cross
  0.255, wiki cross 0.086). Expect t2 (120M) ≈ 0.45 — swarm4 tests exactly that.
- **470m_k A/B (392M, paired seed 7, 970 steps, seq ramp to 8192): PARITY, not a win.**
  Passport leads early (+3.18% @ step 100), topk edges ahead after step 400, final
  −0.27% (4.3723 vs 4.3604 — single-seed noise range). The passport quality advantage
  is confirmed ≤120M and unproven at 392M in a short window. **Decision: t3 launches
  with passport anyway** — parity-on-quality + free modularity (the actual product)
  justifies it, and the full 4000-step run is itself the long-window test the 970-step
  probe can't answer. AMD_PITCH.md updated with the honest 392M row.
- Evidence: lab/imported/swarm3/ (commit e22c90b). Spend: $13.24 settled.
- Droplet quota discovery (agent-15): **account allows ONE GPU droplet** — sessions
  serialize. swarm4 holds the slot now; t3 launches when swarm4 finishes (its
  passport_dim sweep result feeds the t3 config).

---

## 2026-10-08 goal-mode run — deliverables 1–3 DONE (of 4)

- **D2 Hippocampus demo on true KDA+MLA: PASS** (f64ed1b). tiny_agent_k step_3000,
  HIPPO_TARGET_PROFILE=kda_mla + HIPPO_CKPT (21bf76d). Margins all up
  (−4.01→−3.00 / −4.18→−2.73 / −3.77→−2.93), digit-NLL 2.18→3.20, word-NLL
  4.77→4.01; poison batch ROLLBACK (target collapse), incumbent byte-identical.
  34 LoRA wraps = exact kda_mla target set. Evidence: lab/imported/hippo_demo_vesperk.log.
- **KDA/MLA CPU shim set COMPLETE** (97b8d90): added pure-torch causal_conv1d
  (+update/step), rotary (ref+offsets), rms_norm_ref wraps to BOTH Immune/cpu_backend
  and Hippocampus/consolidate; patch via sys.modules (bare import binds the FUNCTION).
  **Pre-existing sigmoid gated-norm bug fixed** (consolidate.py computed swish in the
  sigmoid branch — error 2.14 on unit tensors, load-bearing under KDA o_norm).
  41 PASS / 0 FAIL suite on real step_500 weights; gla_gqa backward compat verified.
  Known non-blockers: state_v_first ignored on CPU (self-consistent), conv/rotary shims
  reject packed cu_seqlens, archived gqa_buggy ckpt has wrong full_type in metadata.
- **Cosmopedia root-caused + regened** (8d30a2e): the 0-byte shard was NOT the suspend —
  config "full" never existed on the hub (ShardWriter creates the file before the stream
  raises). Config → "cosmopedia-v2"; 2 shards, 1.5B tokens, indexed. Mix complete.
- **Speedrun branch READY (not merged)**: `speedrun-opts` (pushed; worktree
  /home/tliao/VesperLM-speedrun) adds env-gated VESPER_COMPILE (dense-submodule
  torch.compile), VESPER_FUSED_CE (fla FusedLinearCrossEntropyLoss, pad==eos masking
  preserved exactly — biggest single win: 65536-vocab logits at seq 8192 ≈ 2.1GB fp32
  per micro-batch), VESPER_VALUE_EMBED (modded-nanogpt value embeds on first/last
  layers; KDA/MLA/GQA sites). Byte-compat sha256-verified when flags off; CPU smoke
  suite green (Pretrain/tests_speedrun_flags.py). **Merge gate: MI300X canary
  (~10 steps each flag) on a swarm session, then merge to main.** Est: +5-15% compile,
  +5-15% fused-CE at long seq, value-embed is a seeded-A/B quality bet (~1-3%).
- **AMD pitch committed**: AMD_PITCH.md (4c17adb) — 4-pt scaling table, plug-in gates,
  honest negatives, costed ladder t3≈$41 → t4≈$150–250 → t5 cluster, repro appendix.
- **Spend: $10.41 settled, $0 open.** ~~t3 (~$41) crosses the user's $40 soft-pause~~
  **USER LIFTED THE CAP (2026-10-08 ~23:59 PDT): "you can have multiple droplets if you
  need and test a few things at once, if it makes it go faster"** — the $40 soft-pause
  is answered; MULTIPLE CONCURRENT DROPLETS approved. Remaining hard rules unchanged:
  $180 ledger cap, TTL + destroy + verify on every droplet, everything pulled home
  and committed. t3 GO: 470m_k full pretrain, passport router, ~21h ≈ $42.

---

## 2026-10-08 — SWARM SESSION 2: multi-plug-in SOLVED ($1.34, destroyed+verified)

**Protocol D55 passes all 9 purity gates at t0** (own ≥0.5 / cross ≤0.3 / CE beats masked):
code .773/.297 (+1.04 nats), math .590/.290 (+0.11), wiki .676/.185 (+0.23).
Recipe: (1) train experts independently with contrastive reject (as before), (2) insert
SEQUENTIALLY with ~100 router-only recalibration steps between insertions, (3) final joint
calibration 800 steps, router-only, over all domains, with a **mutual-exclusion target**:
on each domain the owner expert gets routing mass **0.55**, original base experts share
0.45, and ALL other plug-in passport rows get exactly zero. owner_mass is the dial:
0.5 misses wiki-own (0.491), 0.6+ blows cross gates. **0.55 is the operating point.**
Structural finding: independent reject training clusters plug-in passport rows (each only
trained against the base), so top-2 becomes {owner, other-plug-in}; rows can only be
partitioned in a model where they COEXIST (hence the final joint cal). Joint reject (A),
sequential-only (B), joint training (C), and joint+mutual-excl without sequential (C2)
all FAIL — the full D55 chain is necessary. Cost: ~5× phase-2 wall vs independent
plug-in, entirely calibration. Evidence: lab/imported/swarm2/ (9 protocols ablated,
commit 4ec55c5). Note: swarm2_purity.py has a report-field bug (`cal_steps: 0` in JSON
for D55 — actual 800, see log); cosmetic, fix when next touching that script.
**This unblocks the N-expert library vision: repeat insert+recalibrate per new expert,
periodic joint calibration with mutual exclusion.**

---

## 2026-10-08 — SWARM SESSION 1 ($0.83, 6-way parallel, droplet destroyed+verified)

- **t1 second seed PASS**: passport +2.42% (5.968 vs 6.116) vs +1.95% unseeded — the
  advantage is NOT seed luck. Scaling table now 4 points: t0 +2.67% / t1 +1.95% /
  t1-seed2 +2.42% / t2 +2.83%. Trainer gained `VESPER_SEED` env knob (bff5cdb) for
  paired runs (the droplet-local seed hack is gone; use the committed one).
- **reject_w curve at 33M mapped**: passing window **rw10–15** (rw10 best: code util
  0.785, web 0.199, CE −1.16 nats). Slope smooth/monotone → scale→optimal-reject_w
  mapping: 11M ~15, 33M ~10–15 (best 10), 120M ~9. **Production heuristic: larger model
  → lower reject_w; start at 10 for ≥33M.**
- **Growth upcycle PASS at t0**: exact 4→6 expert surgery (clones + dup passport rows +
  top_k 2→4) trains on and matches unexpanded control (Δval −0.0008). Modular growth
  without retraining works at toy scale.
- **MULTI-PLUG-IN FAIL (the important negative)**: 3 independently reject-trained experts
  (code/math/wiki) plugged into one frozen base → each domain gets a CE win from its own
  expert (+0.68/+0.10/+0.05 nats — specialists are real) but routing PURITY fails:
  code↔math cross-talk ~0.5 (gate ≤0.3), wiki expert underused (0.17). Independent
  reject training makes plug-ins COMPETE instead of partition. **Next protocol to try
  (swarm session 2): joint plug-in training — reject terms against the OTHER plug-ins'
  domains too, or sequential plug-in with router recalibration between insertions.**
  This is the gating question for the "9999 plug-in experts" vision.
- Evidence: lab/imported/swarm1/ (results/, plugin/, expand/, logs/, scripts/;
  commits 9dcdf40, 9b417f2). Program spend: $9.07 settled, $0 open.

---

## 2026-10-08 late — laptop suspend-proofed + true Vesper-K restart

- GPU wedge (post-suspend "Unable to determine device handle") fixed by user reboot.
  **Suspend now hard-blocked**: systemd user service `vesper-no-suspend.service`
  (block inhibitor on sleep:idle, auto-starts at login) + PowerDevil AC/Battery/
  LowBattery set to never-suspend via kwriteconfig6. Trainings also launch wrapped
  in their own systemd-inhibit.
- **Old step_500 ckpt ARCHIVED as `vesper_linear_checkpoints_tiny_agent_k_gqa_buggy_archived`**
  — it predates the arch_keys fix (GQA weights in full layers; load fails against the
  fixed trainer's MLA). Do NOT resume from it. Fresh true KDA+MLA tiny_agent_k run
  started 21:35 PDT, micro_batch 2 (3 OOMs the dummy pass), ~4k tok/s at seq-256 ramp.
- **cosmopedia_0.bin was 0 bytes** (builder died at source start, pre-wedge) → deleted;
  `pod/rebuild_index.sh` now SKIPS empty shards with a warning (was: blindly included
  → "cannot mmap an empty file" crash). Cosmopedia regen queued (heartbeat item a).
- Mix now 18 bins (~33GB) until cosmopedia returns; phase-2 index rebuild picks it up
  automatically once regenerated.

## 2026-10-09 — t2 VALIDATED: passport advantage is SCALE-STABLE (all evidence committed)

Droplet session ($~1.8 open at handoff; droplet destroyed+verified after):
- **t2 head-to-head** (tiny_agent_k = 120.2M total / 82.5M active, vocab 65536, 600 steps,
  real fineweb/dclm, 2 parallel jobs ~30k tok/s each): **passport +2.83%**
  (val 6.5410 vs 6.7316). Gap widens late (passport 6.526@400→6.541@500; baseline
  6.816→6.732). Hybrid stack confirmed in trainer: {'kda': 6, 'mla': 2}.
- **SCALING TABLE (the AMD pitch)**: t0 11M **+2.67%** → t1 33M **+1.95%** → t2 120M
  **+2.83%** — the t1 attenuation REVERSED; advantage is scale-stable. Single-job 103M
  bf16 throughput: 103k tok/s @ seq 1024.
- **t2 plug-in test**: train 800 steps fineweb → freeze → separately train expert #5 on
  python code with contrastive passport loss → zero-shot plug-in. The t1 recipe (rw15)
  OVER-SUPPRESSES at 120M (code util 0.378 FAIL). Exact phase-A checkpoint replay sweep
  (lab/imported/t2/plugin_t2_tune.py, verify_ce delta 0.000000): **rw9 PASSES ALL GATES**
  (util_code 0.617 >0.5, util_web 0.171 <0.3, CE_code 7.285 vs 8.725 masked = −1.44 nats);
  rw5 fails web (0.308). **Scale→reject_w: 33M needs rw15, 120M needs rw9 — optimal reject
  weight FALLS as models grow. Production default at 120M: REJECT_W=9.**
- Evidence: `lab/imported/t2/` (head-to-head jsons+logs, plugin jsons for rw15/rw5/rw9,
  both scripts). Commits 53e8912, 29cfb17.

**NEXT DECISIONS (user's), now that the thesis holds at 3 scales:**
(a) AMD-credits pitch package — scaling table + plug-in evidence + costed ladder, all in
    lab/imported/ (user mentioned: beg AMD for credits → cluster → paper).
(b) t3 = 470m_k full pretrain (~$41 projected, single MI300X ~20h) — possibly with
    passport router as the DEFAULT (it's now the better router at every scale tested).
(c) Hippocampus/Immune teach-by-talking demo on fresh Vesper-K ckpt — KDA/MLA port
    LANDED (fbf2fad: target profiles, lazy fla import, MLA CPU fallback); KDA shim
    runtime smoke test on the training box is the only remaining gate.
(d) Speedrun optimizations: torch.compile env-gate, fused linear+CE, value embeddings.

**DROPLET SPEND PRE-APPROVED (2026-10-08, user):** autonomous swarm farm sessions on
MI300X are expected to run without asking. Standing caps: ≤$12 per session, ledger cap
$180 total, TTL mandatory on every droplet, destroy+verify (`pod/devcloud.py list` →
"no vesper-* droplets") after EVERY session, results pulled home + committed+pushed.
Session 1 launched 21:45 PDT (agent-9, droplet farm): t1 second-seed head-to-head,
reject_w midpoint curve (rw10/rw12 at t1), Growth exact-upcycle live test at t0,
3-expert multi-plug-in purity test at t0.

---

## 2026-10-09 — MODULAR-MOE THESIS VALIDATED AT t1 (all evidence committed)

Full session on one MI300X droplet ($3.66; day total $6.93; droplet destroyed+verified):
- **KDA+MLA re-probe after arch_keys fix**: {'kda': 8, 'mla': 2} confirmed in trainer;
  57.0k tok/s steady @ seq 8192 (−3.5% vs GQA — MLA is free on ROCm). 470m_k full run ≈ $41.
- **t0r real-data farm batch** (lab_tiny 11M, fineweb/dclm): passport beats top-k in ALL 5
  variants; best plain passport **+2.67% val** (6.950 vs 7.141). Random-data t0 batch was
  noise-floor (ln vocab) — plumbing-only; always use real data for quality signals.
- **t1 head-to-head** (lab_small 33M, 1590 steps): passport **+1.95%** (6.092 vs 6.213).
  Attenuating with scale (2.67→1.95) — watch at t2; consider 2-seed confirmation.
- **t1 PLUG-IN TEST (the verdict)**: train 600 steps fineweb → freeze → separately train
  expert #5 + passport on python-code with contrastive loss → zero-shot plug-in:
  **rw15/600 steps PASSES ALL GATES**: util_code 0.645 (>0.5), util_web 0.159 (<0.3),
  CE_code 7.64 vs 8.53 without (−0.89 nats). rw5 under-rejects (web 0.345), rw40
  over-suppresses (code 0.28). **Production plug-in default: REJECT_W≈15, phase-B 600 steps,
  reject examples from every non-target domain.** Script: lab/imported/t1_plugin/
  plugin_expert_test_t1.py (env knobs PLUGIN_T1_*). Per-layer finding: selectivity deepens
  with depth at low rw; layer 0 keeps target preference best under high reject pressure.
- Passport throughput cost ≈ 3-4% (72.8k → 70.4k tok/s at t1) — negligible.

**t2 DECISION TAKEN + DONE — see the t2 section above: validated at 120M, rw9 is the
new plug-in default at that scale.**

---

## 2026-10-08 evening — MI300X (AMD Dev Cloud) bring-up: WORKS

Droplet via `pod/devcloud.py` (mandatory TTL, self-destruct timer + laptop watchdog).
Ubuntu/py3.12, ROCm driver 6.19, gfx942 MI300X VF 192GB. Recipe that worked:
1. `pip install torch --index-url https://download.pytorch.org/whl/rocm6.3`
   → torch 2.9.1+rocm6.3, triton 3.5.1 (pytorch-triton-rocm).
2. fla 0.6.0 MUST be pinned by **commit**, not tag: tag `v0.6.0` does not exist
   (max tag v0.5.2 = PyPI max). Use
   `pip install --no-deps "git+https://github.com/fla-org/flash-linear-attention.git@37a6b1c6290e5240f6f0d80419d08a7aac27e548"`
   (matches laptop/box installs). Plus `pip install einops` (fla dep we rely on,
   not in requirements.txt).
3. `python tools/patch_fla.py` — now 3 patches; the new third one (ROCm-only,
   gated on torch.version.hip) caps KDA autotune `num_stages` at 2 in
   fla/ops/kda/{chunk_bwd,chunk_intra,gate,wy_fast}.py. Without it, KDA chunk
   kernels fail to compile on the AMD triton backend: "'tt.load' op operation
   destroyed but still has uses" in make_ttgir (upstream triton#9815 — AMD
   software-pipeliner bug with 4+ loads at num_stages>=3). With the cap: KDA
   chunk fwd+bwd passes in fp32 AND bf16; full tiny_agent_k model trains.
4. patch_fla.py no longer imports fla (find_spec only) — works on GPU-less hosts.

Probe gotchas that cost cycles (do not repeat):
- Probes must pass `vocab_size=65536` (or 65523): model default is 32000 and
  out-of-range token ids surface on ROCm as HSA_STATUS_ERROR_EXCEPTION hardware
  aborts, not a clean assert. Looked exactly like a kernel crash.
- Do NOT name a script `bisect.py` (shadows stdlib bisect → torch import dies
  with a confusing circular-import error).
- Trainer requires phase1 AND phase2 files in data/index.txt (nemotron
  curriculum). Synthetic probe data: two uint16 bins named *phase1*/*phase2*.
- `Pretrain/custom_tokenizer` now tracked in git (7046650) — was silently
  missing on fresh clones.
- p01's stats banner hardcodes "Config: small_v2" — cosmetic, ignore; 02's
  ACTIVE_CONFIG_NAME (env VESPER_CONFIG) is the real one. Hybrid-stack print
  now shows the true layer mix.

**Modular MoE landed (91091fc).** `PassportRouter` (per-expert passport embeddings,
dot-product scoring, expert dropout forcing passport reliance, `register_expert()` hot-plug)
+ `MoEFeedForward.add_expert()` in Common/vesper_model.py — defaults byte-compatible with
running checkpoints. Env overrides VESPER_ROUTER_TYPE/PASSPORT_DIM/ROUTER_EXPERT_DROPOUT/
NUM_EXPERTS. `lab/` experiment farm: queue/runner/promote + tier ladder (t0 lab_tiny →
t3 470m_k), sandboxed per-run cwd (never touches production data/ckpts). MODULAR_MOE.md
spec. **First science result** (lab/plugin_expert_test.py, CPU): zero-shot plug-in of a
separately-trained 5th expert — CE on its domain 7.38 vs 11.07 masked (thesis core holds),
router prefers it on-domain 63% vs 40% chance, but does NOT exclude it off-domain (43% vs
<0.3 target = chance for top-2-of-5) → passport loss needs negative (reject-off-domain)
pressure; that's the next t0 variant. **v2 RESULT (c17e6f5): PASSES** — contrastive
passport loss (reject-on-A term, weight 5.0, 400 phase-B steps): util_A 0.297 (<0.3 ✓,
thin), util_B 0.555 (>0.5 ✓), CE_B 8.01 vs 11.07 without plug-in, domain A undamaged
(4.95 vs 4.86). Mechanism confirmed at CPU scale but margins are thin on
barely-separable synthetic domains — production rule: plug-in must include reject
examples from every non-target domain, and gates should measure utilization margins,
not single cutoffs. **BUG FOUND+FIXED: trainer arch_keys dropped
full_type/kda_head_dim/kv_lora_rank/v_head_dim — all trainer runs so far (overnight
tiny_agent_k, MI300X 470m_k probe) silently built GQA full layers, not MLA.** KDA-on-ROCm
validation stands (linear_type was honored; MLA wrapper passed isolation separately), but
the next droplet session must re-probe KDA+MLA end-to-end with the fixed trainer.

**Throughput probe RESULT (KDA+GQA — see arch_keys bug note above; MLA re-probe pending) (470m_k, bf16, micro 8 / accum 16, synthetic random tokens):
~59.1k tok/s steady-state at full seq 8192, VRAM 14.1GB/192GB, CE ~11.095 ≈ ln(65523) on
noise (correct).** Projected full 4.2B-token 470m_k run: ~20h ≈ **$40** at $2/hr single
MI300X — well inside budget; headroom for micro_batch 32+ or grad-checkpoint-off tuning
would cut it further. Droplet destroyed, $3.27 settled for the whole bring-up.

---

Box: `192.168.1.153` (poweredge-r740, 3× Tesla P40 sm_61). Repo `/home/tliao/VesperLM` there,
AND a fresh local clone on the user's laptop `/home/tliao/VesperLM` (RTX 3070 Laptop 8GB,
torch 2.7.1+cu126, fla 0.6.0@git + local patches via `tools/patch_fla.py`, bitsandbytes user-site).
GitHub `git@github.com:datacrystals/VesperLM.git` is the sync point; laptop clones via HTTPS.

## STATE AS OF 2026-10-08 (newest first)

**Vesper-K exists and trains.** `Common/vesper_linear_model.py` now takes `linear_type="kda"`
(KimiDeltaAttention) and `full_type="mla"` (MultiheadLatentAttention, internal RoPE, no
incremental cache yet — full forward only). Configs: `tiny_agent_k` (103M) and `470m_k` (392M,
param-neutral vs 470m's 395M). Committed `c99c013`. **bf16 trainer path**: `VESPER_AMP=bf16`
env-gated autocast in 02_pretrain_linear.py (dense parts; linear layers stay fp32). Also env
overrides VESPER_CONFIG / VESPER_MICRO_BATCH / VESPER_ACCUM / VESPER_TOTAL_STEPS. Tested
end-to-end on the 3070: 200 steps bf16 on real shards, CE 10.4→7.0.

**RUNNING OVERNIGHT on the laptop 3070**: fresh `tiny_agent_k` pretrain, 6000 steps,
`bash overnight_3070.sh` (driver log `Pretrain/overnight_3070.log`). micro_batch auto-fell
back 4→3→2 (dummy-pass OOM at 4 and 3). Phase 2 auto-extends to 12000 steps if phase 1
completes. Checkpoints `Pretrain/vesper_linear_checkpoints_tiny_agent_k/`.
NEXT MORNING: adapt Hippocampus LoRA targets + Immune cpu_backend shims for KDA/MLA (both
are GLA/GQA-specific right now) and run the integrated teach→consolidate→canary-gate demo
on the fresh checkpoint. GPU probes locally are fine (no politeness needed on the laptop).

**Corpus building on the laptop** (CPU, `Dataset/11_vesperk_corpus.py`, log
`Dataset/corpus_vesperk.log`): ~19.5B-token mix → `Pretrain/data/vesperk/*.bin`
(fineweb_edu 8B, dclm 4B, code 3B, finemath 2B, cosmopedia 1.5B, wikipedia 1B; uint16+eos,
same convention as 03_fineweb.py). Box's 4B curriculum already rsynced to
`Pretrain/data/pretrain/`. Convention: raw text + <|endoftext|>, packed, no header.

**Spot pod** (`pod/`, committed `2349418`): `bootstrap.sh` = one-command cloud-instance setup
(CUDA/ROCm autodetect, fla@v0.6.0+patch_fla, background rsync of data from home, manifest-gated
index build, ckpt pull, auto-resume train loop, uploader). `MANIFEST` + `rebuild_index.sh`
weight sources that have shards PRESENT. `upload_ckpts.sh` streams step_* home, keeps last 2.
Home data dir = the repo's `Pretrain/data` (Pretrain/data symlinks into Dataset/data).
Plan: MI300X via user's AMD dev credits ($2/GPU/hr, $200 total). Phase 1 bring-up (~$5,
2-3h throwaway instance) validates KDA+MLA on ROCm FIRST, then kill; full run only after.
fp32 → bf16 note: trainer default is fp32; always set VESPER_AMP=bf16 off-P40.

**Box 429M run (unchanged, the verdict gate)**: step ~1900/4000 at handoff, val 1600→2.729
(step_best), 1700→2.755, 1800→2.846. DECISION (made): if val@1900-2000 > ~2.95, resume from
step_best@1600 with max_lr halved; else hands off. ETA ~Oct 12-13. CPU probes: loops
tightening, no factual pins yet at 1600.

**Env gotchas (both machines)**: box venv had triton 3.1.0 installed 2026-10-08 ~10:47 UTC
(shadowing user-site triton 3.4.0, breaking fresh `import fla`) — FIXED by installing
triton==3.3.1 into the venv + two cache.py compat patches (kwargs filter, getattr hooks —
harmless no-ops under 3.3.1). Running trainings were never affected (imports cached in
memory). user-site also has triton 3.4.0 (PYTHONPATH=~/.local/... works too). fla 0.6.0 needs
`tools/patch_fla.py` on any fresh machine (find_spec parent-package probe + MLA SDPA fallback
— no flash-attn needed anywhere). Box disk 93% full — mind checkpoints.

**Subagent deliverables (all committed)**: `Growth/` (expert expansion surgery, 14/14 —
exact upcycle = bit-exact clones + duplicate gate rows + top_k DOUBLED; production mode
top_k=2+noise drifts, needs warmup recipe; NO shared-expert slot exists — K2's "+1 shared"
needs a model-class change first). `Immune/` (canary gate + probes + drift; corruption drill
passes, auto-rollback works). `Hippocampus/` (quarantined session log + manual LoRA q/o +
reward-weighted-NLL consolidation; demo: 118M learned "math in words" preference from 6
turns; poison batch rolled back). Moonshot blueprint: `~/moonshot/TEACH_BY_TALKING.md`
(laptop). Design docs: `LMBUS_DESIGN.md`.

## What is running RIGHT NOW (original 2026-10-03 entry below)

**429M pretrain ("470m" config, ACTIVE_CONFIG_NAME="470m")** — launched 2026-10-05 19:11 UTC,
log `/home/tliao/pretrain_470m.log`. Fresh from scratch. dim 1024, 10 layers, 8 experts top-2,
429M total / 193M active. **8k context** (max_seq_len 8192 — first run at >2048), grad_checkpoint
on, micro_batch 1, accum 128 → 1.048M tokens/step, total_steps 4000 → **4.19B token budget**.
Data: fineweb_v2 1.5B + nemotron_phase1 2B (NEW to mix) + nemotron_phase2 0.5B.
Checkpoints: `Pretrain/vesper_linear_checkpoints_470m/` — numbered every 500 (5GB each),
step_best on val improvement. fp16 + Muon (as v2). Watch for: NaN (fp16), disk (75GB free), phase-1→2 switch, seq-len ramp to 8192 by step 800.

Rate/ETA (measured 2026-10-06 01:37, step ~310): 7.4k tok/s total at seq 3200, gentle knee,
~55s/step at 410k tok/step; VRAM 8.1GB. Projects ~5-6 days total (completion ~Oct 10-11).
Mid-run eval samples at steps 1000-2000 give the early semantics read. ckpt_dir bug fixed
(1faf75a): eval_samples_step{N}.json + loss_curve.png now live at checkpoint_dir ROOT —
numbered step_N dirs exist ONLY at %500 saves, do not create others (breaks resume).

Why: the 118M SFT refresh (results below) capped at format-without-semantics on held-out
prompts from TWO bases → 118M = capability ceiling → scale is the lever. This run tests
whether semantics emerge at 429M. SFT of the 429M comes after; `SFT/01_sft_train.py` config
selection will need pointing at the 470m checkpoint dir (it currently resolves v2 step_best).

## CPU probe (Pretrain/cpu_probe.py)

Runs any 470m checkpoint on CPU (~23 tok/s, KV-cached greedy). Shims fla Triton-only ops for
CPU: chunk_gla/fused_recurrent_gla -> naive_recurrent_gla (transpose state, v_first), and
fused gated RMSNorm -> torch (y = rmsnorm(x)*w*(g*sigmoid(g)), norm-first, fp32). Results
@step 600 (236M tok): topical, grammatical, degenerate loops, no factual recall yet — more
coherent than the 118M base was at 100% trained, too early for semantics verdict. NOTE:
cpu_probe prompts are ALSO held-out from SFT EVAL_PROMPTS now — do not reuse across that boundary.

## Probe findings 2026-10-03 (why this refresh exists)

Live probes of the v1 SFT model (`SFT/sft_checkpoints_118m_v1/step_2900/chat_model`, trained
from step_best@4100):

- Valid tool-call JSON and fluent English, BUT one dominant "describe a small Python project"
  script for nearly every prompt.
- `847*392` → wrong tool (`search_players`) + confabulated query. Identity question → fake
  file listing. String-reverser → thought-block repetition loop. hello.py task → malformed
  truncated tool call.
- Base model (pretrain step_best@5200, no SFT): token-salad loops, multilingual garbage,
  broken pseudo-Python.
- **Verdict: SFT v1 bought format/control, not task semantics.** The distill mix + better
  base (@5200) in the refresh are aimed at the semantics gap.

## Refresh RESULTS (2026-10-04, harness on step_2900 — Agent/eval_refresh_5200.log)

- "list all files incl hidden" -> correct `ls -la` first call, then DEGENERATE markdown-table
  loop when summarizing the observation. Observation-integration broken.
- "17*23+145" -> `python -c "print(17*23+145)"` -> 536 -> "The result is 536." PERFECT — but
  this is the templated/memorized one.
- "create notes.txt containing hello" (HELD-OUT) -> malformed `echo 'Hello', "string_word"`,
  no file created. FAIL.
- "13 * 12?" (HELD-OUT) -> MISCOPY: emitted `11 * 12`, ran it raw in bash (no python -c),
  got "command not found", repeated the same broken call, then confabulated about "17 * 12". FAIL.
- **Verdict: refresh = better CE (3.55->~1.5) + kept format, but held-out semantics STILL fail,
  same as v1.** The memorized-vs-novel contrast (python -c for the templated problem, raw bash
  for the novel one) is the cleanest memorization demonstration we have. Two SFT runs from
  different bases both cap at format -> **118M is a capability ceiling, not a data-mix problem.
  Scale is the next lever (429M config exists).** Infra proven end-to-end: auto-resume after
  disk-full, chained held-out eval fired correctly.

## CORRECTION: earlier eval success was contamination

The "17*23+145 = 536" success cited earlier is **not evidence of generalization**: that exact
prompt was in `EVAL_PROMPTS` (SFT/01_sft_train.py) AND the agent harness, and it matches the
templated math examples in the synthetic tooluse data → memorization. Fixed 2026-10-03:
`EVAL_PROMPTS` now uses 4 held-out prompts (capital of Japan, 91*7, count .py files, today's
date), verified absent from `Agent/agent_harness.py`. The harness chain additionally uses 2
fresh prompts invented today (notes.txt creation, 13*12). Treat any future eval success on
prompts resembling training templates with suspicion; prefer the fresh ones.

## State

- **Pretrain v2 COMPLETE**: 1.42B tokens. `Pretrain/vesper_linear_checkpoints_v2/step_best`
  @5200 (val CE **3.272**, was 3.55 @4100). Trainer never writes `step_6000`; exits after
  final eval. Val curve was still descending at the end — ran out of steps, not capacity.
- v1 SFT preserved at `SFT/sft_checkpoints_118m_v1/` (incl. step_2900/chat_model).
  384-param toy SFT at `SFT/sft_checkpoints_tiny384/`.
- Distill pipeline: `Dataset/10_sft_distill.py` (commit `a0d334a`), output
  `Dataset/data/sft/distill_chat_sft.bin`, wired into `Dataset/data/sft/index.txt` @ weight 1.0.
- Recent commits: `1a4a8ad` (permutation MoE, grad-ckpt flag, micro_batch 6, SFT accum fix),
  `802e57b` (KV cache inference, bf16 SFT, NaN guard), `1542633` (eval-sample crash fixes),
  `f106526` (harness step_best parse, chain threshold >= 2900), `a0d334a` (distill).

## Traps (still live — read before touching anything)

- **SFT resume trap**: any `SFT/sft_checkpoints/step_*` is resumed IN PREFERENCE to pretrain
  init. For a fresh SFT, mv the dir aside first (that is exactly what was done for this
  refresh). For crash recovery the same mechanism is your friend.
- Trainer saves the final model inside the last `step_N` dir (e.g. step_2900), never
  `step_3000`. Completion check = `>= 2900` or "Training complete".
- **bf16 required** for anything loading v2 checkpoints (`"amp_dtype": "bfloat16"`): the v2
  residual stream hits 40–60k → fp16 overflows in the last MoE layer → NaN. GradScaler is
  fp16-only.
- `find_unused_parameters=True` in DDP is INTENTIONAL (MoE idle experts). The torch warning
  suggesting removal is a false positive.
- GLA/Mamba2 stay fp32 on P40 (fla fp16 Triton kernels crash sm_61) — wrappers handle it.
- ssh+nohup: give jobs `</dev/null >log 2>&1`; `pkill -f` matches your own ssh cmdline — kill
  by pid.
- `get_latest_checkpoint` ignores `step_best` — pretrain resumes from highest `step_N`.
- **MixedDataStream probs are raw weights** — normalized inside __init__ now (2f22eb5).
  Phase buckets are name-matched ('phase1'/'phase2' substring): nemotron_phase1.bin lands in
  the phase1 stream alongside fineweb. val_probs group-normalized 0.8/0.2.
- **LINEAR_CHECKPOINT_DIR is now per-config** (`vesper_linear_checkpoints_{ACTIVE_CONFIG_NAME}`)
  — switching ACTIVE_CONFIG_NAME no longer resumes the wrong model.
- **Disk-full kills silently** (happened 2026-10-03 23:56 at SFT step 700): checkpoint save
  crashes the run; worse, torchrun relaunches die instantly AND silently because the log file
  itself cant be written. Check `df -h /` FIRST when a run vanishes. Pruned to 119GB free by
  deleting intermediate step_* (kept finals + step_best). Elephant: `/home/tliao/.cache/
  huggingface` is 403GB — candidates for reclaim if needed (distill dumps already converted
  to .bin). SFT run needs ~1.4GB per step_N dir, ~32GB for the remaining 2300 steps.

## Next steps (priority order)

1. Verify refresh results (morning checklist). Compare v2-refresh vs v1 probes — did the
   distill mix + @5200 base buy task semantics, or still format-only?
2. Switch `Agent/agent_harness.py` generation to `forward_incremental` (KV cache; server has
   it, harness still full-recompute — big interactive win).
3. Decide `nemotron_phase1.bin` (2B tokens, finished Oct 1) — still NOT in the pretrain mix.
   Any mix change belongs with the NEXT pretrain, not mid-run.
4. Scale-up pretrain: 429M config (`"470m"` = 429M total / 193M active) with **8k context
   from the start** (max_seq_len 2048 is the binding constraint; changing it needs a retrain
   anyway). Val curve says the 118M run was step-limited — budget more steps/tokens.
5. Fused linear+CE (fla `FusedLinearCrossEntropy`) — vocab-65523 logits are ~10% of step;
   needs care with the SFT mask and pad ignore_index.
6. **Vesper-K pretrain (APPROVED, queued behind current run)**: GLA->KDA linear layers +
   GQA->MLA full layers, MoE unchanged — full spec in LMBUS_DESIGN.md "Next-pretrain spec".
   Fresh pretrain at 429M scale for matched-token comparison vs this run, then scale.

7. LMbus biomimetic sensory stack — see LMBUS_DESIGN.md (full proposal: canonical semantic
   space + per-model bridge, foveated heterogeneous-MoE vision with LM-driven gaze,
   certification = held-out-modality demo on the 118M, staged toward grafted support packs
   on GLM/K-class open models).

## Hardware notes (settled — do not reopen unless user asks)

User weighs MI210 (~$4k) vs 8× Gaudi2 (~$16k) later. fla has first-class ROCm → MI210
(gfx90a, 64GB) is the safe pick; Gaudi2 needs a full TPC/SynapseAI port of GLA/Mamba2.
Advice already given: free vLLM config pass on existing 8× MI50 first → 1–2× MI100 ($1400 ea)
→ used MI300X when budget allows. User has 8× MI50s serving other models (ports Mimo/GLM
flashes to vLLM on them), hates cloud, mortal budget, wants ~1TB VRAM long-term for ~1T-param
4-bit models. Serving note: Chinese frontier models are mostly 4-bit (K3), GLM 8-bit — a
768B GLM-5.3 at 8-bit needs ~800GB → 10× MI210 (640GB) does NOT fit; 8× MI300X (1.5TB) does.
