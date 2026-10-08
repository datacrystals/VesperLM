# VesperLM spot-pod — checkpoint-and-migrate training across providers

The pod turns any fresh GPU instance (DO MI300X droplet, GCP/Azure/AWS
free credits, Lambda, RunPod, Vast — CUDA or ROCm) into a self-managing
training node that:

- pulls corpus shards from home while already training on what it has,
- uploads every checkpoint home within a minute of it being written,
- resumes from the latest checkpoint after a spot kill or crash,
- needs exactly one command to set up.

## Home side (once)

1. Build the corpus into the repo checkout's `Pretrain/data` (the corpus
   builder writes there directly: `vesperk/*.bin`; the proven 4B-token
   curriculum lives in `pretrain/*.bin`).
2. `mkdir -p ~/vesper_runs` (checkpoint archive lands here, per RUN_NAME).
3. Authorize the instance key (printed by bootstrap on first run).

## Instance side

```bash
git clone https://github.com/datacrystals/VesperLM.git && cd VesperLM
HOME_SSH=tliao@home.lan HOME_DATA=/home/tliao/VesperLM/Pretrain/data bash pod/bootstrap.sh
```

Env knobs: `RUN_NAME` / `VESPER_CONFIG` (default `470m_k`),
`VESPER_MICRO_BATCH`, `VESPER_ACCUM` (throughput tuning per GPU),
`HOME_DATA`, `HOME_RUNS`, `SYNC_GATE_SECS`.

## Files

- `bootstrap.sh` — env + repo + fla + patches + data sync + train loop +
  uploader. Idempotent; safe to re-run after a reboot.
- `MANIFEST` — data sources and relative weights.
- `rebuild_index.sh` — regenerates `Pretrain/data/index.txt` from MANIFEST
  limited to shards actually present (partial sync still trains sanely).
- `upload_ckpts.sh` — checkpoint streaming + local pruning (keeps last 2).

## Notes / known limits

- The trainer resumes from the newest `step_*` in its checkpoint dir on
  every (re)start; the train loop restarts it forever. Spot kill → new
  instance → bootstrap → training continues with <1 checkpoint-interval
  of lost compute.
- If training started on a partial sync, the loop restarts it once after
  the full sync lands so the complete mix activates.
- The trainer is fp32 (P40 heritage). bf16 autocast for the dense parts
  (KDA/MLA wrappers already self-manage dtype) is the MI300X phase-1.5
  task — do it after fp32 bring-up validates on ROCm.
- flash-attn is NOT required (MLA falls back to SDPA via
  `tools/patch_fla.py`), which is what makes ROCm bring-up cheap.
