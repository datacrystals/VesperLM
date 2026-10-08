# Lab — experiment farm

Filesystem-only harness for small training experiments. Agent workers drop
experiment JSONs into `queue/`; `runner.py` executes them N-at-a-time on one
GPU; `promote.py` compares results against the tier baseline and queues
follow-ups at the next tier. No daemons, no database.

## Layout

| dir | purpose |
|---|---|
| `queue/` | incoming experiment JSONs (you write here) |
| `running/` | claimed experiments, moved here while executing |
| `results/` | one result JSON per finished run |
| `logs/` | full training logs |
| `sandbox/` | per-run working dirs (gitignored, disposable) |
| `experiments/` | seed / archived experiment definitions |
| `data_synth/` | shared synthetic token bins, generated once by the runner |

## Experiment schema

```json
{
  "id": "t0-passport-vs-topk-a",
  "tier": "t0",
  "baseline": false,
  "hypothesis": "one sentence",
  "config": "lab_tiny",
  "env": {"VESPER_ROUTER_TYPE": "passport", "VESPER_ROUTER_EXPERT_DROPOUT": "0.1"},
  "budget_minutes": 30,
  "mutation_of": null
}
```

| field | meaning |
|---|---|
| `id` | unique, filesystem-safe; becomes `results/<id>.json` and `logs/<id>.log` |
| `tier` | `t0` \| `t1` \| `t2` \| `t3` |
| `baseline` | exactly one baseline per tier — the reference all others are judged against |
| `hypothesis` | one sentence; keep it falsifiable |
| `config` | registered trainer config name (see `Pretrain/configs/model_configs.py`) |
| `env` | env overrides merged on top of the runner defaults |
| `budget_minutes` | hard wall-clock cap for the run |
| `mutation_of` | parent experiment id, or `null` |

Trainer env overrides you can set in `env`: `VESPER_ROUTER_TYPE`,
`VESPER_PASSPORT_DIM`, `VESPER_ROUTER_EXPERT_DROPOUT`, `VESPER_NUM_EXPERTS`,
plus `VESPER_MICRO_BATCH`, `VESPER_ACCUM`, `VESPER_TOTAL_STEPS`. The runner
always sets `VESPER_AMP=bf16`, `VESPER_CONFIG=<config>`, and
`VESPER_TOTAL_STEPS` scaled from `budget_minutes` (30 min = 400 steps); your
`env` wins on conflict.

## How to submit

1. Write one JSON per experiment into `queue/` (schema above).
2. Wait for `results/<id>.json`. Nothing is ever deleted; if a run is killed
   mid-flight the file stays in `running/` and its partial log in `logs/`.
3. Read `val_loss_final` — that is the only judgment metric.

Run the farm:

```
python3 lab/runner.py --slots 8 --once    # drain the queue, then exit
python3 lab/runner.py --slots 8 --watch   # keep polling the queue every 30s
```

The runner refuses to start a new slot while free VRAM is under 4 GB, and kills
any run at `budget_minutes` (result is still written with `"stopped": "budget"`).
A run that dies with nonzero rc and no validation loss gets `"failed": true`
plus the last 40 log lines embedded in the result.

Each run executes the trainer with cwd `sandbox/<id>/` (synthetic data index,
tokenizer symlink, its own checkpoint dir). Lab sandboxes must never touch
`Pretrain/data/` or production checkpoints — everything a run reads or writes
lives under its sandbox.

## Tier ladder

| tier | config | budget | notes |
|---|---|---|---|
| t0 | `lab_tiny` | 30 min | plumbing + direction; 4 experts, tiny dims |
| t1 | ~30M | ~2 h | real pretraining at small scale, expert-set changes |
| t2 | `tiny_agent_k` | ~8 h | real hybrid architecture (KDA + MLA) |
| t3 | `470m_k` | ~24 h | 470M active; only after t2 passes |

## Promotion

```
python3 lab/promote.py --tier t0 --threshold 0.01
```

Reads all `results/` entries for the tier, finds the `baseline: true` result,
and for every non-baseline that improved `val_loss_final` by more than the
relative threshold writes **one** next-tier entry into `queue/`:

- `tier` bumps t0 → t1 → t2 → t3
- `id` = `<parent-id>-p<next-tier>` (e.g. `t0-passport-p1`)
- `mutation_of` = parent id
- `budget_minutes` = 2 × next-tier default (t1 240, t2 960, t3 2880)
- `env` and `hypothesis` copied from the parent

Idempotent: if a queue/running/result entry with that promoted id already
exists, nothing is written.

## Rules

- Judgment is always by `val_loss_final` at a fixed step budget — never by
  vibes, never by train loss, never by wall-clock luck.
- Exactly one baseline per tier; comparisons are relative to it.
- Sandboxes are disposable and self-contained. Never point a lab run at
  `Pretrain/data/` or production checkpoint directories.
- Nothing is deleted: claim = move `queue/` → `running/`, finish = move
  `running/` → `results/`.
