# AMD OneClick free GPU instances — mechanics, limits, automation + exfil design

Investigation date: 2026-10-09. No instances were created ("documentation only" round);
all findings are from read-only probes of the public HTTP API, launcher HTML, public
source mirrors, and public docs. Anything not directly observed is marked **UNVERIFIED**.

Target URL pattern from the user:

    https://ocr.oneclickamd.ai/github/AMD-AIM/AMD-OneClick/blob/pd_ocr/notebooks/ppocr_vl_demo.ipynb

---

## 1. Executive summary

- The launcher is a thin server-rendered page over a **public, unauthenticated FastAPI**
  ("AMD OneClick Notebook Manager"). Full API surface is exposed at
  `https://ocr.oneclickamd.ai/openapi.json` and `/docs`. Instance creation is a single
  `POST /api/github/notebook/create` — **headless automation is possible: YES**.
- Instances are Kubernetes pods (JupyterLab) fronted by the manager; the runtime
  `url` comes back in the create/status JSON and is the host that actually serves Jupyter
  (mirror source builds `http://<node>:<nodeport>/lab?token=...`; production form
  **UNVERIFIED** but returned as `data.url`).
- Lifecycle per `/api/config` on `ocr.oneclickamd.ai` (VERIFIED, public endpoint):
  **max lifetime 6 h, idle timeout 10 min** — not "1 h TTL". The "~1 h" experience is
  consistent with instances being reaped on *idle* (see §2.5): a long silent cell can get
  the pod killed. A keepalive is mandatory for long runs.
- Arbitrary public GitHub notebooks do load via the URL pattern (documented by AMD
  itself in huggingface/transformers notebook badges and by third-party runbooks).
  Canonical form is `/github/<org>/<repo>/blob/<branch>/<path>`; the branch-less form
  mis-parses.
- GPU spec of the free launcher is **UNVERIFIED** (user reports 8xMI300X; related AMD
  Developer Cloud sells 8xMI300X nodes and the cloud.oneclickamd.ai template catalog says
  "MI300x"). First notebook cell must run `rocm-smi` and record ground truth.
- Exfil recommendation: pull-job-config + push small results (JSON/logs) to **an endpoint
  we control** with a per-run token; GB checkpoints to Hugging Face with a **fine-grained,
  single-repo, revocable** write token (via `hf-mirror.com` if the egress is PRC-side);
  no broad PATs on the box (§5).

---

## 2. Launcher mechanics

### 2.1 What the URL does

`GET https://ocr.oneclickamd.ai/github/{full_path}` returns a small HTML page
("AMD OneClick - Loading Notebook", ~14 KB) with the GitHub coordinates **baked into
inline JS constants** (server-side templating):

```js
const githubOrg = "AMD-AIM";
const githubRepo = "AMD-OneClick";
const githubPath = "notebooks/ppocr_vl_demo.ipynb";
const githubBranch = "pd_ocr";
```

On `DOMContentLoaded` the page itself POSTs to the create API and polls status every 2 s
until `running`/`ready`, then `window.location.href = data.url`. So the browser does
nothing you cannot do with curl. Note `HEAD` returns `405 allow: GET` — use GET/POST.

### 2.2 API surface (from the public OpenAPI spec)

Manager title: "AMD OneClick Notebook Manager — Kubernetes-based Jupyter Notebook
instance management". Endpoints on `ocr.oneclickamd.ai`:

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/github/{full_path}` | none | launcher page (parses the GitHub path) |
| POST | `/api/github/notebook/create` | none | create instance for a GitHub notebook |
| GET | `/api/github/notebook/status?instance_id=` | none | poll until ready |
| POST | `/api/notebook/request` | none (email required) | plain instance request (`{"email","image"}`) |
| GET | `/api/notebook/status?email=` | none | poll the email-flow instance |
| GET | `/api/config` | none | **public config: images, max lifetime, idle timeout** |
| GET | `/health` | none | `{"status":"healthy"}` |
| GET | `/admin`, `/api/admin/instances` | HTTP Basic | admin list |
| DELETE | `/api/admin/instance/{id}`, `/api/admin/instances/all`; POST `/api/admin/cleanup` | HTTP Basic | destroy / cleanup |

Create request body (OpenAPI `GitHubNotebookRequest`, VERIFIED against the JS):

```json
{"org": "datacrystals", "repo": "VesperLM", "branch": "main", "path": "pod/oneclick_boot.ipynb"}
```

Response (`NotebookStatus`): `{"status", "message", "url", "email", "instance_id"}`.
Statuses seen in the JS state machine: `pending, creating, initializing, downloading,
jupyter_starting, running, ready, exists, error, not_found`. If the create response is
already `running|ready|exists` it includes `url` and you can skip polling. Errors are
FastAPI-shaped (`{"detail": ...}`), 422 on missing fields (VERIFIED with an empty POST —
no instance was created).

Per-user tracking uses a cookie `amd_oneclick_gh_instance` (OpenAPI: "Each user gets its
own instance (tracked via cookie)"). Keep a cookie jar; the JS comment "important for
per-user instances" indicates production instance IDs are per-user, unlike the older
mirror which used a deterministic ID per notebook path (§2.7).

### 2.3 URL pattern rules

- Canonical: `https://<host>/github/<org>/<repo>/blob/<branch>/<path/to/file.ipynb>`.
  VERIFIED: `.../github/datacrystals/VesperLM/blob/main/README.md` parses to
  org=datacrystals, repo=VesperLM, branch=main, path=README.md (the launcher accepts any
  file, not just `.ipynb`).
- Branch-less form `.../github/<org>/<repo>/<dir>/<branch>/<file>` **mis-parses**: probed
  with a third-party runbook URL, the server set `branch="oneclickamd"` and
  `path="lfm2_rocm_post_training_pipeline.ipynb"` (it appears to take branch=`parts[-2]`,
  path=`parts[-1]` when there is no `blob` segment). Do not rely on it.
- Branch names containing `/` will break the positional parser (**UNVERIFIED** but the
  mirror parser is naive `split("/")`).
- The path need not exist at parse time — parsing is pure string handling; a missing file
  fails later at instance startup when the pod downloads it.

### 2.4 What the instance actually is (public source mirror)

`AMD-AIM/AMD-OneClick` is **private** (404 even authenticated), but an older public copy
of the service source exists at `github.com/tywuAMD/AMD-OneClick` (README:
"Kubernetes-based Jupyter Notebook instance management with automatic lifecycle control …
Auto-cleanup: 10min idle timeout, 6h max lifetime"). Treat it as highly indicative but
not authoritative for the current production build. From its `app/k8s_client.py`:

- One K8s **Pod + NodePort Service** per instance; node port allocated from `30000+`.
- Pod resources (config-map defaults): CPU request 40 / limit 128, memory request 48Gi /
  limit 256Gi, `amd.com/gpu` request=limit=`GPU_LIMIT` (default **1** in the mirror;
  production value **UNVERIFIED** — user reports 8xMI300X), `/dev/shm` 64Gi
  `emptyDir(memory)`, toleration `amd.com/gpu:NoSchedule`.
- Startup script: pins a Tsinghua PyPI mirror in `~/.pip/pip.conf` (+ `/etc/hosts` pin),
  `pip install jupyter ihighlight`, downloads the notebook from
  `https://raw.githubusercontent.com/<org>/<repo>/<branch>/<path>` **with TLS
  verification disabled**, then runs
  `jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --ServerApp.token='<NOTEBOOK_TOKEN>' --notebook-dir=/app/notebooks`.
- Instance URL is built as
  `http://<SERVICE_HOST>:<nodeport>/lab/tree/<notebook_file>?token=<NOTEBOOK_TOKEN>`
  (mirror default token `amd-oneclick`; production token/URL form **UNVERIFIED**).
  The manager host (`ocr.oneclickamd.ai` → 134.199.132.159) is only the API/launcher
  front door; the Jupyter host returned in `data.url` is a different address.
- Mirror-era GitHub flow: instance ID = `gh-` + `md5("<org>/<repo>/<path>".lower())[:8]`,
  email placeholder `github-<id>@oneclick.local`. (For the AMD demo notebook that is
  `gh-bc9ee351`; probed read-only — no instance existed at the time.)

### 2.5 Lifecycle — the operationally important part

From `/api/config` (VERIFIED on `ocr.oneclickamd.ai`, 2026-10-09):

```json
{
  "available_images": ["docker.io/vivienfanghua/vllm_paddle:ppocr-oneclick"],
  "default_image": "docker.io/vivienfanghua/vllm_paddle:ppocr-oneclick",
  "max_lifetime_hours": 6,
  "idle_timeout_minutes": 10
}
```

Cleanup semantics (mirror `cleanup_idle_instances`, runs every `IDLE_TIMEOUT_MINUTES`):
a pod is destroyed if `uptime >= max_lifetime` **or** the timestamp of its **last
container log line** is older than the idle timeout. Consequences:

1. Absolute ceiling is **6 h**, not 1 h.
2. A long-running cell that prints nothing can be reaped ~10–20 min in. **Keepalive is
   mandatory**: background loop echoing timestamps to container stdout
   (e.g. `while :; do date; sleep 240; done >> /proc/1/fd/1 &`) or continuous training
   logs. This idle-reaping is the most likely explanation of the observed "~1 h" life.
3. Quirk: if the log-timestamp parse fails, `check_pod_activity` returns `None` and the
   pod is *not* idle-killed (only max-lifetime applies). Do not rely on this.
4. Jupyter-server log lines also count as activity, so an open notebook making periodic
   HTTP calls helps, but do not depend on it.

### 2.6 The oneclickamd.ai service family (each host is a separate deployment)

| Host | What it is | Notes |
|---|---|---|
| `ocr.oneclickamd.ai` | unauthenticated GitHub-notebook launcher (the target) | image `vllm_paddle:ppocr-oneclick`; 6 h / 10 min idle; backed the PaddleOCR-VL-1.5 demo |
| `cloud.oneclickamd.ai` | larger "AMD Developer Cloud" product | login (GitHub/ModelScope/email), credits, templates, image catalog (rocm/pytorch 7.2.4, verl-dev, specforge, ComfyUI…), model APIs, token factory, `/rate-limited` page; `/api/config`: 6 h / 30 min idle |
| `radeon.oneclickamd.ai` | AMD Radeon Cloud (credits-based) | Radeon Pro W7900 48 GB per official Zhihu guide; "Add Template" with GitHub repo + notebook path; 1 credit = 1 GPU-hour; 1 active instance per account |
| `workshop.oneclickamd.ai` | 301 → `amd-ai-academy.com/github/wangxunx/ai_sprint_shanghai/blob/shenzhen/...ipynb` | same URL pattern on yet another host |
| `amd-devpod.oneclickamd.ai` | devpod variant (302 at `/`) | **UNVERIFIED** purpose |
| `oneclickamd.ai` / `www` | hub (500 at time of probe) | AMD-AIM org blog points here |
| `api.devcloud.amd.com/v2` | DigitalOcean-style VM API used by `pod/devcloud.py` | paid/credit MI300X VMs (~$2/h in our ledger), not the free launcher |

### 2.7 Docs found (official + third-party)

- **Official AMD**: `huggingface/transformers` `notebooks/README.md` carries
  "Open in AMD Dev Cloud" badges pointing at
  `https://oneclickamd.ai/github/huggingface/notebooks/blob/main/transformers_doc/en/training.ipynb`
  — i.e. AMD markets the pattern for **arbitrary third-party repos**.
- **Official AMD**: `AMD-AIM/hf-radeon-gpu-notebooks` automates notebook execution in
  "Radeon Global One-Click Pods" from GitHub Actions (branch `hf_oneclick_radeon_global`;
  AMD's own headless usage, on the credit-based Radeon Cloud).
- **PaddlePaddle community task doc** (`PaddlePaddle/community`
  `pfcc/paddle-hardware/AMD-PaddleOCR-VL-GPU打卡任务.md`) links the exact demo URL as the
  "Quick Start" for PaddleOCR-VL-1.5 on AMD GPU.
- Third-party runbooks: `joelhenwang/lfm2-training-rocm-eaft-comt-luspo-sdft`
  (`docs/oneclickamd_runbook.md`, `scripts/print_oneclickamd_url.sh` — note its URL form
  lacks `/blob/<branch>` and per §2.3 mis-parses), `hmtxj/Sulphur-2-base-Notebook` README
  (Radeon Cloud template flow).
- Public service source mirror: `tywuAMD/AMD-OneClick` (README + `app/`). Its README
  also shows default admin creds (`admin/admin123`) for the *mirror* — do not attempt to
  use admin endpoints against production.

---

## 3. Confirmed specs and limits

| Item | Value | Status |
|---|---|---|
| Max instance lifetime | 6 h | VERIFIED (`/api/config`) |
| Idle timeout | 10 min (ocr) / 30 min (cloud) | VERIFIED (`/api/config`) |
| Idle definition | last container log timestamp (mirror code) | UNVERIFIED for prod |
| Image (ocr) | `docker.io/vivienfanghua/vllm_paddle:ppocr-oneclick` only | VERIFIED (`/api/config`) |
| GPU type/count | MI300X per image tags/user report; **count UNVERIFIED** (user says 8; mirror default 1; cloud template mentions 4 Instinct GPUs; AMD Dev Cloud VMs offer 1 or 8 MI300X) | UNVERIFIED |
| CPU / RAM / disk | mirror: 40–128 CPU, 48–256Gi RAM, 64Gi shm; disk UNVERIFIED | UNVERIFIED for prod |
| Instance URL form | returned in `data.url`; mirror: `http://<node>:<port>/lab?token=…` | UNVERIFIED for prod |
| Jupyter token | mirror default `amd-oneclick`; comes embedded in `data.url` | UNVERIFIED for prod |
| Concurrency per user | mirror: 1 instance per notebook path (deterministic id); prod: per-cookie | UNVERIFIED |
| Rate limits on launcher | none in OpenAPI; related Radeon Cloud docs: launches ~1/min, 3/10 min, 5/h | UNVERIFIED for ocr |
| Auth required | none for create/status | VERIFIED |
| ToS / acceptable use | no ToS found on the unauthenticated launcher | UNVERIFIED / none published |
| "1 hour TTL" | contradicted: 6 h max + 10 min idle | VERIFIED |

Adjacent, for context: the *authenticated* AMD Developer Cloud gives "qualified
developers" an initial **25 complimentary credit hours** (8xMI300X node burns 8 GPU-h/h)
— application-based, GitHub login, SSH keys ([AMD blog](https://www.amd.com/en/developer/resources/technical-articles/2025/how-to-get-started-on-the-amd-developer-cloud-.html)).
Radeon Cloud is credits via the AMD AI Developer Program (ADP), 1 point = 1 GPU-hour,
max 20 points/day redemption (official Zhihu guide in `AMD-AIM/zhihu_rednote_articles`).

---

## 4. Do arbitrary notebooks work? — YES (documentation-level confirmation)

Evidence (no instance burned):

1. AMD's own badge links in `huggingface/transformers` `notebooks/README.md` launch
   notebooks from `huggingface/notebooks` (a non-AMD repo) through the pattern.
2. `workshop.oneclickamd.ai` redirects to a notebook in `wangxunx/ai_sprint_shanghai`.
3. Third-party runbook (`joelhenwang/...`) documents launching a private-user notebook
   through the same service family.
4. Probed parse of `.../github/datacrystals/VesperLM/blob/main/README.md` — coordinates
   accepted and templated (creation would work the same; not executed).

Caveats:

- The demo repo `AMD-AIM/AMD-OneClick` itself is **private**; the service apparently
  fetches it with server-side GitHub credentials (or the demo is currently broken).
  For us: use `datacrystals/VesperLM` (public) — no special case needed.
- Only one image is offered on `ocr`; you cannot choose the container.
- The notebook is downloaded at pod startup; if the raw URL 404s the instance comes up
  without the file (likely still with Jupyter) — treat as failure mode.

---

## 5. Persistence / exfil for a 1-hour-to-6-hour box

### 5.1 What outbound networking we know

VERIFIED from the pod startup script (mirror) that at boot the box can reach:
`raw.githubusercontent.com` (HTTPS, **with TLS verification disabled** — expect
interception/cert weirdness and add retries), and `pypi.tuna.tsinghua.edu.cn` (pip is
pinned to the Tsinghua mirror; `/etc/hosts` is edited). This strongly suggests
**PRC-side egress** (consistent with the demo's Baidu/Paddle affiliations and a
China-Telecom `SERVICE_HOST` in the mirror config). UNVERIFIED from inside a live box:
`github.com:443` (clone/push), `huggingface.co`, arbitrary HTTPS to our own host, DNS,
UDP. Plan for: `pip install` works (mirror), HF may need `HF_ENDPOINT=https://hf-mirror.com`,
GitHub HTTPS likely works but flaky (retries).

### 5.2 Options

| Option | Fit | Security |
|---|---|---|
| (a) `git push` via PAT | OK for small JSON; needs `github.com` egress | Token visible to operator + anyone with the Jupyter URL. Only a **fine-grained PAT, contents:write on one throwaway results repo, ≤7-day expiry, revoke after run**. Never a classic PAT, never a token on other repos. |
| (b) HF hub upload, scoped write token | Best for GB checkpoints | Fine-grained token scoped to a single `datacrystals/vesperlm-oneclick-results`-style dataset repo, write role, created per campaign and revoked after. May need `HF_ENDPOINT=https://hf-mirror.com`. Same exposure caveat as (a). |
| (c) Anonymous transfer (`0x0.st`, `transfer.sh`, `file.io`) | Last resort for non-sensitive logs | No confidentiality, no retention/reliability guarantee, may be blocked; results are public to the service operator and the transfer host. |
| (d) Box polls/POSTs to a location we control | **Best primary channel** | We set the policy: per-run bearer token minted in the job config (or pulled at runtime from our endpoint), rotate per job, revoke by deleting the endpoint route. No standing credential on the box. Requires our endpoint reachable from the pod (**UNVERIFIED** — must be probed in the first smoke run). |

### 5.3 Recommendation

Threat model: the instance environment is visible to the service operator (and to anyone
holding the instance URL + Jupyter token, which is itself a bearer secret in `data.url`).
Everything on the box may be disclosed. Therefore:

1. **Small results (metrics JSON, logs — the common case): (d) HTTPS POST to an endpoint
   we control**, auth = short per-run bearer token carried in the job config fetched at
   runtime (config itself can live in a gist; the token is minted/rotated per job and
   dies with the run). Fall back to (a) with a fresh fine-grained PAT on a dedicated
   throwaway results repo if our endpoint is unreachable.
2. **GB checkpoints: (b)** fine-grained HF write token, one repo, one campaign, revoke
   after pull; resume-friendly upload (`huggingface_hub` `upload_folder` with
   `allow_patterns`), via `hf-mirror.com` fallback. Alternatively ship checkpoints home
   through the existing `pod/upload_ckpts.sh` channel if the pod can reach home.
3. **Never** plant a broad GitHub/HF credential; prefer the *pull* pattern (box fetches
   job config + short-lived exfil token from us at start) so no long-lived secret is ever
   committed or cached; assume every planted token will leak and design it to be
   revocable within the run window.
4. (c) only for throwaway, non-sensitive artifacts.

---

## 6. Design draft: `pod/oneclick_boot.ipynb` (do not run yet)

Purpose: one committed notebook that turns a fresh free OneClick instance into a
VesperLM experiment node and gets the results off-box before the reaper fires.

Launch URL for us:

    https://ocr.oneclickamd.ai/github/datacrystals/VesperLM/blob/main/pod/oneclick_boot.ipynb

Headless alternative (no browser): `POST /api/github/notebook/create` with
`{"org":"datacrystals","repo":"VesperLM","branch":"main","path":"pod/oneclick_boot.ipynb"}`,
poll `/api/github/notebook/status?instance_id=…`, then execute the notebook on the
returned Jupyter via its kernel websocket API (orchestrator sketch in §6.4).

### 6.1 Cell plan

1. **Probe** — `rocm-smi`, `amd-smi`, `ls /opt/rocm/.info/version`, `python -c "import torch…"`
   / `paddle`, `nproc`, `free -h`, `df -h`, curl reachability matrix
   (raw.githubusercontent.com, github.com, huggingface.co, hf-mirror.com, our endpoint).
   Print one JSON "environment report" (this settles the 8xMI300X question on run #1).
2. **Fetch job config** — `GET CONFIG_URL` (see §6.2), retry x3, validate schema, echo it.
3. **Bootstrap** — `git clone $repo_url` at `$ref` (or `git pull` if present — pods are
   ephemeral so clone is fine); `pip install` from config (the pod already pins the
   Tsinghua mirror; fall back to `--index-url https://pypi.org/simple`).
4. **Checkpoint** — download `$checkpoint.url` with `curl -L -C -` (resume) + sha256 check,
   skip if already present with matching hash.
5. **Run** — write `$cmd` to `run.sh`, launch with `setsid nohup` under `tee` to
   `logs/<job_id>.log`; start the **keepalive** (`date >> /proc/1/fd/1` every 240 s) and
   a watchdog that stops the job at `max_minutes` and forces the exfil path.
6. **Exfil** — collect `$results.paths` (JSON/logs) → POST to `$exfil.endpoint` with
   per-run bearer; fallback chain: fine-grained git push → HF upload (for big artifacts)
   → anonymous upload for non-sensitive only. Write an `exfil_receipt.json` and POST it
   too (last message before exit).
7. **Summary** — print metrics + receipt + wall-clock; keepalive stops.

### 6.2 Parameterization (same committed notebook, different jobs)

The launcher accepts only `org/repo/branch/path` — no env passthrough — so the job
description must live at a **stable external URL** that we edit without touching the repo:

- Primary: gist raw URL with a stable name,
  `https://gist.githubusercontent.com/<user>/<gist_id>/raw/vesperlm_job.json`
  (raw always serves latest revision; edit gist per job).
- Override order: `ONECLICK_JOB_URL` env var (if we ever inject one) → gist URL constant →
  fallback `https://raw.githubusercontent.com/datacrystals/VesperLM/main/pod/oneclick_job.json`.
- Config schema (all fields explicit; notebook hard-fails on unknown/missing):

```json
{
  "job_id": "g3_ctrlB_n8",
  "repo_url": "https://github.com/datacrystals/VesperLM.git",
  "ref": "main",
  "pip": ["fla", "tqdm"],
  "checkpoint": {"url": "https://…/ckpt.tar", "sha256": "…", "dest": "lab/ckpt"},
  "env": {"VESPER_CONFIG": "470m_k"},
  "cmd": "bash lab/run_exp.sh --config g3_ctrlB --n 8",
  "max_minutes": 300,
  "results": {"paths": ["lab/results/*.json", "lab/logs/*.log"]},
  "exfil": {"mode": "https_post", "endpoint": "https://exfil.our.host/v1/ingest/<job_id>",
            "token_env": "EXFIL_TOKEN_PLACEHOLDER", "hf_repo": null}
}
```

Secrets rule: the gist holds at most a **per-run** token (or nothing — the endpoint URL
itself can be unguessable and single-use). Nothing long-lived is ever embedded in the
committed notebook.

### 6.3 Failure / lifecycle handling

- Treat `status: exists` as "reuse the box" and make cells idempotent (skip clone/hash-
  matched downloads).
- Never assume 1 h: budget `max_minutes ≤ 330` (6 h ceiling minus boot/download time).
- Always keep the keepalive on from cell 5 onward; a reaped pod loses unsynced work.
- Everything is designed so a killed pod leaves results already exfiltrated (exfil every
  N minutes during the run, not only at the end, for the small JSON/logs).

### 6.4 Local orchestrator sketch (`pod/oneclick_run.py`, to be written later)

`create → poll (2–5 s) → open kernel websocket on data.url → execute notebook cells
(same order as §6.1, or `papermill`-style parameter injection of CONFIG_URL) → stream
outputs → verify exfil receipt → optional admin-less teardown (let the reaper do it)`.
Reuse the safety conventions from `pod/devcloud.py`: embed `ttl` + epoch in a local
ledger entry per launch, refuse to launch when a previous box is still alive, one
instance at a time out of politeness to a free shared service.

---

## 7. Risks and etiquette

- Free, unauthenticated, and clearly a demo/hackathon funnel: expect rate limits, image
  changes, or shutdown without notice. Do not make it the only path for anything critical;
  the primary plan stays `pod/devcloud.py` / spot-pod.
- The operator can see the box contents and network traffic; §5.3 rules apply.
- One notebook path per user-ish instance (cookie); other people running our notebook
  path could share/reuse instances — keep our jobs in our repo namespace and treat the
  box as dirty.
- Do not probe or use `/api/admin/*` (mirror ships default creds; production is not the
  mirror, and trying would be abuse).
- Repo hygiene: never commit `Dataset/data/index.txt`.

## 8. Source list

- Launcher/API (live): `https://ocr.oneclickamd.ai/` (`/openapi.json`, `/docs`,
  `/api/config`, `/health`) — probed 2026-10-09.
- Service source mirror: `https://github.com/tywuAMD/AMD-OneClick`
- AMD badge usage: `https://github.com/huggingface/transformers/blob/main/notebooks/README.md`
- AMD CI automation example: `https://github.com/AMD-AIM/hf-radeon-gpu-notebooks`
- PaddleOCR-VL task doc: `https://github.com/PaddlePaddle/community/blob/main/pfcc/paddle-hardware/AMD-PaddleOCR-VL-GPU打卡任务.md`
- AMD Developer Cloud blog (specs/credits): `https://www.amd.com/en/developer/resources/technical-articles/2025/how-to-get-started-on-the-amd-developer-cloud-.html`
- Radeon Cloud docs (rate limits): `https://github.com/AMD-AIM/radeon-cloud-docs` (`src/content/docs/api/rate-limits.md`)
- Radeon Cloud credits guide (official Zhihu): `https://github.com/AMD-AIM/zhihu_rednote_articles` (`zhihu/05-radeon-cloud-credits-guide`)
- Third-party runbook: `https://github.com/joelhenwang/lfm2-training-rocm-eaft-comt-luspo-sdft` (`docs/oneclickamd_runbook.md`)
