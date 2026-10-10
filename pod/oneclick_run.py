#!/usr/bin/env python3
"""AMD OneClick free-instance orchestrator (laptop side).

Creates a free OneClick GPU notebook instance for a notebook in this repo,
drives its cells over the Jupyter REST + websocket kernel API, streams cell
stdout back to a local log, and enforces a hard wall-clock cap.

    python3 pod/oneclick_run.py --notebook pod/oneclick_boot.ipynb \
        --job-url https://raw.githubusercontent.com/datacrystals/VesperLM/main/lab/oneclick_jobs/smoke.json \
        --minutes 50

Mechanics (see pod/ONECLICK.md):
  * create:  POST https://ocr.oneclickamd.ai/api/github/notebook/create
             json {org, repo, branch, path}; keep cookie amd_oneclick_gh_instance
  * poll:    GET /api/github/notebook/status?instance_id=... until running/ready
  * drive:   data.url is the Jupyter host; token is embedded in its query string
             (SECRET — never commit; this script redacts it from the local log)

Teardown: the manager exposes destroy only under HTTP Basic /api/admin/*
(which we do not use). There is no unauthenticated delete/stop endpoint
(openapi.json checked 2026-10-09). After we go quiet the operator idle reaper
(~10 min without container log output) reclaims the pod; the notebook-side
keepalive is bounded so it cannot pin a pod past job["max_minutes"].

Dev aid: --attach URL skips creation and drives an already-running Jupyter
(for local protocol dry-runs).
"""

import argparse
import datetime
import http.cookiejar
import json
import os
import queue
import re
import sys
import threading
import time
import uuid
from urllib.parse import parse_qs, urlparse

try:
    import requests
except ImportError:
    sys.exit("pod/oneclick_run.py needs requests: pip install --user requests")

try:
    import websocket
except ImportError:
    sys.exit("pod/oneclick_run.py needs websocket-client: "
             "pip install --user websocket-client")

DEFAULT_BASE = "https://ocr.oneclickamd.ai"
NOTEBOOK_TOKEN_PLACEHOLDER = "__ONECLICK_JOB_URL__"
READY_STATES = ("running", "ready", "exists")
TERMINAL_BAD_STATES = ("error", "not_found", "failed")


class Log:
    """Timestamped log to file + stdout, with secret redaction."""

    def __init__(self, path):
        self.path = path
        self.secrets = []
        self.t0 = time.time()
        self._lock = threading.Lock()
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.f = open(path, "a", buffering=1)

    def add_secret(self, s):
        if s and s not in self.secrets:
            self.secrets.append(s)

    def redact(self, text):
        if not text:
            return text
        for s in self.secrets:
            text = text.replace(s, "REDACTED_TOKEN")
        text = re.sub(r"(token=)[^&\s'\"]+", r"\1REDACTED_TOKEN", text, flags=re.I)
        text = re.sub(r"(Authorization:\s*token\s+)\S+", r"\1REDACTED_TOKEN", text, flags=re.I)
        return text

    def line(self, msg):
        elapsed = time.time() - self.t0
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S")
        out = self.redact(f"[T+{elapsed/60:04.1f}m {stamp}] {msg}")
        with self._lock:
            self.f.write(out + "\n")
            print(out, flush=True)

    def chunk(self, text):
        """Raw cell stdout chunk (possibly multi-line)."""
        for ln in text.splitlines():
            self.line("  | " + ln)


def norm_payload(payload):
    """Flatten {data:{...}}-wrapped manager responses; tolerate flat shape."""
    d = dict(payload or {})
    inner = d.get("data")
    if isinstance(inner, dict):
        for k, v in inner.items():
            d.setdefault(k, v)
    return d


def parse_jupyter_url(url, token=None):
    """Return (http_base, ws_base, token) from a Jupyter URL.

    Handles both bare-host form (http://node:port/lab?token=…) and the
    production proxied form
    (https://ocr.oneclickamd.ai/instance/<id>/lab/tree/<nb>.ipynb?token=…),
    where the Jupyter REST API lives under the proxy prefix
    (https://…/instance/<id>/api/…).
    """
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        raise ValueError(f"not an http(s) Jupyter URL: {url!r}")
    path = p.path or ""
    cut = len(path)
    for marker in ("/lab", "/tree", "/notebooks", "/doc", "/api", "/files"):
        idx = path.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    prefix = path[:cut].rstrip("/")
    http_base = f"{p.scheme}://{p.netloc}{prefix}"
    ws_base = ("wss" if p.scheme == "https" else "ws") + f"://{p.netloc}{prefix}"
    if token is None:
        token = (parse_qs(p.query).get("token") or [None])[0]
    return http_base, ws_base, token


class OneClickManager:
    """Unauthenticated launcher API client with a persistent cookie jar."""

    def __init__(self, base, log, max_create_attempts=3):
        self.base = base.rstrip("/")
        self.log = log
        self.max_create_attempts = max_create_attempts
        self.session = requests.Session()
        self.session.cookies = http.cookiejar.CookieJar()

    def _post_create(self, org, repo, branch, path):
        return self.session.post(
            f"{self.base}/api/github/notebook/create",
            json={"org": org, "repo": repo, "branch": branch, "path": path},
            timeout=60,
        )

    def create(self, org, repo, branch, path, wall_deadline):
        """Bounded-retry create. Returns normalized NotebookStatus dict."""
        last_err = "unknown"
        for attempt in range(1, self.max_create_attempts + 1):
            if time.time() > wall_deadline:
                raise RuntimeError("wall clock cap reached during instance creation")
            self.log.line(f"create attempt {attempt}/{self.max_create_attempts}: "
                          f"{org}/{repo}@{branch}:{path}")
            try:
                r = self._post_create(org, repo, branch, path)
            except requests.RequestException as e:
                last_err = f"network: {type(e).__name__}: {e}"
                self.log.line(f"  create network error, retrying: {last_err}")
                time.sleep(min(60, 5 * attempt))
                continue
            try:
                d = norm_payload(r.json())
            except ValueError:
                d = {"detail": r.text[:300]}
            status = str(d.get("status") or "").lower()
            detail = str(d.get("detail") or d.get("message") or "")
            if r.status_code == 422:
                raise RuntimeError(f"create rejected by API (422): {detail}")
            retryable = (
                r.status_code in (408, 425, 429, 500, 502, 503, 504)
                or status in ("busy", "queue", "queued", "pending", "error", "failed")
                or any(w in detail.lower() for w in ("busy", "queue", "full", "too many"))
            )
            self.log.line(f"  -> http {r.status_code} status={status or '-'} "
                          f"instance_id={d.get('instance_id') or '-'} "
                          f"message={detail[:120]!r}")
            if r.ok and status in READY_STATES and d.get("url"):
                return d
            if r.ok and (d.get("instance_id") or d.get("url")):
                return d  # pending/creating/... -> caller polls
            last_err = f"http {r.status_code} status={status} detail={detail[:200]}"
            if not retryable:
                raise RuntimeError(f"instance creation failed: {last_err}")
            time.sleep(min(90, 10 * attempt))
        raise RuntimeError(f"instance creation failed after "
                           f"{self.max_create_attempts} attempts: {last_err}")

    def wait_ready(self, instance_id, wall_deadline, poll_secs=5, boot_timeout_secs=900):
        boot_deadline = min(wall_deadline, time.time() + boot_timeout_secs)
        last_status = "?"
        while time.time() < boot_deadline:
            try:
                r = self.session.get(
                    f"{self.base}/api/github/notebook/status",
                    params={"instance_id": instance_id}, timeout=30)
                d = norm_payload(r.json())
            except (requests.RequestException, ValueError) as e:
                self.log.line(f"  status poll error (transient): {type(e).__name__}: {e}")
                time.sleep(poll_secs)
                continue
            status = str(d.get("status") or "").lower()
            message = str(d.get("message") or "")
            if status != last_status:
                self.log.line(f"  status={status} message={message[:120]!r}")
                last_status = status
            if status in READY_STATES and d.get("url"):
                return d
            if status in TERMINAL_BAD_STATES:
                raise RuntimeError(f"instance creation failed: status={status} "
                                   f"message={message[:200]!r}")
            time.sleep(poll_secs)
        raise TimeoutError(f"instance {instance_id} not ready within boot timeout "
                           f"(last status={last_status})")


class RemoteKernel:
    """Drive one Jupyter kernel over REST + /api/kernels/{id}/channels websocket.

    Single reader thread demultiplexes iopub/shell messages onto per-request
    queues keyed by parent msg_id.
    """

    def __init__(self, jupyter_url, log, token=None, keepalive_secs=300, cookies=None):
        self.http_base, self.ws_base, self.token = parse_jupyter_url(jupyter_url, token)
        self.log = log
        self.keepalive_secs = keepalive_secs
        self.cookies = dict(cookies or {})
        self.session_id = uuid.uuid4().hex
        self.kernel_id = None
        self.ws = None
        self.waiters = {}
        self.background_ids = set()
        self._send_lock = threading.Lock()
        self._stop = threading.Event()
        self._ka_thread = None
        self._reader_thread = None
        self.s = requests.Session()
        if self.token:
            self.s.headers["Authorization"] = f"token {self.token}"
        if self.cookies:
            self.s.cookies.update(self.cookies)

    # ---------- HTTP helpers ----------

    def _url(self, path):
        return self.http_base + path

    def wait_server(self, timeout_secs=180):
        deadline = time.time() + timeout_secs
        last_err = None
        last_log = 0.0
        while time.time() < deadline:
            try:
                r = self.s.get(self._url("/api/status"), params=self._auth(), timeout=15)
                if r.ok:
                    self.log.line(f"jupyter server up at {self.http_base}")
                    return
                last_err = f"http {r.status_code}"
            except requests.RequestException as e:
                last_err = f"{type(e).__name__}: {e}"
            if time.time() - last_log > 30:
                self.log.line(f"  waiting for jupyter at {self.http_base}: {last_err}")
                last_log = time.time()
            time.sleep(3)
        raise TimeoutError(f"jupyter server not reachable at {self.http_base}: {last_err}")

    def _auth(self):
        return {"token": self.token} if self.token else {}

    def start(self):
        name = "python3"
        try:
            r = self.s.get(self._url("/api/kernelspecs"), params=self._auth(), timeout=30)
            if r.ok:
                specs = r.json()
                default = specs.get("default")
                available = list((specs.get("kernelspecs") or {}).keys())
                if default in available:
                    name = default
                elif "python3" in available:
                    name = "python3"
                elif available:
                    name = available[0]
                self.log.line(f"kernelspecs: default={default} available={available} -> {name}")
        except (requests.RequestException, ValueError) as e:
            self.log.line(f"kernelspecs fetch failed ({e}); assuming 'python3'")
        r = self.s.post(self._url("/api/kernels"), params=self._auth(),
                        json={"name": name}, timeout=60)
        r.raise_for_status()
        self.kernel_id = r.json()["id"]
        self.log.line(f"kernel started: id={self.kernel_id} name={name}")
        ws_url = (f"{self.ws_base}/api/kernels/{self.kernel_id}/channels"
                  f"?session_id={self.session_id}")
        if self.token:
            ws_url += f"&token={self.token}"
        headers = []
        if self.token:
            headers.append(f"Authorization: token {self.token}")
        if self.cookies:
            headers.append("Cookie: " + "; ".join(f"{k}={v}" for k, v in self.cookies.items()))
        self.ws = websocket.create_connection(
            ws_url, header=headers, timeout=20, enable_multithread=True)
        self._reader_thread = threading.Thread(target=self._reader, daemon=True)
        self._reader_thread.start()
        self._ka_thread = threading.Thread(target=self._keepalive, daemon=True)
        self._ka_thread.start()
        self.log.line(f"kernel websocket connected ({self.ws_base})")

    def interrupt(self):
        if not self.kernel_id:
            return
        try:
            r = self.s.post(self._url(f"/api/kernels/{self.kernel_id}/interrupt"),
                            params=self._auth(), timeout=30)
            self.log.line(f"kernel interrupt -> http {r.status_code}")
        except requests.RequestException as e:
            self.log.line(f"kernel interrupt failed: {e}")

    def close(self):
        self._stop.set()
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
        if self.kernel_id:
            try:
                self.s.delete(self._url(f"/api/kernels/{self.kernel_id}"),
                              params=self._auth(), timeout=15)
                self.log.line("kernel deleted (Jupyter kernel only — the pod itself "
                              "is left to the operator idle reaper)")
            except requests.RequestException as e:
                self.log.line(f"kernel delete failed: {e}")

    # ---------- message plumbing ----------

    def _msg(self, msg_type, content, channel, msg_id=None):
        return json.dumps({
            "header": {
                "msg_id": msg_id or uuid.uuid4().hex,
                "username": "oneclick_run",
                "session": self.session_id,
                "msg_type": msg_type,
                "version": "5.3",
                "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
            "parent_header": {},
            "metadata": {},
            "content": content,
            "channel": channel,
            "buffers": [],
        })

    def _send(self, msg):
        with self._send_lock:
            self.ws.send(msg)

    def _reader(self):
        while not self._stop.is_set():
            try:
                raw = self.ws.recv()
            except Exception as e:
                if not self._stop.is_set():
                    self.log.line(f"kernel websocket closed: {type(e).__name__}: {e}")
                return
            if raw is None or raw == "":
                continue
            if isinstance(raw, bytes):
                try:
                    raw = raw.decode("utf-8")
                except UnicodeDecodeError:
                    continue  # binary buffer frame we do not need
            try:
                m = json.loads(raw)
            except ValueError:
                continue
            parent = (m.get("parent_header") or {}).get("msg_id")
            mtype = (m.get("header") or {}).get("msg_type", "")
            content = m.get("content") or {}
            if parent in self.waiters:
                self.waiters[parent].put((mtype, content, m.get("channel") or ""))
            elif parent in self.background_ids:
                if mtype == "execute_reply":
                    st = content.get("status")
                    if st not in ("ok",):
                        self.log.line(f"keepalive cell reply status={st}")
            elif mtype == "stream":
                self.log.chunk(f"[kernel:{content.get('name')}] {content.get('text', '')}")
            elif mtype == "error":
                self.log.chunk("[kernel:error] " + "\n".join(content.get("traceback") or []))

    def _keepalive(self):
        """Trivial no-op cell every keepalive_secs (queues behind busy cells)."""
        while not self._stop.wait(self.keepalive_secs):
            try:
                self.execute("1  # oneclick keepalive", deadline=time.time() + 60,
                             label="keepalive", background=True)
            except Exception as e:
                self.log.line(f"keepalive send failed: {e}")

    def execute(self, code, deadline, label="cell", background=False):
        """Run one cell; stream output to the log. Returns (ok, error_text)."""
        msg_id = uuid.uuid4().hex
        q = queue.Queue()
        if background:
            self.background_ids.add(msg_id)
        else:
            self.waiters[msg_id] = q
        try:
            self._send(self._msg("execute_request", {
                "code": code,
                "silent": False,
                "store_history": not background,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": True,
            }, "shell", msg_id=msg_id))
        except Exception as e:
            self.waiters.pop(msg_id, None)
            self.background_ids.discard(msg_id)
            return False, f"websocket send failed: {type(e).__name__}: {e}"
        if background:
            return True, ""
        self.log.line(f"[{label}] executing ({len(code)} chars)")
        ok = True
        err_text = []
        idle = busy = reply = False
        last_wait_log = time.time()
        while not (reply or (idle and busy)):
            if time.time() > deadline:
                self.waiters.pop(msg_id, None)
                return False, f"{label}: wall clock cap reached while cell was running"
            try:
                mtype, content, channel = q.get(timeout=15)
            except queue.Empty:
                if time.time() - last_wait_log > 55:
                    self.log.line(f"[{label}] still running...")
                    last_wait_log = time.time()
                continue
            if mtype == "stream":
                self.log.chunk(f"[{label}] {content.get('text', '')}")
            elif mtype == "error":
                ok = False
                tb = "\n".join(content.get("traceback") or
                               [f"{content.get('ename')}: {content.get('evalue')}"])
                err_text.append(tb)
                self.log.chunk(f"[{label}] TRACEBACK\n{tb}")
            elif mtype == "execute_result":
                txt = (content.get("data") or {}).get("text/plain", "")
                if txt:
                    self.log.chunk(f"[{label}] => {txt}")
            elif mtype == "status":
                state = content.get("execution_state")
                if state == "busy":
                    busy = True
                elif state == "idle" and busy:
                    idle = True
            elif mtype == "execute_reply":
                reply = True
                if content.get("status") not in (None, "ok"):
                    ok = False
                    err_text.append(f"execute_reply status={content.get('status')}")
        self.waiters.pop(msg_id, None)
        self.log.line(f"[{label}] done ok={ok}")
        return ok, "\n".join(err_text)


def extract_tagged_json(stream_text, tag):
    """Last occurrence of '<tag><json>' in captured stdout."""
    found = None
    for line in stream_text.splitlines():
        idx = line.find(tag)
        if idx != -1:
            blob = line[idx + len(tag):].strip()
            try:
                found = json.loads(blob)
            except ValueError:
                continue
    return found


class StreamCapture:
    """Tee of cell stdout for RESULT_JSON / PROBE_JSON scraping."""

    def __init__(self, log):
        self.log = log
        self.buf = []
        self._lock = threading.Lock()
        self._orig_chunk = log.chunk

        def chunk(text):
            with self._lock:
                self.buf.append(text)
            self._orig_chunk(text)

        log.chunk = chunk

    def text(self):
        with self._lock:
            return "".join(self.buf)


def main():
    ap = argparse.ArgumentParser(description="AMD OneClick free-instance orchestrator")
    ap.add_argument("--notebook", default="pod/oneclick_boot.ipynb",
                    help="local notebook path (cells are driven from this copy)")
    ap.add_argument("--job-url", required=True,
                    help="public raw URL of the job JSON fetched by the boot notebook")
    ap.add_argument("--minutes", type=float, default=50.0,
                    help="hard wall-clock cap for the whole run (default 50)")
    ap.add_argument("--org", default="datacrystals")
    ap.add_argument("--repo", default="VesperLM")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--base", default=DEFAULT_BASE, help="manager base URL")
    ap.add_argument("--log", default=None, help="local log path "
                    "(default lab/logs/oneclick_<utc>.log)")
    ap.add_argument("--max-create-attempts", type=int, default=3)
    ap.add_argument("--boot-timeout-min", type=float, default=15.0)
    ap.add_argument("--keepalive-secs", type=int, default=300,
                    help="period of trivial no-op keepalive cells (default 300)")
    ap.add_argument("--attach", default=None,
                    help="skip creation; drive this Jupyter URL (dev/dry-run)")
    ap.add_argument("--token", default=None,
                    help="Jupyter token if not embedded in --attach URL")
    args = ap.parse_args()

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = args.log or os.path.join("lab", "logs", f"oneclick_{ts}.log")
    log = Log(log_path)
    capture = StreamCapture(log)
    log.line(f"oneclick_run start: notebook={args.notebook} job-url={args.job_url} "
             f"minutes={args.minutes} base={args.base} log={log_path}")

    wall_deadline = log.t0 + args.minutes * 60
    notebook_path = args.notebook
    if not os.path.isfile(notebook_path):
        sys.exit(f"notebook not found: {notebook_path}")
    nb = json.load(open(notebook_path))
    cells = []
    for c in nb.get("cells", []):
        if c.get("cell_type") != "code":
            continue
        src = c.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        cells.append(src.replace(NOTEBOOK_TOKEN_PLACEHOLDER, args.job_url))
    if not cells:
        sys.exit("notebook has no code cells")
    log.line(f"notebook loaded: {len(cells)} code cells; job URL placeholder substituted")

    # ---- acquire an instance (or attach) ----
    if args.attach:
        jupyter_url, token = args.attach, args.token
        log.line(f"attach mode: {jupyter_url}")
    else:
        mgr = OneClickManager(args.base, log, max_create_attempts=args.max_create_attempts)
        try:
            created = mgr.create(args.org, args.repo, args.branch,
                                 notebook_path.replace(os.sep, "/"), wall_deadline)
            instance_id = created.get("instance_id") or ""
            status = str(created.get("status") or "").lower()
            jupyter_url = created.get("url") or ""
            if status not in READY_STATES or not jupyter_url:
                if not instance_id:
                    raise RuntimeError(f"create returned neither ready url nor "
                                       f"instance_id: {created}")
                log.line(f"create status={status or '-'} — polling until ready")
                ready = mgr.wait_ready(instance_id, wall_deadline,
                                       boot_timeout_secs=args.boot_timeout_min * 60)
                jupyter_url = ready.get("url") or jupyter_url
                status = str(ready.get("status") or "").lower()
            if not jupyter_url:
                raise RuntimeError("instance ready but no url returned")
            log.line(f"instance ready: id={instance_id or '-'} status={status} "
                     f"url={jupyter_url}")
        except (RuntimeError, TimeoutError) as e:
            log.line(f"FATAL: {e}")
            log.line("no unauthenticated teardown endpoint exists; nothing to clean up")
            return 2

    # The URL embeds the Jupyter bearer token — keep it out of the log.
    try:
        _, _, url_token = parse_jupyter_url(args.attach or jupyter_url, args.token)
    except ValueError as e:
        log.line(f"FATAL: bad jupyter url: {e}")
        return 2
    if url_token:
        log.add_secret(url_token)
    log.add_secret(args.token)

    # ---- drive cells ----
    # The manager tracks instances via cookie amd_oneclick_gh_instance; the
    # front proxy may require it on /instance/<id>/ routes — always send it.
    m = re.search(r"/instance/([^/]+)", urlparse(args.attach or jupyter_url).path or "")
    cookies = {"amd_oneclick_gh_instance": m.group(1)} if m else {}
    if cookies:
        log.line(f"instance cookie set for: {m.group(1)}")
    try:
        k = RemoteKernel(args.attach or jupyter_url, log, token=args.token or url_token,
                         keepalive_secs=args.keepalive_secs, cookies=cookies)
        k.wait_server(timeout_secs=min(600, max(30, wall_deadline - time.time())))
        k.start()
    except (TimeoutError, requests.RequestException, websocket.WebSocketException,
            ValueError, OSError) as e:
        log.line(f"FATAL: cannot start kernel: {type(e).__name__}: {e}")
        return 2

    cell_ok = []
    cap_hit = False
    try:
        for i, src in enumerate(cells, 1):
            if time.time() > wall_deadline:
                cap_hit = True
                log.line(f"wall clock cap reached before cell {i}; stopping")
                break
            ok, err = k.execute(src, deadline=wall_deadline, label=f"cell{i}")
            cell_ok.append(ok)
            if not ok and "wall clock cap" in err:
                cap_hit = True
                log.line("wall clock cap hit mid-cell; interrupting kernel")
                k.interrupt()
                time.sleep(10)
                # Best-effort: let the cell's own cleanup write its stop file.
                k.execute("import os, pathlib; pathlib.Path(os.environ.get("
                          "'ONECLICK_REPO_DIR','/tmp')).mkdir(parents=True, exist_ok=True)",
                          deadline=min(wall_deadline + 30, time.time() + 30),
                          label="post-interrupt", background=True)
                break
    finally:
        k.close()

    # ---- summary ----
    text = capture.text()
    probe = extract_tagged_json(text, "PROBE_JSON:")
    result = extract_tagged_json(text, "RESULT_JSON:")
    log.line("---- run summary ----")
    log.line(f"cells ok: {cell_ok}")
    if probe:
        keys = ("gpu_count", "gpu_count_torch", "gpu_count_rocm", "gpu_names",
                "torch", "torch_cuda", "hip", "rocm_version", "cpu_count")
        log.line("probe: " + json.dumps({k: probe.get(k) for k in keys if k in probe}))
        egr = {k: v for k, v in probe.items() if isinstance(v, dict) and "ok" in v}
        if egr:
            log.line("egress: " + json.dumps(egr))
    else:
        log.line("probe: PROBE_JSON not found in cell output")
    if result:
        log.line("result: " + json.dumps(result)[:2000])
    else:
        log.line("result: RESULT_JSON not found in cell output")
    log.line(f"teardown: no unauthenticated delete/stop endpoint exists on the "
             f"manager (openapi.json 2026-10-09); going quiet so the operator idle "
             f"reaper (~10 min without container log output) reclaims the pod")
    log.line(f"full log: {log_path}")

    if cap_hit:
        return 3
    if not cell_ok or not all(cell_ok):
        return 4
    if result is None:
        return 5
    if result.get("job_rc") not in (0, None) or result.get("ok") is False:
        return 6
    return 0


if __name__ == "__main__":
    sys.exit(main())
