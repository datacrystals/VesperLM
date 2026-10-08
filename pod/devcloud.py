#!/usr/bin/env python3
"""devcloud.py — AMD Dev Cloud (DigitalOcean-compatible API) VM manager
with hard budget + liveness guards for VesperLM runs.

SAFETY MODEL (the point of this file):
  - Every droplet name embeds its TTL and creation epoch:
        vesper-<tag>-ttl<N>m-<epoch>
  - A system-cron watchdog (every 5 min) destroys any vesper-* droplet
    past its TTL, and ANY vesper-* droplet older than 24h regardless.
  - The droplet also self-destructs at TTL via a systemd timer calling
    the API (survives the laptop being asleep; token lives 600 on the
    droplet only for this).
  - A local ledger (~/.amd_devcloud_ledger.jsonl) meters every cent;
    create refuses once estimated+actual spend exceeds the cap.

Commands:
  balance                       API balance + ledger summary
  list                          droplets with TTL status
  create --ttl-min N [--tag T]  create MI300X droplet (TTL is MANDATORY)
  destroy <id|name>             destroy one
  destroy-all                   destroy every vesper-* droplet
  watchdog                      destroy expired droplets (for cron)
  ledger                        show spend ledger

Token: $DEVCLOUD_TOKEN or ~/.amd_devcloud_token (chmod 600).
"""

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.request
import urllib.error

API = "https://api.devcloud.amd.com/v2"
TOKEN_FILE = os.path.expanduser("~/.amd_devcloud_token")
LEDGER = os.path.expanduser("~/.amd_devcloud_ledger.jsonl")
HOURLY_USD = 2.00                 # gpu-mi300x1-192gb-devcloud
DEFAULT_CAP_USD = 180.00          # refuse creates past this ledger total
HARD_KILL_AGE_S = 24 * 3600       # watchdog kills any vesper-* this old

CREATE_PAYLOAD = {
    "region": "atl1",
    "size": "gpu-mi300x1-192gb-devcloud",
    "image": "amddeveloperclou-rocm101",
    "ssh_keys": [55875564, 55814795],
    "backups": False,
    "ipv6": False,
    "monitoring": False,
    "tags": ["vesperlm"],
    "vpc_uuid": "c93a05d0-79d4-43c0-b371-b8f1ecae84be",
}


def token():
    t = os.environ.get("DEVCLOUD_TOKEN")
    if t:
        return t.strip()
    with open(TOKEN_FILE) as f:
        return f.read().strip()


def req(method, path, payload=None, timeout=30):
    r = urllib.request.Request(
        f"{API}{path}", method=method,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Authorization": f"Bearer {token()}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:300]
        raise SystemExit(f"API {method} {path} -> {e.code}: {body}")


def ledger_write(event):
    with open(LEDGER, "a") as f:
        f.write(json.dumps({**event, "ts": time.time()}) + "\n")


def ledger_summary():
    est = actual = 0.0
    open_cost = {}
    if os.path.exists(LEDGER):
        for line in open(LEDGER):
            e = json.loads(line)
            if e["event"] == "create":
                est += e["est_cost"]
                open_cost[e["id"]] = e
            elif e["event"] == "destroy":
                actual += e.get("cost", 0.0)
                open_cost.pop(e.get("id"), None)
    pending = sum(e["est_cost"] for e in open_cost.values())
    return est, actual, pending


def droplets():
    return req("GET", "/droplets?per_page=100").get("droplets", [])


def ours():
    return [d for d in droplets() if d.get("name", "").startswith("vesper-")]


def parse_ttl(name):
    # vesper-<tag>-ttl<N>m-<epoch>
    try:
        *_, ttl, epoch = name.split("-")
        return int(ttl[3:-1]), int(epoch)
    except (ValueError, IndexError):
        return None, None


def self_destruct_user_data(ttl_min):
    secs = ttl_min * 60 + 120
    return f"""#cloud-config
write_files:
  - path: /root/.dc_token
    permissions: '0600'
    content: |
      {token()}
  - path: /usr/local/bin/pod_selfdestruct.sh
    permissions: '0755'
    content: |
      #!/bin/bash
      T=$(cat /root/.dc_token)
      ID=$(curl -sS http://169.254.169.254/metadata/v1/id 2>/dev/null)
      if [ -z "$ID" ]; then
        NAME=$(hostname)
        ID=$(curl -sS -H "Authorization: Bearer $T" "{API}/droplets?per_page=100" \\
             | python3 -c "import sys,json;print(next((d['id'] for d in json.load(sys.stdin)['droplets'] if d['name']=='$NAME'),''))")
      fi
      [ -n "$ID" ] && curl -sS -X DELETE -H "Authorization: Bearer $T" "{API}/droplets/$ID"
runcmd:
  - systemd-run --on-active={secs} --unit=pod-selfdestruct /usr/local/bin/pod_selfdestruct.sh
"""


def cmd_balance(_):
    bal = req("GET", "/customers/my/balance")
    est, actual, pending = ledger_summary()
    print(f"API account_balance: {bal.get('account_balance')}  "
          f"MTD usage: {bal.get('month_to_date_usage')}")
    print(f"Ledger: committed(est) ${est:.2f} | settled(actual) ${actual:.2f} | "
          f"open ${pending:.2f} | cap ${DEFAULT_CAP_USD:.2f}")
    print("NOTE: AMD promo credits are NOT visible via API; ledger is the source of truth.")


def cmd_list(_):
    ds = ours()
    if not ds:
        print("no vesper-* droplets")
        return
    now = time.time()
    for d in ds:
        ttl, epoch = parse_ttl(d["name"])
        left = (epoch + ttl * 60 - now) / 60 if ttl else float("nan")
        ip = next((n["ip_address"] for n in d.get("networks", {}).get("v4", [])
                   if n["type"] == "public"), "?")
        print(f'{d["id"]}  {d["name"]}  {d["status"]}  ip={ip}  ttl_left={left:.0f}m')


def cmd_create(a):
    ttl_min = a.ttl_min
    est_cost = ttl_min / 60 * HOURLY_USD
    est, actual, pending = ledger_summary()
    if est + est_cost > DEFAULT_CAP_USD and not a.force:
        raise SystemExit(
            f"REFUSED: ledger ${est:.2f} + this ${est_cost:.2f} > cap ${DEFAULT_CAP_USD:.2f}. "
            "Use --force only if you mean it.")
    name = f"vesper-{a.tag}-ttl{ttl_min}m-{int(time.time())}"
    payload = {**CREATE_PAYLOAD, "name": name,
               "user_data": self_destruct_user_data(ttl_min)}
    d = req("POST", "/droplets", payload)["droplet"]
    ledger_write({"event": "create", "id": d["id"], "name": name,
                  "ttl_min": ttl_min, "est_cost": est_cost})
    print(f"created {d['id']} {name} (${est_cost:.2f} committed, ttl {ttl_min}m)")
    if a.wait:
        deadline = time.time() + 900
        while time.time() < deadline:
            d = req("GET", f"/droplets/{d['id']}")["droplet"]
            ip = next((n["ip_address"] for n in d.get("networks", {}).get("v4", [])
                       if n["type"] == "public"), None)
            if d["status"] == "active" and ip:
                print(f"active ip={ip}")
                print(f"ssh root@{ip}")
                return
            time.sleep(10)
        print("WARN: not active after 15 min — check list; watchdog will still enforce TTL")


def _destroy(d):
    req("DELETE", f"/droplets/{d['id']}", timeout=60)
    created = d.get("created_at", "")
    try:
        t0 = dt.datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
        hours = max((time.time() - t0) / 3600, 1 / 60)  # minimum 1 minute billing
    except ValueError:
        hours = 0.0
    cost = hours * HOURLY_USD
    ledger_write({"event": "destroy", "id": d["id"], "name": d["name"],
                  "hours": hours, "cost": cost})
    print(f"destroyed {d['id']} {d['name']} (${cost:.2f} actual)")


def cmd_destroy(a):
    target = a.target
    for d in ours():
        if str(d["id"]) == str(target) or d["name"] == target:
            _destroy(d)
            return
    raise SystemExit(f"no vesper-* droplet matching {target!r}")


def cmd_destroy_all(_):
    ds = ours()
    if not ds:
        print("nothing to destroy")
        return
    for d in ds:
        _destroy(d)


def cmd_watchdog(_):
    now = time.time()
    for d in ours():
        ttl, epoch = parse_ttl(d["name"])
        expired = ttl is not None and now > epoch + ttl * 60
        ancient = False
        try:
            t0 = dt.datetime.fromisoformat(
                d["created_at"].replace("Z", "+00:00")).timestamp()
            ancient = now - t0 > HARD_KILL_AGE_S
        except ValueError:
            pass
        if expired or ttl is None or ancient:
            print(f"watchdog destroying {d['name']} "
                  f"(expired={expired} ttl_parse={ttl is not None} ancient={ancient})")
            _destroy(d)


def cmd_ledger(_):
    if not os.path.exists(LEDGER):
        print("empty ledger")
        return
    for line in open(LEDGER):
        e = json.loads(line)
        ts = dt.datetime.fromtimestamp(e["ts"]).strftime("%m-%d %H:%M")
        if e["event"] == "create":
            print(f'{ts} CREATE  {e["name"]} est ${e["est_cost"]:.2f}')
        else:
            print(f'{ts} DESTROY {e["name"]} actual ${e.get("cost", 0):.2f} '
                  f'({e.get("hours", 0):.2f}h)')
    est, actual, pending = ledger_summary()
    print(f"--- est ${est:.2f} | actual settled ${actual:.2f} | open ${pending:.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("balance").set_defaults(f=cmd_balance)
    sub.add_parser("list").set_defaults(f=cmd_list)
    sub.add_parser("destroy-all").set_defaults(f=cmd_destroy_all)
    sub.add_parser("watchdog").set_defaults(f=cmd_watchdog)
    sub.add_parser("ledger").set_defaults(f=cmd_ledger)
    c = sub.add_parser("create")
    c.add_argument("--ttl-min", type=int, required=True,
                   help="MANDATORY self-destroy TTL in minutes")
    c.add_argument("--tag", default="run")
    c.add_argument("--wait", action="store_true", help="wait for active + print ssh")
    c.add_argument("--force", action="store_true", help="override budget cap")
    c.set_defaults(f=cmd_create)
    d = sub.add_parser("destroy")
    d.add_argument("target")
    d.set_defaults(f=cmd_destroy)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    sys.exit(main())
