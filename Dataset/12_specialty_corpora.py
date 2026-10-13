#!/usr/bin/env python3
"""Build the three specialty calibration corpora for the flagship grab-kit
build (MODULAR_MOE §10.7 step FEED: 50-200k specialty tokens per wrap).

Produces, under --out-root (default Dataset/data/specialties/):

  cybersec/   exploit-dev / CTF / defense writeups
  hacking/    pentest methodology / recon
  writing/    high-quality prose (public-domain fiction + curated modern)

each with corpus.jsonl (per-doc text + provenance), corpus.txt (flat concat,
door10-style), and manifest.json (counts, lineage, filter stats).  The bins
under Dataset/data/ are NEVER committed; the builder and the human-readable
summary Dataset/specialties_manifest.md are.

License policy: public-domain / permissive sources only.  Sources named in the
grab brief that are NC or share-alike (HackTricks CC BY-NC-SA, security
StackExchange / OWASP CC BY-SA) are deliberately NOT pulled into the bins;
license-clean substitutes stand in for them and every source is recorded with
its license and attribution in the manifest.

Usage:
  python3 Dataset/12_specialty_corpora.py --tokens 200000
  python3 Dataset/12_specialty_corpora.py --specialty writing --tokens 200000
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import html
import io
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import zipfile
from datetime import date, datetime, timezone

UA = "VesperLM-grabkit-specialties/1.0 (calibration corpus builder)"
ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(ROOT, "data", "specialties")
TRACKED_MANIFEST = os.path.join(ROOT, "specialties_manifest.md")
DEFAULT_TOK_PRIMARY = "/home/tliao/Desktop/Kimi-K3"          # flagship spine
DEFAULT_TOK_SECONDARY = "/home/tliao/.cache/vesper_d0/OLMoE-1B-7B-0924"

MIN_DOC_CHARS = 80
MAX_DOC_CHARS = 20000
JUNK_LINE_RE = re.compile(
    r"(?i)^\W{0,3}(cookie(s)? (policy|consent)|privacy policy|terms of "
    r"(use|service)|table of contents|edit on github|skip to (main|content)"
    r"|navigation menu|subscribe|sign in|all rights reserved|©\s*20\d\d)\b")
SYMBOL_RE = re.compile(r"[\W_]")
TAG_RE = re.compile(r"<[^>]+>")
EN_STOP = {"the", "of", "and", "to", "in", "is", "for", "that", "with", "as"}


# ---------------------------------------------------------------------------
# fetch helpers
# ---------------------------------------------------------------------------

def fetch(url: str, cache_path: str | None = None, timeout: int = 60,
          binary: bool = False) -> bytes:
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return f.read()
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            f.write(data)
    return data


def fetch_json(url: str, cache_path: str | None = None) -> dict:
    return json.loads(fetch(url, cache_path).decode("utf-8", "replace"))


# ---------------------------------------------------------------------------
# text cleaning / quality filter / dedup
# ---------------------------------------------------------------------------

def clean_text(raw: str) -> str:
    raw = html.unescape(TAG_RE.sub(" ", raw))
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    out_lines = []
    for line in raw.split("\n"):
        s = line.strip()
        if not s:
            out_lines.append("")
            continue
        if JUNK_LINE_RE.match(s):
            continue
        if len(s) >= 40 and len(SYMBOL_RE.findall(s)) / len(s) > 0.6:
            continue  # symbol soup / ASCII art / table rules
        out_lines.append(s)
    text = "\n".join(out_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def looks_english(text: str) -> bool:
    letters = sum(c.isalpha() for c in text)
    if letters < 40 or letters / max(1, len(text)) < 0.5:
        return False
    ascii_ok = sum(1 for c in text if ord(c) < 128 or c.isspace()) / max(1, len(text))
    if ascii_ok < 0.88:
        return False
    words = re.findall(r"[A-Za-z']+", text.lower()[:4000])
    if not words:
        return False
    return sum(1 for w in words if w in EN_STOP) >= 3


def split_long(text: str) -> list[str]:
    if len(text) <= MAX_DOC_CHARS:
        return [text]
    paras = text.split("\n\n")
    docs, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) + 2 > MAX_DOC_CHARS:
            docs.append(cur.strip())
            cur = ""
        cur += p + "\n\n"
        while len(cur) > MAX_DOC_CHARS:  # pathological single paragraph
            docs.append(cur[:MAX_DOC_CHARS])
            cur = cur[MAX_DOC_CHARS:]
    if cur.strip():
        docs.append(cur.strip())
    return docs


def shingle_sig(text: str, n: int = 5, k: int = 8) -> frozenset:
    """Winnowed-ish signature: k smallest word-5-gram hashes of the doc."""
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    if len(words) < n:
        words += ["_"] * (n - len(words))
    hs = []
    for i in range(len(words) - n + 1):
        g = " ".join(words[i:i + n])
        hs.append(int(hashlib.md5(g.encode()).hexdigest()[:12], 16))
    return frozenset(sorted(set(hs))[:k])


class DocKeeper:
    """exact-md5 + near-dup (shingle-signature overlap) gate."""

    def __init__(self):
        self.seen: set = set()
        self.sigs: list = []
        self.stats = {"short": 0, "nonenglish": 0, "dup_exact": 0,
                      "dup_near": 0, "kept": 0}

    def admit(self, text: str) -> bool:
        if len(text) < MIN_DOC_CHARS:
            self.stats["short"] += 1
            return False
        if not looks_english(text):
            self.stats["nonenglish"] += 1
            return False
        norm = re.sub(r"\s+", " ", text.lower())
        h = hashlib.md5(norm.encode()).hexdigest()
        if h in self.seen:
            self.stats["dup_exact"] += 1
            return False
        sig = shingle_sig(text)
        for prev in self.sigs:
            if len(sig & prev) >= 6:
                self.stats["dup_near"] += 1
                return False
        self.seen.add(h)
        self.sigs.append(sig)
        self.stats["kept"] += 1
        return True


# ---------------------------------------------------------------------------
# source fetchers: each yields (doc_text, meta_dict)
# ---------------------------------------------------------------------------

def src_nvd(count: int, start_index: int = 0, keywords=None):
    """NVD CVE descriptions — US NIST work, public domain."""
    yielded = 0
    idx = start_index
    while yielded < count:
        url = (f"https://services.nvd.nist.gov/rest/json/cves/2.0"
               f"?resultsPerPage=500&startIndex={idx}")
        data = None
        for attempt in range(4):
            try:
                data = fetch_json(url, None)
                break
            except Exception:
                time.sleep(8 * (attempt + 1))
        if data is None:
            print(f"[nvd] giving up at startIndex={idx}", file=sys.stderr)
            return
        vulns = data.get("vulnerabilities", [])
        if not vulns:
            # error payloads have no vulnerabilities key; totalResults proves
            # there is more, so back off and retry before ending the stream
            if data.get("totalResults", 0) > idx:
                time.sleep(10)
                continue
            return
        for v in vulns:
            cve = v.get("cve", {})
            cid = cve.get("id", "?")
            descs = cve.get("descriptions", [])
            desc = next((d["value"] for d in descs if d.get("lang") == "en"), "")
            if keywords:
                blob = desc.lower()
                if not any(k in blob for k in keywords):
                    continue
            cwes = []
            for w in cve.get("weaknesses", [])[:2]:
                for d in w.get("description", []):
                    if d.get("lang") == "en":
                        cwes.append(d.get("value", ""))
            text = f"{cid}: {desc}"
            if cwes:
                text += f"\nWeaknesses: {', '.join(cwes)}"
            yield text, {"url": f"https://nvd.nist.gov/vuln/detail/{cid}"}
            yielded += 1
            if yielded >= count:
                return
        idx += len(vulns)
        time.sleep(7)  # NVD unauthenticated rate limit
    return


def src_attack(tactics: set | None, kinds: set, cache_dir: str, count: int):
    """MITRE ATT&CK enterprise STIX — mitre/cti is Apache-2.0."""
    raw = fetch("https://raw.githubusercontent.com/mitre/cti/master/"
                "enterprise-attack/enterprise-attack.json",
                os.path.join(cache_dir, "enterprise-attack.json"))
    doc = json.loads(raw)
    n = 0
    for obj in doc.get("objects", []):
        if obj.get("type") not in kinds or obj.get("revoked"):
            continue
        if tactics:
            phases = {kc.get("phase_name")
                      for kc in obj.get("kill_chain_phases", [])
                      if kc.get("kill_chain_name") == "mitre-attack"}
            if not (phases & tactics):
                continue
        desc = obj.get("description") or ""
        name = obj.get("name", "")
        if not desc or not name:
            continue
        mitre_id = ""
        for ref in obj.get("external_references", []):
            if ref.get("source_name", "").startswith("mitre-attack"):
                mitre_id = ref.get("external_id", "")
                break
        text = f"{name} ({mitre_id})\n{desc}"
        if obj.get("type") == "attack-pattern" and obj.get("x_mitre_detection"):
            text += f"\nDetection: {obj['x_mitre_detection']}"
        yield text, {"url": f"https://attack.mitre.org/techniques/{mitre_id.lower().replace('.', '/')}"
                    if mitre_id else "https://attack.mitre.org"}
        n += 1
        if n >= count:
            return


def src_metasploit(kind: str, count: int):
    """rapid7/metasploit-framework module docs — BSD-3-Clause."""
    base = ("https://raw.githubusercontent.com/rapid7/metasploit-framework/master/modules/")
    if kind == "exploit":
        paths = [
            "exploits/windows/smb/ms17_010_eternalblue.rb",
            "exploits/windows/smb/ms08_067_netapi.rb",
            "exploits/multi/http/struts2_content_type_ognl.rb",
            "exploits/multi/http/apache_mod_cgi_bash_env_exec.rb",
            "exploits/linux/http/apache_mod_cgi_bash_env_exec.rb",
            "exploits/unix/webapp/drupal_drupageddon.rb",
            "exploits/windows/dcerpc/ms03_026_dcom.rb",
            "exploits/windows/http/exchange_proxylogon_rce.rb",
            "exploits/windows/rdp/cve_2019_0708_bluekeep_rce.rb",
            "exploits/linux/http/cve_2021_44228_log4shell.rb",
            "exploits/multi/http/log4shell_header_injection.rb",
            "exploits/windows/http/gitlab_cve_2021_22205.rb",
            "exploits/linux/http/cisco_hyperflex_hx_vmtools_rce.rb",
            "exploits/unix/ftp/proftpd_133c_backdoor.rb",
            "exploits/windows/isapi/ms03_007_nsiislog.rb",
            "exploits/windows/http/icecast_header.rb",
            "exploits/linux/samba/usermap_script.rb",
            "exploits/linux/http/netgear_unauth_exec.rb",
            "exploits/windows/http/http_sys_ods.rb",
            "exploits/multi/browser/java_signed_applet.rb",
        ]
    else:  # auxiliary = recon / scanning methodology
        paths = [
            "auxiliary/scanner/http/dir_scanner.rb",
            "auxiliary/scanner/http/robots_txt.rb",
            "auxiliary/scanner/portscan/tcp.rb",
            "auxiliary/scanner/portscan/syn.rb",
            "auxiliary/scanner/smb/smb_version.rb",
            "auxiliary/scanner/smb/smb_enumshares.rb",
            "auxiliary/scanner/ssh/ssh_version.rb",
            "auxiliary/scanner/ssl/openssl_heartbleed.rb",
            "auxiliary/scanner/http/http_version.rb",
            "auxiliary/scanner/http/wordpress_scanner.rb",
            "auxiliary/scanner/discovery/udp_sweep.rb",
            "auxiliary/scanner/snmp/snmp_enum.rb",
            "auxiliary/scanner/db/mysql_version.rb",
            "auxiliary/scanner/ftp/ftp_version.rb",
            "auxiliary/scanner/smtp/smtp_version.rb",
            "auxiliary/scanner/dns/dns_amp.rb",
            "auxiliary/scanner/vnc/vnc_none_auth.rb",
            "auxiliary/scanner/rdp/rdp_scanner.rb",
            "auxiliary/scanner/http/options.rb",
            "auxiliary/scanner/telnet/telnet_version.rb",
        ]
    n = 0
    for p in paths:
        try:
            body = fetch(base + p, None, timeout=30).decode("utf-8", "replace")
        except Exception:
            continue
        yield body, {"url": base + p}
        n += 1
        if n >= count:
            return


def src_capec(count: int, cache_dir: str):
    """MITRE CAPEC attack patterns — MITRE terms (attribution; see runbook)."""
    raw = fetch("https://capec.mitre.org/data/xml/capec_latest.xml",
                os.path.join(cache_dir, "capec_latest.xml"))
    import xml.etree.ElementTree as ET
    root = ET.fromstring(raw)
    n = 0
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"
    for ap in root.iter(f"{ns}Attack_Pattern"):
        name = ap.get("Name", "")
        cid = ap.get("ID", "")
        desc = (ap.findtext(f"{ns}Description") or "")
        summary = (ap.findtext(f"{ns}Summary") or "")
        sol = (ap.findtext(f"{ns}Solutions") or "")
        text = clean_text(f"CAPEC-{cid}: {name}\n{summary}\n{desc}\nMitigations: {sol}")
        if not text:
            continue
        yield text, {"url": f"https://capec.mitre.org/data/definitions/{cid}.html"}
        n += 1
        if n >= count:
            return


def src_cwe(count: int, cache_dir: str):
    """MITRE CWE weakness descriptions — MITRE terms (attribution)."""
    import csv
    raw = fetch("https://cwe.mitre.org/data/csv/1000.csv.zip",
                os.path.join(cache_dir, "cwe_1000.csv.zip"), binary=True)
    zf = zipfile.ZipFile(io.BytesIO(raw))
    name = zf.namelist()[0]
    rows = list(csv.reader(io.StringIO(zf.read(name).decode("utf-8", "replace"))))
    n = 0
    for row in rows[1:]:
        if len(row) < 5:
            continue
        cid, cname, desc, ext = row[0].strip(), row[1].strip(), row[4].strip(), \
            row[5].strip() if len(row) > 5 else ""
        text = clean_text(f"CWE-{cid}: {cname}\n{desc}\n{ext}")
        if len(text) < MIN_DOC_CHARS:
            continue
        yield text, {"url": f"https://cwe.mitre.org/data/definitions/{cid}.html"}
        n += 1
        if n >= count:
            return


def src_nist_800_115(cache_dir: str):
    """NIST SP 800-115 technical security testing guide — US gov, PD."""
    pdf = fetch("https://nvlpubs.nist.gov/nistpubs/Legacy/SP/"
                "nistspecialpublication800-115.pdf",
                os.path.join(cache_dir, "nist-sp800-115.pdf"))
    pdf_path = os.path.join(cache_dir, "nist-sp800-115.pdf")
    try:
        out = subprocess.run(["pdftotext", "-layout", pdf_path, "-"],
                             capture_output=True, timeout=120)
        text = out.stdout.decode("utf-8", "replace")
    except Exception:
        return
    # cut the front matter / TOC noise: keep from chapter 1
    m = re.search(r"\n\s*1\.\s+INTRODUCTION", text, re.I)
    if m:
        text = text[m.start():]
    for chunk in split_long(re.sub(r"\n{3,}", "\n\n", text)):
        if looks_english(chunk):
            yield chunk, {"url": "https://csrc.nist.gov/pubs/sp/800/115/final"}


def src_gutenberg(pg_id: int, cache_dir: str):
    """Project Gutenberg plain text — public domain (US)."""
    url = f"https://www.gutenberg.org/cache/epub/{pg_id}/pg{pg_id}.txt"
    try:
        raw = fetch(url, os.path.join(cache_dir, f"pg{pg_id}.txt")).decode(
            "utf-8", "replace")
    except Exception:
        return
    meta = {"url": url, "pg_id": pg_id}
    mt = re.search(r"^Title:\s*(.+)$", raw, re.M)
    ma = re.search(r"^Author:\s*(.+)$", raw, re.M)
    if mt:
        meta["work"] = mt.group(1).strip()
    if ma:
        meta["author"] = ma.group(1).strip()
    m = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", raw)
    if m:
        raw = raw[m.end():]
    m = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", raw)
    if m:
        raw = raw[:m.start()]
    raw = re.sub(r"\[Illustration[^\]]*\]", "", raw)
    raw = re.sub(r"\[Transcriber[^\]]*\]", "", raw, flags=re.I)
    for chunk in split_long(clean_text(raw)):
        if looks_english(chunk):
            yield chunk, meta


# ---------------------------------------------------------------------------
# specialty plans
# ---------------------------------------------------------------------------

PLAN = {
    "cybersec": {
        "goal": "exploit-dev / CTF / defense writeups",
        "streams": [
            ("nvd-cve", "NVD CVE descriptions", "Public Domain (US NIST)",
             "https://nvd.nist.gov", lambda c, cd: src_nvd(3500, 0,
              keywords=["buffer", "overflow", "injection", "rce", "remote code",
                        "use-after-free", "privilege", "bypass", "sql", "xss",
                        "deserial", "out-of-bounds", "heap", "stack", "csrf"])),
            ("attack-attack", "MITRE ATT&CK offensive techniques/malware/tools",
             "Apache-2.0 (MITRE cti)", "https://github.com/mitre/cti",
             lambda c, cd: src_attack(None,
              {"attack-pattern", "malware", "tool"}, cd, 350)),
            ("metasploit-exploits", "Metasploit exploit module docs",
             "BSD-3-Clause (rapid7)", "https://github.com/rapid7/metasploit-framework",
             lambda c, cd: src_metasploit("exploit", 25)),
            ("cwe-weaknesses", "MITRE CWE weakness descriptions",
             "MITRE CWE terms (attribution)",
             "https://cwe.mitre.org/data/csv/1000.csv.zip",
             lambda c, cd: src_cwe(700, cd)),
        ],
    },
    "hacking": {
        "goal": "pentest methodology / recon",
        "streams": [
            ("attack-recon", "MITRE ATT&CK recon/initial-access/lateral phases",
             "Apache-2.0 (MITRE cti)", "https://github.com/mitre/cti",
             lambda c, cd: src_attack(
                 {"reconnaissance", "resource-development", "initial-access",
                  "execution", "lateral-movement", "credential-access"},
                 {"attack-pattern"}, cd, 260)),
            ("capec-patterns", "MITRE CAPEC attack patterns",
             "MITRE CAPEC terms (attribution)", "https://capec.mitre.org",
             lambda c, cd: src_capec(200, cd)),
            ("nist-sp800-115", "NIST SP 800-115 security testing guide",
             "Public Domain (US NIST)",
             "https://csrc.nist.gov/pubs/sp/800/115/final",
             lambda c, cd: src_nist_800_115(cd)),
            ("metasploit-aux", "Metasploit scanner/recon module docs",
             "BSD-3-Clause (rapid7)", "https://github.com/rapid7/metasploit-framework",
             lambda c, cd: src_metasploit("auxiliary", 25)),
            ("nvd-recon-flavor", "NVD CVE descriptions (recon/auth/network)",
             "Public Domain (US NIST)", "https://nvd.nist.gov",
             lambda c, cd: src_nvd(400, 20000,
              keywords=["authentication", "authorization", "info-leak",
                        "information disclosure", "directory", "discovery",
                        "enumeration", "brute", "default credential",
                        "misconfiguration", "path traversal", "ssrf"])),
        ],
    },
    "writing": {
        "goal": "high-quality prose (PD fiction + curated modern)",
        "streams": [
            # curated classics
            ("pg-pride-prejudice", "Pride and Prejudice (Austen 1813)",
             "Public Domain (Project Gutenberg)", "https://www.gutenberg.org/ebooks/1342",
             lambda c, cd: src_gutenberg(1342, cd)),
            ("pg-frankenstein", "Frankenstein (Shelley 1818)",
             "Public Domain (Project Gutenberg)", "https://www.gutenberg.org/ebooks/84",
             lambda c, cd: src_gutenberg(84, cd)),
            ("pg-jane-eyre", "Jane Eyre (Bronte 1847)",
             "Public Domain (Project Gutenberg)", "https://www.gutenberg.org/ebooks/1260",
             lambda c, cd: src_gutenberg(1260, cd)),
            ("pg-moby-dick", "Moby-Dick (Melville 1851)",
             "Public Domain (Project Gutenberg)", "https://www.gutenberg.org/ebooks/2701",
             lambda c, cd: src_gutenberg(2701, cd)),
            # curated modern prose (published 1900-1930, PD in the US)
            ("pg-ulysses", "Ulysses (Joyce 1922)",
             "Public Domain (Project Gutenberg)", "https://www.gutenberg.org/ebooks/4300",
             lambda c, cd: src_gutenberg(4300, cd)),
            ("pg-gatsby", "The Great Gatsby (Fitzgerald 1925)",
             "Public Domain (Project Gutenberg)", "https://www.gutenberg.org/ebooks/64317",
             lambda c, cd: src_gutenberg(64317, cd)),
            ("pg-modern-mix", "curated early-20c prose (see manifest)",
             "Public Domain (Project Gutenberg)",
             "https://www.gutenberg.org",
             lambda c, cd: _gutenberg_mix(cd)),
        ],
    },
}


def _gutenberg_mix(cache_dir: str):
    """Early-modern prose short works/selections (all PD, 1900-1930 era)."""
    for pg_id, label in ((52881, "1920s short prose"),
                         (76872, "1920s novel"),
                         (28054, "The Good Soldier (Ford 1915)"),
                         (1080, "A Modest Proposal (Swift)")):
        for text, meta in src_gutenberg(pg_id, cache_dir):
            yield text, {**meta, "work": label, "pg_id": pg_id}


# ---------------------------------------------------------------------------
# build + manifest
# ---------------------------------------------------------------------------

def load_tokenizer(path_or_name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path_or_name, trust_remote_code=True)


def build_specialty(name: str, plan: dict, out_root: str, cache_dir: str,
                    tok, tok_name: str, target_tokens: int,
                    keeper: DocKeeper) -> dict:
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    t_docs = []
    source_stats = []
    n_tokens = 0
    t0 = time.time()

    # Round-robin across sources so early streams cannot starve later ones
    # (e.g. writing must reach the modern-prose books, not stop after two
    # classics).  Per-source cap 45% of the budget keeps lineage diverse.
    stream_cap = int(target_tokens * 0.45)
    streams = []
    for sid, label, license_, url, fetcher in plan["streams"]:
        try:
            gen = fetcher(target_tokens, cache_dir)
        except TypeError:
            gen = fetcher(target_tokens)
        except Exception as e:
            source_stats.append({"source_id": sid, "error": f"{type(e).__name__}: {e}"})
            continue
        streams.append({
            "sid": sid, "label": label, "license": license_, "url": url,
            "gen": gen, "docs": 0, "tok": 0, "dropped": 0, "done": False,
        })

    pending = [s for s in streams if not s["done"]]
    while pending and n_tokens < target_tokens:
        for s in pending:
            if n_tokens >= target_tokens or s["tok"] >= stream_cap:
                s["done"] = True
                continue
            try:
                text, meta = next(s["gen"])
            except StopIteration:
                s["done"] = True
                continue
            for doc in split_long(clean_text(text)):
                if n_tokens >= target_tokens:
                    break
                if not keeper.admit(doc):
                    s["dropped"] += 1
                    continue
                # K3's tiktoken wrapper prints a debug line per encode call
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    ids = tok.encode(doc, add_special_tokens=False)
                t_docs.append({
                    "text": doc,
                    "source_id": s["sid"],
                    "license": s["license"],
                    "url": meta.get("url", s["url"]),
                    "sha256": hashlib.sha256(doc.encode()).hexdigest(),
                    "n_chars": len(doc),
                    "n_tokens": len(ids),
                })
                s["docs"] += 1
                s["tok"] += len(ids)
                n_tokens += len(ids)
        pending = [s for s in streams if not s["done"]]

    for s in streams:
        source_stats.append({
            "source_id": s["sid"], "label": s["label"], "license": s["license"],
            "home": s["url"], "docs_kept": s["docs"], "tokens": s["tok"],
            "docs_dropped": s["dropped"],
        })

    with open(os.path.join(out_dir, "corpus.jsonl"), "w") as f:
        for d in t_docs:
            f.write(json.dumps(d) + "\n")
    with open(os.path.join(out_dir, "corpus.txt"), "w") as f:
        for d in t_docs:
            f.write(d["text"] + "\n\n")
    manifest = {
        "specialty": name,
        "goal": plan["goal"],
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "builder": "Dataset/12_specialty_corpora.py",
        "target_tokens": target_tokens,
        "tokenizer": {"name": tok_name, "vocab_size": int(tok.vocab_size)},
        "n_docs": len(t_docs),
        "n_tokens": n_tokens,
        "n_chars": sum(d["n_chars"] for d in t_docs),
        "filter_stats": dict(keeper.stats),
        "sources": source_stats,
        "files": ["corpus.jsonl", "corpus.txt"],
        "wall_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def write_tracked_manifest(manifests: dict, tok_name: str) -> None:
    lines = [
        "# Specialty calibration corpora — grab-kit manifest",
        "",
        "Built by `Dataset/12_specialty_corpora.py` (MODULAR_MOE §10.7 step FEED:",
        "50-200k specialty tokens per wrap; door10 zero-crossing ~52k, gap 1.35 ->",
        "0.039 at 50k). **Bins under `Dataset/data/specialties/` are local-only —",
        "never commit them**; this manifest and the builder are the tracked record.",
        "",
        f"- build date (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}",
        f"- counting tokenizer: `{tok_name}`",
        "- license policy: public domain / permissive only. The grab brief's",
        "  examples security StackExchange (CC BY-SA) and HackTricks (CC BY-NC-SA)",
        "  are share-alike/non-commercial and were deliberately NOT pulled;",
        "  license-clean substitutes (NVD, ATT&CK, CAPEC, CWE, Metasploit, NIST,",
        "  Project Gutenberg) stand in. Provenance is recorded per document in",
        "  `corpus.jsonl` and per source below.",
        "",
        "| specialty | docs | tokens | chars | goal |",
        "|---|---|---|---|---|",
    ]
    for name in ("cybersec", "hacking", "writing"):
        m = manifests.get(name)
        if not m:
            continue
        lines.append(f"| {name} | {m['n_docs']} | {m['n_tokens']} | "
                     f"{m['n_chars']} | {m['goal']} |")
    lines += ["", "## Sources and lineage", ""]
    for name in ("cybersec", "hacking", "writing"):
        m = manifests.get(name)
        if not m:
            continue
        lines += [f"### {name}", "",
                  "| source | license | docs | tokens | origin |",
                  "|---|---|---|---|---|"]
        for s in m["sources"]:
            if "error" in s:
                lines.append(f"| {s['source_id']} | — | — | — | "
                             f"ERROR {s['error']} |")
                continue
            lines.append(f"| {s['label']} | {s['license']} | {s['docs_kept']} | "
                         f"{s['tokens']} | {s['home']} |")
        fs = m["filter_stats"]
        lines += ["", f"filter stats: kept {fs['kept']}, short {fs['short']}, "
                  f"non-English {fs['nonenglish']}, exact-dup {fs['dup_exact']}, "
                  f"near-dup {fs['dup_near']}.", ""]
    with open(TRACKED_MANIFEST, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-root", default=DEFAULT_OUT)
    ap.add_argument("--tokens", type=int, default=200_000,
                    help="target tokens per specialty")
    ap.add_argument("--specialty", default="all",
                    choices=["all", "cybersec", "hacking", "writing"])
    ap.add_argument("--tokenizer", default="",
                    help="tokenizer path/name for counting (default: K3 spine "
                         "if present, else the OLMoE host tokenizer)")
    args = ap.parse_args()

    tok_name = args.tokenizer
    if not tok_name:
        tok_name = (DEFAULT_TOK_PRIMARY if os.path.isdir(DEFAULT_TOK_PRIMARY)
                    else DEFAULT_TOK_SECONDARY)
    tok = load_tokenizer(tok_name)
    cache_dir = os.path.join(args.out_root, "_cache")
    os.makedirs(cache_dir, exist_ok=True)

    names = (list(PLAN) if args.specialty == "all" else [args.specialty])
    manifests = {}
    for name in names:
        print(f"== building {name} (target {args.tokens} tokens, tok={tok_name})")
        keeper = DocKeeper()
        m = build_specialty(name, PLAN[name], args.out_root, cache_dir,
                            tok, tok_name, args.tokens, keeper)
        manifests[name] = m
        print(f"   docs={m['n_docs']} tokens={m['n_tokens']} "
              f"chars={m['n_chars']} wall={m['wall_s']}s")
    for name in PLAN:  # partial rebuilds keep the other specialties in the table
        if name not in manifests:
            p = os.path.join(args.out_root, name, "manifest.json")
            if os.path.exists(p):
                manifests[name] = json.load(open(p))
    write_tracked_manifest(manifests, tok_name)
    print(f"tracked manifest -> {TRACKED_MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
