# Specialty calibration corpora — grab-kit manifest

Built by `Dataset/12_specialty_corpora.py` (MODULAR_MOE §10.7 step FEED:
50-200k specialty tokens per wrap; door10 zero-crossing ~52k, gap 1.35 ->
0.039 at 50k). **Bins under `Dataset/data/specialties/` are local-only —
never commit them**; this manifest and the builder are the tracked record.

- build date (UTC): 2026-10-13 00:26
- counting tokenizer: `/home/tliao/Desktop/Kimi-K3`
- license policy: public domain / permissive only. The grab brief's
  examples security StackExchange (CC BY-SA) and HackTricks (CC BY-NC-SA)
  are share-alike/non-commercial and were deliberately NOT pulled;
  license-clean substitutes (NVD, ATT&CK, CAPEC, CWE, Metasploit, NIST,
  Project Gutenberg) stand in. Provenance is recorded per document in
  `corpus.jsonl` and per source below.

| specialty | docs | tokens | chars | goal |
|---|---|---|---|---|
| cybersec | 2009 | 200030 | 832033 | exploit-dev / CTF / defense writeups |
| hacking | 804 | 200060 | 923573 | pentest methodology / recon |
| writing | 41 | 200690 | 805895 | high-quality prose (PD fiction + curated modern) |

## Sources and lineage

### cybersec

| source | license | docs | tokens | origin |
|---|---|---|---|---|
| NVD CVE descriptions | Public Domain (US NIST) | 1053 | 58232 | https://nvd.nist.gov |
| MITRE ATT&CK offensive techniques/malware/tools | Apache-2.0 (MITRE cti) | 294 | 32617 | https://github.com/mitre/cti |
| Metasploit exploit module docs | BSD-3-Clause (rapid7) | 15 | 46798 | https://github.com/rapid7/metasploit-framework |
| MITRE CWE weakness descriptions | MITRE CWE terms (attribution) | 647 | 62383 | https://cwe.mitre.org/data/csv/1000.csv.zip |

filter stats: kept 2009, short 3, non-English 343, exact-dup 0, near-dup 47.

### hacking

| source | license | docs | tokens | origin |
|---|---|---|---|---|
| MITRE ATT&CK recon/initial-access/lateral phases | Apache-2.0 (MITRE cti) | 258 | 82936 | https://github.com/mitre/cti |
| MITRE CAPEC attack patterns | MITRE CAPEC terms (attribution) | 189 | 19412 | https://capec.mitre.org |
| NIST SP 800-115 security testing guide | Public Domain (US NIST) | 13 | 43352 | https://csrc.nist.gov/pubs/sp/800/115/final |
| Metasploit scanner/recon module docs | BSD-3-Clause (rapid7) | 17 | 26932 | https://github.com/rapid7/metasploit-framework |
| NVD CVE descriptions (recon/auth/network) | Public Domain (US NIST) | 327 | 27428 | https://nvd.nist.gov |

filter stats: kept 804, short 6, non-English 7, exact-dup 0, near-dup 7.

### writing

| source | license | docs | tokens | origin |
|---|---|---|---|---|
| Pride and Prejudice (Austen 1813) | Public Domain (Project Gutenberg) | 6 | 28760 | https://www.gutenberg.org/ebooks/1342 |
| Frankenstein (Shelley 1818) | Public Domain (Project Gutenberg) | 6 | 27490 | https://www.gutenberg.org/ebooks/84 |
| Jane Eyre (Bronte 1847) | Public Domain (Project Gutenberg) | 6 | 29783 | https://www.gutenberg.org/ebooks/1260 |
| Moby-Dick (Melville 1851) | Public Domain (Project Gutenberg) | 6 | 30444 | https://www.gutenberg.org/ebooks/2701 |
| Ulysses (Joyce 1922) | Public Domain (Project Gutenberg) | 6 | 31617 | https://www.gutenberg.org/ebooks/4300 |
| The Great Gatsby (Fitzgerald 1925) | Public Domain (Project Gutenberg) | 6 | 29393 | https://www.gutenberg.org/ebooks/64317 |
| curated early-20c prose (see manifest) | Public Domain (Project Gutenberg) | 5 | 23203 | https://www.gutenberg.org |

filter stats: kept 41, short 0, non-English 0, exact-dup 0, near-dup 0.

