"""R4 — GROWTH-250 under the corrected §11.2 judge (laptop RTX 3070).

Resume of the bank ladder per lab/PHOME_REJUDGE.md: growth-250 was halted
2026-10-16 on the metric-miscalibrated p_home tripwire; the corrected judge
(rejudge at n=128) clears saturation, so n=250 gets its first honest measure.

Phases:
  check-dequant  device dequant == lab/v05_core_prep.py CPU dequant (per
                 family, exact) — the only new math in this file.
  assembled      membership / p_home / ceiling / p_base for n in
                 {32,64,128,250} — fitted rows + cached eval taps (CPU).
                 The crowding slope.
  ce             masked-vs-plugged CE per home + base at n=250 on this
                 laptop's GPU. Cores are NOT RAM/VRAM-resident: packed
                 core.safetensors stay in RAM (~4.2GB for all 250); a
                 per-host-block hook dequantizes that block's plugs straight
                 to GPU around the block forward (peak = one site's plugs,
                 ~2GB into the 8GB card). Windows identical to growth_128 /
                 R2 (6 batches per home + base, seq 256, grouped bs=6 token-
                 exact); masked reused from the verified R2 probe cell with
                 a one-home re-computed guard. λ-balance watch: writing-home
                 plugs swept λ in {0.5, 0.35, 0.25} measuring writing + base
                 + hacking control. pairwise harm sampled 24 pairs (rng 0,
                 growth protocol).

Writes /home/tliao/rung1/results/growth_250.json (laptop-local rung record;
never committed — it contains per-route scalars only but lives with the
weights). Aggregate slope table is re-printed into the mission report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "lab"))
sys.path.insert(0, os.path.join(REPO_ROOT, "Common"))

import vk_graft as vk  # noqa: E402
import graft_d0_parity as d0  # noqa: E402
import graft_d1_plugin as d1  # noqa: E402
import door25_wrap_expert as d25  # noqa: E402
import grafted_spine_v0 as g0  # noqa: E402
import v05_core_prep as prep  # noqa: E402
import vk_final_build as vb  # noqa: E402
import bank_rung1 as br1  # noqa: E402
import phome_calibration as pcl  # noqa: E402
import phome_rejudge as prj  # noqa: E402

OUT = "/home/tliao/rung1/results/growth_250.json"
PROBE_JSON = os.path.join(REPO_ROOT, "lab", "results",
                          "phome_calibration.json")

E2M1 = prep.E2M1_TO_FLOAT


def _key(st: dict, suffix: str) -> str:
    hits = [k for k in st if k.endswith(suffix)]
    if len(hits) != 1:
        raise KeyError(f"expected exactly 1 key ending {suffix!r}, got {hits}")
    return hits[0]


def unpack_int4_dev(packed, shape, packed_dim=1):
    """prep.unpack_int4_signed, device-propagating (prep allocates CPU)."""
    dev = packed.device
    if packed_dim == 1:
        rows, cols = shape
        w = packed.to(torch.int64)
        if w.shape != (rows, (cols + 7) // 8):
            raise ValueError(f"packed {tuple(packed.shape)} vs shape {shape}")
        out = torch.empty(rows, cols, dtype=torch.int8, device=dev)
        for k in range(8):
            out[:, k::8] = ((w >> (4 * k)) & 0xF).to(torch.int8) - 8
        return out
    rows, cols = shape
    w = packed.to(torch.int64)
    if w.shape != ((rows + 7) // 8, cols):
        raise ValueError(f"packed {tuple(packed.shape)} vs shape {shape}")
    out = torch.empty(rows, cols, dtype=torch.int8, device=dev)
    for k in range(8):
        out[k::8, :] = ((w >> (4 * k)) & 0xF).to(torch.int8) - 8
    return out


def unpack_fp4_dev(packed, shape):
    dev = packed.device
    rows, cols = shape
    b = packed.reshape(-1).to(torch.int64)
    low = b & 0xF
    high = (b >> 4) & 0xF
    codes = torch.empty(b.numel() * 2, dtype=torch.int64, device=dev)
    codes[0::2] = low
    codes[1::2] = high
    sign = (codes & 0x8) != 0
    mag = codes & 0x7
    lut = torch.tensor(E2M1, dtype=torch.float32, device=dev)
    vals = lut[mag] * torch.where(
        sign, torch.tensor(-1.0, device=dev), torch.tensor(1.0, device=dev))
    return vals.reshape(rows, cols)


def dequant_core_dev(st: dict, fam: str, device):
    """prep.dequant_* math on `device`; st assumed already on device."""
    if fam == "k3":
        ws = {}
        for member, wname in (("w1", "w1"), ("w2", "w2"), ("w3", "w3")):
            k_packed = _key(st, f".{member}.weight_packed")
            root = k_packed[: -len("weight_packed")]
            packed = st[root + "weight_packed"]
            scale_u8 = st[root + "weight_scale"]
            out_f = packed.shape[0]
            in_f = packed.shape[1] * 2
            g = in_f // scale_u8.shape[-1]
            w = unpack_fp4_dev(packed, (out_f, in_f))
            scale = (2.0 ** (scale_u8.to(torch.int64) - 127)).float()
            w = w * scale.repeat_interleave(g, dim=1)
            ws[wname] = w
    else:
        ws = {}
        for member, wname in prep.MEMBER_TO_W.items():
            try:
                k_packed = _key(st, f".{member}.weight_packed")
            except KeyError:
                continue
            root = k_packed[: -len("weight_packed")]
            packed = st[root + "weight_packed"]
            scale = st[root + "weight_scale"].float()
            zp_p = st[root + "weight_zero_point"]
            shape = st[root + "weight_shape"].tolist()
            out_f, in_f = int(shape[0]), int(shape[1])
            g = in_f // scale.shape[-1]
            # unpack_* read shapes only; the shift math is device-native
            q = unpack_int4_dev(packed, (out_f, in_f), packed_dim=1)
            zp = unpack_int4_dev(zp_p, (out_f, in_f // g), packed_dim=0)
            w = (q.float() - zp.float().repeat_interleave(g, dim=1)) * \
                scale.repeat_interleave(g, dim=1)
            ws[wname] = w
    w_gu = torch.cat([ws["w1"], ws["w3"]], dim=0)
    return w_gu, ws["w2"]


# ---------------------------------------------------------------------------

def phase_check_dequant(args):
    for cid, fam in (("k3_L1E0", "k3"), ("glm_L3E25", "glm"),
                     ("mimo_L32E54", "mimo")):
        cp = f"/home/tliao/core_bank/" + {
            "k3": "kimi-k3", "glm": "glm-5.3-flash",
            "mimo": "mimo-v2.6-flash"}[fam] + "/" + cid.split("_", 1)[1] + "/core"
        st, prov = prep.load_core_tensors(cp)
        if fam == "k3":
            g0_, d0_, _cm = prep.dequant_mxfp4_core(st)
        else:
            g0_, d0_, _cm = prep.dequant_int4_core(st)
        st_dev = {k: v.to(args.device) for k, v in st.items()}
        g1, d1_ = dequant_core_dev(st_dev, prep.family_of(st, prov),
                                   args.device)
        e1 = (g1.cpu() - g0_).abs().max().item()
        e2 = (d1_.cpu() - d0_).abs().max().item()
        print(f"[check] {cid} ({fam}) max|dgu| {e1:.3e} max|ddown| {e2:.3e}",
              flush=True)
        assert e1 == 0.0 and e2 == 0.0, "device dequant mismatch"
    print("[check] device dequant EXACT vs CPU oracle for all 3 families",
          flush=True)


def phase_assembled(args, out):
    man = json.load(open(os.path.join(args.root, "manifest.json")))
    banks_reg, _ = d0.load_registry(args.registry)
    streams = {name: pcl.load_cache(f"eval_{name}")
               for name in br1.HOMES + ["base"]}
    slope = {}
    for n in (32, 64, 128, 250):
        rows = prj.fitted_rows(man, n, args.root)
        st = prj.assembled_stats(banks_reg, rows, streams)
        slope[str(n)] = {k: v for k, v in st.items() if k != "detail"}
        print(f"[assembled] n={n}: memb {st['membership_median']} "
              f"ceilrel {st['ceiling_relative_med_over_med']} "
              f"p_base_max {st['p_base_max']}", flush=True)
    out["slope"] = slope
    return out


class CoreStager:
    """Per-host-block hooks: dequant this block's plugs to GPU just before
    the block forward, drop them after. Adapters ride e.to(device)."""

    def __init__(self, handles, packed, fams, device):
        self.packed = packed
        self.fams = fams
        self.device = device
        self.removers = []
        for blk, _sf in handles:
            pm = getattr(blk, "_plugin_merge", None)
            if pm is None:
                continue

            def pre(mod, inputs, pm=pm, self=self):
                for e in pm.experts:
                    cid = e._cid
                    st_dev = {k: v.to(self.device, non_blocking=True)
                              for k, v in self.packed[cid].items()}
                    w_gu, w_down = dequant_core_dev(st_dev, self.fams[cid],
                                                    self.device)
                    e.core_w_gu.data = w_gu.half()
                    e.core_w_down.data = w_down.half()
                    e.to(self.device)
                    del st_dev, w_gu, w_down

            def post(mod, inputs, _out, pm=pm):
                for e in pm.experts:
                    e.core_w_gu.data = torch.empty(0, dtype=torch.float16)
                    e.core_w_down.data = torch.empty(0, dtype=torch.float16)
                    e.to("cpu")
                torch.cuda.empty_cache()

            self.removers.append(blk.register_forward_pre_hook(pre))
            self.removers.append(blk.register_forward_hook(post))

    def remove(self):
        for r in self.removers:
            r.remove()


def phase_ce(args, out):
    probe = json.load(open(PROBE_JSON))
    model, tok = vk.load_vk(args.model, args.device, torch.float16)
    blocks = vk.install_vk_host(model)
    banks_reg, _ = d0.load_registry(args.registry)
    man = json.load(open(os.path.join(args.root, "manifest.json")))

    seq = 256
    home_eval = {}
    for home in br1.HOMES:
        b = br1.feed_batches(tok, home, seq, 1, 300_000)[:6]
        if b:
            home_eval[home] = pcl._group_eval(b, 6, seq)
    base_eval = pcl._group_eval(
        br1.base_batches(tok, seq, 1, 40_000)[:6], 6, seq)

    # wraps: constructed once; core buffers dropped after construction
    plugs, packed, fams, rows_by_site = [], {}, {}, {}
    for c in man["cores"][:250]:
        pp = br1.plugin_path(args.root, c["id"])
        d = torch.load(pp, map_location="cpu", weights_only=False)
        spec = d["wrap_spec"]
        cp = d["meta"]["core_prefix"].replace("/data/tliao", "/home/tliao", 1)
        st, prov = prep.load_core_tensors(cp)
        fam = prep.family_of(st, prov)
        if fam == "k3":
            w_gu, w_down, cmeta = prep.dequant_mxfp4_core(st)
        else:
            w_gu, w_down, cmeta = prep.dequant_int4_core(st)
        wrap = d25.WrappedForeignExpert(
            spec["d_model"], spec["d_canon"], w_gu, w_down, seed=0,
            act=spec.get("act", cmeta.get("act", "swiglu")),
            situ_beta=spec.get("situ_beta", cmeta.get("situ_beta", 1.0)),
            situ_linear_beta=spec.get("situ_linear_beta",
                                      cmeta.get("situ_linear_beta")))
        wrap.load_state_dict(d["wrap_adapters"], strict=False)
        wrap.core_w_gu.data = torch.empty(0, dtype=torch.float16)
        wrap.core_w_down.data = torch.empty(0, dtype=torch.float16)
        wrap.to("cpu")
        wrap._cid = c["id"]
        del w_gu, w_down
        packed[c["id"]] = st
        fams[c["id"]] = fam
        meta = {**d["meta"], "id": c["id"], "home": c["home"]}
        plugs.append((wrap, d["row"].float(), meta))
    print(f"[ce] {len(plugs)} wraps built; packed cores in RAM "
          f"~{sum(v.nelement() * v.element_size() for s in packed.values() for v in s.values()) / 1e9:.1f}GB",
          flush=True)

    site_entries: dict[int, list] = {}
    for wrap, row, meta in plugs:
        host = int(meta["host_layer"])
        site_entries.setdefault(host, []).append({
            "expert": wrap, "lam": torch.tensor(args.default_lam),
            "tag": meta["id"], "num_stock": banks_reg[host].num_experts})
        rows_by_site.setdefault(host, []).append(row)
    gates, _ = vb._build_vk_gates(blocks, banks_reg, rows_by_site)
    handles = g0.install_mix(blocks, vk.FAM, site_entries, mode="off")
    stager = CoreStager(handles, packed, fams, args.device)

    def ce_on(batches, which):
        return g0.run_in_state(
            model, blocks, vk.FAM, gates, handles, which,
            lambda: d1.ce_in_state(model, blocks, batches, args.device,
                                   None, None, "off"))

    # masked guard: recompute one control home vs probe's verified value
    mk = probe["ce"]["masked"]
    guard = ce_on(home_eval["hacking"], "off")
    assert abs(guard - mk["homes"]["hacking"]) < 0.01, \
        f"masked guard drift: {guard} vs {mk['homes']['hacking']}"
    out.setdefault("ce", {})["masked_guard"] = {
        "hacking_recomputed": round(guard, 4),
        "probe_value": mk["homes"]["hacking"], "drift": round(
            guard - mk["homes"]["hacking"], 4)}
    masked = {"homes": mk["homes"], "base": mk["base"]}

    def run_config(writing_lam, streams):
        for blk, _sf in handles:
            pm = getattr(blk, "_plugin_merge", None)
            if pm is None:
                continue
            for j, tag in enumerate(pm.tags):
                pm.active[j] = True
        writing_ids = {m["id"] for _w, _r, m in plugs if m["home"] == "writing"}
        for blk, _sf in handles:
            pm = getattr(blk, "_plugin_merge", None)
            if pm is None:
                continue
            for j, tag in enumerate(pm.tags):
                pm.lams[j] = torch.tensor(
                    writing_lam if tag in writing_ids else args.default_lam)
        vals = {}
        for home in streams:
            t0 = time.time()
            b = home_eval[home] if home != "base" else base_eval
            v = ce_on(b, "both")
            vals[home] = round(v, 4)
            print(f"[ce] lam_w={writing_lam} {home} {v:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        return vals

    base_vals = run_config(args.default_lam, br1.HOMES + ["base"])
    out["ce"]["lam_default"] = {
        "plugged": base_vals,
        "home_delta": {h: round(base_vals[h] - masked["homes"][h], 4)
                       for h in br1.HOMES},
        "home_quality_worst": round(max(
            base_vals[h] - masked["homes"][h] for h in br1.HOMES), 4),
        "worst_home": max(br1.HOMES,
                          key=lambda h: base_vals[h] - masked["homes"][h]),
        "base_tax": round(base_vals["base"] - masked["base"], 4)}
    json.dump(out, open(OUT + ".partial", "w"), indent=1)

    sweep = {}
    for lam_w in (0.35, 0.25):
        vals = run_config(lam_w, ["writing", "base", "hacking"])
        sweep[str(lam_w)] = {
            "plugged": vals,
            "writing_delta": round(vals["writing"] - masked["homes"]["writing"], 4),
            "hacking_delta": round(vals["hacking"] - masked["homes"]["hacking"], 4),
            "base_tax": round(vals["base"] - masked["base"], 4)}
        json.dump(out, open(OUT + ".partial", "w"), indent=1)
    out["ce"]["lam_default"]["writing_delta"] = round(
        base_vals["writing"] - masked["homes"]["writing"], 4)
    out["ce"]["writing_lam_sweep"] = sweep

    # pairwise harm: 24 sampled pairs (growth protocol, rng 0)
    rng = np.random.default_rng(0)
    idx = list(range(len(plugs)))
    pairs = []
    for _ in range(min(24, len(idx) * (len(idx) - 1) // 2)):
        i, j = rng.choice(idx, size=2, replace=False)
        pairs.append((int(i), int(j)))
    harms = []
    for i, j in pairs:
        hi = plugs[i][2]["home"]
        b = home_eval.get(hi)
        if not b:
            continue
        all_ce = ce_on(b, "both")
        cid_j = plugs[j][2]["id"]
        for blk, _sf in handles:
            pm = getattr(blk, "_plugin_merge", None)
            if pm is not None and cid_j in pm.tags:
                pm.active[pm.tags.index(cid_j)] = False
        off_ce = ce_on(b, "both")
        for blk, _sf in handles:
            pm = getattr(blk, "_plugin_merge", None)
            if pm is not None and cid_j in pm.tags:
                pm.active[pm.tags.index(cid_j)] = True
        harms.append({"i": plugs[i][2]["id"], "j": cid_j,
                      "harm": round(all_ce - off_ce, 4)})
        print(f"[ce] harm {plugs[i][2]['id']} x {cid_j}: "
              f"{all_ce - off_ce:+.4f}", flush=True)
    worst = max(harms, key=lambda d: d["harm"]) if harms else None
    out["ce"]["pairwise_harm"] = {
        "sampled": len(harms), "worst": worst,
        "pass": bool(worst["harm"] <= 0.02) if worst else None}
    stager.remove()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", required=True,
                    choices=["check-dequant", "assembled", "ce", "all"])
    ap.add_argument("--root", default=pcl.DEFAULT_ROOT)
    ap.add_argument("--model", default=pcl.DEFAULT_MODEL)
    ap.add_argument("--registry", default=pcl.DEFAULT_REGISTRY)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--default-lam", type=float, default=0.5)
    args = ap.parse_args()

    out = {}
    if os.path.exists(OUT + ".partial"):
        out = json.load(open(OUT + ".partial"))
    out.setdefault("phase", "growth")
    out["format"] = "vesper-bank-rung1-v1"
    out["n"] = 250
    out["measurement_host"] = "laptop-rtx3070 (torch 2.7.1+cu126 fp16)"
    out["judge"] = ("corrected §11.2 per lab/PHOME_CALIBRATION.md + "
                    "lab/PHOME_REJUDGE.md (R2/R3 erratum); masked windows "
                    "reused from verified R2 probe cell")

    if args.phase in ("check-dequant", "all"):
        phase_check_dequant(args)
    if args.phase in ("assembled", "all"):
        phase_assembled(args, out)
        json.dump(out, open(OUT + ".partial", "w"), indent=1)
    if args.phase in ("ce", "all"):
        phase_ce(args, out)
        # corrected break status
        ld = out["ce"]["lam_default"]
        sl = out["slope"]["250"]
        out["home_quality"] = {
            h: {"plugged": ld["plugged"][h],
                "delta": ld["home_delta"][h]}
            for h in br1.HOMES}
        out["base_tax"] = {"delta": ld["base_tax"],
                           "pass": bool(ld["base_tax"] <= 0.05)}
        out["pairwise_harm"] = out["ce"]["pairwise_harm"]
        out["saturation_corrected"] = {
            "membership_median": sl["membership_median"],
            "membership_trend_32_64_128_250": [
                out["slope"][str(n)]["membership_median"]
                for n in (32, 64, 128, 250)],
            "ceiling_relative_p_home": sl["ceiling_relative_med_over_med"],
            "p_base_max": sl["p_base_max"],
            "pass": bool(sl["ceiling_relative_med_over_med"] >= 0.5)}
        out["break_status"] = {
            "1_home_quality": bool(ld["home_quality_worst"] <= 0.02),
            "2_pairwise_harm": out["pairwise_harm"]["pass"],
            "3_saturation_corrected": out["saturation_corrected"]["pass"],
            "4_base_tax": out["base_tax"]["pass"]}
        out["phase"] = "growth_done"
        with open(OUT + ".tmp", "w") as f:
            json.dump(out, f, indent=1)
        os.replace(OUT + ".tmp", OUT)
        if os.path.exists(OUT + ".partial"):
            os.remove(OUT + ".partial")
        print("break_status:", out["break_status"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
