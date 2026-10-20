#!/usr/bin/env python3
"""Feature-bank feasibility measure: cross-model residual-stream alignment
between the vk-family and 470m-family checkpoints on the laptop, CPU fp32.

Part (b) of lab/TRANSPLANT_FEATURE_BANK.md (concept-door-4: transplant as
service). Question: at analogous layers, is there a learnable map between a
vk hidden h_vk(t, layer l) and a 470m h_470m(t, layer m) on the SAME shared
token windows? Reported series: ridge-map held cosine (Door-9 comparable),
whitened-CCA canonical-correlation spectrum, identity floor, shuffled null,
and a within-model cross-layer reference — train/select/held split BY WINDOW
so the pairing has no leakage.

Materials are the program's own (paths in the JSON); no corpus text is
printed anywhere. Verified by counts and shapes only, per house rules.

Usage:
  PYTHONPATH=$HOME/.local/lib/python3.12/site-packages VESPER_CPU_THREADS=12 \
    python3 -u lab/feature_bank_cca.py \
      --out lab/results/feature_bank_cca.json [--smoke]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "Common"))
sys.path.insert(0, os.path.join(_REPO, "Hippocampus"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_VK = os.path.join(_REPO, "lab", "imported", "vk_sft_r3", "step_1800",
                          "chat_model", "vesper_chat.pt")
DEFAULT_470M = os.path.join(_REPO, "lab", "imported", "sft_470m", "step_best",
                            "checkpoint.pt")
DEFAULT_BIN = os.path.join(_REPO, "Dataset", "data", "pretrain",
                           "phase2_pretrain.bin")

LORA_SCALING = 16.0 / 8.0  # Hippocampus/lora.py: alpha 16, rank 8 (470m SFT)


def load_vk(path: str):
    """vk-family chat checkpoint: plain keys, value-embed auto-detect."""
    from consolidate import enable_cpu_gla_shims, enable_cpu_kda_mla_shims
    enable_cpu_gla_shims()
    enable_cpu_kda_mla_shims()
    from vesper_linear_model import VesperLinearLM

    ck = torch.load(path, map_location="cpu", weights_only=False)
    mc = dict(ck["model_config"])
    state = {k.replace("module.", ""): v for k, v in ck["model_state_dict"].items()}
    model = build_model(VesperLinearLM, mc, state, vocab_from=state)
    return model, state


def load_470m(path: str):
    """470m-family SFT checkpoint: trainer raw ckpt with LoRA wrappers.

    Fold `scaling * lora_B @ lora_A` into each base weight and un-`.base.`
    base:  checkpoint for the measure.
    """
    from consolidate import enable_cpu_gla_shims, enable_cpu_kda_mla_shims
    enable_cpu_gla_shims()
    enable_cpu_kda_mla_shims()
    from vesper_linear_model import VesperLinearLM

    ck = torch.load(path, map_location="cpu", weights_only=False)
    mc = dict(ck["model_config"])
    raw = {k.replace("module.", ""): v for k, v in ck["model"].items()}

    state = {}
    for k, v in raw.items():
        if k.endswith(".lora_A") or k.endswith(".lora_B"):
            continue
        state[k.replace(".base.", ".")] = v.clone()
    n_fold = 0
    for k in raw:
        if not k.endswith(".lora_A"):
            continue
        stem = k[: -len(".lora_A")]
        b_key = stem + ".lora_B"
        tgt_key = (stem + ".weight").replace(".base.", ".")
        if b_key not in raw:
            continue
        delta = LORA_SCALING * (raw[b_key].float() @ raw[k].float())
        if tgt_key not in state:
            raise KeyError(f"LoRA target missing: {tgt_key} ({stem})")
        state[tgt_key] = state[tgt_key].float() + delta
        n_fold += 1

    model = build_model(VesperLinearLM, mc, state, vocab_from=state)
    return model, state, n_fold


def build_model(ModelClass, mc: dict, state: dict, vocab_from: dict):
    vocab = int(vocab_from["tok_embeddings.weight"].shape[0])
    value_embed = any("value_emb" in k for k in state)
    model = ModelClass(
        vocab_size=vocab,
        dim=mc["dim"], n_layers=mc["n_layers"], n_heads=mc["n_heads"],
        n_kv_heads=mc["n_kv_heads"], hidden_dim=mc["hidden_dim"],
        num_experts=mc["num_experts"], top_k=mc["top_k"],
        max_seq_len=mc["max_seq_len"], pad_id=0,
        linear_type=mc.get("linear_type", "gla"),
        full_type=mc.get("full_type", "gqa"),
        attention_every=mc.get("attention_every", 4),
        gla_head_dim=mc.get("gla_head_dim", 64),
        qk_norm=mc.get("qk_norm", True),
        kda_head_dim=mc.get("kda_head_dim", 64),
        kda_short_conv=mc.get("kda_short_conv", True),
        mamba2_state_size=mc.get("mamba2_state_size", 128),
        kv_lora_rank=mc.get("kv_lora_rank"),
        v_head_dim=mc.get("v_head_dim", 128),
        linear_force_fp32=mc.get("linear_force_fp32", True),
        router_type=mc.get("router_type", "topk"),
        passport_dim=mc.get("passport_dim", 64),
        router_expert_dropout=mc.get("router_expert_dropout", 0.0),
        value_embed=value_embed,
    )
    missing, unexpected = model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def tap_hiddens(model, tokens: torch.Tensor, layers: list, check: bool = False):
    """Mirror VesperLinearLM.forward's eval branch; capture per-block
    residual stream x after each layer. Returns dict layer -> (T, dim)."""
    x = model.tok_embeddings(tokens)
    taps = {}
    for i, layer in enumerate(model.layers):
        is_linear = model.layer_types[i] in ("gla", "mamba2", "kda")
        if is_linear:
            attn_out = layer["attn"](layer["attn_norm"](x), value_tokens=tokens)
        else:
            attn_out = layer["attn"](layer["attn_norm"](x), model.freqs_cis,
                                     value_tokens=tokens)
        x = x + attn_out
        ffn_out, _aux = layer["ffn"](layer["ffn_norm"](x))
        x = x + ffn_out
        if i in layers:
            taps[i] = x[0].float().clone()  # (T, dim)
    err = None
    if check:
        # forward-contract self-check: the mirror must reproduce the stock fwd
        logits_mine = model.output(model.norm(x))
        ref, _, _ = model(tokens)
        err = float((logits_mine - ref).abs().max())
    return taps, err


def collect_windows(state_bin: str, n_windows: int, window: int, offset_frac: float):
    arr = np.load(state_bin, mmap_mode="r", allow_pickle=False) \
        if state_bin.endswith(".npy") else \
        np.memmap(state_bin, dtype=np.uint16, mode="r")
    n = arr.shape[0]
    start = int(n * offset_frac)
    step = (n - start - window - 1) // n_windows
    outs = []
    for j in range(n_windows):
        seg = np.asarray(arr[start + j * step: start + j * step + window],
                         dtype=np.int64)
        outs.append(seg)
    return outs


def ridge_perm_block_stats(Xf, Yf, Xs, Xh, Ys, Yh, lam_frac):
    """Fit W: X->Y on train (standardized on fit), ridge lam = lam_frac of
    mean diagonal variance term. Returns held/select cosine stats + R2."""
    d = Xf.shape[1]
    lam = lam_frac * float(torch.diagonal(Xf.T @ Xf).mean())
    A = (Xf.T @ Xf + lam * torch.eye(d, dtype=torch.float64))
    B = Xf.T @ Yf
    W = torch.linalg.solve(A, B)  # d x d

    def stats(X, Y):
        P = X @ W
        pn = P / P.pow(2).sum(1, keepdim=True).sqrt().clamp_min(1e-9)
        yn = Y / Y.pow(2).sum(1, keepdim=True).sqrt().clamp_min(1e-9)
        cos = (pn * yn).sum(1)
        ss_res = float((Y - P).pow(2).sum())
        ss_tot = float((Y - Y.mean(0, keepdim=True)).pow(2).sum())
        return {"cos_mean": float(cos.mean()),
                "cos_p10": float(torch.quantile(cos, 0.1)),
                "cos_p90": float(torch.quantile(cos, 0.9)),
                "r2": 1.0 - ss_res / max(ss_tot, 1e-12)}

    return {"fit": stats(Xf, Yf), "select": stats(Xs, Ys), "held": stats(Xh, Yh),
            "lam_frac": lam_frac}


def whitened_cca(Xf, Yf, Xh, Yh, k: int):
    """PCA-whiten each view on fit split to k comps, canonical SVD; report
    canonical correlations on fit and held (transform through fit bases)."""
    Cx = (Xf.T @ Xf) / (Xf.shape[0] - 1)
    Cy = (Yf.T @ Yf) / (Yf.shape[0] - 1)
    evx, ex = torch.linalg.eigh(Cx)
    evy, ey = torch.linalg.eigh(Cy)
    kx = min(k, Xf.shape[1]); ky = min(k, Yf.shape[1])
    sx = evx.clamp_min(1e-9)
    sy = evy.clamp_min(1e-9)
    Px = ex[:, -kx:] @ torch.diag(sx[-kx:].rsqrt())
    Py = ey[:, -ky:] @ torch.diag(sy[-ky:].rsqrt())

    Zx = Xf @ Px
    Zy = Yf @ Py
    M = (Zx.T @ Zy) / (Zx.shape[0] - 1)
    U, S, Vt = torch.linalg.svd(M)
    n_show = 16
    out = {"k": int(kx), "n_canonical": int(S.numel()),
           "fit_top": [round(float(v), 6) for v in S[:n_show]],
           "fit_mean_all": float(S.mean())}

    Zh_x = (Xh - 0) @ Px
    Zh_y = (Yh - 0) @ Py
    cx = Zh_x @ U
    cy = Zh_y @ Vt.T
    rhos = []
    for j in range(int(S.numel())):
        a = cx[:, j] - cx[:, j].mean()
        b = cy[:, j] - cy[:, j].mean()
        num = float((a * b).mean())
        den = float(a.pow(2).mean().sqrt() * b.pow(2).mean().sqrt()) + 1e-12
        rhos.append(num / den)
    out["held_top"] = [round(v, 6) for v in rhos[:n_show]]
    out["held_mean_all"] = float(np.mean(rhos))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vk-ckpt", default=DEFAULT_VK)
    ap.add_argument("--m470-ckpt", default=DEFAULT_470M)
    ap.add_argument("--token-bin", default=DEFAULT_BIN)
    ap.add_argument("--out", default=os.path.join(_REPO, "lab", "results",
                                                  "feature_bank_cca.json"))
    ap.add_argument("--n-windows", type=int, default=16)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--offset-frac", type=float, default=0.70)
    ap.add_argument("--layers", default="2,5,8")
    ap.add_argument("--k-cca", type=int, default=64)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    t0 = time.time()
    layers = sorted(int(x) for x in args.layers.split(","))

    if args.smoke:
        args.n_windows = 3
        args.window = 128

    windows = collect_windows(args.token_bin, args.n_windows, args.window,
                              args.offset_frac)

    res = {"args": vars(args), "materials": {
        "vk_ckpt": os.path.relpath(args.vk_ckpt, _REPO),
        "m470_ckpt": os.path.relpath(args.m470_ckpt, _REPO),
        "token_bin": os.path.relpath(args.token_bin, _REPO),
        "n_windows": args.n_windows, "window": args.window}}

    print(f"[load] vk: {args.vk_ckpt}", flush=True)
    model_vk, vk_state = load_vk(args.vk_ckpt)
    print(f"[load] 470m: {args.m470_ckpt}", flush=True)
    model_470, m470_state, n_fold = load_470m(args.m470_ckpt)
    res["materials"]["lora_folded"] = n_fold
    res["materials"]["vk_keys"] = len(vk_state)
    res["materials"]["m470_keys"] = len(m470_state)

    hid = {"vk": {}, "m470": {}}
    layers_plus = sorted(set(layers) | {l + 2 for l in layers if l + 2 < 10})
    errs = []
    for wi, seg in enumerate(windows):
        ids = torch.from_numpy(seg)[None]  # (1, T)
        taps_v, ev = tap_hiddens(model_vk, ids, layers_plus, check=(wi == 0))
        taps_m, em = tap_hiddens(model_470, ids, layers_plus, check=(wi == 0))
        if wi == 0:
            errs.append({"fv": ev, "fm": em})
        for l in layers_plus:
            hid["vk"].setdefault(l, []).append(taps_v[l])
            hid["m470"].setdefault(l, []).append(taps_m[l])
        if wi == 0:
            print(f"[fwd] window 1/{len(windows)} fv_maxabs={ev:.2e} "
                  f"fm_maxabs={em:.2e}", flush=True)
        else:
            print(f"[fwd] window {wi + 1}/{len(windows)}", flush=True)
    res["forward_contract_maxerr"] = errs[0]["fv"]
    res["forward_contract_maxerr_470"] = errs[0]["fm"]

    # window split: fit 10 / select 3 / held 3 (no leakage; pairing by t)
    n = len(windows)
    if n >= 16:
        fit_idx = list(range(0, n - 6))
        sel_idx = list(range(n - 6, n - 3))
        held_idx = list(range(n - 3, n))
    else:  # smoke: one window each
        fit_idx, sel_idx, held_idx = [0], [1], [2]

    def split(tensors, idxs):
        return torch.cat([tensors[i] for i in idxs], 0).to(torch.float64)

    def standardize(train, *others):
        mu = train.mean(0, keepdim=True)
        sd = train.std(0, keepdim=True).clamp_min(1e-8)
        return (train - mu) / sd, *[(o - mu) / sd for o in others]

    lam_grid = [1e-3, 1e-2, 1e-1]

    series = {}
    for l in layers:
        vk_f = split(hid["vk"][l], fit_idx)
        vk_s = split(hid["vk"][l], sel_idx)
        vk_h = split(hid["vk"][l], held_idx)
        m_f = split(hid["m470"][l], fit_idx)
        m_s = split(hid["m470"][l], sel_idx)
        m_h = split(hid["m470"][l], held_idx)

        # identity floor: raw cos between the two models' same-position hiddens
        p = vk_h / vk_h.pow(2).sum(1, keepdim=True).sqrt().clamp_min(1e-9)
        q = m_h / m_h.pow(2).sum(1, keepdim=True).sqrt().clamp_min(1e-9)
        ident = float((p * q).sum(1).mean())

        # shuffled-pair null: vk window order permuted against 470m
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(len(windows), generator=g)
        vk_null = torch.cat([torch.cat([hid["vk"][l][perm[i]] for i in held_idx], 0)], 0).to(torch.float64)
        m_null_fit = torch.cat([torch.cat([hid["m470"][l][perm[i]] for i in fit_idx], 0)], 0).to(torch.float64)
        m_null_h = m_h
        v_nf, v_nh = standardize(vk_f, vk_null)
        m_nf, m_nh = standardize(m_null_fit, m_null_h)
        pn = v_nh / v_nh.pow(2).sum(1, keepdim=True).sqrt().clamp_min(1e-9)
        qn = m_nh / m_nh.pow(2).sum(1, keepdim=True).sqrt().clamp_min(1e-9)
        ident_null = float((pn * qn).sum(1).mean())

        Xf, Xs, Xh = standardize(vk_f, vk_s, vk_h)      # view X = vk
        Yf, Ys, Yh = standardize(m_f, m_s, m_h)          # view Y = 470m

        maps = {}
        for lf in lam_grid:
            maps[f"lam_{lf}"] = ridge_perm_block_stats(Xf, Yf, Xs, Xh, Ys, Yh, lf)

        cca = whitened_cca(Xf, Yf, Xh, Yh, args.k_cca)

        # within-model reference: vk layer l -> vk layer l+2
        ref = None
        if (l + 2) in hid["vk"]:
            rf = split(hid["vk"][l + 2], fit_idx)
            rs = split(hid["vk"][l + 2], sel_idx)
            rh = split(hid["vk"][l + 2], held_idx)
            Rf, Rs, Rh = standardize(rf, rs, rh)
            ref = ridge_perm_block_stats(Xf, Rf, Xs, Xh, Rs, Rh, 1e-2)

        series[f"layer_{l}"] = {
            "identity_held_cos_mean": ident,
            "identity_shuffled_null": ident_null,
            "ridge_vk_to_470m": maps,
            "cca_vk_vs_470m": cca,
            "within_model_l_to_l2": ref,
        }
        best_map = max(maps.values(), key=lambda m: m["select"]["cos_mean"])
        line = (f"[L{l}] ident={ident:+.4f} null={ident_null:+.4f} "
                f"ridge held cos={best_map['held']['cos_mean']:.4f} "
                f"(fit {best_map['fit']['cos_mean']:.4f}) "
                f"cca held top1={cca['held_top'][0]:.4f} "
                f"mean={cca['held_mean_all']:.4f}")
        if ref:
            line += f" | within-model ref held cos={ref['held']['cos_mean']:.4f}"
        print(line, flush=True)

    res["series"] = series
    res["split"] = {"fit_windows": fit_idx, "select_windows": sel_idx,
                    "held_windows": held_idx}
    res["wall_s"] = round(time.time() - t0, 1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1)
    os.replace(tmp, args.out)
    print(f"[done] {args.out} wall {res['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
