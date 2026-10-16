#!/usr/bin/env python3
"""KDA/MLA CPU shim smoke test (roadmap item (b), part 1).

Proves the Vesper-K (KDA + MLA) model code runs on plain CPU with no GPU
stack in play: fla's Triton-only ops are replaced by the pure-torch shims in
Immune/cpu_backend.install_cpu_shims(kda_mla=True) (the same set is mirrored
in Hippocampus/consolidate.enable_cpu_kda_mla_shims), then a real tiny
kda_mla checkpoint is loaded, run forward, generated with, and trained
through one LoRA micro-step.

Sections
  S1  shim unit checks (causal_conv1d + update/step, rotary+offsets,
      rms_norm, gated-norm swish vs sigmoid branches)
  S2  Immune.cpu_backend.load_model on the tiny kda_mla ckpt: layer mix,
      finite CE, greedy generation
  S3  Hippocampus path: load_base_model + attach_lora(kda_mla) + one
      micro-session step (the teach path) + delta save/load roundtrip
  S4  gla_gqa backward compat (tiny random GLA/GQA model still forwards)

Usage (the PYTHONPATH prefix is the handoff's user-site trick; on hosts where
~/.local is already on sys.path it is a harmless no-op):

  PYTHONPATH=$HOME/.local/lib/python3.12/site-packages \
    python3 lab/kda_cpu_shim_smoke.py [--ckpt PATH]

Default checkpoint: Pretrain/vesper_linear_checkpoints_tiny_agent_k/step_500
(the fresh true KDA+MLA run — never the archived *_gqa_buggy_* tree, whose
metadata carries the wrong full_type).

Runs on CPU only. CUDA_VISIBLE_DEVICES is deliberately NOT hidden: fla's
import path initialises Triton's autotuner and wants a visible driver
(Immune/cpu_backend.py note); tensors are simply never moved to a device.
Exit code 0 iff every check passes.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

REPO = os.environ.get("VESPER_REPO", "/home/tliao/VesperLM")
for _p in (os.path.join(REPO, "Common"), os.path.join(REPO, "Pretrain"),
           os.path.join(REPO, "Immune"), os.path.join(REPO, "Hippocampus")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(int(os.environ.get("VESPER_CPU_THREADS", "16")))

RESULTS = []  # (name, ok, detail)


def check(name: str, fn):
    try:
        detail = fn()
        if detail is None:
            detail = "ok"
        RESULTS.append((name, True, str(detail)))
        print(f"  PASS  {name}  {detail}")
    except Exception as e:
        RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"  FAIL  {name}  {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)


def section(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
# S1 — shim unit checks
# ---------------------------------------------------------------------------
def s1_shim_units():
    sys.path.insert(0, os.path.join(REPO, "Immune"))
    from cpu_backend import install_cpu_shims
    status = install_cpu_shims(kda_mla=True)
    print(f"  install_cpu_shims: {status}")

    import importlib
    cconv = importlib.import_module("fla.modules.conv.causal_conv1d")
    ctrit = importlib.import_module("fla.modules.conv.triton.ops")
    frot = importlib.import_module("fla.modules.rotary")
    lnorm = importlib.import_module("fla.modules.layernorm")
    fng = importlib.import_module("fla.modules.fused_norm_gate")
    import fla.layers.kda as fla_kda

    def t_causal_conv():
        torch.manual_seed(0)
        B, T, D, W = 2, 7, 4, 3
        x = torch.randn(B, T, D)
        w = torch.randn(D, W)
        b = torch.randn(D)
        y, _ = cconv.causal_conv1d(x, weight=w, bias=b, activation=None,
                                  output_final_state=True)
        ref = F.conv1d(F.pad(x.transpose(1, 2), (W - 1, 0)),
                       w.unsqueeze(1), b, groups=D).transpose(1, 2)
        assert torch.allclose(y, ref, atol=1e-5), (y - ref).abs().max()
        return f"max|y-ref|={(y - ref).abs().max():.2e}"

    def t_causal_conv_state():
        torch.manual_seed(1)
        B, T, D, W = 1, 6, 3, 4
        x = torch.randn(B, T, D)
        w = torch.randn(D, W)
        y_full, st = cconv.causal_conv1d(x, weight=w, bias=None, activation="silu",
                                         output_final_state=True)
        # step through token-by-token with causal_conv1d_update; the last step's
        # output must match the full forward's last token.
        cache = torch.zeros(B, D, W)
        y_last = None
        for t in range(T):
            y_last, cache = ctrit.causal_conv1d_update(x[:, t:t + 1, :], cache,
                                                       weight=w, activation="silu")
        assert torch.allclose(y_full[:, -1, :], y_last, atol=1e-5), \
            (y_full[:, -1, :] - y_last).abs().max()
        assert torch.allclose(st[:, :, -1], x[:, -1, :], atol=1e-6)
        return f"step-vs-full last token ok, max|d|={(y_full[:, -1, :] - y_last).abs().max():.2e}"

    def t_rotary():
        torch.manual_seed(2)
        B, T, H, Dh = 1, 5, 2, 8
        x = torch.randn(B, T, H, Dh)
        Tm = 16
        ang = torch.randn(Tm, Dh // 2)
        cos, sin = ang.cos(), ang.sin()
        y = frot.rotary_embedding(x, cos, sin, interleaved=False, seqlen_offsets=0)
        ref = frot.rotary_embedding_ref(x, cos[:T], sin[:T], interleaved=False)
        assert torch.allclose(y, ref, atol=1e-6), (y - ref).abs().max()
        off = 3
        y_off = frot.rotary_embedding(x, cos, sin, interleaved=False,
                                      seqlen_offsets=off)
        ref_off = frot.rotary_embedding_ref(x, cos[off:off + T], sin[off:off + T],
                                            interleaved=False)
        assert torch.allclose(y_off, ref_off, atol=1e-6), (y_off - ref_off).abs().max()
        return "offsets 0 and 3 match rotary_embedding_ref"

    def t_rms_norm():
        torch.manual_seed(3)
        x = torch.randn(2, 5, 8)
        w = torch.randn(8)
        y = lnorm.rms_norm(x, w, None, eps=1e-5)
        xf = x.float()
        ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5) * w.float()
        assert torch.allclose(y, ref.to(y.dtype), atol=1e-5), (y - ref).abs().max()
        return "matches manual rmsnorm"

    def t_gated_norm_branches():
        torch.manual_seed(4)
        x = torch.randn(2, 5, 8)
        g = torch.randn(2, 5, 8)
        w = torch.ones(8)
        y_swish = fng.rms_norm_gated(x, g, w, activation="swish")
        y_sig = fng.rms_norm_gated(x, g, w, activation="sigmoid")
        xf = x.float()
        base = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
        gf = g.float()
        assert torch.allclose(y_swish, (base * gf * torch.sigmoid(gf)).to(x.dtype),
                              atol=1e-5)
        # sigmoid branch must NOT include the extra g factor (the fixed bug)
        assert torch.allclose(y_sig, (base * torch.sigmoid(gf)).to(x.dtype),
                              atol=1e-5)
        return "swish=g*sigmoid(g) scale, sigmoid=sigmoid(g) scale (bug fix held)"

    def t_kda_shim_forward():
        torch.manual_seed(5)
        B, T, H, K, V = 1, 6, 2, 4, 4
        q = torch.randn(B, T, H, K, requires_grad=True)
        k = torch.randn(B, T, H, K, requires_grad=True)
        v = torch.randn(B, T, H, V, requires_grad=True)
        g = torch.randn(B, T, H, K, requires_grad=True)
        beta = torch.randn(B, T, H, requires_grad=True)
        A_log = torch.randn(H)
        dt_bias = torch.randn(H * K)
        o, h = fla_kda.chunk_kda(q, k, v, g, beta, A_log=A_log, dt_bias=dt_bias,
                                 use_qk_l2norm_in_kernel=True,
                                 use_gate_in_kernel=True,
                                 use_beta_sigmoid_in_kernel=True,
                                 output_final_state=True)
        assert o.shape == (B, T, H, V), o.shape
        assert torch.isfinite(o).all()
        assert h is not None and torch.isfinite(h).all()
        # backward through the naive recurrence (the teach path needs grads)
        o.sum().backward()
        for name, t in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)):
            assert t.grad is not None and torch.isfinite(t.grad).all(), name
        return f"forward+backward finite, out {tuple(o.shape)}"

    check("causal_conv1d vs manual conv", t_causal_conv)
    check("causal_conv1d_update step vs full forward", t_causal_conv_state)
    check("rotary shim + seqlen_offsets vs ref", t_rotary)
    check("rms_norm shim vs manual", t_rms_norm)
    check("gated-norm swish/sigmoid branches", t_gated_norm_branches)
    check("KDA chunk_kda pure-torch fwd+bwd", t_kda_shim_forward)


# ---------------------------------------------------------------------------
# S2 — Immune loader on a real tiny kda_mla checkpoint
# ---------------------------------------------------------------------------
def s2_immune_load(ckpt: str):
    import cpu_backend as cb

    def t_load_forward():
        model, ck = cb.load_model(ckpt, target_profile="kda_mla")
        mc = ck["model_config"]
        assert mc.get("linear_type") == "kda" and mc.get("full_type") == "mla", \
            f"not a kda_mla checkpoint: {mc.get('linear_type')}/{mc.get('full_type')}"
        mix = {}
        for lt in getattr(model, "layer_types", ()):
            mix[lt] = mix.get(lt, 0) + 1
        ids = torch.randint(3, 1000, (1, 16))
        logits, aux, _ = model(ids)
        assert torch.isfinite(logits).all()
        ce = F.cross_entropy(logits[0, :-1].float(), ids[0, 1:])
        assert torch.isfinite(ce)
        return f"layer mix {mix}, CE(random ids) {float(ce):.3f} finite"

    def t_generate():
        model, _ = cb.load_model(ckpt, target_profile="kda_mla")
        tok = cb.load_tokenizer()
        out = cb.generate(model, tok, "The capital of France is", max_new=8)
        assert isinstance(out, str)
        return f"greedy 8 tok -> {out!r}"

    check("Immune.load_model kda_mla ckpt (arch assert + finite CE)", t_load_forward)
    check("Immune.generate greedy decode on CPU", t_generate)


# ---------------------------------------------------------------------------
# S3 — Hippocampus teach path (LoRA attach + one micro step)
# ---------------------------------------------------------------------------
def s3_hippo_path(ckpt: str):
    sys.path.insert(0, os.path.join(REPO, "Hippocampus"))
    import consolidate as cons
    import lora as lora_mod

    def t_load():
        model, tok, mc = cons.load_base_model(ckpt, device="cpu",
                                              target_profile="kda_mla")
        assert mc["linear_type"] == "kda" and mc["full_type"] == "mla"
        return f"loaded, vocab {len(tok)}, pad_id {tok.pad_token_id}"

    def t_lora_microstep():
        model, tok, _ = cons.load_base_model(ckpt, device="cpu",
                                             target_profile="kda_mla")
        wrapped = lora_mod.attach_lora(model, rank=8, alpha=16.0,
                                       include_router=False,
                                       target_profile="kda_mla")
        # tiny_agent_k: 6 kda layers x 4 proj + 2 mla layers x 5 proj = 34
        assert len(wrapped) == 34, f"expected 34 kda_mla wraps, got {len(wrapped)}"
        lora_mod.freeze_trunk(model)
        n_train = sum(p.numel() for p in lora_mod.lora_parameters(model))
        from session_log import TrainingTriple
        triples = [TrainingTriple(prompt="What is 2 + 2?",
                                  response="The answer is four.", reward=1.0,
                                  session_id="smoke", turn_id=0,
                                  source="explicit_feedback",
                                  confidence=1.0)]
        stats = cons.train_lora_microsession(
            model, tok, triples, steps=2, lr=2e-3, batch_size=1,
            kl_coef=0.05, device="cpu", log_every=1)
        assert all(torch.isfinite(torch.tensor(v)) for v in
                   (stats["loss0"], stats["loss_final"], stats["final_kl"]))
        return (f"34 wraps / {n_train:,} trainable, "
                f"loss {stats['loss0']:+.4f} -> {stats['loss_final']:+.4f}, "
                f"kl {stats['final_kl']:.4f}")

    def t_delta_roundtrip(tmp="/tmp/kda_shim_smoke_delta.pt"):
        model, tok, _ = cons.load_base_model(ckpt, device="cpu",
                                             target_profile="kda_mla")
        wrapped = lora_mod.attach_lora(model, rank=8, alpha=16.0,
                                       include_router=False,
                                       target_profile="kda_mla")
        assert len(wrapped) == 34
        torch.manual_seed(7)
        for p in lora_mod.lora_parameters(model):
            p.data.normal_(0, 0.01)
        lora_mod.save_delta(model, tmp,
                            meta={"target_profile": "kda_mla", "user_id": "smoke"})
        model2, tok2, _ = cons.load_base_model(ckpt, device="cpu",
                                               target_profile="kda_mla")
        lora_mod.attach_lora(model2, rank=8, alpha=16.0, include_router=False,
                             target_profile="kda_mla")
        lora_mod.load_delta(model2, tmp, target_profile="kda_mla")
        d1 = dict(lora_mod.lora_modules(model))
        d2 = dict(lora_mod.lora_modules(model2))
        assert set(d1) == set(d2) and d1
        worst = 0.0
        for k in d1:
            worst = max(worst, (d1[k].lora_A - d2[k].lora_A).abs().max().item(),
                        (d1[k].lora_B - d2[k].lora_B).abs().max().item())
        assert worst == 0.0, worst
        os.remove(tmp)
        return f"{len(d1)} modules roundtrip bit-exact"

    check("Hippocampus.load_base_model kda_mla", t_load)
    check("attach_lora(kda_mla) + 2-step micro-session fwd/bwd", t_lora_microstep)
    check("delta save/load roundtrip bit-exact", t_delta_roundtrip)


# ---------------------------------------------------------------------------
# S4 — gla_gqa backward compat (shims must not break the old stack)
# ---------------------------------------------------------------------------
def s4_gla_gqa_compat():
    from cpu_backend import install_cpu_shims
    install_cpu_shims(kda_mla=False)
    from vesper_linear_model import VesperLinearLM

    def t_random_gla_gqa():
        torch.manual_seed(11)
        model = VesperLinearLM(
            vocab_size=256, dim=64, n_layers=2, n_heads=4, n_kv_heads=2,
            hidden_dim=128, num_experts=2, top_k=2, max_seq_len=64,
            pad_id=0, linear_type="gla", full_type="gqa",
            attention_every=2, gla_head_dim=16, qk_norm=True,
            linear_force_fp32=True, router_type="topk")
        model.eval()
        ids = torch.randint(3, 200, (1, 8))
        logits, aux, _ = model(ids)
        assert torch.isfinite(logits).all()
        ce = F.cross_entropy(logits[0, :-1].float(), ids[0, 1:])
        assert torch.isfinite(ce)
        return f"random gla_gqa forward finite, CE {float(ce):.3f}"

    check("gla_gqa stack still forwards on CPU (compat)", t_random_gla_gqa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(
        REPO, "Pretrain", "vesper_linear_checkpoints_tiny_agent_k", "step_500"))
    args = ap.parse_args()

    print("KDA/MLA CPU shim smoke test")
    print(f"  repo     : {REPO}")
    print(f"  ckpt     : {args.ckpt}")
    print(f"  torch    : {torch.__version__}")
    import fla
    print(f"  fla      : {getattr(fla, '__version__', '?')} ({fla.__file__})")
    try:
        import triton
        print(f"  triton   : {triton.__version__} ({triton.__file__})")
    except Exception as e:
        print(f"  triton   : unavailable ({e})")
    print(f"  device   : CPU only (tensors never moved to GPU)")
    assert os.path.exists(os.path.join(args.ckpt, "checkpoint.pt")), \
        f"no checkpoint.pt under {args.ckpt}"

    s1_shim_units()
    s2_immune_load(args.ckpt)
    s3_hippo_path(args.ckpt)
    s4_gla_gqa_compat()

    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    n_fail = len(RESULTS) - n_pass
    section("SUMMARY")
    for name, ok, detail in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
    print(f"\n  {n_pass} PASS / {n_fail} FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
