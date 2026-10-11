"""GPU compatibility smoke gate for the MI50 box (gfx906) — run BEFORE any training.

Builds tiny VesperLinearLM instances and runs forward+backward through the fla
GLA Triton kernels (and the 470m_k training path: KDA + MLA + fused CE) on one
GPU. Any failure here means stop and report — do not train.

  HIP_VISIBLE_DEVICES=0 python3 lab/smi50_smoke_gate.py
"""
import os
import sys
import traceback

os.environ.setdefault("HIPBLASLT_DISABLE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Common"))

import torch
from vesper_linear_model import VesperLinearLM


def report(name, fn):
    try:
        out = fn()
        print(f"PASS  {name}: {out}")
        return True
    except Exception as e:
        print(f"FAIL  {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def main():
    dev = "cuda:0"
    print(f"torch {torch.__version__} | hip {torch.version.hip} | "
          f"device {torch.cuda.get_device_name(0)}")
    ok = True

    def tiny_gla():
        torch.manual_seed(0)
        m = VesperLinearLM(
            vocab_size=512, dim=128, n_layers=2, n_heads=4, n_kv_heads=2,
            hidden_dim=256, num_experts=2, top_k=1, max_seq_len=256,
            linear_type="gla", full_type="gqa", gla_head_dim=64,
            grad_checkpoint=False, fused_ce=False,
        ).to(dev)
        x = torch.randint(0, 512, (2, 128), device=dev)
        y = torch.randint(0, 512, (2, 128), device=dev)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, ce, aux = m(x, y)
        ce.backward()
        gnorm = sum(p.grad.float().norm() for p in m.parameters()
                    if p.grad is not None).item()
        assert torch.isfinite(ce), f"non-finite ce {ce}"
        assert torch.isfinite(torch.tensor(gnorm)), f"non-finite gnorm {gnorm}"
        return f"ce={ce.item():.4f} aux={aux.item():.4f} gnorm={gnorm:.3f}"

    def tiny_gla_fp32():
        # GLA runs fp32 internally even under autocast (P40-safe path).
        torch.manual_seed(1)
        m = VesperLinearLM(
            vocab_size=512, dim=128, n_layers=2, n_heads=4, n_kv_heads=2,
            hidden_dim=256, num_experts=2, top_k=1, max_seq_len=256,
            linear_type="gla", full_type="gqa", gla_head_dim=64,
            grad_checkpoint=True, fused_ce=False,
        ).to(dev)
        x = torch.randint(0, 512, (2, 128), device=dev)
        y = torch.randint(0, 512, (2, 128), device=dev)
        logits, ce, aux = m(x, y)  # fp32, grad-checkpointed
        ce.backward()
        return f"ce={ce.item():.4f}"

    def tiny_470m_k_path():
        # The real training arch (kda + mla + passport) at tiny scale.
        torch.manual_seed(2)
        m = VesperLinearLM(
            vocab_size=512, dim=128, n_layers=2, n_heads=4, n_kv_heads=2,
            hidden_dim=256, num_experts=2, top_k=1, max_seq_len=256,
            linear_type="kda", full_type="mla", kda_head_dim=32,
            kv_lora_rank=64, v_head_dim=32,
            router_type="passport", passport_dim=16,
            grad_checkpoint=True, fused_ce=False,
        ).to(dev)
        x = torch.randint(0, 512, (2, 128), device=dev)
        y = torch.randint(0, 512, (2, 128), device=dev)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, ce, aux = m(x, y)
        ce.backward()
        assert torch.isfinite(ce)
        return f"ce={ce.item():.4f} aux={aux.item():.4f}"

    def fused_ce_path():
        # VESPER_FUSED_CE=1 training path: fla FusedLinearCrossEntropyLoss.
        os.environ["VESPER_FUSED_CE"] = "1"
        try:
            torch.manual_seed(3)
            m = VesperLinearLM(
                vocab_size=512, dim=128, n_layers=2, n_heads=4, n_kv_heads=2,
                hidden_dim=256, num_experts=2, top_k=1, max_seq_len=256,
                linear_type="kda", full_type="mla", kda_head_dim=32,
                kv_lora_rank=64, v_head_dim=32,
                router_type="passport", passport_dim=16,
                grad_checkpoint=True,
            ).to(dev)
            assert m.fused_ce, f"fused_ce not enabled: {m._fused_ce_decision}"
            x = torch.randint(0, 512, (2, 128), device=dev)
            y = torch.randint(0, 512, (2, 128), device=dev)
            y[0, :10] = m.pad_id  # ignore-masking path
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, ce, aux = m(x, y)
            ce.backward()
            assert torch.isfinite(ce)
            return f"ce={ce.item():.4f} ({m._fused_ce_decision})"
        finally:
            os.environ.pop("VESPER_FUSED_CE", None)

    ok &= report("tiny VesperLinearLM GLA fwd+bwd bf16 autocast", tiny_gla)
    ok &= report("tiny VesperLinearLM GLA fwd+bwd fp32 + grad ckpt", tiny_gla_fp32)
    ok &= report("tiny 470m_k path (KDA+MLA+passport) fwd+bwd bf16", tiny_470m_k_path)
    ok &= report("VESPER_FUSED_CE fla linear+CE", fused_ce_path)

    print("SMOKE GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
