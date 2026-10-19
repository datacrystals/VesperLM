"""CPU smoke for the speedrun-style training opts (roadmap item c).

Everything here is CPU-only (torch.cuda.is_available is forced False so
neither inductor nor fla can touch the live GPU). Proves:

  1. VESPER_FUSED_CE numerics — the pure-torch chunked logsumexp linear+CE
     (`_torch_chunked_linear_ce` / `_ChunkedLinearCE`) matches plain
     F.cross_entropy on tiny fp32 tensors: max abs diff of the loss AND of
     the input/head gradients < 1e-3 (measured well below that; see output).
     Also: the fused Function retains NO [N, V] logits between forward and
     backward (the 470M 2 GiB OOM fix class) — checked structurally via the
     autograd node's saved tensors — and an all-ignored batch is 0.0
     (fused semantics) instead of the unfused NaN.
  2. Gates default OFF — with VESPER_COMPILE / VESPER_FUSED_CE /
     VESPER_VALUE_EMBED unset, a tiny VesperLinearLM reports every flag off,
     has no extra state_dict keys, and a seeded forward pass is
     byte-identical (sha256 of logits bytes) to a model explicitly built
     with the gates off. Each gate is a one-line env flip.

Full flag matrix + flags-off state_dict golden digests live in
Pretrain/tests_speedrun_flags.py (run that too for the byte-compat proof
against the pre-change tree).

Run:
    python lab/speedrun_smoke.py
"""
import hashlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "Common"))
sys.path.insert(0, _ROOT)

# Point Immune.cpu_backend's repo paths at THIS tree
os.environ.setdefault("VESPER_REPO", _ROOT)

from Immune.cpu_backend import install_cpu_shims  # noqa: E402

install_cpu_shims(kda_mla=True)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# Never touch the live GPU: inductor pattern inits (pad_mm, sfdp) allocate on
# CUDA whenever torch.cuda.is_available() is true (pytorch#97894 workaround).
torch.cuda.is_available = lambda: False

# This suite asserts default-OFF behavior: drop ambient gates (each test sets
# explicitly what it needs).
for _k in ("VESPER_COMPILE", "VESPER_FUSED_CE", "VESPER_VALUE_EMBED"):
    os.environ.pop(_k, None)

from vesper_linear_model import (  # noqa: E402
    VesperLinearLM,
    _ChunkedLinearCE,
    _torch_chunked_linear_ce,
)

TINY = dict(vocab_size=128, dim=128, n_layers=4, n_heads=4, n_kv_heads=2,
            hidden_dim=384, num_experts=4, top_k=2, max_seq_len=64, pad_id=0)
KDA_MLA = dict(TINY, linear_type="kda", full_type="mla", kda_head_dim=64,
               kv_lora_rank=64, v_head_dim=64)

MAX_ABS_DIFF = 1e-3  # task bar for the fp32 CE numerics check


def batch():
    torch.manual_seed(123)
    x = torch.randint(0, 128, (2, 32))
    y = torch.randint(1, 128, (2, 32))
    y[:, :5] = 0    # pad_id masking, as the trainer's pad/eos convention
    y[:, -3:] = 0
    return x, y


def forward_hash(model, x, y):
    model.eval()
    with torch.no_grad():
        logits, ce, _ = model(x, y)
    h = hashlib.sha256()
    h.update(logits.detach().numpy().tobytes())
    h.update(f"{float(ce):.17e}".encode())
    return h.hexdigest(), float(ce), logits


def test_fused_ce_numerics():
    """(2) chunked logsumexp CE == plain CE on tiny fp32 tensors."""
    torch.manual_seed(0)
    worst_loss = worst_grad = 0.0

    cases = [
        (8, 4, 16, 4),      # literal tiny tensor, 4 chunks
        (200, 32, 512, 8),  # N not divisible by num_chunks -> chunk edges
    ]
    for N, D, V, num_chunks in cases:
        for _ in range(3):
            x = torch.randn(N, D)
            w = torch.randn(V, D)
            t = torch.randint(1, V, (N,))
            t[: max(1, N // 8)] = 0     # pad_id-style ignore prefix
            t[-(max(1, N // 10)):] = 0  # ...and suffix

            ref = F.cross_entropy(x @ w.t(), t, ignore_index=0)
            got = _torch_chunked_linear_ce(x, t, w, ignore_index=0,
                                           num_chunks=num_chunks)
            d_loss = abs(float(got) - float(ref))
            worst_loss = max(worst_loss, d_loss)
            assert d_loss < MAX_ABS_DIFF, (N, D, V, d_loss)

            x1, w1 = x.clone().requires_grad_(), w.clone().requires_grad_()
            F.cross_entropy(x1 @ w1.t(), t, ignore_index=0).backward()
            x2, w2 = x.clone().requires_grad_(), w.clone().requires_grad_()
            _torch_chunked_linear_ce(x2, t, w2, ignore_index=0,
                                     num_chunks=num_chunks).backward()
            d_grad = max((x1.grad - x2.grad).abs().max().item(),
                         (w1.grad - w2.grad).abs().max().item())
            worst_grad = max(worst_grad, d_grad)
            assert d_grad < MAX_ABS_DIFF, (N, D, V, d_grad)

    # all-ignored batch: fused semantics = 0.0 (unfused F.cross_entropy = NaN)
    x = torch.randn(16, 8)
    w = torch.randn(32, 8)
    t = torch.zeros(16, dtype=torch.long)
    got = float(_torch_chunked_linear_ce(x, t, w, ignore_index=0))
    assert got == 0.0, got

    print(f"PASS fused-CE numerics: max |dloss|={worst_loss:.3e}, "
          f"max |dgrad|={worst_grad:.3e} (bar {MAX_ABS_DIFF}, fp32, "
          f"2 shapes x 3 trials, ignore_index=pad_id)")
    print("     all-ignored batch -> 0.0 (fused semantics; unfused is NaN)")


def test_no_logits_retained():
    """(2) fix-class proof: no [N, V] logits kept between fwd and bwd."""
    N, D, V, num_chunks = 64, 16, 32, 8
    x = torch.randn(N, D, requires_grad=True)
    w = torch.randn(V, D, requires_grad=True)
    t = torch.randint(1, V, (N,))
    t[:3] = 0
    out = _torch_chunked_linear_ce(x, t, w, ignore_index=0, num_chunks=num_chunks)
    shapes = [tuple(s.shape) for s in out.grad_fn.saved_tensors]
    assert (N, V) not in shapes, f"full logits retained for backward: {shapes}"
    assert set(shapes) == {(N, D), (V, D), (N,)}, shapes
    # peak logits working set is one chunk, not N: C = ceil(N/num_chunks)
    C = (N + num_chunks - 1) // num_chunks
    assert C < N
    out.backward()
    assert x.grad.abs().sum() > 0 and w.grad.abs().sum() > 0
    print(f"PASS no logits retained: saved {shapes} (no ({N}, {V})); "
          f"backward works chunkwise (C={C} < N={N})")


def test_gates_default_off():
    """Gates unset == gates explicitly off: same flags, byte-identical fwd."""
    x, y = batch()

    torch.manual_seed(0)
    m_default = VesperLinearLM(**KDA_MLA)  # every gate unset
    torch.manual_seed(0)
    m_off = VesperLinearLM(**KDA_MLA, compile_mode="0", fused_ce=False,
                           value_embed=False)

    assert not m_default.fused_ce
    assert m_default._compile_decision.startswith("off"), m_default._compile_decision
    assert m_default.value_embed_decision.startswith("off"), m_default.value_embed_decision
    assert not any("value_emb" in k for k in m_default.state_dict())
    assert set(m_default.state_dict()) == set(m_off.state_dict())

    h_def, ce_def, lg_def = forward_hash(m_default, x, y)
    h_off, ce_off, lg_off = forward_hash(m_off, x, y)
    assert h_def == h_off, "gate-unset forward differs from explicit-off"
    assert ce_def == ce_off and torch.equal(lg_def, lg_off)

    # unfused loss formula unchanged with gates off
    ref = F.cross_entropy(lg_def.view(-1, 128), y.view(-1), ignore_index=0)
    assert abs(float(ref) - ce_def) < 1e-5, (float(ref), ce_def)

    # gates-off train step is finite and uses the unfused path (logits materialized)
    m_default.train()
    logits, ce, aux = m_default(x, y)
    assert logits is not None and logits.shape == (2, 32, 128)
    (ce + 0.01 * aux).backward()
    assert torch.isfinite(ce)

    # one-line flips
    flips = {}
    for k in ("VESPER_COMPILE", "VESPER_FUSED_CE", "VESPER_VALUE_EMBED"):
        os.environ[k] = "1"
        try:
            m = VesperLinearLM(**KDA_MLA)
        finally:
            del os.environ[k]
        if k == "VESPER_COMPILE":
            flips[k] = "compiled" in m._compile_decision
        elif k == "VESPER_FUSED_CE":
            flips[k] = m.fused_ce
        else:
            flips[k] = any("value_emb" in n for n in m.state_dict())
        assert flips[k], k

    print(f"PASS gates default OFF: flags off, no extra keys, fwd sha256 "
          f"{h_def[:16]}... == explicit-off (bit-identical logits+ce); "
          f"unfused CE formula intact; train step finite")
    print(f"     one-line env flips work: {flips}")


def test_all_three_on():
    os.environ["VESPER_FUSED_CE"] = "1"
    try:
        torch.manual_seed(0)
        m = VesperLinearLM(**KDA_MLA, compile_mode="1", value_embed=True)
    finally:
        del os.environ["VESPER_FUSED_CE"]
    x, y = batch()
    m.train()
    logits, ce, aux = m(x, y)
    assert logits is None and torch.isfinite(ce)
    (ce + 0.01 * aux).backward()
    print(f"PASS all three on: fwd+bwd finite (ce={float(ce):.4f}), "
          f"logits=None as contracted")


if __name__ == "__main__":
    test_fused_ce_numerics()
    test_no_logits_retained()
    test_gates_default_off()
    test_all_three_on()
    print("\nALL SPEEDRUN SMOKES PASSED (CPU-only)")
