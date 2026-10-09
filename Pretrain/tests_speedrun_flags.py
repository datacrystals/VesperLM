"""CPU smoke for the nanogpt-speedrun training opts (all env-gated, default OFF).

Flags and what they change (known limitations in brackets):

  VESPER_COMPILE=1    torch.compile of dense submodules (MoE expert MLPs +
                      GQA). [fla wrappers stay eager: Triton ops don't trace.
                      "whole" tier is best-effort and may graph-break.]
  VESPER_FUSED_CE=1   fla FusedLinearCrossEntropyLoss replaces
                      logits+softmax+CE; forward returns logits=None when
                      targets are given. Masking = ignore_index=pad_id, same
                      as the trainer. [fla FLCE is Triton-only: CPU falls back
                      to a pure-torch chunked equivalent. All-ignored batch is
                      0.0 fused vs NaN unfused. Generation path unaffected.]
  VESPER_VALUE_EMBED=1 token-id value tables added to attention values on
                      first/last layers. [kda/mla/gqa only; gla/mamba2 sites
                      are skipped with a warning. Flag-on checkpoints have 2
                      extra keys and cannot load into a flag-off model.]

Byte-compat guarantee: with every flag OFF, state_dict names+shapes are
identical to the pre-change model (checked against golden sha256 digests
captured from the tree at 92c75e1, before the speedrun-opts commits).

Run (CPU only; never touches CUDA):
    python Pretrain/tests_speedrun_flags.py
"""
import hashlib
import io
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "Common"))
sys.path.insert(0, _ROOT)

# Point Immune.cpu_backend's repo paths at THIS tree (not the main checkout)
os.environ.setdefault("VESPER_REPO", _ROOT)

from Immune.cpu_backend import install_cpu_shims  # noqa: E402

install_cpu_shims(kda_mla=True)

import torch  # noqa: E402

# Test-only guard: inductor's pattern inits (pad_mm, sfdp) allocate on CUDA
# whenever torch.cuda.is_available() (pytorch#97894 workaround). This box's
# GPU is running live training — keep compile smoke off it entirely. fla is
# already imported and CPU tensors never reach its Triton kernels (shims).
torch.cuda.is_available = lambda: False

# Deterministic defaults: this suite asserts default-OFF behavior, so drop
# any ambient gates from the calling environment (each test re-sets what it
# needs explicitly).
for _k in ("VESPER_COMPILE", "VESPER_FUSED_CE", "VESPER_VALUE_EMBED"):
    os.environ.pop(_k, None)

from vesper_linear_model import VesperLinearLM, _torch_chunked_linear_ce  # noqa: E402

# Golden state_dict digests of the pre-change model (sha256 of sorted
# "name:shape" lines), captured from the tree at 92c75e1 for the two tiny
# configs below. Any flag-off key/shape drift breaks these.
GOLDEN_KDA_MLA = "baafb70e38f108af62d0df808b38a60247eee3ff34b96042b7386105218de384"
GOLDEN_GLA_GQA = "42cdf0f82c242e11eb96ce01330fc48da14e3f24b07767055f932a88ea8539e9"

TINY = dict(vocab_size=128, dim=128, n_layers=4, n_heads=4, n_kv_heads=2,
            hidden_dim=384, num_experts=4, top_k=2, max_seq_len=64, pad_id=0)
KDA_MLA = dict(TINY, linear_type="kda", full_type="mla", kda_head_dim=64,
               kv_lora_rank=64, v_head_dim=64)
GLA_GQA = dict(TINY, linear_type="gla", full_type="gqa")


def sdigest(model):
    items = [f"{n}:{tuple(p.shape)}" for n, p in sorted(model.state_dict().items())]
    return hashlib.sha256("\n".join(items).encode()).hexdigest()


def batch():
    torch.manual_seed(123)
    x = torch.randint(0, 128, (2, 32))
    y = torch.randint(1, 128, (2, 32))
    y[:, :5] = 0    # pad_id masking, as the trainer's pad/eos convention
    y[:, -3:] = 0
    return x, y


def step(model, x, y):
    model.train()
    logits, ce, aux = model(x, y)
    loss = ce + 0.01 * aux
    loss.backward()
    assert torch.isfinite(loss), f"non-finite loss {loss}"
    return logits, float(ce)


def test_py_compile():
    import py_compile
    for rel in ("Common/vesper_linear_model.py", "Common/vesper_model.py",
                "Pretrain/02_pretrain_linear.py", "Pretrain/tests_speedrun_flags.py"):
        py_compile.compile(os.path.join(_ROOT, rel), doraise=True)
    print("PASS py_compile all edited files")


def test_flags_off_byte_compat():
    for name, kw, golden in (("kda_mla", KDA_MLA, GOLDEN_KDA_MLA),
                             ("gla_gqa", GLA_GQA, GOLDEN_GLA_GQA)):
        m = VesperLinearLM(**kw)  # every gate defaults to OFF
        assert not m.fused_ce
        assert sdigest(m) == golden, f"{name}: state_dict drifted from pre-change tree"
        # state_dict round-trip
        buf = io.BytesIO()
        torch.save(m.state_dict(), buf)
        buf.seek(0)
        m2 = VesperLinearLM(**kw)
        m2.load_state_dict(torch.load(buf, weights_only=True), strict=True)
        assert sdigest(m2) == golden
    print("PASS flags-off: state_dict names/shapes match pre-change golden digests,")
    print("     save/load round-trip strict-clean (both tiny configs)")


def test_compile_flag():
    x, y = batch()
    torch.manual_seed(0)
    m = VesperLinearLM(**KDA_MLA, compile_mode="1")
    assert "compiled" in m._compile_decision
    _, ce = step(m, x, y)
    # env gate
    os.environ["VESPER_COMPILE"] = "1"
    try:
        m2 = VesperLinearLM(**KDA_MLA)
    finally:
        del os.environ["VESPER_COMPILE"]
    assert "compiled" in m2._compile_decision
    m3 = VesperLinearLM(**KDA_MLA, compile_mode="0")
    assert m3._compile_decision.startswith("off")
    print(f"PASS VESPER_COMPILE=1: fwd+bwd finite (ce={ce:.4f}), env gate works,")
    print(f"     decision: {m._compile_decision}")


def test_fused_ce_flag():
    x, y = batch()
    torch.manual_seed(0)
    m_on = VesperLinearLM(**KDA_MLA, fused_ce=True)
    torch.manual_seed(0)
    m_off = VesperLinearLM(**KDA_MLA, fused_ce=False)
    m_on.load_state_dict(m_off.state_dict(), strict=True)

    m_on.eval(), m_off.eval()
    with torch.no_grad():
        logits, ce_on, _ = m_on(x, y)
        _, ce_off, _ = m_off(x, y)
    assert logits is None, "fused path must not materialize logits"
    assert abs(ce_on - ce_off) < 1e-5, (ce_on, ce_off)

    # bare masking equivalence vs the trainer's formula
    xt = torch.randn(64, 16)
    w = torch.randn(32, 16)
    t = torch.randint(1, 32, (64,))
    t[:7] = 0
    ref = torch.nn.functional.cross_entropy(xt @ w.t(), t, ignore_index=0)
    got = _torch_chunked_linear_ce(xt, t, w, ignore_index=0)
    assert abs(float(got) - float(ref)) < 1e-5

    _, ce = step(m_on, x, y)
    assert m_on.output.weight.grad.abs().sum() > 0  # tied emb/head gets grads

    # generation path still materializes logits
    m_on.eval()
    with torch.no_grad():
        lg, ce_none, _ = m_on(x)
    assert lg is not None and lg.shape == (2, 32, 128) and ce_none is None

    os.environ["VESPER_FUSED_CE"] = "1"
    try:
        m_env = VesperLinearLM(**KDA_MLA)
    finally:
        del os.environ["VESPER_FUSED_CE"]
    assert m_env.fused_ce
    print(f"PASS VESPER_FUSED_CE=1: loss {ce_on:.6f} == unfused {ce_off:.6f} "
          f"(ignore_index=pad_id masking preserved); train step finite (ce={ce:.4f});")
    print("     logits=None with targets, generation logits intact; env gate works")


def test_value_embed_flag():
    x, y = batch()
    torch.manual_seed(0)
    m_off = VesperLinearLM(**KDA_MLA, value_embed=False)
    torch.manual_seed(0)
    m_on = VesperLinearLM(**KDA_MLA, value_embed=True)

    base = {n: tuple(p.shape) for n, p in m_off.state_dict().items()}
    new = {n: tuple(p.shape) for n, p in m_on.state_dict().items()}
    added = {k: v for k, v in new.items() if k not in base}
    assert set(added) == {"layers.0.attn.value_emb.weight",
                          "layers.3.attn.value_emb.weight"}, added
    assert all(new[k] == base[k] for k in base), "pre-existing key shape changed"
    # kda value_dim == dim == 128; mla == n_heads(4) * v_head_dim(64) == 256
    assert added["layers.0.attn.value_emb.weight"] == (128, 128)
    assert added["layers.3.attn.value_emb.weight"] == (128, 256)

    # zero tables => bit-identical to the flag-off twin; random tables change it
    m_on.load_state_dict({**m_off.state_dict(),
                          "layers.0.attn.value_emb.weight": torch.zeros(128, 128),
                          "layers.3.attn.value_emb.weight": torch.zeros(128, 256)},
                         strict=True)
    m_on.eval(), m_off.eval()
    with torch.no_grad():
        _, ce_zero, _ = m_on(x, y)
        _, ce_ref, _ = m_off(x, y)
    assert abs(ce_zero - ce_ref) < 1e-6, (ce_zero, ce_ref)
    with torch.no_grad():
        m_on.layers[0].attn.value_emb.weight.normal_(0, 1.0)
        _, ce_rnd, _ = m_on(x, y)
    assert abs(ce_rnd - ce_ref) > 1e-4

    _, ce = step(m_on, x, y)
    g0 = m_on.layers[0].attn.value_emb.weight.grad
    g3 = m_on.layers[3].attn.value_emb.weight.grad
    assert g0.abs().sum() > 0 and g3.abs().sum() > 0

    # gla first layer: skipped with warning, last layer (gqa) still wired
    torch.manual_seed(0)
    m_g = VesperLinearLM(**GLA_GQA, value_embed=True)
    assert [n for n in m_g.state_dict() if "value_emb" in n] == \
        ["layers.3.attn.value_emb.weight"]
    step(m_g, x, y)

    os.environ["VESPER_VALUE_EMBED"] = "1"
    try:
        m_env = VesperLinearLM(**KDA_MLA)
    finally:
        del os.environ["VESPER_VALUE_EMBED"]
    assert "layers.0.attn.value_emb.weight" in m_env.state_dict()
    print(f"PASS VESPER_VALUE_EMBED=1: +2 keys only ({added}); zero tables == "
          f"flag-off loss bit-exactly; random tables change loss; both sites")
    print(f"     backprop (grad norms {float(g0.norm()):.4f}/{float(g3.norm()):.4f}); "
          f"gla site skipped; env gate works")


def test_all_flags_together():
    x, y = batch()
    torch.manual_seed(0)
    m = VesperLinearLM(**KDA_MLA, compile_mode="1", fused_ce=True, value_embed=True)
    logits, ce = step(m, x, y)
    assert logits is None
    # flags-off digest check is unaffected by flags-on extras: pre-existing
    # keys keep name+shape (covered in test_value_embed_flag)
    print(f"PASS all three flags on: fwd+bwd finite (ce={ce:.4f})")


if __name__ == "__main__":
    test_py_compile()
    test_flags_off_byte_compat()
    test_compile_flag()
    test_fused_ce_flag()
    test_value_embed_flag()
    test_all_flags_together()
    print("\nALL SPEEDRUN-FLAG SMOKES PASSED")
