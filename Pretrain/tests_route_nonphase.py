"""CPU unit check for the t3-postmortem data-pipeline fix (route_nonphase).

The bug: MixedDataStream phase bucketing name-matches 'phase1'/'phase2' in the
index.txt filename, so all vesperk/* bins (~18B tokens) matched neither stream
and were silently dropped; stream weights were hardcoded to {k: 1.0}, so
index.txt weights were ignored too.

This suite builds a tiny synthetic data/ tree (fake .bin shards + fake
index.txt) and asserts:

  1. flag OFF (default): stream construction + drawn batches are bit-identical
     to the legacy inline pipeline (pre-fix 02_pretrain_linear.py lines), and
     vesperk-named files are still dropped — i.e. t0/t1/t2/t3 ladder
     comparability is preserved;
  2. flag ON (VESPER_ROUTE_NONPHASE=1 / route_nonphase=True): vesperk-named
     files enter training through the always-on stream at the configured
     share, for both phase streams;
  3. flag ON: index.txt weights are honored (per-file draw counts track the
     weights within every stream, non-phase included);
  4. flag ON: val is evaluated per group (phase1/phase2/nonphase) and the val
     mix composition table names every file explicitly.

Run (CPU only; never touches CUDA):
    python3 Pretrain/tests_route_nonphase.py
"""
import os
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "Common"))
sys.path.insert(0, _HERE)

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("p01", os.path.join(_HERE, "01_pretrain.py"))
p01 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(p01)

# Token-id markers per fake shard (uint16), so a batch's ids identify its source
TOKEN_IDS = {
    "pretrain/phase1_pretrain.bin": 101,
    "pretrain/nemotron_phase1.bin": 102,
    "pretrain/nemotron_phase2.bin": 201,
    "pretrain/phase2_pretrain.bin": 202,
    "vesperk/fineweb_edu_0.bin": 301,
    "vesperk/wikipedia_0.bin": 302,
}

# index.txt weights (relative). The two phase1 files deliberately differ (1:3)
# so "weights honored" is observable on a phase stream too, not only on the
# non-phase stream (3:1).
WEIGHTS = {
    "pretrain/phase1_pretrain.bin": 1.0,
    "pretrain/nemotron_phase1.bin": 3.0,
    "pretrain/nemotron_phase2.bin": 1.0,
    "pretrain/phase2_pretrain.bin": 1.0,
    "vesperk/fineweb_edu_0.bin": 3.0,
    "vesperk/wikipedia_0.bin": 1.0,
}

BIN_TOKENS = 4096
BATCH = 2
SEQ = 16          # max_seq_len=start_len=16 => seq len constant 16
ACCUM = 2


def make_fake_data(root):
    for rel, tid in TOKEN_IDS.items():
        path = os.path.join(root, "data", rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.full(BIN_TOKENS, tid, dtype=np.uint16).tofile(path)
    with open(os.path.join(root, "data", "index.txt"), "w") as f:
        f.write("# synthetic index for tests_route_nonphase.py\n")
        for rel, w in WEIGHTS.items():
            f.write(f"{rel}, {w:g}\n")


def build_via_factory(datasets_dict, index_probs, route_nonphase, nonphase_share=0.3):
    return p01.build_curriculum_streams(
        datasets_dict, index_probs,
        batch_size=BATCH, start_step=0, phase2_start_step=0,
        accumulation_steps=ACCUM, seq_len_warmup=0, max_seq_len=SEQ,
        seq_len_start=SEQ, is_distributed=False,
        route_nonphase=route_nonphase, nonphase_share=nonphase_share)


def build_legacy(datasets_dict):
    """Verbatim pre-fix stream construction from 02_pretrain_linear.py."""
    phase1_train = {n: d for n, d in datasets_dict['train'].items() if 'phase1' in n}
    phase2_train = {n: d for n, d in datasets_dict['train'].items() if 'phase2' in n}
    assert phase1_train and phase2_train
    val_datasets = {n: d for n, d in datasets_dict['val'].items()
                    if 'phase1' in n or 'phase2' in n}
    _n_p1 = sum(1 for n in val_datasets if 'phase1' in n)
    _n_p2 = max(len(val_datasets) - _n_p1, 1)
    val_probs = {n: (0.8 / max(_n_p1, 1) if 'phase1' in n else 0.2 / _n_p2)
                 for n in val_datasets}
    phase1_stream = p01.MixedDataStream(
        phase1_train, {k: 1.0 for k in phase1_train}, BATCH, 0, ACCUM,
        0, SEQ, SEQ, False, resume_state=None)
    phase2_stream = p01.MixedDataStream(
        phase2_train, {k: 1.0 for k in phase2_train}, BATCH, 0, ACCUM,
        0, SEQ, SEQ, False, resume_state=None)
    val_stream = p01.MixedDataStream(
        val_datasets, val_probs, BATCH, 0, 1, 0, SEQ, SEQ, False,
        resume_state=None)
    return phase1_stream, phase2_stream, val_stream


def draw(stream, n):
    batches = []
    sources = []
    for _ in range(n):
        x, y = next(stream)
        batches.append((x.numpy().copy(), y.numpy().copy()))
        sources.append(stream.last_source)
    return batches, sources


def test_load_and_bucket(tmp):
    os.chdir(tmp)
    datasets_dict, index_probs = p01.load_dataset_index("data/index.txt")
    assert set(datasets_dict['train']) == set(TOKEN_IDS), datasets_dict['train'].keys()
    assert set(datasets_dict['val']) == set(TOKEN_IDS)
    for rel, w in WEIGHTS.items():
        assert abs(index_probs[rel] - w / sum(WEIGHTS.values())) < 1e-12, rel

    p1, p2, np_ = p01.bucket_curriculum_datasets(datasets_dict['train'],
                                                 route_nonphase=False)
    assert set(p1) == {"pretrain/phase1_pretrain.bin", "pretrain/nemotron_phase1.bin"}
    assert set(p2) == {"pretrain/nemotron_phase2.bin", "pretrain/phase2_pretrain.bin"}
    assert np_ == {}, f"flag-off must still drop non-phase files, got {set(np_)}"

    p1, p2, np_ = p01.bucket_curriculum_datasets(datasets_dict['train'],
                                                 route_nonphase=True)
    assert set(np_) == {"vesperk/fineweb_edu_0.bin", "vesperk/wikipedia_0.bin"}
    print("PASS bucketing: phase1/phase2 name-match unchanged; vesperk/* is the")
    print("     non-phase bucket, dropped when route off, routed when on")


def test_flag_off_bit_identical(tmp):
    os.chdir(tmp)
    datasets_dict, index_probs = p01.load_dataset_index("data/index.txt")

    # legacy construction + draws, fixed seed
    np.random.seed(1234)
    leg_p1, leg_p2, leg_val = build_legacy(datasets_dict)
    leg_out = (draw(leg_p1, 40)[0], draw(leg_p2, 40)[0], draw(leg_val, 40)[0])
    leg_rng = np.random.get_state()

    # new construction with the flag off must reproduce every batch bit-exactly
    np.random.seed(1234)
    s = build_via_factory(datasets_dict, index_probs, route_nonphase=False)
    assert s.nonphase_stream is None
    assert s.val_stream is not None and s.val_group_streams == {}
    assert s.train_pre is s.phase1_stream and s.train_post is s.phase2_stream
    assert s.nonphase_probs == {}
    new_out = (draw(s.train_pre, 40)[0], draw(s.train_post, 40)[0],
               draw(s.val_stream, 40)[0])
    new_rng = np.random.get_state()

    for name, a, b in zip(("phase1", "phase2", "val"), leg_out, new_out):
        for i, ((ax, ay), (bx, by)) in enumerate(zip(a, b)):
            assert np.array_equal(ax, bx) and np.array_equal(ay, by), \
                f"{name} batch {i} diverged from legacy pipeline"
    assert leg_rng[0] == new_rng[0] and np.array_equal(leg_rng[1], new_rng[1]) \
        and leg_rng[2:] == new_rng[2:], "numpy RNG state diverged"

    # and vesperk tokens never appear (still dropped)
    for part in new_out:
        for x, y in part:
            assert x.max() < 300 and y.max() < 300
    print("PASS flag-off: 40x3 batches bit-identical to the legacy pipeline")
    print("     (same rng state afterwards); vesperk/* still dropped")


def test_flag_on_routing_and_weights(tmp):
    os.chdir(tmp)
    datasets_dict, index_probs = p01.load_dataset_index("data/index.txt")

    # share=1.0 -> every micro-batch from the always-on non-phase stream
    s = build_via_factory(datasets_dict, index_probs, route_nonphase=True,
                          nonphase_share=1.0)
    assert s.nonphase_stream is not None
    np.random.seed(7)
    b1, _ = draw(s.train_pre, 60)
    b2, _ = draw(s.train_post, 60)
    for batches in (b1, b2):
        for x, y in batches:
            assert x.min() >= 300 and y.min() >= 300, "non-phase tokens missing"

    # share=0.0 -> always-on stream exists but phase streams never splice it in
    s0 = build_via_factory(datasets_dict, index_probs, route_nonphase=True,
                           nonphase_share=0.0)
    np.random.seed(7)
    b1, _ = draw(s0.train_pre, 60)
    for x, y in b1:
        assert x.max() < 300, "share=0.0 must not splice non-phase batches"

    # share=0.5 -> both populations present, roughly half/half
    s5 = build_via_factory(datasets_dict, index_probs, route_nonphase=True,
                           nonphase_share=0.5)
    np.random.seed(7)
    _, src = draw(s5.train_pre, 400)
    from_nonphase = sum(1 for n in src if n.startswith("vesperk/"))
    assert 140 < from_nonphase < 260, from_nonphase
    print(f"PASS flag-on routing: share 1.0 -> 100% vesperk batches, 0.0 -> 0%, "
          f"0.5 -> {from_nonphase}/400 non-phase")

    # weights: non-phase stream draws 3:1 (fineweb_edu_0 : wikipedia_0);
    # phase-1 stream draws 3:1 (nemotron_phase1 : phase1_pretrain) too —
    # index.txt weights are honored within every stream when the flag is on.
    np.random.seed(11)
    _, src_np = draw(s5.nonphase_stream, 400)
    fw = sum(1 for n in src_np if "fineweb" in n)
    assert abs(fw / 400 - 0.75) < 0.06, fw
    _, src_p1 = draw(s5.phase1_stream, 400)
    nem = sum(1 for n in src_p1 if "nemotron_phase1" in n)
    assert abs(nem / 400 - 0.75) < 0.06, nem
    print(f"PASS flag-on weights: non-phase fineweb:wikipedia = {fw}:{400-fw} (~3:1),")
    print(f"     phase1 nemotron:phase1_pretrain = {nem}:{400-nem} (~3:1)")


def test_flag_on_val_groups(tmp):
    os.chdir(tmp)
    datasets_dict, index_probs = p01.load_dataset_index("data/index.txt")
    s = build_via_factory(datasets_dict, index_probs, route_nonphase=True)
    assert s.val_stream is None
    assert set(s.val_group_streams) == {"phase1", "phase2", "nonphase"}
    expected = {
        "phase1": {101, 102},
        "phase2": {201, 202},
        "nonphase": {301, 302},
    }
    np.random.seed(3)
    for g, gs in s.val_group_streams.items():
        batches, _ = draw(gs, 20)
        seen = set()
        for x, y in batches:
            seen |= set(np.unique(x).tolist()) | set(np.unique(y).tolist())
        assert seen == expected[g], (g, seen)

    rows = {r["name"]: r for r in s.val_mix_table()}
    assert set(rows) == set(TOKEN_IDS), rows.keys()
    assert rows["vesperk/fineweb_edu_0.bin"]["bucket"] == "nonphase"
    # probs are within-stream: the non-phase group stream draws 3:1
    assert abs(rows["vesperk/fineweb_edu_0.bin"]["prob"] - 0.75) < 1e-12
    assert abs(rows["vesperk/wikipedia_0.bin"]["prob"] - 0.25) < 1e-12
    train_rows = {r["name"]: r for r in s.train_mix_table()}
    assert set(train_rows) == set(TOKEN_IDS)
    assert train_rows["vesperk/wikipedia_0.bin"]["bucket"] == "nonphase"
    print("PASS flag-on val: per-group streams cover phase1/phase2/nonphase")
    print("     disjointly; val_mix_table() names every file with its group/prob")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="vesper_route_nonphase_") as tmp:
        make_fake_data(tmp)
        test_load_and_bucket(tmp)
        test_flag_off_bit_identical(tmp)
        test_flag_on_routing_and_weights(tmp)
        test_flag_on_val_groups(tmp)
    print("\nALL ROUTE-NONPHASE DATA-PIPELINE CHECKS PASSED")
