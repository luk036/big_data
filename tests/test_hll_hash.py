import numpy as np
import pytest

from big_data.hll_hash import (
    fmix64,
    hash_array_u64,
    hash_bytes,
    hash_strings,
    hash_value,
)

try:
    from numba import cuda
except ImportError:  # pragma: no cover
    cuda = None

cuda_required = pytest.mark.skipif(
    cuda is None or not cuda.is_available(), reason="CUDA device required"
)


def test_hash_is_deterministic_and_64_bit():
    for value in ["item_0", "item_1", 42, ("a", 1)]:
        result = hash_value(value)
        assert result == hash_value(value)
        assert 0 <= result < (1 << 64)


def test_symbols_sharing_a_prefix_hash_differently():
    # Guards against reducing only the first 8 bytes, which would collapse all
    # keys of the form "item_XXXXXX".
    hashes = {hash_value(f"item_{i}") for i in range(5000)}
    assert len(hashes) == 5000


def test_hash_bytes_folds_every_byte():
    assert hash_bytes(b"item_00000001") != hash_bytes(b"item_00000002")


def test_fmix64_zero_is_zero():
    assert fmix64(0) == 0


def test_vectorised_mixer_matches_scalar():
    values = np.array([0, 1, 2, 12345, 2**63, 2**64 - 1], dtype=np.uint64)
    expected = np.array([fmix64(int(v)) for v in values], dtype=np.uint64)
    np.testing.assert_array_equal(hash_array_u64(values), expected)


def test_vectorised_string_hasher_matches_scalar():
    items = ["", "a", "item_0", "item_123", "item_00000000", "xyz", 42]
    expected = np.array([hash_value(x) for x in items], dtype=np.uint64)
    np.testing.assert_array_equal(hash_strings(items), expected)


def test_cpu_hll_implementations_agree_beyond_small_range():
    # n is chosen so the small-range linear-counting correction does NOT apply,
    # which is where the rank computation (rank = leading zeros + 1) matters.
    from big_data.hyper_loglog_demo import HyperLogLog
    from big_data.hyper_loglog_gpu_demo import CPUHyperLogLog

    n = 100000
    data = [f"item_{i}" for i in range(n)]
    reference = HyperLogLog(p=12)
    comparison = CPUHyperLogLog(p=12)
    for item in data:
        reference.add(item)
        comparison.add(item)

    assert reference.count() == pytest.approx(comparison.count(), rel=1e-6)
    assert comparison.count() == pytest.approx(n, rel=0.08)


@cuda_required
def test_fused_device_hashing_matches_cpu():
    from big_data.hyper_loglog_gpu_demo import CPUHyperLogLog, GPUHyperLogLog, HLLConfig

    rng = np.random.default_rng(0)
    keys = rng.integers(0, 500000, size=200000, dtype=np.uint64)
    gpu = GPUHyperLogLog(HLLConfig(p=12))
    gpu.add_keys(keys)
    cpu = CPUHyperLogLog(p=12)
    cpu.add_hashes(hash_array_u64(keys))

    assert gpu.count() == pytest.approx(cpu.count(), rel=1e-6)


@cuda_required
def test_gpu_and_cpu_hll_agree_on_same_data():
    from big_data.hyper_loglog_gpu_demo import CPUHyperLogLog, GPUHyperLogLog, HLLConfig

    data = [f"item_{i}" for i in range(100000)]
    gpu = GPUHyperLogLog(HLLConfig(p=12))
    gpu.add_batch(data)
    cpu = CPUHyperLogLog(p=12)
    cpu.add_batch(data)

    assert gpu.count() == pytest.approx(cpu.count(), rel=0.05)
