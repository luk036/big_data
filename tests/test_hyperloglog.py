import numpy as np
import pytest


def test_cpu_hll_variants_are_bit_identical():
    from big_data.hyper_loglog_demo import HyperLogLog
    from big_data.hyper_loglog_gpu_demo import CPUHyperLogLog

    items = [f"node_{i}" for i in range(20000)]
    reference = HyperLogLog(p=12)
    cpu = CPUHyperLogLog(p=12)
    for item in items:
        reference.add(item)
        cpu.add(item)

    np.testing.assert_array_equal(reference.registers, cpu.registers)
    assert reference.count() == cpu.count()


def test_eda_variant_estimates_distinct_element_count():
    from big_data.hyper_loglog_eda_demo import EDAHyperLogLog

    n = 50000
    eda = EDAHyperLogLog(p=12)
    for i in range(n):
        eda.add_element("NET", i, {"layer": i % 4})

    assert eda._estimate_cardinality() == pytest.approx(n, rel=0.08)


def test_rank_and_register_index_helpers():
    from big_data.sketches.hyperloglog import rank, register_index

    p = 12
    assert register_index(0, p) == 0
    assert register_index(1 << 63, p) == 1 << (p - 1)
    assert rank(0, p) == (64 - p) + 1
    assert rank(1, p) == 64 - p
    assert rank(1 << (64 - p - 1), p) == 1


def test_from_relative_error_sizes_registers():
    from big_data.hyper_loglog_demo import HyperLogLog

    hll = HyperLogLog.from_relative_error(0.02)
    assert hll.m == 1 << hll.p
    assert 1.04 / (hll.m**0.5) <= 0.02


def test_from_relative_error_rejects_out_of_range():
    from big_data.hyper_loglog_demo import HyperLogLog

    with pytest.raises(ValueError):
        HyperLogLog.from_relative_error(0.0)
    with pytest.raises(ValueError):
        HyperLogLog.from_relative_error(1.5)


def test_estimate_is_dtype_safe_for_unsigned_registers():
    from big_data.sketches.hyperloglog import estimate

    signed = np.array([0, 3, 5, 7], dtype=np.int64)
    unsigned = signed.astype(np.uint32)

    expected = estimate(signed, 3, small_range=False, large_range=False)
    actual = estimate(unsigned, 3, small_range=False, large_range=False)

    assert expected > 0
    assert actual == pytest.approx(expected)
