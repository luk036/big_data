import numpy as np
import pytest

from big_data.sketches.count_min import CountMinSketch


def test_hash_is_deterministic_across_instances():
    assert CountMinSketch(100, 5).hash_params == CountMinSketch(100, 5).hash_params


def test_rejects_invalid_dimensions_and_mode():
    with pytest.raises(ValueError):
        CountMinSketch(width=0)
    with pytest.raises(ValueError):
        CountMinSketch(depth=0)
    with pytest.raises(ValueError):
        CountMinSketch(mode="bogus")


def test_update_tracks_total_weight():
    cms = CountMinSketch(width=64, depth=3)
    cms.update(1)
    cms.update(2, 3.0)
    assert cms.total == pytest.approx(4.0)


def test_query_never_underestimates_under_collisions():
    cms = CountMinSketch(width=32, depth=4)
    truth = {}
    rng = np.random.default_rng(0)
    for key in range(500):
        count = int(rng.integers(1, 20))
        truth[key] = count
        for _ in range(count):
            cms.update(key)
    for key, count in truth.items():
        assert cms.query(key) >= count


def test_mean_min_reduces_overcounting_bias():
    cms = CountMinSketch(width=64, depth=5)
    truth = {}
    rng = np.random.default_rng(1)
    for key in range(1000):
        count = int(rng.integers(1, 50))
        truth[key] = count
        for _ in range(count):
            cms.update(key)

    min_err = np.mean([cms.query(k) - v for k, v in truth.items()])
    mean_min_err = np.mean([cms.query_mean_min(k) - v for k, v in truth.items()])

    assert min_err > 0
    assert abs(mean_min_err) < abs(min_err)


def test_conservative_update_counts_no_more_than_classic():
    classic = CountMinSketch(width=32, depth=4)
    conservative = CountMinSketch(width=32, depth=4, mode="conservative")
    rng = np.random.default_rng(2)
    truth = {}
    for key in range(300):
        count = int(rng.integers(1, 10))
        truth[key] = count
        for _ in range(count):
            classic.update(key)
            conservative.update(key)

    for key, count in truth.items():
        assert conservative.query(key) >= count
        assert conservative.query(key) <= classic.query(key)


def test_merge_is_exact_associative_and_commutative():
    a = CountMinSketch(width=50, depth=3)
    a.update(1)
    a.update(2)
    b = CountMinSketch(width=50, depth=3)
    b.update(2)
    b.update(3)
    c = CountMinSketch(width=50, depth=3)
    c.update(4, 2.0)

    single_pass = CountMinSketch(width=50, depth=3)
    for key, value in [(1, 1.0), (2, 1.0), (2, 1.0), (3, 1.0), (4, 2.0)]:
        single_pass.update(key, value)

    merged = a.merge(b).merge(c)
    np.testing.assert_array_equal(merged.sketch, single_pass.sketch)
    assert merged.total == pytest.approx(single_pass.total)

    np.testing.assert_array_equal(a.merge(b).sketch, b.merge(a).sketch)
    np.testing.assert_array_equal(
        a.merge(b).merge(c).sketch, a.merge(b.merge(c)).sketch
    )


def test_merge_rejects_mismatched_dimensions():
    with pytest.raises(ValueError):
        CountMinSketch(width=10, depth=2).merge(CountMinSketch(width=11, depth=2))


def test_reset_clears_counters_and_total():
    cms = CountMinSketch(width=16, depth=2)
    cms.update(1, 5.0)
    cms.reset()
    assert cms.total == 0.0
    assert not cms.sketch.any()


def test_estimate_with_error_reports_additive_bound():
    cms = CountMinSketch(width=100, depth=3)
    for _ in range(50):
        cms.update(7)
    estimate, bound = cms.estimate_with_error(7)
    assert estimate == cms.query(7)
    assert bound == pytest.approx(np.e * cms.total / 100)
