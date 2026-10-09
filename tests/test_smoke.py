import importlib
import pkgutil

import big_data


def _all_modules():
    return sorted(m.name for m in pkgutil.iter_modules(big_data.__path__))


def test_all_modules_importable():
    modules = _all_modules()
    assert modules, "no modules discovered in the big_data package"
    for name in modules:
        importlib.import_module(f"big_data.{name}")


def test_count_min_sketch_never_underestimates():
    from big_data.basic_count_min_sketch import CountMinSketch

    cms = CountMinSketch(width=256, depth=4)
    for _ in range(5):
        cms.update(42, 1.0)
    assert cms.query(42) >= 5.0


def test_hyperloglog_estimates_cardinality():
    from big_data.hyper_loglog_demo import HyperLogLog

    hll = HyperLogLog(p=14)
    n = 1000
    for i in range(n):
        hll.add(i)
    estimate = hll.count()
    assert abs(estimate - n) / n < 0.1
