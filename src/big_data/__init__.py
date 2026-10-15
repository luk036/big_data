"""big_data: sketch algorithms and GPU experiments for EDA.

This package bundles a collection of self-contained experiments covering:

* probabilistic streaming data structures -- Count-Min Sketch and HyperLogLog;
* GPU-accelerated cardinality estimation with Numba CUDA;
* switching-activity / power analysis demos built on top of Count-Min Sketch;
* a GPU Mandelbrot renderer.

Each experiment lives in its own module and can be executed directly, e.g.::

    python -m big_data.switching_analysis_demo

Shared algorithms live under :mod:`big_data.sketches`; the demo modules build
self-contained experiments on top of them and stay runnable on their own.
"""

from importlib.metadata import PackageNotFoundError, version

from big_data.sketches import CountMinSketch

try:
    __version__ = version("big_data")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0"

__all__ = ["CountMinSketch", "__version__"]
