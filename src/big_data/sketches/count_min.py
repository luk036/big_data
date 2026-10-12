"""Count-Min Sketch with bias-reducing estimators for EDA stream analytics.

This module hosts the single canonical implementation used by every demo in the
package.  It wraps the classic Count-Min Sketch (CMS) with three additions that
make it usable for distribution-tail and conservative analyses:

* ``total`` weight tracking, so the additive ``eps * N`` error bound is
  available and the Count-Mean-Min estimator can be computed;
* :meth:`CountMinSketch.query_mean_min` -- the Count-Mean-Min estimator, which
  subtracts the expected collision noise and is therefore (nearly) unbiased,
  instead of the strictly over-counting ``min`` estimator;
* an opt-in conservative-update mode for :meth:`CountMinSketch.update`, which
  only touches the row counters that currently hold the minimum estimate and
  thereby sharply reduces over-estimation on heavy hitters and tails.

The hash functions are deterministic functions of ``(width, depth)`` only, so
equal-sized sketches built in different processes produce identical counter
layouts and :meth:`CountMinSketch.merge` stays exact (associative and
commutative).
"""

import math
import random

import numpy as np

_HASH_MODULUS = 2147483647


class CountMinSketch:
    """Count-Min Sketch for switching-activity / event-frequency tracking."""

    def __init__(self, width: int = 1000, depth: int = 5, mode: str = "min"):
        """
        Initialize Count-Min Sketch

        Args:
            width: Number of counters in each hash array
            depth: Number of hash functions (rows)
            mode: Update strategy -- ``"min"`` (classic CMS) or
                ``"conservative"`` (conservative update; lower over-estimation)
        """
        if width <= 0 or depth <= 0:
            raise ValueError("width and depth must be positive")
        if mode not in ("min", "conservative"):
            raise ValueError("mode must be 'min' or 'conservative'")

        self.width = width
        self.depth = depth
        self.mode = mode
        self.sketch = np.zeros((depth, width), dtype=np.float32)

        # Total weight of every update seen: needed for the ``eps * N`` error
        # bound and for the Count-Mean-Min noise estimate.
        self.total = 0.0

        # Deterministic hash functions derived from (width, depth): reproducible
        # across runs, and shared by equal-sized sketches so merge() stays correct.
        rng = random.Random((width << 20) ^ depth)
        self.hash_params = [
            (rng.randint(1, 2**31), rng.randint(1, 2**31)) for _ in range(depth)
        ]

    def _hash(self, key: int, row: int) -> int:
        """Hash function for a given key and row"""
        a, b = self.hash_params[row]
        # Universal hash: (a*x + b) mod p mod width
        return ((a * key + b) % _HASH_MODULUS) % self.width

    def update(self, key: int, value: float = 1.0):
        """
        Update sketch with a value for a key

        Args:
            key: Node ID or hash of node name
            value: Switching activity value to add
        """
        self.total += value

        if self.mode == "conservative":
            # Increment only the row counters that currently hold the minimum;
            # this keeps estimates much closer to the true count.
            cols = [self._hash(key, d) for d in range(self.depth)]
            current_min = min(self.sketch[d][cols[d]] for d in range(self.depth))
            for d in range(self.depth):
                if self.sketch[d][cols[d]] == current_min:
                    self.sketch[d][cols[d]] += value
            return

        for d in range(self.depth):
            col = self._hash(key, d)
            self.sketch[d][col] += value

    def query(self, key: int) -> float:
        """
        Query estimated value for a key

        Returns:
            Minimum estimate (CMS property: always overestimates)
        """
        return float(min(self.sketch[d][self._hash(key, d)] for d in range(self.depth)))

    def query_mean_min(self, key: int) -> float:
        """
        Query the Count-Mean-Min estimate for a key.

        Each row counter is reduced by the expected collision noise
        ``(N - counter) / (width - 1)`` before the median across rows is taken,
        which removes the systematic over-counting of :meth:`query`.  The result
        may fall below the true count, but it is (nearly) unbiased.

        Returns:
            Bias-corrected estimate (never negative).
        """
        estimates = []
        for d in range(self.depth):
            counter = self.sketch[d][self._hash(key, d)]
            if self.width > 1:
                counter = counter - (self.total - counter) / (self.width - 1)
            estimates.append(max(0.0, counter))
        return float(np.median(estimates))

    def estimate_with_error(self, key: int) -> tuple:
        """
        Return ``(estimate, additive_error_bound)`` for a key.

        The bound is the standard CMS guarantee ``eps * N`` with
        ``eps = e / width`` (exact for the ``width = ceil(e/eps)``,
        ``depth = ceil(ln(1/delta))`` parameterization).
        """
        return self.query(key), self.additive_error_bound()

    def additive_error_bound(self) -> float:
        """Additive error bound ``e * N / width`` over the whole stream."""
        return math.e * self.total / self.width

    def merge(self, other: "CountMinSketch") -> "CountMinSketch":
        """Merge another sketch into this one (element-wise addition)"""
        if self.width != other.width or self.depth != other.depth:
            raise ValueError("Sketches must have same dimensions for merging")

        merged = CountMinSketch(self.width, self.depth, mode=self.mode)
        merged.sketch = self.sketch + other.sketch
        merged.total = self.total + other.total
        return merged

    def reset(self):
        """Reset all counters to zero"""
        self.sketch.fill(0)
        self.total = 0.0
