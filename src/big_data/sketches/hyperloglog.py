"""Shared HyperLogLog core: precision constants, register maths and estimators.

Every HyperLogLog variant in this package shares the same precision constants,
register indexing, rank computation and cardinality estimator; only the register
storage differs (a NumPy array on the CPU, a CUDA device array on the GPU).
Centralising the maths here keeps the variants from drifting apart.
"""

import math
from typing import Any, Iterable

import numpy as np

from big_data.hll_hash import hash_value

_ALPHA_SMALL = {16: 0.673, 32: 0.697, 64: 0.709}


def alpha_for(m: int) -> float:
    return _ALPHA_SMALL.get(m, 0.7213 / (1 + 1.079 / m))


def register_index(h: int, p: int) -> int:
    return (h >> (64 - p)) & ((1 << p) - 1)


def rank(h: int, p: int) -> int:
    """Number of leading zeros in the low ``64 - p`` bits, plus one."""
    remaining_bits = 64 - p
    remaining = h & ((1 << remaining_bits) - 1)
    if remaining == 0:
        return remaining_bits + 1
    return remaining_bits - remaining.bit_length() + 1


def estimate(
    registers, p: int, *, small_range: bool = True, large_range: bool = True
) -> float:
    """Harmonic-mean cardinality estimate with the standard corrections."""
    registers = np.asarray(registers, dtype=np.float64)
    m = 1 << p
    z = np.sum(2.0**-registers)
    e = alpha_for(m) * m * m / z

    if small_range and e <= 2.5 * m:
        zeros = int(np.count_nonzero(registers == 0))
        if zeros:
            e = m * math.log(m / zeros)
    elif large_range and e > (1.0 / 30.0) * 2**64:
        e = -(2**64) * math.log(1.0 - e / 2**64)

    return float(e)


class HyperLogLogBase:
    """HyperLogLog over a NumPy register array.

    Subclasses only need to override :meth:`_hash` when they hash a different
    key representation; the update, merge and estimation logic is shared.
    """

    def __init__(self, p: int = 14):
        if p < 4 or p > 16:
            raise ValueError("p must be between 4 and 16")
        self.p = p
        self.m = 1 << p
        self.alpha = alpha_for(self.m)
        self.registers = np.zeros(self.m, dtype=np.int32)

    @classmethod
    def from_relative_error(cls, error: float) -> "HyperLogLogBase":
        """Size a sketch whose standard error is about ``error`` (e.g. ``0.02`` for 2%).

        ``p`` is capped at 16, so errors below about ``0.004`` are not achievable.
        """
        if not 0 < error < 1:
            raise ValueError("error must be in (0, 1)")
        m = math.ceil((1.04 / error) ** 2)
        p = max(4, min(16, math.ceil(math.log2(m))))
        return cls(p=p)

    def _hash(self, value: Any) -> int:
        return hash_value(value)

    def _update(self, h: int) -> None:
        idx = register_index(h, self.p)
        value = rank(h, self.p)
        if value > self.registers[idx]:
            self.registers[idx] = value

    def add(self, value: Any) -> None:
        self._update(self._hash(value))

    def add_hashes(self, hashes: Iterable[int]) -> None:
        for h in hashes:
            self._update(int(h))

    def add_batch(self, data: Iterable[Any]) -> None:
        for item in data:
            self.add(item)

    def count(self) -> float:
        return estimate(self.registers, self.p)

    def merge(self, other: "HyperLogLogBase") -> None:
        if self.p != other.p:
            raise ValueError("Cannot merge HyperLogLog with different precision")
        np.maximum(self.registers, other.registers, out=self.registers)
