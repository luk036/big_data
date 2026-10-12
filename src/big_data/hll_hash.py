"""Single 64-bit hash used by every HyperLogLog implementation.

Before this module each HLL used a different hash (SHA-256, a full MurmurHash3,
a weak xorshift, and an inline Murmur finalizer), so the "reference" CPU and GPU
estimators disagreed on the same input.  Everything now reduces to a 64-bit
integer and runs it through :func:`fmix64`, the MurmurHash3 64-bit finalizer,
which exists in three interchangeable forms:

* :func:`fmix64`            -- pure Python (arbitrary precision ints)
* :func:`hash_array_u64`    -- vectorised NumPy (CPU baselines, batch hashing)
* :func:`fmix64_device`     -- ``@cuda.jit(device=True)`` mirror for kernels

:func:`hash_value` hashes an arbitrary hashable object by folding *all* of its
UTF-8 bytes with FNV-1a and then finalising with :func:`fmix64`; folding all
bytes (rather than the first eight) is required so that keys sharing a prefix
still differ.
"""

import numpy as np
from numba import cuda, uint64

MASK64 = (1 << 64) - 1

_FNV_PRIME = 0x100000001B3
_MIX_C1 = 0xFF51AFD7ED558CCD
_MIX_C2 = 0xC4CEB9FE1A85EC53


def fmix64(x: int) -> int:
    """MurmurHash3 64-bit finalizer on a Python integer."""
    x &= MASK64
    x ^= x >> 33
    x = (x * _MIX_C1) & MASK64
    x ^= x >> 33
    x = (x * _MIX_C2) & MASK64
    x ^= x >> 33
    return x


def hash_bytes(data: bytes) -> int:
    """FNV-1a fold over every byte, finalised with :func:`fmix64`."""
    h = 0
    for byte in data:
        h = ((h ^ byte) * _FNV_PRIME) & MASK64
    return fmix64(h ^ (len(data) << 32))


def hash_value(value) -> int:
    """Hash any hashable value to a 64-bit integer, deterministically."""
    return hash_bytes(str(value).encode("utf-8"))


def hash_array_u64(values) -> np.ndarray:
    """Vectorised :func:`fmix64` over a NumPy array of integers."""
    x = np.asarray(values, dtype=np.uint64).copy()
    x ^= x >> np.uint64(33)
    x *= np.uint64(_MIX_C1)
    x ^= x >> np.uint64(33)
    x *= np.uint64(_MIX_C2)
    x ^= x >> np.uint64(33)
    return x


def hash_strings(items) -> np.ndarray:
    """Vectorised :func:`hash_value` over a batch of items.

    Folds every item's UTF-8 bytes column-by-column in NumPy and returns the
    same 64-bit hashes as ``[hash_value(x) for x in items]`` -- replacing the
    per-item Python loop that otherwise dominates batch hashing.
    """
    encoded = [str(item).encode("utf-8") for item in items]
    count = len(encoded)
    if count == 0:
        return np.empty(0, dtype=np.uint64)

    lengths = np.fromiter((len(e) for e in encoded), dtype=np.int64, count=count)
    flat = np.frombuffer(b"".join(encoded), dtype=np.uint8)
    starts = np.zeros(count, dtype=np.int64)
    np.cumsum(lengths[:-1], out=starts[1:])

    h = np.zeros(count, dtype=np.uint64)
    prime = np.uint64(_FNV_PRIME)
    for column in range(int(lengths.max())):
        active = np.nonzero(lengths > column)[0]
        h[active] = (
            h[active] ^ flat[starts[active] + column].astype(np.uint64)
        ) * prime

    h ^= lengths.astype(np.uint64) << np.uint64(32)
    return hash_array_u64(h)


@cuda.jit(device=True)
def fmix64_device(x):
    """CUDA mirror of :func:`fmix64` for ``uint64`` values."""
    x ^= x >> uint64(33)
    x *= uint64(_MIX_C1)
    x ^= x >> uint64(33)
    x *= uint64(_MIX_C2)
    x ^= x >> uint64(33)
    return x
