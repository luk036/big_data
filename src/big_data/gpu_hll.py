"""Shared CUDA HyperLogLog kernel for the simple and tuned GPU demos.

The minimal (:mod:`big_data.simple_gpu_demo`) and tuned
(:mod:`big_data.better_gpu_demo`) demos differ only in whether the small-range
correction is applied; they share this kernel and estimator.
"""

import numpy as np
from numba import cuda, uint64

from big_data.hll_hash import fmix64_device
from big_data.sketches.hyperloglog import estimate


@cuda.jit
def hll_kernel(data, registers, b):
    pos = cuda.grid(1)

    if pos < data.size:
        x = fmix64_device(uint64(data[pos]))

        m_bits = uint64(b)
        idx = x >> (uint64(64) - m_bits)

        count = uint64(1)
        temp_w = x << m_bits
        for _ in range(64 - b):
            if (temp_w & uint64(0x8000000000000000)) == 0:
                count += 1
                temp_w <<= 1
            else:
                break

        cuda.atomic.max(registers, idx, count)


def run_gpu_hll(data_array, b: int = 12, *, correct: bool = False) -> float:
    m = 2**b
    d_registers = cuda.to_device(np.zeros(m, dtype=np.uint32))
    d_data = cuda.to_device(data_array)

    threads_per_block = 256
    blocks_per_grid = (data_array.size + (threads_per_block - 1)) // threads_per_block

    hll_kernel[blocks_per_grid, threads_per_block](d_data, d_registers, b)

    registers = d_registers.copy_to_host()
    return estimate(registers, b, small_range=correct, large_range=False)
