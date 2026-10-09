import math

import numpy as np
from numba import cuda, uint64


# A more robust 64-bit hash (MurmurHash3-style mixer)
@cuda.jit(device=True)
def murmur_mixer(key):
    key ^= key >> 33
    key *= uint64(0xFF51AFD7ED558CCD)
    key ^= key >> 33
    key *= uint64(0xC4CEB9FE1A85EC53)
    key ^= key >> 33
    return key


@cuda.jit
def tuned_hll_kernel(data, registers, b):
    pos = cuda.grid(1)
    if pos < data.size:
        # 1. Robust Hashing
        x = murmur_mixer(uint64(data[pos]))

        # 2. Bucket Index
        m_bits = uint64(b)
        idx = x >> (uint64(64) - m_bits)

        # 3. Leading Zeros on the remaining 64-b bits
        (x << m_bits) | (uint64(1) << (m_bits - uint64(1)))  # Ensure non-zero

        # Count leading zeros using a fast bit-counting approach
        count = uint64(1)
        # Check bits from left to right
        temp_w = x << m_bits
        for i in range(64 - b):
            if (temp_w & uint64(0x8000000000000000)) == 0:
                count += 1
                temp_w <<= 1
            else:
                break

        cuda.atomic.max(registers, idx, count)


def run_tuned_gpu_hll(data_array, b=14):
    m = 2**b
    d_registers = cuda.to_device(np.zeros(m, dtype=np.uint32))
    d_data = cuda.to_device(data_array)

    threads_per_block = 256
    blocks_per_grid = (data_array.size + (threads_per_block - 1)) // threads_per_block

    tuned_hll_kernel[blocks_per_grid, threads_per_block](d_data, d_registers, b)

    h_registers = d_registers.copy_to_host()

    # --- Advanced Estimation Logic ---
    alpha_m = 0.7213 / (1 + 1.079 / m)
    E = alpha_m * (m**2) / sum(2.0 ** -float(r) for r in h_registers)

    # Small range correction (Linear Counting)
    if E <= 2.5 * m:
        v = np.count_nonzero(h_registers == 0)
        if v != 0:
            E = m * math.log(m / v)

    return E


# Test with 10 million items
if __name__ == "__main__":
    n_elements = 10_000_000
    data = np.arange(n_elements, dtype=np.uint64)

    # Using b=14 gives 16,384 buckets (approx 1% error)
    # Using b=16 gives 65,536 buckets (approx 0.4% error)
    b_val = 14
    estimate = run_tuned_gpu_hll(data, b=b_val)

    error = abs(estimate - n_elements) / n_elements * 100
    print(f"Buckets: {2**b_val}")
    print(f"Estimate: {int(estimate):,}")
    print(f"Actual:   {n_elements:,}")
    print(f"Error:    {error:.2f}%")
