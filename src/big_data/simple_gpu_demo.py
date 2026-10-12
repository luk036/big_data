import numpy as np
from numba import cuda, uint64

from big_data.hll_hash import fmix64_device


@cuda.jit
def hll_kernel(data, registers, b):
    pos = cuda.grid(1)

    if pos < data.size:
        x = fmix64_device(uint64(data[pos]))

        m_bits = uint64(b)
        idx = x >> (uint64(64) - m_bits)

        # Count leading zeros on the remaining 64-b bits
        count = uint64(1)
        temp_w = x << m_bits
        for i in range(64 - b):
            if (temp_w & uint64(0x8000000000000000)) == 0:
                count += 1
                temp_w <<= 1
            else:
                break

        cuda.atomic.max(registers, idx, count)


def run_gpu_hll(data_array, b=12):
    m = 2**b
    # Initialize registers on the GPU
    d_registers = cuda.to_device(np.zeros(m, dtype=np.uint32))
    d_data = cuda.to_device(data_array)

    # Configure GPU threads (blocks and grids)
    threads_per_block = 256
    blocks_per_grid = (data_array.size + (threads_per_block - 1)) // threads_per_block

    # Launch the kernel
    hll_kernel[blocks_per_grid, threads_per_block](d_data, d_registers, b)

    # Copy registers back to CPU for final calculation
    h_registers = d_registers.copy_to_host()

    # Harmonic mean calculation (standard HLL)
    alpha_m = 0.7213 / (1 + 1.079 / m)
    estimate = alpha_m * (m**2) / sum(2.0 ** -float(r) for r in h_registers)
    return estimate


# --- Execution ---
if __name__ == "__main__":
    # Simulate 10 million unique items
    n_elements = 10_000_000
    print(f"Generating {n_elements} items...")
    data = np.arange(n_elements, dtype=np.uint64)

    print("Running GPU HyperLogLog...")
    import time

    start = time.time()
    result = run_gpu_hll(data, b=14)  # 16,384 buckets
    end = time.time()

    print(f"Estimated Unique Count: {int(result):,}")
    print(f"Actual Unique Count:    {n_elements:,}")
    print(f"Time Taken on GPU:      {end - start:.4f} seconds")
