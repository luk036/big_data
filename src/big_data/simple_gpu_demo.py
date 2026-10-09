import numpy as np
from numba import cuda, uint64


# 1. GPU Kernel: This runs in parallel on thousands of GPU cores
@cuda.jit
def hll_kernel(data, registers, b):
    # Calculate the unique thread ID
    pos = cuda.grid(1)

    if pos < data.size:
        # A simple, fast Xorshift-like hash for demonstration
        # In production, use a more robust GPU-compatible MurmurHash
        x = uint64(data[pos])
        x ^= x << 13
        x ^= x >> 7
        x ^= x << 17

        # Use first b bits for bucket index
        m = 1 << b
        idx = x % m

        # Use remaining bits to find leading zeros
        # We simulate this by looking at the bit pattern
        w = x >> b
        if w == 0:
            count = 64 - b
        else:
            # Count leading zeros (manual implementation for GPU)
            count = 1
            while (w & 1) == 0 and count < (64 - b):
                count += 1
                w >>= 1

        # ATOMIC MAX: Critical for GPU.
        # Ensures multiple threads don't overwrite each other.
        cuda.atomic.max(registers, idx, count)


def run_gpu_hll(data_array, b=12):
    m = 2**b
    # Initialize registers on the GPU
    d_registers = cuda.to_device(np.zeros(m, dtype=np.int32))
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
    estimate = alpha_m * (m**2) / sum(2.0**-r for r in h_registers)
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
