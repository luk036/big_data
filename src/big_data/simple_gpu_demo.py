import time

import numpy as np

from big_data.gpu_hll import run_gpu_hll


if __name__ == "__main__":
    n_elements = 10_000_000
    print(f"Generating {n_elements} items...")
    data = np.arange(n_elements, dtype=np.uint64)

    print("Running GPU HyperLogLog...")
    start = time.time()
    result = run_gpu_hll(data, b=14)
    end = time.time()

    print(f"Estimated Unique Count: {int(result):,}")
    print(f"Actual Unique Count:    {n_elements:,}")
    print(f"Time Taken on GPU:      {end - start:.4f} seconds")
