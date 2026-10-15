import numpy as np

from big_data.gpu_hll import run_gpu_hll


def run_tuned_gpu_hll(data_array, b: int = 14) -> float:
    return run_gpu_hll(data_array, b, correct=True)


if __name__ == "__main__":
    n_elements = 10_000_000
    data = np.arange(n_elements, dtype=np.uint64)

    b_val = 14
    estimate = run_tuned_gpu_hll(data, b=b_val)

    error = abs(estimate - n_elements) / n_elements * 100
    print(f"Buckets: {2**b_val}")
    print(f"Estimate: {int(estimate):,}")
    print(f"Actual:   {n_elements:,}")
    print(f"Error:    {error:.2f}%")
