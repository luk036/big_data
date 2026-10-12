"""
GPU-Accelerated HyperLogLog using Numba CUDA
No PyTorch dependency - pure Python with direct CUDA access
"""

import random
import time
from dataclasses import dataclass
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from numba import cuda

from big_data.hll_hash import fmix64_device, hash_array_u64, hash_strings, hash_value


@dataclass
class HLLConfig:
    """HyperLogLog configuration."""

    p: int = 12  # Precision bits (2^p registers)
    hash_bits: int = 64  # Output hash bits
    batch_size: int = 1000000  # Default batch size for GPU


@cuda.jit(device=True)
def device_count_leading_zeros(x, total_bits):
    """Count leading zeros on GPU device."""
    if x == 0:
        return total_bits + 1

    # Find position of most significant 1
    # GPU has __clz intrinsic but we implement manually for portability
    count = 0
    mask = 1 << (total_bits - 1)

    while mask > 0 and (x & mask) == 0:
        count += 1
        mask >>= 1

    return count + 1


@cuda.jit
def hyperloglog_kernel(hashes, registers, p, hash_bits):
    """
    CUDA kernel for HyperLogLog.
    Each thread processes one hash value.
    """
    # Thread index
    idx = cuda.grid(1)

    if idx < hashes.size:
        # Get hash value
        h = hashes[idx]

        # Extract register index from first p bits
        reg_idx = (h >> (hash_bits - p)) & ((1 << p) - 1)

        # Count leading zeros in remaining bits
        remaining_bits = hash_bits - p
        remaining = h & ((1 << remaining_bits) - 1)

        leading_zeros = device_count_leading_zeros(remaining, remaining_bits)

        # Atomic max to update register
        cuda.atomic.max(registers, reg_idx, leading_zeros)


@cuda.jit
def hyperloglog_key_kernel(keys, registers, p):
    """Hash uint64 keys on the device, then fold them into the registers."""
    idx = cuda.grid(1)
    if idx < keys.size:
        h = fmix64_device(keys[idx])

        reg_idx = (h >> (64 - p)) & ((1 << p) - 1)
        remaining_bits = 64 - p
        remaining = h & ((1 << remaining_bits) - 1)

        rank = device_count_leading_zeros(remaining, remaining_bits)

        cuda.atomic.max(registers, reg_idx, rank)


class GPUHyperLogLog:
    """GPU-accelerated HyperLogLog using Numba CUDA."""

    def __init__(self, config: Optional[HLLConfig] = None):
        self.config = config or HLLConfig()
        self.m = 1 << self.config.p  # Number of registers: 2^p

        # Bias correction constants
        if self.m == 16:
            self.alpha = 0.673
        elif self.m == 32:
            self.alpha = 0.697
        elif self.m == 64:
            self.alpha = 0.709
        else:
            self.alpha = 0.7213 / (1 + 1.079 / self.m)

        self._reset_registers()

        print("GPU HyperLogLog Initialized:")
        print(f"  Precision p={self.config.p}, registers m={self.m}")
        print(f"  Registers on GPU: {self.registers_gpu.size} elements")
        print(f"  Memory: {self.m * 4 / 1024:.2f} KB")

    def _reset_registers(self):
        """Reset all registers to zero on GPU."""
        self.registers_gpu = cuda.to_device(np.zeros(self.m, dtype=np.int32))

    def _hash_batch_cpu(self, data: List[Any]) -> np.ndarray:
        """Hash a batch of data on CPU (before sending to GPU)."""
        print(f"Hashing {len(data):,} items on CPU...")

        return hash_strings(data)

    def add_keys(self, keys: np.ndarray) -> None:
        """Add a batch of uint64 keys, hashing on the device."""
        if keys.size == 0:
            return
        d_keys = cuda.to_device(np.ascontiguousarray(keys, dtype=np.uint64))
        threads = 256
        blocks = (keys.size + threads - 1) // threads
        hyperloglog_key_kernel[blocks, threads](
            d_keys, self.registers_gpu, self.config.p
        )
        cuda.synchronize()

    def add_batch(self, data: List[Any]) -> None:
        """
        Add a batch of items using GPU acceleration.

        Args:
            data: List of items to add.
        """
        if not data:
            return

        batch_size = len(data)

        # Step 1: Hash on CPU (could also be done on GPU)
        hashes_cpu = self._hash_batch_cpu(data)

        # Step 2: Copy hashes to GPU
        hashes_gpu = cuda.to_device(hashes_cpu)

        # Step 3: Configure CUDA kernel
        threads_per_block = 256
        blocks_per_grid = (batch_size + threads_per_block - 1) // threads_per_block

        # Step 4: Launch kernel
        hyperloglog_kernel[blocks_per_grid, threads_per_block](
            hashes_gpu, self.registers_gpu, self.config.p, self.config.hash_bits
        )

        # Step 5: Wait for kernel to complete
        cuda.synchronize()

    def add_stream(self, data_stream, batch_size: Optional[int] = None):
        """
        Process a stream of data in batches.

        Args:
            data_stream: Iterable of data items.
            batch_size: Batch size for GPU processing.
        """
        if batch_size is None:
            batch_size = self.config.batch_size

        current_batch = []

        for item in data_stream:
            current_batch.append(item)

            if len(current_batch) >= batch_size:
                self.add_batch(current_batch)
                current_batch = []

        # Process remaining items
        if current_batch:
            self.add_batch(current_batch)

    def count(self) -> float:
        """
        Estimate cardinality from GPU registers.

        Returns:
            Estimated cardinality.
        """
        # Copy registers from GPU to CPU
        registers_cpu = self.registers_gpu.copy_to_host()

        # Calculate harmonic mean
        z = np.sum(2.0**-registers_cpu)

        # Raw estimate
        E = self.alpha * (self.m**2) / z

        # Apply small-range correction
        if E <= (5 / 2) * self.m:
            V = np.sum(registers_cpu == 0)
            if V != 0:
                E = self.m * np.log(self.m / V)

        # Apply large-range correction (for 64-bit hashes)
        elif E > (1 / 30) * (2**64):
            E = -(2**64) * np.log(1 - E / (2**64))

        return float(E)

    def merge(self, other: "GPUHyperLogLog") -> None:
        """
        Merge another HyperLogLog into this one.

        Args:
            other: Another GPUHyperLogLog with same configuration.
        """
        if self.m != other.m:
            raise ValueError("Cannot merge HyperLogLogs with different m")

        # Copy other's registers to CPU
        other_registers = other.registers_gpu.copy_to_host()

        # Copy our registers to CPU
        our_registers = self.registers_gpu.copy_to_host()

        # Element-wise max
        merged = np.maximum(our_registers, other_registers)

        # Copy back to GPU
        self.registers_gpu = cuda.to_device(merged)

    def get_register_stats(self):
        """Get statistics about register values."""
        registers_cpu = self.registers_gpu.copy_to_host()

        stats = {
            "min": np.min(registers_cpu),
            "max": np.max(registers_cpu),
            "mean": np.mean(registers_cpu),
            "std": np.std(registers_cpu),
            "zeros": np.sum(registers_cpu == 0),
            "non_zeros": np.sum(registers_cpu > 0),
        }

        return stats


class CPUHyperLogLog:
    """Reference CPU implementation for comparison."""

    def __init__(self, p: int = 12):
        self.p = p
        self.m = 1 << p
        self.registers = np.zeros(self.m, dtype=np.int32)

        if self.m == 16:
            self.alpha = 0.673
        elif self.m == 32:
            self.alpha = 0.697
        elif self.m == 64:
            self.alpha = 0.709
        else:
            self.alpha = 0.7213 / (1 + 1.079 / self.m)

    def _update(self, h: int) -> None:
        reg_idx = (h >> (64 - self.p)) & ((1 << self.p) - 1)
        remaining_bits = 64 - self.p
        remaining = h & ((1 << remaining_bits) - 1)

        if remaining == 0:
            leading_zeros = remaining_bits + 1
        else:
            count = 0
            mask = 1 << (remaining_bits - 1)
            while mask > 0 and (remaining & mask) == 0:
                count += 1
                mask >>= 1
            leading_zeros = count + 1

        if leading_zeros > self.registers[reg_idx]:
            self.registers[reg_idx] = leading_zeros

    def add(self, item: Any) -> None:
        """Add a single item."""
        self._update(hash_value(item))

    def add_hashes(self, hashes: np.ndarray) -> None:
        """Add pre-computed 64-bit hashes (skips hashing)."""
        for value in hashes:
            self._update(int(value))

    def add_batch(self, data: List[Any]) -> None:
        """Add a batch of items."""
        for item in data:
            self.add(item)

    def count(self) -> float:
        """Estimate cardinality."""
        z = np.sum(2.0**-self.registers)
        E = self.alpha * (self.m**2) / z

        if E <= (5 / 2) * self.m:
            V = np.sum(self.registers == 0)
            if V != 0:
                E = self.m * np.log(self.m / V)

        return float(E)


def _median(values: List[float]) -> float:
    return float(np.median(values)) if values else 0.0


def _time_cpu_hashes(hashes: np.ndarray, p: int, repeats: int):
    """Median CPU register-update time over pre-computed hashes."""
    times = []
    estimate = 0.0
    for _ in range(repeats):
        hll = CPUHyperLogLog(p=p)
        start = time.perf_counter()
        hll.add_hashes(hashes)
        times.append(time.perf_counter() - start)
        estimate = hll.count()
    return _median(times), estimate


def benchmark_comparison(repeats: int = 3):
    """Benchmark GPU vs CPU HyperLogLog end-to-end on uint64 keys.

    The GPU path hashes *on the device* (:func:`hyperloglog_key_kernel`), so the
    reported time is a true end-to-end measure rather than one that hides CPU
    string hashing; the CPU path uses vectorised hashing plus a Python register
    loop. The GPU time is split into transfer, kernel, and readout.
    """
    print("=" * 60)
    print("GPU vs CPU PERFORMANCE COMPARISON")
    print("=" * 60)

    if not cuda.is_available():
        print("CUDA not available! Running CPU-only benchmark.")
        return []

    total_items = 5_000_000
    unique_items = 500_000

    print(f"Generating {total_items:,} keys ({unique_items:,} unique)...")

    rng = np.random.default_rng(42)
    keys = rng.integers(0, unique_items, size=total_items, dtype=np.uint64)
    keys[:unique_items] = np.arange(unique_items, dtype=np.uint64)
    rng.shuffle(keys)

    p = 12
    threads = 256

    hash_start = time.perf_counter()
    hashes = hash_array_u64(keys)
    hash_time = time.perf_counter() - hash_start
    print(
        f"CPU hashing (vectorised): {hash_time:.3f}s "
        f"({total_items / hash_time / 1e6:.1f} M keys/sec)"
    )

    cpu_register_time, cpu_estimate = _time_cpu_hashes(hashes, p=p, repeats=repeats)
    cpu_total = hash_time + cpu_register_time

    batch_sizes = [1000, 10000, 100000, 500000, 1000000]
    results = []

    for batch_size in batch_sizes:
        print(f"\n{'=' * 40}")
        print(f"Batch size: {batch_size:,}")
        print(f"{'=' * 40}")

        gpu_hll = GPUHyperLogLog(HLLConfig(p=p))

        warmup = keys[:4096]
        gpu_hll._reset_registers()
        gpu_hll.add_keys(warmup)

        h2d_samples = []
        kernel_samples = []
        for _ in range(repeats):
            gpu_hll._reset_registers()
            cuda.synchronize()
            h2d = kernel = 0.0
            for i in range(0, keys.size, batch_size):
                batch = keys[i : i + batch_size]
                t0 = time.perf_counter()
                d_batch = cuda.to_device(batch)
                t1 = time.perf_counter()
                blocks = (batch.size + threads - 1) // threads
                hyperloglog_key_kernel[blocks, threads](
                    d_batch, gpu_hll.registers_gpu, p
                )
                cuda.synchronize()
                t2 = time.perf_counter()
                h2d += t1 - t0
                kernel += t2 - t1
            h2d_samples.append(h2d)
            kernel_samples.append(kernel)

        gpu_h2d = _median(h2d_samples)
        gpu_kernel = _median(kernel_samples)

        d2h_start = time.perf_counter()
        gpu_estimate = gpu_hll.count()
        gpu_d2h = time.perf_counter() - d2h_start

        gpu_total = gpu_h2d + gpu_kernel + gpu_d2h
        speedup = cpu_total / gpu_total if gpu_total > 0 else 0.0
        register_speedup = cpu_register_time / gpu_kernel if gpu_kernel > 0 else 0.0
        gpu_error = abs(gpu_estimate - unique_items) / unique_items * 100
        cpu_error = abs(cpu_estimate - unique_items) / unique_items * 100

        print(f"GPU Results (median of {repeats} runs):")
        print(f"  Host -> device:     {gpu_h2d:.4f}s")
        print(f"  Kernel (hash+fold): {gpu_kernel:.5f}s")
        print(f"  Device -> host:     {gpu_d2h:.5f}s")
        print(f"  End-to-end:         {gpu_total:.4f}s")
        print(f"  Estimate: {gpu_estimate:,.0f}  Error: {gpu_error:.2f}%")

        print("\nCPU Results:")
        print(f"  Hashing:            {hash_time:.3f}s")
        print(f"  Register updates:   {cpu_register_time:.3f}s")
        print(f"  Total:              {cpu_total:.3f}s")
        print(f"  Estimate: {cpu_estimate:,.0f}  Error: {cpu_error:.2f}%")

        print(f"\nSpeedup (end-to-end): {speedup:.1f}x")
        print(f"Speedup (register update vs kernel): {register_speedup:.1f}x")

        results.append(
            {
                "batch_size": batch_size,
                "hash_time": hash_time,
                "gpu_h2d": gpu_h2d,
                "gpu_kernel": gpu_kernel,
                "gpu_d2h": gpu_d2h,
                "gpu_time": gpu_total,
                "cpu_time": cpu_total,
                "speedup": speedup,
                "register_speedup": register_speedup,
                "gpu_estimate": gpu_estimate,
                "cpu_estimate": cpu_estimate,
                "gpu_error": gpu_error,
                "cpu_error": cpu_error,
            }
        )

    return results


def visualize_results(results):
    """Visualize benchmark results."""
    if not results:
        print("No results to visualize.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    batch_sizes = [r["batch_size"] for r in results]

    # 1. Speedup vs Batch Size
    ax = axes[0, 0]
    speedups = [r["speedup"] for r in results]
    ax.plot(batch_sizes, speedups, "bo-", linewidth=2, markersize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Speedup (CPU Time / GPU Time)")
    ax.set_title("GPU Speedup vs Batch Size")
    ax.grid(True, alpha=0.3)

    for i, (bs, sp) in enumerate(zip(batch_sizes, speedups)):
        ax.annotate(
            f"{sp:.1f}x",
            (bs, sp),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    # 2. Processing Time
    ax = axes[0, 1]
    gpu_times = [r["gpu_time"] for r in results]
    cpu_times = [r["cpu_time"] for r in results]

    width = 0.35
    x = np.arange(len(batch_sizes))

    ax.bar(x - width / 2, cpu_times, width, label="CPU", alpha=0.8, color="red")
    ax.bar(x + width / 2, gpu_times, width, label="GPU", alpha=0.8, color="blue")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{bs:,}" for bs in batch_sizes], rotation=45)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Processing Time Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 3. Accuracy Comparison
    ax = axes[1, 0]
    gpu_errors = [r["gpu_error"] for r in results]
    cpu_errors = [r["cpu_error"] for r in results]

    ax.plot(
        batch_sizes, gpu_errors, "ro-", linewidth=2, markersize=8, label="GPU Error"
    )
    ax.plot(
        batch_sizes, cpu_errors, "go-", linewidth=2, markersize=8, label="CPU Error"
    )

    ax.set_xscale("log")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Error (%)")
    ax.set_title("Accuracy Comparison (Lower is Better)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Throughput Comparison
    ax = axes[1, 1]
    total_items = 5_000_000  # From benchmark

    gpu_throughput = [total_items / r["gpu_time"] / 1e6 for r in results]
    cpu_throughput = [total_items / r["cpu_time"] / 1e6 for r in results]

    ax.plot(
        batch_sizes,
        gpu_throughput,
        "mo-",
        linewidth=2,
        markersize=8,
        label="GPU Throughput",
    )
    ax.plot(
        batch_sizes,
        cpu_throughput,
        "co-",
        linewidth=2,
        markersize=8,
        label="CPU Throughput",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput (Million items/sec)")
    ax.set_title("Processing Throughput (Higher is Better)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "HyperLogLog: GPU vs CPU Performance (Numba CUDA)",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def memory_efficiency_demo():
    """Demonstrate memory efficiency of HyperLogLog."""
    print("\n" + "=" * 60)
    print("MEMORY EFFICIENCY DEMONSTRATION")
    print("=" * 60)

    # Test different precision values
    precisions = [8, 10, 12, 14, 16]

    print("\nMemory usage vs accuracy trade-off:")
    print("-" * 50)
    print(
        f"{'Precision (p)':<12} {'Registers (m)':<15} {'Memory':<12} {'Std Error':<12}"
    )
    print("-" * 50)

    for p in precisions:
        m = 1 << p
        memory_kb = m * 4 / 1024  # 4 bytes per register
        std_error = 1.04 / np.sqrt(m)

        print(f"{p:<12} {m:<15,} {memory_kb:<12.2f} KB {std_error * 100:<12.2f}%")

    # Generate data for visualization
    x = np.array([1 << p for p in range(8, 17)])
    memory = x * 4 / 1024  # KB
    errors = 1.04 / np.sqrt(x) * 100  # Percentage

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Memory vs Precision
    ax1.semilogy(range(8, 17), memory, "bo-", linewidth=2, markersize=8)
    ax1.set_xlabel("Precision (p)")
    ax1.set_ylabel("Memory (KB) - Log Scale")
    ax1.set_title("Memory Usage vs Precision")
    ax1.grid(True, alpha=0.3)

    for i, (p, mem) in enumerate(zip(range(8, 17), memory)):
        ax1.annotate(
            f"{mem:.1f}KB",
            (p, mem),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=9,
        )

    # Error vs Memory
    ax2.plot(memory, errors, "ro-", linewidth=2, markersize=8)
    ax2.set_xscale("log")
    ax2.set_xlabel("Memory (KB) - Log Scale")
    ax2.set_ylabel("Standard Error (%)")
    ax2.set_title("Accuracy vs Memory Trade-off")
    ax2.grid(True, alpha=0.3)

    for i, (mem, err) in enumerate(zip(memory, errors)):
        if i % 2 == 0:  # Annotate every other point
            ax2.annotate(
                f"{err:.2f}%",
                (mem, err),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
            )

    plt.tight_layout()
    plt.show()


def streaming_demo():
    """Demonstrate streaming data processing."""
    print("\n" + "=" * 60)
    print("STREAMING DATA PROCESSING DEMONSTRATION")
    print("=" * 60)

    if not cuda.is_available():
        print("CUDA not available for streaming demo.")
        return

    # Simulate a data stream
    print("Simulating a data stream with changing cardinality...")

    hll = GPUHyperLogLog(HLLConfig(p=12, batch_size=100000))

    # Create a stream with three phases:
    # 1. Increasing unique items
    # 2. Plateau
    # 3. New unique items

    stream_size = 3_000_000
    batch_size = 10000

    print(f"\nProcessing {stream_size:,} items in {batch_size:,}-item batches...")

    estimates = []
    actual_counts = []
    batch_numbers = []

    unique_so_far = 0

    for batch_num in range(stream_size // batch_size):
        # Generate batch with changing characteristics
        if batch_num < 100:  # Phase 1: Increasing
            new_unique = 8000
            duplicates = 2000
        elif batch_num < 200:  # Phase 2: Plateau
            new_unique = 2000
            duplicates = 8000
        else:  # Phase 3: New increase
            new_unique = 6000
            duplicates = 4000

        # Generate batch
        batch = []

        # New unique items
        for i in range(new_unique):
            batch.append(f"item_{unique_so_far + i}")

        # Duplicates from previous items
        for i in range(duplicates):
            batch.append(f"item_{random.randint(0, unique_so_far)}")

        random.shuffle(batch)

        # Update actual count
        unique_so_far += new_unique

        # Process batch
        hll.add_batch(batch)

        # Estimate every 10 batches
        if batch_num % 10 == 0:
            estimate = hll.count()
            estimates.append(estimate)
            actual_counts.append(unique_so_far)
            batch_numbers.append(batch_num)

            if batch_num % 50 == 0:
                error = abs(estimate - unique_so_far) / unique_so_far * 100
                print(
                    f"  Batch {batch_num}: Actual={unique_so_far:,}, "
                    f"Estimate={estimate:,.0f}, Error={error:.2f}%"
                )

    # Plot streaming results
    plt.figure(figsize=(12, 6))

    plt.plot(batch_numbers, actual_counts, "g-", linewidth=3, label="Actual Count")
    plt.plot(batch_numbers, estimates, "b--", linewidth=2, label="HLL Estimate")

    plt.fill_between(
        batch_numbers,
        [a * 0.98 for a in actual_counts],  # 2% lower bound
        [a * 1.02 for a in actual_counts],  # 2% upper bound
        alpha=0.2,
        color="green",
        label="±2% Error Band",
    )

    plt.xlabel("Batch Number")
    plt.ylabel("Distinct Count")
    plt.title("Streaming Cardinality Estimation with HyperLogLog")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def interactive_example():
    """Interactive example of GPU HyperLogLog."""
    print("\n" + "=" * 60)
    print("INTERACTIVE GPU HYPERLOGLOG EXAMPLE")
    print("=" * 60)

    if not cuda.is_available():
        print("CUDA is not available. Using CPU-only mode.")
        config = HLLConfig(p=10)
    else:
        config = HLLConfig(p=12)

    hll = GPUHyperLogLog(config)

    # Example 1: Count distinct words
    print("\nExample 1: Counting distinct words in text")
    print("-" * 40)

    text = """
    HyperLogLog is a probabilistic data structure used to estimate
    the cardinality of a set. It is particularly useful when dealing
    with very large datasets where storing all unique elements is
    impractical due to memory constraints. The algorithm provides
    an approximate count with a small, fixed memory footprint.
    """

    words = text.lower().split()
    words = [word.strip(".,!?;:()[]{}\"'") for word in words]

    print(f"Text has {len(words):,} total words")

    hll.add_batch(words)
    estimate = hll.count()

    actual = len(set(words))
    print(f"Actual unique words: {actual}")
    print(f"HyperLogLog estimate: {estimate:.0f}")
    print(f"Error: {abs(estimate - actual) / actual * 100:.2f}%")

    # Example 2: Simulated network traffic
    print("\nExample 2: Simulated network IP counting")
    print("-" * 40)

    hll._reset_registers()  # Reset for new counting

    # Generate random IP addresses
    num_packets = 1_000_000
    unique_ips = 10000

    print(f"Generating {num_packets:,} packets from ~{unique_ips:,} unique IPs...")

    packets = []
    for i in range(num_packets):
        # Most packets from first 5000 IPs, some from rest
        if i < 800000:
            ip_num = random.randint(1, 5000)
        else:
            ip_num = random.randint(5001, unique_ips)

        ip = f"192.168.1.{ip_num % 256}.{ip_num // 256}"
        packets.append(ip)

    random.shuffle(packets)

    # Process in batches
    batch_size = 100000
    start_time = time.time()

    for i in range(0, len(packets), batch_size):
        batch = packets[i : i + batch_size]
        hll.add_batch(batch)

    if cuda.is_available():
        cuda.synchronize()

    elapsed = time.time() - start_time
    estimate = hll.count()

    print(f"Processing time: {elapsed:.3f} seconds")
    print(f"Throughput: {num_packets / elapsed / 1e6:.2f} M packets/sec")
    print(f"Estimated unique IPs: {estimate:.0f}")
    print(f"Actual unique IPs: {unique_ips}")
    print(f"Error: {abs(estimate - unique_ips) / unique_ips * 100:.2f}%")

    # Show register statistics
    stats = hll.get_register_stats()
    print("\nRegister Statistics:")
    print(f"  Min value: {stats['min']}")
    print(f"  Max value: {stats['max']}")
    print(f"  Mean value: {stats['mean']:.2f}")
    print(f"  Zero registers: {stats['zeros']:,} ({stats['zeros'] / hll.m * 100:.1f}%)")


if __name__ == "__main__":
    print("=" * 70)
    print("GPU-ACCELERATED HYPERLOGLOG DEMO (No PyTorch)")
    print("=" * 70)

    # Check CUDA availability
    if cuda.is_available():
        print("✅ CUDA is available!")
        gpu_info = cuda.get_current_device()
        print(
            f"GPU Device: {gpu_info.name.decode() if hasattr(gpu_info.name, 'decode') else gpu_info.name}"
        )
    else:
        print("⚠️  CUDA is not available. Running in CPU-only mode.")
        print("   Install CUDA toolkit and NVIDIA drivers for GPU acceleration.")

    # Run demos
    try:
        # 1. Memory efficiency demo
        memory_efficiency_demo()

        # 2. Interactive example
        interactive_example()

        # 3. Benchmark comparison (if CUDA available)
        if cuda.is_available():
            results = benchmark_comparison()
            visualize_results(results)

        # 4. Streaming demo
        streaming_demo()

    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)

    # Show installation instructions
    print("\nInstallation requirements for this demo:")
    print("  pip install numba numpy matplotlib")
    print("\nFor CUDA support:")
    print("  1. Install NVIDIA CUDA Toolkit (11.0+)")
    print("  2. Ensure numba can find CUDA:")
    print(
        "     python -c \"from numba import cuda; print('CUDA:', cuda.is_available())\""
    )
