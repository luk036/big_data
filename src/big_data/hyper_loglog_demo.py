import hashlib
import math
import random
from typing import Any

import matplotlib.pyplot as plt


class HyperLogLog:
    """
    A simple implementation of the HyperLogLog algorithm for cardinality estimation.
    """

    def __init__(self, p: int = 14):
        """
        Initialize HyperLogLog with 2^p registers.

        Args:
            p: Precision parameter, typically between 4 and 16.
               Higher p = more accuracy but more memory.
        """
        if p < 4 or p > 16:
            raise ValueError("p must be between 4 and 16")

        self.p = p
        self.m = 1 << p  # Number of registers: 2^p
        self.registers = [0] * self.m

        # Alpha constant for bias correction
        if self.m == 16:
            self.alpha = 0.673
        elif self.m == 32:
            self.alpha = 0.697
        elif self.m == 64:
            self.alpha = 0.709
        else:
            self.alpha = 0.7213 / (1 + 1.079 / self.m)

    def _hash(self, value: Any) -> int:
        """
        Hash the input value to a 64-bit integer.

        Args:
            value: Any hashable value.

        Returns:
            A 64-bit integer hash.
        """
        # Convert to string and hash using SHA256
        value_str = str(value).encode("utf-8")
        hash_bytes = hashlib.sha256(value_str).digest()

        # Take first 8 bytes (64 bits)
        return int.from_bytes(hash_bytes[:8], byteorder="big")

    def _count_leading_zeros(self, x: int, bits: int = 64) -> int:
        """
        Count leading zeros in binary representation.

        Args:
            x: Integer value.
            bits: Total bits to consider.

        Returns:
            Number of leading zeros + 1.
        """
        if x == 0:
            return bits + 1

        # Count leading zeros
        # We're looking at (bits - self.p) bits after the first p bits
        remaining_bits = bits - self.p
        mask = (1 << remaining_bits) - 1

        # Get the relevant bits after the first p bits
        relevant = x & mask

        if relevant == 0:
            return remaining_bits + 1

        # Count leading zeros in relevant bits
        return remaining_bits - relevant.bit_length() + 1

    def add(self, value: Any) -> None:
        """
        Add a value to the HyperLogLog sketch.

        Args:
            value: Value to add.
        """
        # Get hash
        x = self._hash(value)

        # Get register index from first p bits
        register_index = x >> (64 - self.p)

        # Count leading zeros in remaining bits
        leading_zeros = self._count_leading_zeros(x)

        # Update register if new count is larger
        if leading_zeros > self.registers[register_index]:
            self.registers[register_index] = leading_zeros

    def count(self) -> float:
        """
        Estimate the cardinality from the current sketch.

        Returns:
            Estimated cardinality.
        """
        # Calculate harmonic mean
        z = sum(2.0**-r for r in self.registers)

        # Raw estimate
        E = self.alpha * (self.m**2) / z

        # Apply corrections for small and large ranges
        if E <= (5 / 2) * self.m:
            # Small range correction
            V = self.registers.count(0)
            if V != 0:
                E = self.m * math.log(self.m / V)
        elif E > (1 / 30) * (2**64):
            # Large range correction
            E = -(2**64) * math.log(1 - E / (2**64))

        return E

    def merge(self, other: "HyperLogLog") -> None:
        """
        Merge another HyperLogLog into this one.

        Args:
            other: Another HyperLogLog with same precision p.
        """
        if self.p != other.p:
            raise ValueError("Cannot merge HyperLogLog with different precision")

        for i in range(self.m):
            if other.registers[i] > self.registers[i]:
                self.registers[i] = other.registers[i]


def test_hyperloglog() -> None:
    """
    Test and demonstrate the HyperLogLog algorithm.
    """
    print("=" * 60)
    print("HYPERLOGLOG DEMONSTRATION")
    print("=" * 60)

    # Test with different precision values
    precisions = [10, 12, 14]

    for p in precisions:
        print(f"\nTesting with p={p} (m={1 << p} registers)")
        print("-" * 40)

        # Create HyperLogLog instance
        hll = HyperLogLog(p=p)

        # Generate test data - numbers with some duplicates
        actual_count = 10000
        data = []

        # Add 70% unique values
        for i in range(int(actual_count * 0.7)):
            data.append(f"user_{i}")

        # Add 30% duplicates from first half
        for i in range(int(actual_count * 0.3)):
            data.append(f"user_{i % 3500}")

        random.shuffle(data)

        # Add all items to HLL
        for item in data:
            hll.add(item)

        # Get estimate
        estimated_count = hll.count()

        # Calculate error
        error = abs(estimated_count - actual_count) / actual_count * 100

        print(f"Actual distinct count: {actual_count}")
        print(f"Estimated count: {estimated_count:.2f}")
        print(f"Error: {error:.2f}%")
        print(
            f"Memory used: {len(hll.registers)} registers × 6 bits ≈ {len(hll.registers) * 6 / 8:.2f} bytes"
        )

    # Test merge operation
    print("\n" + "=" * 60)
    print("TESTING MERGE OPERATION")
    print("=" * 60)

    hll1 = HyperLogLog(p=12)
    hll2 = HyperLogLog(p=12)

    # Add different sets to each HLL
    set1 = [f"item_{i}" for i in range(5000)]
    set2 = [f"item_{i}" for i in range(5000, 10000)]  # No overlap

    for item in set1:
        hll1.add(item)

    for item in set2:
        hll2.add(item)

    # Merge hll2 into hll1
    hll1.merge(hll2)

    estimated_total = hll1.count()
    actual_total = 10000  # 5000 + 5000, no overlap

    error = abs(estimated_total - actual_total) / actual_total * 100

    print("Set 1 size: 5000")
    print("Set 2 size: 5000")
    print(f"Actual union size: {actual_total}")
    print(f"Estimated union size after merge: {estimated_total:.2f}")
    print(f"Error: {error:.2f}%")

    # Visualization of register distribution
    visualize_registers(hll1, "Register values after merge")


def visualize_registers(hll: HyperLogLog, title: str):
    """
    Visualize the distribution of register values.

    Args:
        hll: HyperLogLog instance.
        title: Plot title.
    """
    plt.figure(figsize=(12, 5))

    # Plot histogram of register values
    plt.subplot(1, 2, 1)
    plt.hist(
        hll.registers,
        bins=range(0, max(hll.registers) + 2),
        edgecolor="black",
        alpha=0.7,
    )
    plt.xlabel("Register Value (max leading zeros + 1)")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Register Values\n{title}")
    plt.grid(True, alpha=0.3)

    # Plot first 100 registers
    plt.subplot(1, 2, 2)
    plt.plot(hll.registers[:100], "b-", marker="o", markersize=3)
    plt.xlabel("Register Index")
    plt.ylabel("Value")
    plt.title("First 100 Register Values")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def accuracy_experiment():
    """
    Run an experiment to show accuracy vs. memory trade-off.
    """
    print("\n" + "=" * 60)
    print("ACCURACY VS MEMORY EXPERIMENT")
    print("=" * 60)

    # Test different precision values
    results = []

    for p in range(8, 17, 2):
        hll = HyperLogLog(p=p)

        # Generate 100,000 items with 50,000 unique
        actual_unique = 50000
        data = []

        # Add unique items
        for i in range(actual_unique):
            data.append(f"element_{i}")

        # Add some duplicates
        for i in range(actual_unique):
            data.append(f"element_{i % 25000}")

        random.shuffle(data)

        # Add to HLL
        for item in data[:100000]:  # Use first 100k
            hll.add(item)

        estimated = hll.count()
        error_pct = abs(estimated - actual_unique) / actual_unique * 100
        memory_bytes = hll.m * 6 / 8  # 6 bits per register

        results.append((p, hll.m, memory_bytes, error_pct))

        print(
            f"p={p:2d}, m={hll.m:5d}, memory={memory_bytes:6.1f} bytes, "
            f"error={error_pct:.2f}%"
        )

    # Plot results
    plt.figure(figsize=(10, 6))

    precisions = [r[0] for r in results]
    errors = [r[3] for r in results]
    memory = [r[2] for r in results]

    fig, ax1 = plt.subplots()

    color1 = "tab:blue"
    ax1.set_xlabel("Precision (p)")
    ax1.set_ylabel("Error (%)", color=color1)
    ax1.plot(precisions, errors, "o-", color=color1, linewidth=2)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("Memory (bytes)", color=color2)
    ax2.plot(precisions, memory, "s--", color=color2, linewidth=2)
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("HyperLogLog: Accuracy vs Memory Trade-off")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Run the demonstration
    test_hyperloglog()

    # Run accuracy experiment
    accuracy_experiment()

    # Simple interactive example
    print("\n" + "=" * 60)
    print("INTERACTIVE EXAMPLE")
    print("=" * 60)

    hll = HyperLogLog(p=12)

    print("Adding 1,000,000 random numbers (with many duplicates)...")

    # Generate 1M random numbers between 0 and 100,000
    # This means about 100,000 unique values expected
    for _ in range(1000000):
        num = random.randint(0, 100000)
        hll.add(num)

    estimated = hll.count()
    print(f"Estimated distinct numbers: {estimated:,.0f}")
    print("Expected distinct numbers: ~100,000")
    print(f"Memory used: ~{len(hll.registers) * 6 / 8:,.0f} bytes")
    print(f"That's {(len(hll.registers) * 6 / 8) / 1024:.2f} KB for 1M items!")
