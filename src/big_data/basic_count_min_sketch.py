import random

import numpy as np


class CountMinSketch:
    """Basic Count-Min Sketch implementation for switching activity tracking"""

    def __init__(self, width: int = 1000, depth: int = 5):
        """
        Initialize Count-Min Sketch

        Args:
            width: Number of counters in each hash array
            depth: Number of hash functions (rows)
        """
        self.width = width
        self.depth = depth
        self.sketch = np.zeros((depth, width), dtype=np.float32)

        # Deterministic hash functions derived from (width, depth): reproducible
        # across runs, and shared by equal-sized sketches so merge() stays correct.
        rng = random.Random((width << 20) ^ depth)
        self.hash_params = [
            (rng.randint(1, 2**31), rng.randint(1, 2**31)) for _ in range(depth)
        ]

    def _hash(self, key: int, row: int) -> int:
        """Hash function for a given key and row"""
        a, b = self.hash_params[row]
        # Universal hash: (a*x + b) mod p mod width
        return ((a * key + b) % 2147483647) % self.width

    def update(self, key: int, value: float = 1.0):
        """
        Update sketch with a value for a key

        Args:
            key: Node ID or hash of node name
            value: Switching activity value to add
        """
        for d in range(self.depth):
            col = self._hash(key, d)
            self.sketch[d][col] += value

    def query(self, key: int) -> float:
        """
        Query estimated value for a key

        Returns:
            Minimum estimate (CMS property: always overestimates)
        """
        estimates = []
        for d in range(self.depth):
            col = self._hash(key, d)
            estimates.append(self.sketch[d][col])

        return min(estimates)

    def merge(self, other: "CountMinSketch") -> "CountMinSketch":
        """Merge another sketch into this one (element-wise addition)"""
        if self.width != other.width or self.depth != other.depth:
            raise ValueError("Sketches must have same dimensions for merging")

        merged = CountMinSketch(self.width, self.depth)
        merged.sketch = self.sketch + other.sketch
        return merged

    def reset(self):
        """Reset all counters to zero"""
        self.sketch.fill(0)


# Demo 1: Basic CMS functionality
def demo_basic_cms():
    """Demonstrate basic CMS functionality for switching counting"""
    print("=" * 60)
    print("DEMO 1: Basic Count-Min Sketch for Switching Activity")
    print("=" * 60)

    # Create CMS with small dimensions for demo
    cms = CountMinSketch(width=50, depth=3)

    # Simulate toggles for different nodes
    nodes = {
        "clk_buffer_1": 1001,
        "clk_buffer_2": 1002,
        "data_reg_1": 2001,
        "data_reg_2": 2002,
        "alu_out": 3001,
        "memory_cell": 4001,
    }

    # Simulate toggle events
    print("\nSimulating toggle events...")
    toggle_events = []

    # Node 1001 toggles frequently (high activity)
    for _ in range(150):
        cms.update(nodes["clk_buffer_1"])
        toggle_events.append(("clk_buffer_1", nodes["clk_buffer_1"]))

    # Node 2001 toggles moderately
    for _ in range(75):
        cms.update(nodes["data_reg_1"])
        toggle_events.append(("data_reg_1", nodes["data_reg_1"]))

    # Node 4001 toggles rarely (low activity)
    for _ in range(10):
        cms.update(nodes["memory_cell"])
        toggle_events.append(("memory_cell", nodes["memory_cell"]))

    print(f"Total toggle events simulated: {len(toggle_events)}")

    # Query estimates
    print("\nEstimated toggle counts:")
    for name, node_id in nodes.items():
        estimate = cms.query(node_id)
        print(f"  {name:15s} (ID: {node_id}): {estimate:.1f}")

    # Show sketch visualization
    print("\nSketch visualization (first 10 counters of each row):")
    for d in range(cms.depth):
        row_preview = cms.sketch[d, :10]
        print(f"  Row {d}: {row_preview}")

    return cms, nodes


if __name__ == "__main__":
    # Run demo 1
    cms_basic, node_ids = demo_basic_cms()
