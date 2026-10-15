"""Switching-power analysis built on the shared Count-Min Sketch.

This is the one home of :class:`SwitchingPowerAnalyzer` and its demo.  The
end-to-end orchestrator :mod:`big_data.switching_analysis_demo` imports and runs
these alongside the other experiment demos.
"""

import time
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from big_data.sketches.count_min import CountMinSketch


class SwitchingPowerAnalyzer:
    """Complete switching power analysis using Count-Min Sketch"""

    def __init__(self, num_nodes: int = 10000, clock_freq: float = 1e9):
        """
        Initialize switching power analyzer

        Args:
            num_nodes: Estimated number of nodes in design
            clock_freq: Clock frequency in Hz
        """
        self.clock_freq = clock_freq
        self.total_cycles = 0

        # Initialize CMS with optimized dimensions
        # Based on error tolerance ε=0.001, δ=0.01
        self.epsilon = 0.001
        self.delta = 0.01
        self.cms = CountMinSketch.from_error(epsilon=self.epsilon, delta=self.delta)
        self.width = self.cms.width
        self.depth = self.cms.depth

        print(f"CMS Dimensions: width={self.width}, depth={self.depth}")
        print(f"Memory usage: {self.width * self.depth * 4 / 1024:.1f} KB")

        # Store node properties
        self.node_capacitances: Dict[int, float] = {}
        self.node_locations: Dict[int, Tuple[float, float]] = {}
        self.node_categories: Dict[int, str] = {}

        # For power calculation
        self.voltage = 1.0  # V
        self.activity_history: Dict[int, List[float]] = defaultdict(list)

    def add_node(
        self,
        node_id: int,
        capacitance: float = 1e-15,
        x: float = 0,
        y: float = 0,
        category: str = "logic",
    ):
        """Add a node with its properties"""
        self.node_capacitances[node_id] = capacitance
        self.node_locations[node_id] = (x, y)
        self.node_categories[node_id] = category

    def process_toggles(self, toggles: List[Tuple[int, int]]):
        """
        Process toggle events

        Args:
            toggles: List of (node_id, cycle) pairs
        """
        max_cycle = 0
        for node_id, cycle in toggles:
            self.cms.update(node_id)
            max_cycle = max(max_cycle, cycle)

        if toggles:
            self.total_cycles = max(self.total_cycles, max_cycle + 1)

    def estimate_activity_factor(self, node_id: int) -> float:
        """Estimate switching activity factor (α) for a node"""
        if self.total_cycles == 0:
            return 0.0

        toggle_count = self.cms.query(node_id)
        return toggle_count / self.total_cycles

    def estimate_power(self, node_id: int) -> float:
        """Estimate switching power for a node"""
        alpha = self.estimate_activity_factor(node_id)
        C = self.node_capacitances.get(node_id, 1e-15)

        # P = α * C * V² * f
        power = alpha * C * (self.voltage**2) * self.clock_freq
        return power

    def find_hot_nodes(self, top_n: int = 10) -> List[Tuple[int, float]]:
        """Find nodes with highest switching activity"""
        # In practice, you'd have a list of all nodes
        # For demo, we'll check a subset or use sampling

        hot_nodes = []
        # Check a representative sample of nodes
        sample_nodes = list(self.node_capacitances.keys())[:1000]

        for node_id in sample_nodes:
            power = self.estimate_power(node_id)
            hot_nodes.append((node_id, power))

        # Sort by power (descending)
        hot_nodes.sort(key=lambda x: x[1], reverse=True)
        return hot_nodes[:top_n]

    def analyze_spatial_distribution(self, grid_size: int = 10):
        """Analyze spatial distribution of switching activity"""
        spatial_grid = np.zeros((grid_size, grid_size))

        # Normalize coordinates to grid
        max_x = (
            max(x for x, _ in self.node_locations.values())
            if self.node_locations
            else 1
        )
        max_y = (
            max(y for _, y in self.node_locations.values())
            if self.node_locations
            else 1
        )

        for node_id, (x, y) in self.node_locations.items():
            grid_x = int(x / max_x * (grid_size - 1))
            grid_y = int(y / max_y * (grid_size - 1))

            activity = self.estimate_activity_factor(node_id)
            spatial_grid[grid_x, grid_y] += activity

        return spatial_grid

    def analyze_by_category(self):
        """Analyze switching activity by node category"""
        category_stats = defaultdict(lambda: {"total_power": 0, "node_count": 0})

        for node_id, category in self.node_categories.items():
            power = self.estimate_power(node_id)
            category_stats[category]["total_power"] += power
            category_stats[category]["node_count"] += 1

        # Calculate averages
        for category in category_stats:
            if category_stats[category]["node_count"] > 0:
                category_stats[category]["avg_power"] = (
                    category_stats[category]["total_power"]
                    / category_stats[category]["node_count"]
                )

        return dict(category_stats)


def demo_switching_power_analysis():
    """Demonstrate switching power analysis"""
    print("\n" + "=" * 60)
    print("DEMO 2: Switching Power Analysis with CMS")
    print("=" * 60)

    # Create analyzer
    analyzer = SwitchingPowerAnalyzer(num_nodes=10000, clock_freq=1e9)

    # Create synthetic design with different node types
    np.random.seed(42)

    print("\nCreating synthetic design...")
    num_nodes = 500
    toggle_events = []

    # Create nodes with different characteristics
    for i in range(num_nodes):
        node_id = 1000 + i

        # Assign properties based on node type
        if i < 50:  # Clock network
            capacitance = 5e-15  # Higher capacitance for clock buffers
            x, y = np.random.uniform(0, 100, 2)
            category = "clock"
            analyzer.add_node(node_id, capacitance, x, y, category)

            # High switching activity for clock nodes
            for cycle in range(0, 1000, 2):  # Toggles every cycle (α=0.5)
                toggle_events.append((node_id, cycle))

        elif i < 200:  # Data path logic
            capacitance = 2e-15
            x, y = np.random.uniform(0, 100, 2)
            category = "logic"
            analyzer.add_node(node_id, capacitance, x, y, category)

            # Moderate switching (α~0.1-0.3)
            toggle_prob = np.random.uniform(0.1, 0.3)
            for cycle in range(1000):
                if np.random.random() < toggle_prob:
                    toggle_events.append((node_id, cycle))

        else:  # Memory and control (low activity)
            capacitance = 1e-15
            x, y = np.random.uniform(0, 100, 2)
            category = "control" if i < 300 else "memory"
            analyzer.add_node(node_id, capacitance, x, y, category)

            # Low switching (α~0.01-0.05)
            toggle_prob = np.random.uniform(0.01, 0.05)
            for cycle in range(1000):
                if np.random.random() < toggle_prob:
                    toggle_events.append((node_id, cycle))

    print(f"Created {num_nodes} nodes")
    print(f"Generated {len(toggle_events):,} toggle events")

    # Process toggles
    print("\nProcessing toggle events...")
    start_time = time.time()
    analyzer.process_toggles(toggle_events)
    process_time = time.time() - start_time
    print(f"Processing time: {process_time:.3f} seconds")
    print(f"Total simulation cycles: {analyzer.total_cycles}")

    # Analyze results
    print("\n" + "-" * 40)
    print("POWER ANALYSIS RESULTS")
    print("-" * 40)

    # Find hot nodes
    hot_nodes = analyzer.find_hot_nodes(top_n=5)
    print("\nTop 5 Power-Hungry Nodes:")
    for rank, (node_id, power) in enumerate(hot_nodes, 1):
        activity = analyzer.estimate_activity_factor(node_id)
        cap = analyzer.node_capacitances[node_id]
        category = analyzer.node_categories[node_id]
        print(f"  {rank}. Node {node_id} ({category}):")
        print(
            f"     Power: {power * 1e6:.2f} μW, Activity: {activity:.3f}, Capacitance: {cap * 1e15:.1f} fF"
        )

    # Analyze by category
    print("\nPower Breakdown by Category:")
    category_stats = analyzer.analyze_by_category()
    for category, stats in category_stats.items():
        print(
            f"  {category:10s}: {stats['total_power'] * 1e6:6.2f} μW "
            f"({stats['node_count']:3d} nodes, avg: {stats.get('avg_power', 0) * 1e6:.2f} μW)"
        )

    # Spatial analysis
    spatial_grid = analyzer.analyze_spatial_distribution(grid_size=8)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot spatial distribution
    im = axes[0].imshow(spatial_grid, cmap="hot", interpolation="nearest")
    axes[0].set_title("Switching Activity Spatial Distribution")
    axes[0].set_xlabel("X Position")
    axes[0].set_ylabel("Y Position")
    plt.colorbar(im, ax=axes[0], label="Total Activity")

    # Plot power distribution by category
    categories = list(category_stats.keys())
    total_powers = [category_stats[c]["total_power"] * 1e6 for c in categories]

    bars = axes[1].bar(
        categories, total_powers, color=["red", "blue", "green", "orange"]
    )
    axes[1].set_title("Total Power by Node Category")
    axes[1].set_xlabel("Category")
    axes[1].set_ylabel("Power (μW)")

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig("switching_power_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()

    return analyzer


if __name__ == "__main__":
    demo_switching_power_analysis()
