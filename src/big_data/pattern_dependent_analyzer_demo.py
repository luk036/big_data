import random
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
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

        # Initialize hash function parameters
        self.hash_params = []
        for i in range(depth):
            # Use different random seeds for each hash function
            a = random.randint(1, 2**31)
            b = random.randint(1, 2**31)
            self.hash_params.append((a, b))

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
        merged.sketch = self.sketch + other.sketch  # type: ignore[assignment]
        return merged

    def reset(self):
        """Reset all counters to zero"""
        self.sketch.fill(0)


class PatternDependentAnalyzer:
    """Analyze pattern-dependent switching activity"""

    def __init__(self, num_patterns: int = 5):
        """
        Initialize pattern-dependent analyzer

        Args:
            num_patterns: Number of input patterns to analyze
        """
        self.num_patterns = num_patterns
        self.pattern_sketches = [
            CountMinSketch(width=800, depth=4) for _ in range(num_patterns)
        ]
        self.pattern_names = [f"Pattern_{i}" for i in range(num_patterns)]

        # Statistics per pattern
        self.pattern_stats = [
            {"total_cycles": 0, "total_toggles": 0} for _ in range(num_patterns)
        ]

        # Node information
        self.node_info: Dict[int, Dict[str, Any]] = {}

    def simulate_pattern(self, pattern_id: int, toggles: List[Tuple[int, int]]):
        """Simulate switching for a specific pattern"""
        sketch = self.pattern_sketches[pattern_id]
        max_cycle = 0

        for node_id, cycle in toggles:
            sketch.update(node_id)
            max_cycle = max(max_cycle, cycle)

            # Update node info
            if node_id not in self.node_info:
                self.node_info[node_id] = {"name": f"Node_{node_id}"}

        # Update pattern statistics
        self.pattern_stats[pattern_id]["total_cycles"] = max(
            self.pattern_stats[pattern_id]["total_cycles"], max_cycle + 1
        )
        self.pattern_stats[pattern_id]["total_toggles"] += len(toggles)

    def analyze_pattern_sensitivity(self, node_id: int) -> Dict:
        """Analyze how sensitive a node is to different patterns"""
        activities = []

        for pattern_id in range(self.num_patterns):
            sketch = self.pattern_sketches[pattern_id]
            cycles = self.pattern_stats[pattern_id]["total_cycles"]

            if cycles > 0:
                toggle_count = sketch.query(node_id)
                activity = toggle_count / cycles
                activities.append(activity)
            else:
                activities.append(0)

        if activities:
            sensitivity_metrics = {
                "min_activity": min(activities),
                "max_activity": max(activities),
                "avg_activity": sum(activities) / len(activities),
                "range": max(activities) - min(activities),
                "std_activity": np.std(activities),
                "pattern_activities": activities,
            }

            # Calculate sensitivity index
            if sensitivity_metrics["avg_activity"] > 0:  # type: ignore[operator]
                sensitivity_metrics["sensitivity_index"] = (
                    sensitivity_metrics["range"] / sensitivity_metrics["avg_activity"]  # type: ignore[operator]
                )
            else:
                sensitivity_metrics["sensitivity_index"] = 0

            # Find worst-case pattern
            worst_pattern = np.argmax(activities)
            sensitivity_metrics["worst_pattern"] = worst_pattern
            sensitivity_metrics["worst_pattern_name"] = self.pattern_names[
                worst_pattern
            ]

            return sensitivity_metrics
        else:
            return {}

    def find_worst_case_nodes(self, top_n: int = 10) -> List[Tuple[int, float]]:
        """Find nodes with highest pattern sensitivity"""
        sensitive_nodes = []

        # Check all nodes
        for node_id in list(self.node_info.keys())[:200]:  # Limit for demo
            sensitivity = self.analyze_pattern_sensitivity(node_id)
            if sensitivity and "sensitivity_index" in sensitivity:
                sensitive_nodes.append((node_id, sensitivity["sensitivity_index"]))

        # Sort by sensitivity (descending)
        sensitive_nodes.sort(key=lambda x: x[1], reverse=True)
        return sensitive_nodes[:top_n]

    def analyze_pattern_correlation(self) -> np.ndarray:
        """Analyze correlation between patterns"""
        correlation_matrix = np.zeros((self.num_patterns, self.num_patterns))

        # For each node, check if it toggles in both patterns
        sample_nodes = list(self.node_info.keys())[:100]  # Sample for efficiency

        for node_id in sample_nodes:
            pattern_activities = []

            for pattern_id in range(self.num_patterns):
                sketch = self.pattern_sketches[pattern_id]
                cycles = self.pattern_stats[pattern_id]["total_cycles"]

                if cycles > 0:
                    activity = sketch.query(node_id) / cycles
                    pattern_activities.append(activity > 0.1)  # Threshold
                else:
                    pattern_activities.append(False)

            # Update correlation matrix
            for i in range(self.num_patterns):
                for j in range(self.num_patterns):
                    if pattern_activities[i] and pattern_activities[j]:
                        correlation_matrix[i, j] += 1

        # Normalize
        if len(sample_nodes) > 0:
            correlation_matrix = correlation_matrix / len(sample_nodes)  # type: ignore[assignment]

        return correlation_matrix


class MultiCornerAnalyzer:
    """Analyze switching across different PVT corners"""

    def __init__(self, corners: List[str]):
        """
        Initialize multi-corner analyzer

        Args:
            corners: List of corner names (e.g., ['TT_25C', 'FF_125C', 'SS_-40C'])
        """
        self.corners = corners
        self.num_corners = len(corners)

        # Create CMS for each corner
        self.corner_sketches = {
            corner: CountMinSketch(width=800, depth=4) for corner in corners
        }

        # Corner-specific parameters
        self.corner_params = {
            "TT_25C": {"voltage": 1.0, "temp": 25, "process": "typical"},
            "FF_125C": {"voltage": 1.1, "temp": 125, "process": "fast"},
            "SS_-40C": {"voltage": 0.9, "temp": -40, "process": "slow"},
        }

    def simulate_corner(
        self,
        corner_name: str,
        toggles: List[Tuple[int, int]],
        scaling_factor: float = 1.0,
    ):
        """Simulate switching for a specific corner"""
        sketch = self.corner_sketches[corner_name]

        for node_id, cycle in toggles:
            # Apply corner-specific scaling
            scaled_value = scaling_factor
            sketch.update(node_id, scaled_value)

    def analyze_corner_impact(self, node_id: int) -> Dict:
        """Analyze switching across different corners"""
        corner_results = {}

        for corner in self.corners:
            sketch = self.corner_sketches[corner]
            toggle_count = sketch.query(node_id)

            # Get corner parameters
            params = self.corner_params.get(corner, {"voltage": 1.0})
            voltage = float(params.get("voltage", 1.0))  # type: ignore[arg-type]

            # Estimate power (simplified)
            # Assuming capacitance = 1e-15F, frequency = 1GHz
            power = toggle_count * 1e-15 * (voltage**2) * 1e9

            corner_results[corner] = {
                "toggle_count": toggle_count,
                "voltage": voltage,
                "estimated_power": power,
            }

        # Calculate variations
        if corner_results:
            max_power = max(r["estimated_power"] for r in corner_results.values())
            min_power = min(r["estimated_power"] for r in corner_results.values())
            avg_power = sum(
                r["estimated_power"] for r in corner_results.values()
            ) / len(corner_results)

            return {
                "corner_results": corner_results,
                "max_power": max_power,
                "min_power": min_power,
                "avg_power": avg_power,
                "power_variation": (max_power - min_power) / avg_power
                if avg_power > 0
                else 0,
                "worst_case_corner": max(
                    corner_results.items(), key=lambda x: x[1]["estimated_power"]
                )[0],
            }

        return {}


def demo_pattern_corner_analysis():
    """Demonstrate pattern and corner analysis"""
    print("\n" + "=" * 60)
    print("DEMO 4: Pattern Dependency and Multi-Corner Analysis")
    print("=" * 60)

    # Create pattern analyzer
    pattern_analyzer = PatternDependentAnalyzer(num_patterns=4)
    pattern_names = ["Idle", "Reset", "Max_Load", "Worst_Case"]
    pattern_analyzer.pattern_names = pattern_names

    # Create multi-corner analyzer
    corners = ["TT_25C", "FF_125C", "SS_-40C", "Turbo_Mode"]
    corner_analyzer = MultiCornerAnalyzer(corners)

    # Create synthetic nodes
    np.random.seed(42)
    num_nodes = 100

    print(f"\nCreating {num_nodes} synthetic nodes with pattern-dependent behavior...")

    # Generate nodes with different characteristics
    all_pattern_toggles = {i: [] for i in range(4)}
    all_corner_toggles = {corner: [] for corner in corners}

    for node_idx in range(num_nodes):
        node_id = 10000 + node_idx

        # Assign node type
        if node_idx < 20:  # Clock network nodes
            node_type = "clock"
        elif node_idx < 50:  # Data path nodes
            node_type = "data"
        elif node_idx < 80:  # Control logic
            node_type = "control"
        else:  # Memory
            node_type = "memory"

        # Generate pattern-dependent toggles
        for pattern_id in range(4):
            num_toggles = 0

            if node_type == "clock":
                # Clocks toggle in all patterns
                if pattern_id in [0, 1, 2, 3]:  # All patterns
                    num_toggles = 50
            elif node_type == "data":
                # Data toggles more in active patterns
                if pattern_id == 0:  # Idle
                    num_toggles = np.random.randint(5, 15)
                elif pattern_id == 2:  # Max load
                    num_toggles = np.random.randint(40, 60)
                elif pattern_id == 3:  # Worst case
                    num_toggles = np.random.randint(60, 80)
            elif node_type == "control":
                # Control toggles during reset and worst case
                if pattern_id == 1:  # Reset
                    num_toggles = np.random.randint(30, 50)
                elif pattern_id == 3:  # Worst case
                    num_toggles = np.random.randint(20, 40)
            else:  # memory
                # Memory toggles in max load
                if pattern_id == 2:  # Max load
                    num_toggles = np.random.randint(20, 40)

            # Generate toggle cycles
            for _ in range(num_toggles):
                cycle = np.random.randint(0, 100)
                all_pattern_toggles[pattern_id].append((node_id, cycle))

        # Generate corner-dependent toggles (simplified)
        for corner_idx, corner in enumerate(corners):
            # Different corners have different toggle rates
            base_toggles = np.random.randint(20, 60)

            # Apply corner scaling
            if corner == "FF_125C":  # Fast-fast
                scaling = 1.2  # 20% more toggles
            elif corner == "SS_-40C":  # Slow-slow
                scaling = 0.8  # 20% fewer toggles
            elif corner == "Turbo_Mode":
                scaling = 1.5  # 50% more toggles
            else:  # TT
                scaling = 1.0

            num_toggles = int(base_toggles * scaling)

            for _ in range(num_toggles):
                cycle = np.random.randint(0, 100)
                all_corner_toggles[corner].append((node_id, cycle))

    # Process patterns
    print("\nProcessing pattern simulations...")
    for pattern_id in range(4):
        pattern_analyzer.simulate_pattern(pattern_id, all_pattern_toggles[pattern_id])
        print(
            f"  {pattern_names[pattern_id]:12s}: {len(all_pattern_toggles[pattern_id]):6d} toggles"
        )

    # Process corners
    print("\nProcessing corner simulations...")
    for corner in corners:
        scaling = (
            1.2
            if "FF" in corner
            else 0.8
            if "SS" in corner
            else 1.5
            if "Turbo" in corner
            else 1.0
        )
        corner_analyzer.simulate_corner(corner, all_corner_toggles[corner], scaling)
        print(f"  {corner:12s}: {len(all_corner_toggles[corner]):6d} toggles")

    # Analyze pattern sensitivity
    print("\n" + "-" * 40)
    print("PATTERN SENSITIVITY ANALYSIS")
    print("-" * 40)

    # Find most pattern-sensitive nodes
    sensitive_nodes = pattern_analyzer.find_worst_case_nodes(top_n=5)

    print("\nTop 5 Pattern-Sensitive Nodes:")
    for rank, (node_id, sensitivity) in enumerate(sensitive_nodes, 1):
        sensitivity_info = pattern_analyzer.analyze_pattern_sensitivity(node_id)
        node_type = (
            "clock"
            if node_id < 10020
            else "data"
            if node_id < 10050
            else "control"
            if node_id < 10080
            else "memory"
        )

        print(f"\n  {rank}. Node {node_id} ({node_type}):")
        print(f"     Sensitivity Index: {sensitivity:.3f}")
        print(f"     Activity Range: {sensitivity_info['range']:.3f}")
        print(f"     Worst Pattern: {sensitivity_info['worst_pattern_name']}")
        print(
            f"     Pattern Activities: {[f'{a:.3f}' for a in sensitivity_info['pattern_activities']]}"
        )

    # Analyze corner impact
    print("\n" + "-" * 40)
    print("MULTI-CORNER ANALYSIS")
    print("-" * 40)

    # Analyze a representative node
    sample_node = 10025  # A data path node
    corner_impact = corner_analyzer.analyze_corner_impact(sample_node)

    print(f"\nCorner Analysis for Node {sample_node}:")
    for corner, results in corner_impact["corner_results"].items():
        print(
            f"  {corner:12s}: {results['toggle_count']:6.1f} toggles, "
            f"Power: {results['estimated_power'] * 1e6:6.2f} μW"
        )

    print(f"\n  Power Variation: {corner_impact['power_variation'] * 100:.1f}%")
    print(f"  Worst-Case Corner: {corner_impact['worst_case_corner']}")

    # Pattern correlation analysis
    print("\n" + "-" * 40)
    print("PATTERN CORRELATION ANALYSIS")
    print("-" * 40)

    correlation_matrix = pattern_analyzer.analyze_pattern_correlation()

    print("\nPattern Correlation Matrix:")
    print("          " + " ".join(f"{name:10s}" for name in pattern_names))
    for i, pattern_i in enumerate(pattern_names):
        row = f"{pattern_i:10s}"
        for j, pattern_j in enumerate(pattern_names):
            row += f"  {correlation_matrix[i, j]:6.3f}"
        print(row)

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Pattern sensitivity distribution
    all_sensitivities = []
    for node_id in list(pattern_analyzer.node_info.keys())[:100]:
        sensitivity = pattern_analyzer.analyze_pattern_sensitivity(node_id)
        if sensitivity and "sensitivity_index" in sensitivity:
            all_sensitivities.append(sensitivity["sensitivity_index"])

    axes[0, 0].hist(all_sensitivities, bins=20, edgecolor="black", alpha=0.7)
    axes[0, 0].set_title("Pattern Sensitivity Distribution")
    axes[0, 0].set_xlabel("Sensitivity Index")
    axes[0, 0].set_ylabel("Number of Nodes")
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Pattern correlation heatmap
    im = axes[0, 1].imshow(correlation_matrix, cmap="RdYlBu_r", vmin=0, vmax=1)
    axes[0, 1].set_title("Pattern Correlation Heatmap")
    axes[0, 1].set_xticks(range(len(pattern_names)))
    axes[0, 1].set_yticks(range(len(pattern_names)))
    axes[0, 1].set_xticklabels(pattern_names, rotation=45)
    axes[0, 1].set_yticklabels(pattern_names)
    plt.colorbar(im, ax=axes[0, 1], label="Correlation")

    # Add correlation values
    for i in range(len(pattern_names)):
        for j in range(len(pattern_names)):
            axes[0, 1].text(
                j,
                i,
                f"{correlation_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    # Plot 3: Corner comparison for sample node
    corner_names = list(corner_impact["corner_results"].keys())
    corner_powers = [
        corner_impact["corner_results"][c]["estimated_power"] * 1e6
        for c in corner_names
    ]

    bars = axes[1, 0].bar(
        corner_names, corner_powers, color=["blue", "green", "red", "orange"]
    )
    axes[1, 0].set_title(f"Power Across Corners (Node {sample_node})")
    axes[1, 0].set_ylabel("Power (μW)")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axes[1, 0].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
        )

    # Plot 4: Pattern activities for sensitive nodes
    for rank, (node_id, sensitivity) in enumerate(sensitive_nodes[:3], 1):
        sensitivity_info = pattern_analyzer.analyze_pattern_sensitivity(node_id)
        activities = sensitivity_info["pattern_activities"]

        axes[1, 1].plot(
            pattern_names,
            activities,
            marker="o",
            label=f"Node {node_id} (Sens: {sensitivity:.2f})",
        )

    axes[1, 1].set_title("Pattern Activities for Sensitive Nodes")
    axes[1, 1].set_xlabel("Pattern")
    axes[1, 1].set_ylabel("Activity Factor")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("pattern_corner_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()

    return pattern_analyzer, corner_analyzer


if __name__ == "__main__":
    # Run demo 4
    pattern_analyzer, corner_analyzer = demo_pattern_corner_analysis()
