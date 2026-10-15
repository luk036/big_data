"""End-to-end switching-power analysis: run every experiment demo in sequence.

This module is a thin orchestrator.  Each experiment lives in its own module and
keeps a self-contained implementation; this file only wires them together so a
single command exercises the whole story:

    python -m big_data.switching_analysis_demo

The individual steps can also be run on their own, e.g.
``python -m big_data.temporal_switching_analyzer_demo``.
"""

from big_data.basic_count_min_sketch import demo_basic_cms
from big_data.memory_accuracy_tradeoff_demo import demo_memory_accuracy_tradeoff
from big_data.pattern_dependent_analyzer_demo import demo_pattern_corner_analysis
from big_data.switching_power_analysis_demo import demo_switching_power_analysis
from big_data.temporal_switching_analyzer_demo import demo_temporal_analysis

__all__ = [
    "main",
    "demo_basic_cms",
    "demo_switching_power_analysis",
    "demo_temporal_analysis",
    "demo_pattern_corner_analysis",
    "demo_memory_accuracy_tradeoff",
]


def main() -> None:
    demo_basic_cms()
    demo_switching_power_analysis()
    demo_temporal_analysis()
    demo_pattern_corner_analysis()
    demo_memory_accuracy_tradeoff()

    print("\n" + "=" * 60)
    print("ALL DEMOS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nGenerated figures:")
    print("  switching_power_analysis.png")
    print("  temporal_current_analysis.png")
    print("  pattern_corner_analysis.png")
    print("  memory_accuracy_tradeoff.png")


if __name__ == "__main__":
    main()
