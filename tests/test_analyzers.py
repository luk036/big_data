import numpy as np
import pytest

from big_data.pattern_dependent_analyzer_demo import (
    MultiCornerAnalyzer,
    PatternDependentAnalyzer,
)
from big_data.switching_power_analysis_demo import SwitchingPowerAnalyzer
from big_data.temporal_switching_analyzer_demo import (
    CurrentEnvelopeEstimator,
    TemporalSwitchingAnalyzer,
)


def test_switching_power_analyzer_activity_and_power_formulas():
    analyzer = SwitchingPowerAnalyzer(num_nodes=100, clock_freq=1e9)
    analyzer.add_node(1, capacitance=2e-15, x=0.0, y=0.0, category="logic")
    analyzer.add_node(2, capacitance=1e-15, x=1.0, y=1.0, category="memory")

    toggles = [(1, cycle) for cycle in range(0, 100, 2)]
    toggles.append((2, 99))
    analyzer.process_toggles(toggles)

    assert analyzer.total_cycles == 100
    estimated = analyzer.cms.query(1)
    assert analyzer.estimate_activity_factor(1) == pytest.approx(estimated / 100)
    assert analyzer.estimate_power(1) == pytest.approx(
        analyzer.estimate_activity_factor(1) * 2e-15 * 1e9
    )
    assert all(power >= 0 for _, power in analyzer.find_hot_nodes(top_n=10))


def test_switching_power_spatial_and_category_breakdown():
    analyzer = SwitchingPowerAnalyzer(num_nodes=10)
    analyzer.add_node(1, 1e-15, 0.0, 0.0, "clock")
    analyzer.add_node(2, 1e-15, 10.0, 10.0, "logic")
    analyzer.process_toggles([(1, 0), (1, 1), (2, 0)])

    grid = analyzer.analyze_spatial_distribution(grid_size=4)
    assert grid.shape == (4, 4)

    stats = analyzer.analyze_by_category()
    assert set(stats) == {"clock", "logic"}
    assert stats["clock"]["node_count"] == 1
    assert stats["clock"]["avg_power"] == pytest.approx(stats["clock"]["total_power"])


def test_temporal_profile_and_periodicity():
    analyzer = TemporalSwitchingAnalyzer(num_time_windows=2, window_size=20)
    for cycle in range(20):
        analyzer.process_timestep([7] if cycle % 2 == 0 else [])

    profile = analyzer.get_temporal_profile(7)
    assert profile.shape == (2,)
    for window, sketch in enumerate(analyzer.window_sketches):
        assert profile[window] == pytest.approx(sketch.query(7) / 20)

    period = analyzer.detect_periodicity(7)
    assert period["periodic"]
    assert period["period"] == pytest.approx(2.0)


def test_temporal_burstiness_and_hot_windows():
    analyzer = TemporalSwitchingAnalyzer(num_time_windows=3, window_size=2)
    for _ in range(6):
        analyzer.process_timestep([1])

    burst = analyzer.detect_burstiness(1)
    assert burst["burst_count"] >= 0

    hot_windows = analyzer.find_simultaneous_switching(threshold=0.0)
    assert len(hot_windows) == 3
    assert hot_windows == sorted(hot_windows, key=lambda item: item[1], reverse=True)


def test_current_envelope_estimator_peak_and_shape():
    estimator = CurrentEnvelopeEstimator(transition_time=50e-12, voltage=1.0)
    estimator.add_node(1, capacitance=5e-15)
    assert estimator.estimate_peak_current(1) == pytest.approx(5e-15 / 50e-12)

    estimator.process_toggles([(1, cycle) for cycle in range(50)])
    np.random.seed(0)
    time_axis, envelope = estimator.estimate_current_envelope(
        time_resolution=5e-12, duration=200e-12
    )
    assert time_axis.shape == envelope.shape
    assert np.isfinite(envelope).all()
    assert np.all(envelope >= 0)


def test_pattern_sensitivity_metrics():
    analyzer = PatternDependentAnalyzer(num_patterns=3)
    analyzer.pattern_names = ["A", "B", "C"]
    analyzer.simulate_pattern(0, [(1, 0), (1, 9)])
    analyzer.simulate_pattern(1, [(1, 0), (1, 1), (1, 2), (1, 3), (1, 9)])
    analyzer.simulate_pattern(2, [(1, cycle) for cycle in range(8)] + [(1, 9)])

    sensitivity = analyzer.analyze_pattern_sensitivity(1)
    assert sensitivity["pattern_activities"] == pytest.approx([0.2, 0.5, 0.9])
    assert sensitivity["range"] == pytest.approx(0.7)
    assert sensitivity["worst_pattern_name"] == "C"


def test_multi_corner_power_scaling():
    analyzer = MultiCornerAnalyzer(["TT_25C", "FF_125C"])
    analyzer.simulate_corner("TT_25C", [(1, 0)] * 10, scaling_factor=1.0)
    analyzer.simulate_corner("FF_125C", [(1, 0)] * 10, scaling_factor=1.2)

    impact = analyzer.analyze_corner_impact(1)
    assert impact["corner_results"]["TT_25C"]["toggle_count"] == pytest.approx(10.0)
    assert impact["corner_results"]["FF_125C"]["toggle_count"] == pytest.approx(12.0)
    assert impact["worst_case_corner"] == "FF_125C"


def test_orchestrator_reuses_home_modules():
    import big_data.switching_analysis_demo as orchestrator
    from big_data.temporal_switching_analyzer_demo import demo_temporal_analysis

    assert orchestrator.demo_temporal_analysis is demo_temporal_analysis
    assert not hasattr(orchestrator, "SwitchingPowerAnalyzer")
