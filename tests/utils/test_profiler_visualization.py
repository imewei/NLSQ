"""
Tests for Profiler Visualization
=================================

Tests the visualization and dashboard components for performance profiling.
"""

import json
import time

import pytest

from nlsq.utils.profiler import PerformanceProfiler, get_global_profiler
from nlsq.utils.profiler_visualization import (
    HAS_MATPLOTLIB,
    ProfilerVisualization,
    ProfilingDashboard,
)


class TestProfilerVisualization:
    """Test ProfilerVisualization class."""

    def setup_method(self):
        """Set up test profiler with sample data."""
        self.profiler = PerformanceProfiler()

        # Generate sample profiling data
        for i in range(5):
            with self.profiler.profile("test1"):
                time.sleep(0.01)
                self.profiler.update_current(
                    n_iterations=10 + i,
                    n_function_evals=50 + i * 5,
                    n_data_points=1000,
                    n_parameters=3,
                    success=True,
                    method="trf",
                    backend="cpu",
                )

        for i in range(3):
            with self.profiler.profile("test2"):
                time.sleep(0.02)
                self.profiler.update_current(
                    n_iterations=15 + i,
                    n_function_evals=60 + i * 5,
                    n_data_points=2000,
                    n_parameters=4,
                    success=i < 2,  # First 2 succeed, last fails
                    method="lm",
                    backend="cpu",
                )

        self.viz = ProfilerVisualization(self.profiler)

    def test_initialization(self):
        """Test visualization initialization."""
        assert self.viz.profiler is self.profiler

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_plot_timing_series(self):
        """Test timing series plot."""
        fig = self.viz.plot_timing_series("test1")

        assert fig is not None
        assert len(fig.axes) == 4  # 2x2 subplots

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_plot_timing_series_empty(self):
        """Test timing series plot with no data."""
        fig = self.viz.plot_timing_series("nonexistent")

        assert fig is None

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_plot_timing_comparison(self):
        """Test timing comparison plot."""
        fig = self.viz.plot_timing_comparison(["test1", "test2"])

        assert fig is not None
        assert len(fig.axes) == 3  # 3 comparison plots

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_plot_timing_comparison_empty(self):
        """Test timing comparison with no data."""
        fig = self.viz.plot_timing_comparison(["nonexistent1", "nonexistent2"])

        assert fig is None

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_plot_timing_distribution(self):
        """Test timing distribution plot."""
        fig = self.viz.plot_timing_distribution("test1")

        assert fig is not None
        assert len(fig.axes) == 2  # 2 distribution plots

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_plot_timing_distribution_insufficient_data(self):
        """Test distribution plot with insufficient data."""
        # Create profile with only 1 run
        with self.profiler.profile("single_run"):
            time.sleep(0.01)

        fig = self.viz.plot_timing_distribution("single_run")

        assert fig is None

    def test_generate_html_report(self):
        """Test HTML report generation."""
        html = self.viz.generate_html_report()

        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert "NLSQ Performance Profile Report" in html
        assert "test1" in html
        assert "test2" in html
        assert "Total Runs" in html
        assert "Success Rate" in html

    def test_generate_html_report_to_file(self, tmp_path):
        """Test HTML report saved to file."""
        output_path = tmp_path / "report.html"

        html = self.viz.generate_html_report(output_path)

        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert content == html
        assert "NLSQ Performance Profile Report" in content

    def test_export_json(self, tmp_path):
        """Test JSON export."""
        output_path = tmp_path / "profiles.json"

        self.viz.export_json(output_path)

        assert output_path.exists()

        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert "test1" in data
        assert "test2" in data
        assert len(data["test1"]) == 5
        assert len(data["test2"]) == 3

    def test_export_csv(self, tmp_path):
        """Test CSV export."""
        output_path = tmp_path / "test1.csv"

        self.viz.export_csv("test1", output_path)

        assert output_path.exists()

        lines = output_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 6  # Header + 5 data rows

        # Check header
        headers = lines[0].split(",")
        assert "run" in headers
        assert "total_time" in headers
        assert "n_iterations" in headers
        assert "success" in headers

        # Check data row
        first_row = lines[1].split(",")
        assert first_row[0] == "1"  # Run number
        assert first_row[14] == "True"  # Success

    def test_export_csv_empty_profile(self, tmp_path):
        """Test CSV export with no data."""
        output_path = tmp_path / "empty.csv"

        self.viz.export_csv("nonexistent", output_path)

        # Should not create file for empty profile
        assert not output_path.exists()


class TestProfilingDashboard:
    """Test ProfilingDashboard class."""

    def setup_method(self):
        """Set up test dashboard with sample data."""
        self.profiler = PerformanceProfiler()

        # Generate sample profiling data
        for i in range(5):
            with self.profiler.profile("fast"):
                time.sleep(0.01)
                self.profiler.update_current(
                    n_iterations=8 + i, n_function_evals=40 + i * 5, success=True
                )

        for i in range(5):
            with self.profiler.profile("slow"):
                time.sleep(0.02)
                self.profiler.update_current(
                    n_iterations=15 + i, n_function_evals=75 + i * 5, success=True
                )

        for i in range(5):
            with self.profiler.profile("unreliable"):
                time.sleep(0.015)
                self.profiler.update_current(
                    n_iterations=10 + i, n_function_evals=50 + i * 5, success=i < 2
                )

        self.dashboard = ProfilingDashboard(self.profiler)

    def test_initialization(self):
        """Test dashboard initialization."""
        assert self.dashboard.profiler is self.profiler
        assert isinstance(self.dashboard.viz, ProfilerVisualization)
        assert self.dashboard.tracked_profiles == []

    def test_add_profile(self):
        """Test adding profiles to dashboard."""
        self.dashboard.add_profile("fast")
        assert "fast" in self.dashboard.tracked_profiles

        self.dashboard.add_profile("slow")
        assert "slow" in self.dashboard.tracked_profiles
        assert len(self.dashboard.tracked_profiles) == 2

    def test_add_profile_duplicate(self):
        """Test adding duplicate profile."""
        self.dashboard.add_profile("fast")
        self.dashboard.add_profile("fast")

        # Should not add duplicate
        assert self.dashboard.tracked_profiles.count("fast") == 1

    def test_remove_profile(self):
        """Test removing profiles from dashboard."""
        self.dashboard.add_profile("fast")
        self.dashboard.add_profile("slow")
        assert len(self.dashboard.tracked_profiles) == 2

        self.dashboard.remove_profile("fast")
        assert "fast" not in self.dashboard.tracked_profiles
        assert len(self.dashboard.tracked_profiles) == 1

    def test_remove_nonexistent_profile(self):
        """Test removing profile that doesn't exist."""
        self.dashboard.add_profile("fast")

        # Should not raise error
        self.dashboard.remove_profile("nonexistent")
        assert len(self.dashboard.tracked_profiles) == 1

    def test_generate_comparison_report_empty(self):
        """Test comparison report with no profiles."""
        report = self.dashboard.generate_comparison_report()

        assert "No profiles tracked" in report

    def test_generate_comparison_report(self):
        """Test comparison report generation."""
        self.dashboard.add_profile("fast")
        self.dashboard.add_profile("slow")
        self.dashboard.add_profile("unreliable")

        report = self.dashboard.generate_comparison_report()

        assert "Profiling Dashboard" in report
        assert "fast" in report
        assert "slow" in report
        assert "unreliable" in report
        assert "Best Performers" in report
        assert "Fastest" in report
        assert "Most Reliable" in report
        assert "Fewest Iters" in report

    def test_generate_comparison_report_single_profile(self):
        """Test comparison report with single profile."""
        self.dashboard.add_profile("fast")

        report = self.dashboard.generate_comparison_report()

        assert "fast" in report
        # Should not have "Best Performers" section with only 1 profile
        assert "Best Performers" not in report

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_plot_all_comparisons(self):
        """Test comprehensive comparison plots."""
        self.dashboard.add_profile("fast")
        self.dashboard.add_profile("slow")

        fig = self.dashboard.plot_all_comparisons()

        assert fig is not None
        assert len(fig.axes) == 3

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_plot_all_comparisons_empty(self):
        """Test comparison plots with no profiles."""
        fig = self.dashboard.plot_all_comparisons()

        assert fig is None

    def test_save_dashboard(self, tmp_path):
        """Test saving complete dashboard."""
        self.dashboard.add_profile("fast")
        self.dashboard.add_profile("slow")

        output_dir = tmp_path / "dashboard"

        self.dashboard.save_dashboard(output_dir)

        # Check that files were created
        assert (output_dir / "dashboard.html").exists()
        assert (output_dir / "profiles.json").exists()
        assert (output_dir / "comparison_report.txt").exists()

        # Check HTML content
        html_content = (output_dir / "dashboard.html").read_text(encoding="utf-8")
        assert "NLSQ Performance Profile Report" in html_content

        # Check JSON content
        json_data = json.loads(
            (output_dir / "profiles.json").read_text(encoding="utf-8")
        )
        assert "fast" in json_data
        assert "slow" in json_data

        # Check report content
        report_content = (output_dir / "comparison_report.txt").read_text(
            encoding="utf-8"
        )
        assert "Profiling Dashboard" in report_content

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_save_dashboard_with_plots(self, tmp_path):
        """Test saving dashboard with plot images."""
        self.dashboard.add_profile("fast")
        self.dashboard.add_profile("slow")

        output_dir = tmp_path / "dashboard"

        self.dashboard.save_dashboard(output_dir)

        # Check that plot files were created
        assert (output_dir / "fast_series.png").exists()
        assert (output_dir / "slow_series.png").exists()
        assert (output_dir / "comparison.png").exists()


class TestIntegration:
    """Integration tests for profiler visualization."""

    def test_complete_workflow(self, tmp_path):
        """Test complete profiling and visualization workflow."""
        profiler = PerformanceProfiler()

        # Simulate optimization runs
        for run in range(3):
            with profiler.profile("optimization"):
                time.sleep(0.01)
                profiler.update_current(
                    n_iterations=10 + run,
                    n_function_evals=50 + run * 10,
                    n_data_points=1000,
                    n_parameters=3,
                    success=True,
                    method="trf",
                    backend="cpu",
                )

        # Create visualization
        viz = ProfilerVisualization(profiler)

        # Generate HTML report
        html = viz.generate_html_report()
        assert "optimization" in html

        # Export JSON
        json_path = tmp_path / "profiles.json"
        viz.export_json(json_path)
        assert json_path.exists()

        # Export CSV
        csv_path = tmp_path / "optimization.csv"
        viz.export_csv("optimization", csv_path)
        assert csv_path.exists()

        # Create dashboard
        dashboard = ProfilingDashboard(profiler)
        dashboard.add_profile("optimization")

        # Generate comparison report
        report = dashboard.generate_comparison_report()
        assert "optimization" in report

        # Save complete dashboard
        dashboard_dir = tmp_path / "dashboard"
        dashboard.save_dashboard(dashboard_dir)
        assert (dashboard_dir / "dashboard.html").exists()

    def test_global_profiler_integration(self, tmp_path):
        """Test integration with global profiler."""
        profiler = get_global_profiler()
        profiler.clear()  # Clear any existing data

        # Profile some operations
        with profiler.profile("test_global"):
            time.sleep(0.01)
            profiler.update_current(n_iterations=10, success=True)

        # Create visualization from global profiler
        viz = ProfilerVisualization(profiler)

        html = viz.generate_html_report()
        assert "test_global" in html

        # Clean up
        profiler.clear()

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_all_plot_types(self):
        """Test all plot types can be generated."""
        profiler = PerformanceProfiler()

        # Generate sufficient data
        for i in range(10):
            with profiler.profile("plot_test1"):
                time.sleep(0.005)
                profiler.update_current(n_iterations=10 + i, success=True)

            with profiler.profile("plot_test2"):
                time.sleep(0.008)
                profiler.update_current(n_iterations=15 + i, success=True)

        viz = ProfilerVisualization(profiler)

        # Test all plot types
        fig1 = viz.plot_timing_series("plot_test1")
        assert fig1 is not None

        fig2 = viz.plot_timing_comparison(["plot_test1", "plot_test2"])
        assert fig2 is not None

        fig3 = viz.plot_timing_distribution("plot_test1")
        assert fig3 is not None

        # Clean up figures
        import matplotlib.pyplot as plt

        plt.close("all")


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_visualization_without_matplotlib(self):
        """Test visualization gracefully handles missing matplotlib."""
        profiler = PerformanceProfiler()

        with profiler.profile("test"):
            time.sleep(0.01)

        viz = ProfilerVisualization(profiler)

        # These should return None if matplotlib not available
        if not HAS_MATPLOTLIB:
            assert viz.plot_timing_series("test") is None
            assert viz.plot_timing_comparison(["test"]) is None
            assert viz.plot_timing_distribution("test") is None

    def test_empty_profiler(self):
        """Test visualization with empty profiler."""
        profiler = PerformanceProfiler()
        viz = ProfilerVisualization(profiler)

        html = viz.generate_html_report()
        assert "<!DOCTYPE html>" in html

    def test_special_characters_in_profile_name(self, tmp_path):
        """Test handling of special characters in profile names."""
        profiler = PerformanceProfiler()

        with profiler.profile("test-profile_2024"):
            time.sleep(0.01)
            profiler.update_current(n_iterations=10, success=True)

        viz = ProfilerVisualization(profiler)

        html = viz.generate_html_report()
        assert "test-profile_2024" in html

        json_path = tmp_path / "profiles.json"
        viz.export_json(json_path)

        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "test-profile_2024" in data

    def test_large_number_of_runs(self):
        """Test visualization with large number of runs."""
        profiler = PerformanceProfiler()

        # Generate 100 runs
        for i in range(100):
            with profiler.profile("large_test"):
                profiler.update_current(n_iterations=i, success=i % 10 != 0)

        viz = ProfilerVisualization(profiler)

        summary = profiler.get_summary("large_test")
        assert summary["n_runs"] == 100

        html = viz.generate_html_report()
        assert "100" in html or "large_test" in html


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
