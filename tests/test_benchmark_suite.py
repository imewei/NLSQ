"""
Tests for Performance Benchmarking Suite
=========================================

Tests the benchmarking infrastructure for NLSQ performance testing.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from benchmarks.benchmark_suite import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkSuite,
    ExponentialDecayProblem,
    GaussianProblem,
    PolynomialProblem,
    SinusoidalProblem,
)


class TestBenchmarkConfig:
    """Test BenchmarkConfig dataclass."""

    def test_default_initialization(self):
        """Test default config values."""
        config = BenchmarkConfig(name="test")

        assert config.name == "test"
        assert config.problem_sizes == [100, 1000, 10000, 100000]
        assert config.n_repeats == 5
        assert config.warmup_runs == 1
        assert config.methods == ["trf", "lm"]
        assert config.backends == ["cpu"]
        assert config.compare_scipy is True

    def test_custom_initialization(self):
        """Test custom config values."""
        config = BenchmarkConfig(
            name="custom",
            problem_sizes=[10, 100],
            n_repeats=3,
            warmup_runs=0,
            methods=["trf"],
            backends=["cpu"],
            compare_scipy=False,
        )

        assert config.name == "custom"
        assert config.problem_sizes == [10, 100]
        assert config.n_repeats == 3
        assert config.warmup_runs == 0
        assert config.methods == ["trf"]
        assert config.compare_scipy is False


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""

    def test_initialization(self):
        """Test result initialization."""
        result = BenchmarkResult(
            library="nlsq",
            method="trf",
            backend="cpu",
            problem_size=1000,
            n_parameters=3,
            total_time=0.5,
            optimization_time=0.4,
            jit_compile_time=0.1,
            n_iterations=10,
            n_function_evals=50,
            success=True,
            final_cost=1.5,
            speedup_vs_scipy=2.5,
        )

        assert result.library == "nlsq"
        assert result.method == "trf"
        assert result.problem_size == 1000
        assert result.success is True
        assert result.speedup_vs_scipy == 2.5


class TestBenchmarkProblems:
    """Test benchmark problem classes."""

    def test_exponential_decay(self):
        """Test exponential decay problem."""
        problem = ExponentialDecayProblem()

        assert problem.name == "exponential_decay"
        assert problem.n_parameters == 3

        x, y, p_true = problem.generate_data(100)
        assert len(x) == 100
        assert len(y) == 100
        assert len(p_true) == 3

        # Test model
        y_pred = problem.model(x, *p_true)
        assert len(y_pred) == 100

    def test_gaussian(self):
        """Test Gaussian problem."""
        problem = GaussianProblem()

        assert problem.name == "gaussian"
        assert problem.n_parameters == 3

        x, y, p_true = problem.generate_data(100)
        assert len(x) == 100
        assert len(y) == 100
        assert len(p_true) == 3

    def test_polynomial(self):
        """Test polynomial problem."""
        problem = PolynomialProblem()

        assert problem.name == "polynomial"
        assert problem.n_parameters == 5

        x, y, p_true = problem.generate_data(100)
        assert len(x) == 100
        assert len(y) == 100
        assert len(p_true) == 5

    def test_sinusoidal(self):
        """Test sinusoidal problem."""
        problem = SinusoidalProblem()

        assert problem.name == "sinusoidal"
        assert problem.n_parameters == 4

        x, y, p_true = problem.generate_data(100)
        assert len(x) == 100
        assert len(y) == 100
        assert len(p_true) == 4

    def test_initial_guess(self):
        """Test initial parameter guess generation."""
        problem = ExponentialDecayProblem()
        _x, _y, p_true = problem.generate_data(100)

        p0 = problem.get_initial_guess(p_true)

        assert len(p0) == len(p_true)
        # Should be perturbed from true values
        assert not np.allclose(p0, p_true)


class TestBenchmarkSuite:
    """Test BenchmarkSuite class."""

    def setup_method(self):
        """Set up test suite."""
        config = BenchmarkConfig(
            name="test",
            problem_sizes=[50, 100],
            n_repeats=2,
            warmup_runs=1,
            methods=["trf"],
            backends=["cpu"],
            compare_scipy=True,
        )
        self.suite = BenchmarkSuite(config)

    def test_initialization(self):
        """Test suite initialization."""
        assert self.suite.config.name == "test"
        assert len(self.suite.problems) == 0
        assert len(self.suite.results) == 0

    def test_add_problem(self):
        """Test adding problems."""
        problem = ExponentialDecayProblem()
        self.suite.add_problem(problem)

        assert len(self.suite.problems) == 1
        assert self.suite.problems[0].name == "exponential_decay"

    def test_run_benchmarks_nlsq_only(self):
        """Test running benchmarks (NLSQ only)."""
        self.suite.config.compare_scipy = False
        self.suite.add_problem(ExponentialDecayProblem())

        results = self.suite.run_benchmarks(verbose=False)

        # Should have: 2 sizes * 2 repeats = 4 results
        assert len(results) >= 4
        assert all(r.library == "nlsq" for r in results)

    def test_run_benchmarks_with_scipy(self):
        """Test running benchmarks (NLSQ + SciPy)."""
        self.suite.add_problem(ExponentialDecayProblem())

        results = self.suite.run_benchmarks(verbose=False)

        # Should have both NLSQ and SciPy results
        nlsq_results = [r for r in results if r.library == "nlsq"]
        scipy_results = [r for r in results if r.library == "scipy"]

        assert len(nlsq_results) > 0
        assert len(scipy_results) > 0

    def test_multiple_problems(self):
        """Test benchmarking multiple problems."""
        self.suite.config.compare_scipy = False
        self.suite.add_problem(ExponentialDecayProblem())
        self.suite.add_problem(GaussianProblem())

        results = self.suite.run_benchmarks(verbose=False)

        # Should have results for both problems
        exp_results = [
            r for r in results if r.metadata.get("problem") == "exponential_decay"
        ]
        gauss_results = [r for r in results if r.metadata.get("problem") == "gaussian"]

        assert len(exp_results) > 0
        assert len(gauss_results) > 0

    def test_generate_report_empty(self):
        """Test report generation with no results."""
        report = self.suite.generate_report()

        assert "No benchmark results" in report

    def test_generate_report(self):
        """Test report generation."""
        self.suite.config.compare_scipy = False
        self.suite.config.problem_sizes = [50]
        self.suite.config.n_repeats = 2

        self.suite.add_problem(ExponentialDecayProblem())
        self.suite.run_benchmarks(verbose=False)

        report = self.suite.generate_report()

        assert "NLSQ Performance Benchmark Report" in report
        assert "exponential_decay" in report.lower()
        assert "NLSQ (trf)" in report

    def test_save_results(self):
        """Test saving results to directory."""
        self.suite.config.compare_scipy = False
        self.suite.config.problem_sizes = [50]
        self.suite.config.n_repeats = 2

        self.suite.add_problem(ExponentialDecayProblem())
        self.suite.run_benchmarks(verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            self.suite.save_results(tmpdir)

            # Check files created
            output_dir = Path(tmpdir)
            assert (output_dir / "benchmark_report.txt").exists()
            assert (output_dir / "benchmark_results.csv").exists()
            assert (output_dir / "dashboard").exists()

            # Check report content
            report_content = (output_dir / "benchmark_report.txt").read_text(
                encoding="utf-8"
            )
            assert "NLSQ Performance Benchmark Report" in report_content

            # Check CSV content
            csv_content = (output_dir / "benchmark_results.csv").read_text(
                encoding="utf-8"
            )
            lines = csv_content.strip().split("\n")
            assert len(lines) > 1  # Header + data rows
            assert "library,method,backend" in lines[0]

    def test_benchmark_nlsq_success(self):
        """Test NLSQ benchmark successful run."""
        problem = ExponentialDecayProblem()
        x, y, p_true = problem.generate_data(100)
        p0 = problem.get_initial_guess(p_true)

        results = self.suite._benchmark_nlsq(problem, x, y, p0, "trf", "cpu")

        assert len(results) == self.suite.config.n_repeats
        # Most should succeed
        success_count = sum(r.success for r in results)
        assert success_count >= len(results) // 2

    def test_benchmark_scipy_success(self):
        """Test SciPy benchmark successful run."""
        problem = ExponentialDecayProblem()
        x, y, p_true = problem.generate_data(100)
        p0 = problem.get_initial_guess(p_true)

        results = self.suite._benchmark_scipy(problem, x, y, p0, "trf")

        assert len(results) == self.suite.config.n_repeats
        # Most should succeed
        success_count = sum(r.success for r in results)
        assert success_count >= len(results) // 2

    def test_timing_metrics(self):
        """Test that timing metrics are recorded."""
        self.suite.config.compare_scipy = False
        self.suite.config.problem_sizes = [50]
        self.suite.add_problem(ExponentialDecayProblem())

        results = self.suite.run_benchmarks(verbose=False)

        for result in results:
            assert result.total_time > 0
            assert result.optimization_time >= 0

    def test_speedup_calculation(self):
        """Test speedup vs SciPy calculation."""
        self.suite.config.problem_sizes = [50]
        self.suite.config.n_repeats = 2
        self.suite.add_problem(ExponentialDecayProblem())

        results = self.suite.run_benchmarks(verbose=False)

        nlsq_results = [r for r in results if r.library == "nlsq"]

        # Speedup should be calculated
        for result in nlsq_results:
            # Speedup might be > 1 or < 1 depending on problem size
            assert result.speedup_vs_scipy >= 0


class TestIntegration:
    """Integration tests for benchmark suite."""

    def test_complete_benchmark_workflow(self):
        """Test complete benchmarking workflow."""
        config = BenchmarkConfig(
            name="integration_test",
            problem_sizes=[50],
            n_repeats=2,
            warmup_runs=1,
            methods=["trf"],
            backends=["cpu"],
            compare_scipy=True,
        )

        suite = BenchmarkSuite(config)
        suite.add_problem(ExponentialDecayProblem())
        suite.add_problem(GaussianProblem())

        # Run benchmarks
        results = suite.run_benchmarks(verbose=False)

        assert len(results) > 0

        # Generate report
        report = suite.generate_report()
        assert len(report) > 0
        assert "exponential_decay" in report.lower()
        assert "gaussian" in report.lower()

        # Save results
        with tempfile.TemporaryDirectory() as tmpdir:
            suite.save_results(tmpdir)

            # Verify all files created
            output_dir = Path(tmpdir)
            assert (output_dir / "benchmark_report.txt").exists()
            assert (output_dir / "benchmark_results.csv").exists()

    def test_all_problem_types(self):
        """Test all standard problem types."""
        config = BenchmarkConfig(
            name="all_problems",
            problem_sizes=[50],
            n_repeats=2,
            warmup_runs=0,
            methods=["trf"],
            backends=["cpu"],
            compare_scipy=False,
        )

        suite = BenchmarkSuite(config)
        suite.add_problem(ExponentialDecayProblem())
        suite.add_problem(GaussianProblem())
        suite.add_problem(PolynomialProblem())
        suite.add_problem(SinusoidalProblem())

        results = suite.run_benchmarks(verbose=False)

        # Should have results for all 4 problems
        problems = {r.metadata.get("problem") for r in results}
        assert "exponential_decay" in problems
        assert "gaussian" in problems
        assert "polynomial" in problems
        assert "sinusoidal" in problems

    def test_multiple_methods(self):
        """Test benchmarking multiple methods."""
        config = BenchmarkConfig(
            name="multi_method",
            problem_sizes=[50],
            n_repeats=2,
            warmup_runs=0,
            methods=["trf", "lm"],
            backends=["cpu"],
            compare_scipy=False,
        )

        suite = BenchmarkSuite(config)
        suite.add_problem(ExponentialDecayProblem())

        results = suite.run_benchmarks(verbose=False)

        # Should have results for both methods
        trf_results = [r for r in results if r.method == "trf"]
        lm_results = [r for r in results if r.method == "lm"]

        assert len(trf_results) > 0
        assert len(lm_results) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_small_problem_size(self):
        """Test with very small problem size."""
        config = BenchmarkConfig(
            name="small",
            problem_sizes=[10],
            n_repeats=1,
            methods=["trf"],
            compare_scipy=False,
        )

        suite = BenchmarkSuite(config)
        suite.add_problem(ExponentialDecayProblem())

        # Should not crash
        results = suite.run_benchmarks(verbose=False)
        assert len(results) > 0

    def test_no_problems(self):
        """Test benchmark suite with no problems."""
        suite = BenchmarkSuite()

        results = suite.run_benchmarks(verbose=False)

        assert len(results) == 0

        report = suite.generate_report()
        assert "No benchmark results" in report

    def test_single_repeat(self):
        """Test with single repeat (no statistics)."""
        config = BenchmarkConfig(
            name="single",
            problem_sizes=[50],
            n_repeats=1,
            methods=["trf"],
            compare_scipy=False,
        )

        suite = BenchmarkSuite(config)
        suite.add_problem(ExponentialDecayProblem())

        results = suite.run_benchmarks(verbose=False)

        assert len(results) == 1

        # Report should still work
        report = suite.generate_report()
        assert "NLSQ Performance Benchmark Report" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
