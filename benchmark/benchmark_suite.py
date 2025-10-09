"""
NLSQ Performance Benchmarking Suite
====================================

Comprehensive benchmarking suite for comparing NLSQ against SciPy and
measuring performance characteristics across different problem types and sizes.
"""

from __future__ import annotations

import contextlib
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numpy as np
from scipy.optimize import curve_fit as scipy_curve_fit

from nlsq import (
    PerformanceProfiler,
    ProfilerVisualization,
    ProfilingDashboard,
    curve_fit,
)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    name: str
    problem_sizes: list[int] = field(default_factory=lambda: [100, 1000, 10000, 100000])
    n_repeats: int = 5
    warmup_runs: int = 1
    methods: list[str] = field(default_factory=lambda: ["trf", "lm"])
    backends: list[str] = field(default_factory=lambda: ["cpu"])
    compare_scipy: bool = True


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    library: str  # "nlsq" or "scipy"
    method: str
    backend: str
    problem_size: int
    n_parameters: int

    total_time: float
    optimization_time: float
    jit_compile_time: float = 0.0

    n_iterations: int = 0
    n_function_evals: int = 0

    success: bool = False
    final_cost: float = 0.0

    iterations_per_second: float = 0.0
    speedup_vs_scipy: float = 1.0

    metadata: dict = field(default_factory=dict)


class BenchmarkProblem:
    """Base class for benchmark problems."""

    def __init__(self, name: str, n_parameters: int):
        """
        Initialize benchmark problem.

        Parameters
        ----------
        name : str
            Problem name
        n_parameters : int
            Number of parameters to fit
        """
        self.name = name
        self.n_parameters = n_parameters

    def generate_data(self, n_points: int, noise_level: float = 0.1) -> tuple:
        """
        Generate synthetic data for the problem.

        Parameters
        ----------
        n_points : int
            Number of data points
        noise_level : float
            Relative noise level

        Returns
        -------
        xdata, ydata, p_true : tuple
            Independent variable, dependent variable, and true parameters
        """
        raise NotImplementedError

    def model(self, x, *params):
        """Model function."""
        raise NotImplementedError

    def get_initial_guess(self, p_true) -> np.ndarray:
        """Get initial parameter guess."""
        # Default: perturb true parameters by 20%
        return p_true * (1 + 0.2 * np.random.randn(len(p_true)))


class ExponentialDecayProblem(BenchmarkProblem):
    """Exponential decay: y = a * exp(-b * x) + c"""

    def __init__(self):
        super().__init__("exponential_decay", n_parameters=3)

    def model(self, x, a, b, c):
        return a * jnp.exp(-b * x) + c

    def generate_data(self, n_points: int, noise_level: float = 0.1):
        x = np.linspace(0, 10, n_points)
        p_true = np.array([10.0, 0.5, 2.0])
        y_true = p_true[0] * np.exp(-p_true[1] * x) + p_true[2]
        y = y_true + noise_level * np.abs(y_true) * np.random.randn(n_points)
        return x, y, p_true


class GaussianProblem(BenchmarkProblem):
    """Gaussian: y = a * exp(-(x - mu)^2 / (2 * sigma^2))"""

    def __init__(self):
        super().__init__("gaussian", n_parameters=3)

    def model(self, x, a, mu, sigma):
        return a * jnp.exp(-((x - mu) ** 2) / (2 * sigma**2))

    def generate_data(self, n_points: int, noise_level: float = 0.1):
        x = np.linspace(-5, 5, n_points)
        p_true = np.array([5.0, 0.0, 1.0])
        y_true = p_true[0] * np.exp(-((x - p_true[1]) ** 2) / (2 * p_true[2] ** 2))
        y = y_true + noise_level * np.abs(y_true) * np.random.randn(n_points)
        return x, y, p_true


class PolynomialProblem(BenchmarkProblem):
    """Polynomial: y = a + b*x + c*x^2 + d*x^3 + e*x^4"""

    def __init__(self):
        super().__init__("polynomial", n_parameters=5)

    def model(self, x, a, b, c, d, e):
        return a + b * x + c * x**2 + d * x**3 + e * x**4

    def generate_data(self, n_points: int, noise_level: float = 0.1):
        x = np.linspace(-2, 2, n_points)
        p_true = np.array([1.0, 2.0, -0.5, 0.3, -0.1])
        y_true = sum(p_true[i] * x**i for i in range(5))
        y = y_true + noise_level * np.abs(y_true.max()) * np.random.randn(n_points)
        return x, y, p_true


class SinusoidalProblem(BenchmarkProblem):
    """Sinusoidal: y = a * sin(b * x + c) + d"""

    def __init__(self):
        super().__init__("sinusoidal", n_parameters=4)

    def model(self, x, a, b, c, d):
        return a * jnp.sin(b * x + c) + d

    def generate_data(self, n_points: int, noise_level: float = 0.1):
        x = np.linspace(0, 4 * np.pi, n_points)
        p_true = np.array([3.0, 2.0, 0.5, 1.0])
        y_true = p_true[0] * np.sin(p_true[1] * x + p_true[2]) + p_true[3]
        y = y_true + noise_level * np.abs(y_true) * np.random.randn(n_points)
        return x, y, p_true


class BenchmarkSuite:
    """
    Comprehensive benchmarking suite for NLSQ.

    Examples
    --------
    >>> suite = BenchmarkSuite()
    >>> suite.add_problem(ExponentialDecayProblem())
    >>> suite.add_problem(GaussianProblem())
    >>> results = suite.run_benchmarks()
    >>> report = suite.generate_report()
    >>> print(report)
    """

    def __init__(self, config: BenchmarkConfig | None = None):
        """
        Initialize benchmark suite.

        Parameters
        ----------
        config : BenchmarkConfig, optional
            Benchmark configuration
        """
        self.config = config or BenchmarkConfig(name="default")
        self.problems: list[BenchmarkProblem] = []
        self.results: list[BenchmarkResult] = []
        self.profiler = PerformanceProfiler()

    def add_problem(self, problem: BenchmarkProblem) -> None:
        """Add a problem to benchmark."""
        self.problems.append(problem)

    def run_benchmarks(self, verbose: bool = True) -> list[BenchmarkResult]:
        """
        Run all benchmarks.

        Parameters
        ----------
        verbose : bool
            Print progress information

        Returns
        -------
        results : list of BenchmarkResult
            All benchmark results
        """
        self.results = []

        total_runs = (
            len(self.problems)
            * len(self.config.problem_sizes)
            * len(self.config.methods)
            * len(self.config.backends)
            * (2 if self.config.compare_scipy else 1)  # NLSQ + SciPy
            * self.config.n_repeats
        )

        current_run = 0

        for problem in self.problems:
            for size in self.config.problem_sizes:
                if verbose:
                    print(f"\n{problem.name} with {size} points:")

                # Generate data once per size
                x, y, p_true = problem.generate_data(size)
                p0 = problem.get_initial_guess(p_true)

                for method in self.config.methods:
                    for backend in self.config.backends:
                        # Benchmark NLSQ
                        if verbose:
                            print(
                                f"  NLSQ ({method}, {backend})...", end=" ", flush=True
                            )

                        nlsq_results = self._benchmark_nlsq(
                            problem, x, y, p0, method, backend
                        )
                        self.results.extend(nlsq_results)
                        current_run += len(nlsq_results)

                        if verbose:
                            avg_time = np.mean([r.total_time for r in nlsq_results])
                            print(f"{avg_time:.4f}s")

                        # Benchmark SciPy
                        if self.config.compare_scipy:
                            if verbose:
                                print(f"  SciPy ({method})...", end=" ", flush=True)

                            scipy_results = self._benchmark_scipy(
                                problem, x, y, p0, method
                            )
                            self.results.extend(scipy_results)
                            current_run += len(scipy_results)

                            if verbose:
                                avg_time = np.mean(
                                    [r.total_time for r in scipy_results]
                                )
                                print(f"{avg_time:.4f}s")

                            # Calculate speedup
                            nlsq_mean = np.mean([r.total_time for r in nlsq_results])
                            scipy_mean = np.mean([r.total_time for r in scipy_results])
                            speedup = scipy_mean / nlsq_mean if nlsq_mean > 0 else 0

                            for result in nlsq_results:
                                result.speedup_vs_scipy = speedup

                            if verbose:
                                print(f"    Speedup: {speedup:.2f}x")

        return self.results

    def _benchmark_nlsq(
        self,
        problem: BenchmarkProblem,
        x: np.ndarray,
        y: np.ndarray,
        p0: np.ndarray,
        method: str,
        backend: str,
    ) -> list[BenchmarkResult]:
        """Benchmark NLSQ for a specific configuration."""
        results = []

        # Warmup runs
        for _ in range(self.config.warmup_runs):
            with contextlib.suppress(Exception):
                curve_fit(problem.model, x, y, p0=p0, method=method)

        # Actual benchmark runs
        for _ in range(self.config.n_repeats):
            profile_name = f"{problem.name}_{method}_{backend}_{len(x)}"

            start_time = time.perf_counter()

            with self.profiler.profile(profile_name) as metrics:
                try:
                    jit_start = time.perf_counter()
                    popt, _pcov = curve_fit(problem.model, x, y, p0=p0, method=method)
                    jit_time = time.perf_counter() - jit_start

                    success = True
                    final_cost = np.sum((problem.model(x, *popt) - y) ** 2)
                except Exception as e:
                    success = False
                    final_cost = np.inf
                    jit_time = 0

                total_time = time.perf_counter() - start_time

                # Update profiler metrics
                self.profiler.update_current(
                    n_data_points=len(x),
                    n_parameters=problem.n_parameters,
                    success=success,
                    method=method,
                    backend=backend,
                )

            # Get profiler metrics
            profile_metrics = self.profiler.get_metrics(profile_name)[-1]

            result = BenchmarkResult(
                library="nlsq",
                method=method,
                backend=backend,
                problem_size=len(x),
                n_parameters=problem.n_parameters,
                total_time=total_time,
                optimization_time=total_time,  # Approximation
                jit_compile_time=jit_time if _ == 0 else 0,  # Only first run
                n_iterations=profile_metrics.n_iterations,
                n_function_evals=profile_metrics.n_function_evals,
                success=success,
                final_cost=final_cost,
                iterations_per_second=profile_metrics.iterations_per_second(),
                metadata={"problem": problem.name},
            )

            results.append(result)

        return results

    def _benchmark_scipy(
        self,
        problem: BenchmarkProblem,
        x: np.ndarray,
        y: np.ndarray,
        p0: np.ndarray,
        method: str,
    ) -> list[BenchmarkResult]:
        """Benchmark SciPy for a specific configuration."""
        results = []

        # Convert JAX model to NumPy for SciPy
        def numpy_model(x, *params):
            # Convert to JAX, compute, convert back
            x_jax = jnp.array(x)
            result_jax = problem.model(x_jax, *params)
            return np.array(result_jax)

        # Actual benchmark runs (no warmup for SciPy)
        for _ in range(self.config.n_repeats):
            start_time = time.perf_counter()

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _popt, _pcov, infodict, _mesg, ier = scipy_curve_fit(
                        numpy_model,
                        x,
                        y,
                        p0=p0,
                        method=method,
                        full_output=True,
                    )

                success = ier in [1, 2, 3, 4]
                final_cost = np.sum(infodict["fvec"] ** 2) if success else np.inf
                n_function_evals = infodict["nfev"]
            except Exception as e:
                success = False
                final_cost = np.inf
                n_function_evals = 0

            total_time = time.perf_counter() - start_time

            result = BenchmarkResult(
                library="scipy",
                method=method,
                backend="cpu",
                problem_size=len(x),
                n_parameters=problem.n_parameters,
                total_time=total_time,
                optimization_time=total_time,
                n_function_evals=n_function_evals,
                success=success,
                final_cost=final_cost,
                metadata={"problem": problem.name},
            )

            results.append(result)

        return results

    def generate_report(self, detailed: bool = False) -> str:
        """
        Generate benchmark report.

        Parameters
        ----------
        detailed : bool
            Include detailed per-run statistics

        Returns
        -------
        report : str
            Formatted report
        """
        if not self.results:
            return "No benchmark results available."

        lines = [
            "=" * 80,
            "NLSQ Performance Benchmark Report",
            "=" * 80,
            "",
        ]

        # Summary by problem
        for problem in self.problems:
            lines.append(f"\n{problem.name.upper()}")
            lines.append("-" * 80)

            problem_results = [
                r for r in self.results if r.metadata.get("problem") == problem.name
            ]

            for size in self.config.problem_sizes:
                size_results = [r for r in problem_results if r.problem_size == size]
                if not size_results:
                    continue

                lines.append(f"\nProblem Size: {size} points")

                # Group by method
                for method in self.config.methods:
                    method_results = [r for r in size_results if r.method == method]
                    if not method_results:
                        continue

                    # NLSQ results
                    nlsq_results = [r for r in method_results if r.library == "nlsq"]
                    if nlsq_results:
                        nlsq_mean = np.mean([r.total_time for r in nlsq_results])
                        nlsq_std = np.std([r.total_time for r in nlsq_results])
                        success_rate = sum(r.success for r in nlsq_results) / len(
                            nlsq_results
                        )
                        lines.append(
                            f"  NLSQ ({method}):  {nlsq_mean:.4f}s ± {nlsq_std:.4f}s "
                            f"(success: {success_rate:.1%})"
                        )

                    # SciPy results
                    scipy_results = [r for r in method_results if r.library == "scipy"]
                    if scipy_results:
                        scipy_mean = np.mean([r.total_time for r in scipy_results])
                        scipy_std = np.std([r.total_time for r in scipy_results])
                        success_rate = sum(r.success for r in scipy_results) / len(
                            scipy_results
                        )
                        lines.append(
                            f"  SciPy ({method}): {scipy_mean:.4f}s ± {scipy_std:.4f}s "
                            f"(success: {success_rate:.1%})"
                        )

                    # Speedup
                    if nlsq_results and scipy_results:
                        speedup = nlsq_results[0].speedup_vs_scipy
                        lines.append(f"  Speedup: {speedup:.2f}x")

        # Overall statistics
        lines.extend(
            [
                "",
                "",
                "=" * 80,
                "OVERALL STATISTICS",
                "=" * 80,
            ]
        )

        nlsq_times = [r.total_time for r in self.results if r.library == "nlsq"]
        scipy_times = [r.total_time for r in self.results if r.library == "scipy"]

        if nlsq_times:
            lines.append(
                f"NLSQ Average Time:  {np.mean(nlsq_times):.4f}s ± {np.std(nlsq_times):.4f}s"
            )
        if scipy_times:
            lines.append(
                f"SciPy Average Time: {np.mean(scipy_times):.4f}s ± {np.std(scipy_times):.4f}s"
            )

        if nlsq_times and scipy_times:
            overall_speedup = np.mean(scipy_times) / np.mean(nlsq_times)
            lines.append(f"Overall Speedup: {overall_speedup:.2f}x")

        lines.append("=" * 80)

        return "\n".join(lines)

    def save_results(self, output_dir: str) -> None:
        """
        Save benchmark results to directory.

        Parameters
        ----------
        output_dir : str
            Directory to save results
        """
        from pathlib import Path

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save text report
        report = self.generate_report(detailed=True)
        (output_dir / "benchmark_report.txt").write_text(report, encoding="utf-8")

        # Save CSV results
        self._save_csv(output_dir / "benchmark_results.csv")

        # Create dashboard
        dashboard = ProfilingDashboard(self.profiler)
        for profile_name in self.profiler.profiles:
            dashboard.add_profile(profile_name)

        # Save dashboard
        dashboard.save_dashboard(output_dir / "dashboard")

        print(f"Results saved to {output_dir}")

    def _save_csv(self, filepath: str) -> None:
        """Save results to CSV file."""
        from pathlib import Path

        lines = [
            "library,method,backend,problem_size,n_parameters,total_time,"
            "optimization_time,jit_compile_time,n_iterations,n_function_evals,"
            "success,final_cost,speedup_vs_scipy,problem"
        ]

        for r in self.results:
            row = [
                r.library,
                r.method,
                r.backend,
                str(r.problem_size),
                str(r.n_parameters),
                f"{r.total_time:.6f}",
                f"{r.optimization_time:.6f}",
                f"{r.jit_compile_time:.6f}",
                str(r.n_iterations),
                str(r.n_function_evals),
                str(r.success),
                f"{r.final_cost:.6f}",
                f"{r.speedup_vs_scipy:.4f}",
                r.metadata.get("problem", ""),
            ]
            lines.append(",".join(row))

        Path(filepath).write_text("\n".join(lines), encoding="utf-8")


def run_standard_benchmarks(output_dir: str = "./benchmark_results") -> BenchmarkSuite:
    """
    Run standard NLSQ benchmarks.

    Parameters
    ----------
    output_dir : str
        Directory to save results

    Returns
    -------
    suite : BenchmarkSuite
        Benchmark suite with results
    """
    # Configure benchmarks
    config = BenchmarkConfig(
        name="standard",
        problem_sizes=[100, 1000, 10000, 100000],
        n_repeats=5,
        warmup_runs=2,
        methods=["trf", "lm"],
        backends=["cpu"],
        compare_scipy=True,
    )

    # Create suite
    suite = BenchmarkSuite(config)

    # Add standard problems
    suite.add_problem(ExponentialDecayProblem())
    suite.add_problem(GaussianProblem())
    suite.add_problem(PolynomialProblem())
    suite.add_problem(SinusoidalProblem())

    # Run benchmarks
    print("Running NLSQ Standard Benchmarks...")
    print("=" * 80)
    suite.run_benchmarks(verbose=True)

    # Generate report
    print("\n")
    print(suite.generate_report())

    # Save results
    suite.save_results(output_dir)

    return suite


if __name__ == "__main__":
    import sys

    output_dir = sys.argv[1] if len(sys.argv) > 1 else "./benchmark_results"
    suite = run_standard_benchmarks(output_dir)
