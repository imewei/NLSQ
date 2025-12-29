"""Optimization diagnostics and monitoring for NLSQ.

This module provides real-time monitoring, convergence detection,
and diagnostic reporting for optimization processes.
"""

import time
import warnings
from collections import deque
from typing import Any

import numpy as np

from nlsq.config import JAXConfig

_jax_config = JAXConfig()


try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    warnings.warn(
        "psutil not installed, memory monitoring will be limited", UserWarning
    )


class ConvergenceMonitor:
    """Monitor convergence patterns and detect problems.

    Detects:
    - Oscillation in parameter or cost values
    - Stagnation (no progress)
    - Divergence (increasing cost)
    - Slow convergence
    - Numerical instability
    """

    def __init__(self, window_size: int = 10, sensitivity: float = 1.0):
        """Initialize convergence monitor.

        Parameters
        ----------
        window_size : int
            Size of sliding window for pattern detection
        sensitivity : float
            Sensitivity factor (1.0 = normal, <1 = less sensitive, >1 = more sensitive)
        """
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.cost_history: deque[float] = deque(maxlen=window_size)
        self.param_history: deque[np.ndarray] = deque(maxlen=window_size)
        self.gradient_history: deque[float] = deque(maxlen=window_size)
        self.step_size_history: deque[float] = deque(maxlen=window_size)

        # Pattern detection thresholds
        self.oscillation_threshold = 0.7 * sensitivity
        self.stagnation_threshold = 1e-10 * sensitivity
        self.divergence_threshold = 1.5 / sensitivity

    def update(
        self,
        cost: float,
        params: np.ndarray,
        gradient: np.ndarray | None = None,
        step_size: float | None = None,
    ):
        """Update monitor with new iteration data.

        Parameters
        ----------
        cost : float
            Current cost value
        params : np.ndarray
            Current parameter values
        gradient : np.ndarray, optional
            Current gradient
        step_size : float, optional
            Step size taken
        """
        self.cost_history.append(cost)
        self.param_history.append(params.copy())

        if gradient is not None:
            grad_norm = np.linalg.norm(gradient)
            self.gradient_history.append(float(grad_norm))

        if step_size is not None:
            self.step_size_history.append(step_size)

    def add_iteration(
        self,
        cost: float,
        params: np.ndarray,
        gradient: np.ndarray | None = None,
        step_size: float | None = None,
    ):
        """Alias for update() method for backward compatibility.

        Parameters
        ----------
        cost : float
            Current cost value
        params : np.ndarray
            Current parameter values
        gradient : np.ndarray, optional
            Current gradient
        step_size : float, optional
            Step size taken
        """
        self.update(cost, params, gradient, step_size)

    def detect_oscillation(self) -> tuple[bool, float]:
        """Detect oscillation in optimization.

        Returns
        -------
        is_oscillating : bool
            Whether oscillation is detected
        oscillation_score : float
            Oscillation severity score (0-1)
        """
        if len(self.cost_history) < self.window_size:
            return False, 0.0

        costs = np.array(self.cost_history)

        # Check for alternating pattern
        differences = np.diff(costs)
        sign_changes = np.sum(np.diff(np.sign(differences)) != 0)
        oscillation_score = sign_changes / (len(differences) - 1)

        # Also check parameter oscillation
        if len(self.param_history) >= 3:
            params = np.array(self.param_history)
            param_diffs = np.diff(params, axis=0)
            param_sign_changes = np.mean(
                [
                    np.sum(np.diff(np.sign(param_diffs[:, i])) != 0)
                    / max(1, len(param_diffs) - 1)
                    for i in range(params.shape[1])
                ]
            )
            oscillation_score = max(oscillation_score, param_sign_changes)

        return oscillation_score > self.oscillation_threshold, oscillation_score

    def detect_stagnation(self) -> tuple[bool, float]:
        """Detect stagnation in optimization.

        Returns
        -------
        is_stagnant : bool
            Whether stagnation is detected
        stagnation_score : float
            Stagnation severity score
        """
        if len(self.cost_history) < min(5, self.window_size):
            return False, 0.0

        recent_costs = list(self.cost_history)[-5:]
        cost_variance = float(np.var(recent_costs))
        cost_mean = float(np.mean(recent_costs))

        if cost_mean == 0:
            relative_variance = cost_variance
        else:
            relative_variance = cost_variance / (abs(cost_mean) + 1e-10)

        # Check gradient stagnation
        if len(self.gradient_history) >= 3:
            recent_grads = list(self.gradient_history)[-3:]
            grad_stagnation = (
                float(np.mean(recent_grads)) < self.stagnation_threshold * 10
            )
        else:
            grad_stagnation = False

        is_stagnant = relative_variance < self.stagnation_threshold or grad_stagnation
        stagnation_score = 1.0 - min(1.0, relative_variance / self.stagnation_threshold)

        return is_stagnant, stagnation_score

    def detect_divergence(self) -> tuple[bool, float]:
        """Detect divergence in optimization.

        Returns
        -------
        is_diverging : bool
            Whether divergence is detected
        divergence_score : float
            Divergence severity score
        """
        if len(self.cost_history) < 3:
            return False, 0.0

        costs = np.array(self.cost_history)

        # Check if cost is increasing
        if len(costs) >= 5:
            recent = costs[-5:]
            older = costs[-10:-5] if len(costs) >= 10 else costs[:5]

            mean_recent = np.mean(recent)
            mean_older = np.mean(older)

            if mean_older > 0:
                divergence_ratio = mean_recent / mean_older
                is_diverging = divergence_ratio > self.divergence_threshold
                divergence_score = min(
                    1.0, (divergence_ratio - 1.0) / self.divergence_threshold
                )
            else:
                is_diverging = mean_recent > mean_older * self.divergence_threshold
                divergence_score = 0.5 if is_diverging else 0.0
        else:
            is_diverging = costs[-1] > costs[0] * self.divergence_threshold
            divergence_score = 0.5 if is_diverging else 0.0

        return is_diverging, divergence_score

    def get_convergence_rate(self) -> float | None:
        """Estimate convergence rate.

        Returns
        -------
        rate : float or None
            Convergence rate (negative = diverging, positive = converging)
        """
        if len(self.cost_history) < 3:
            return None

        costs = np.array(self.cost_history)

        # Fit exponential decay: cost = a * exp(-rate * iteration)
        iterations = np.arange(len(costs))

        # Use log-linear regression for stability
        with np.errstate(divide="ignore", invalid="ignore"):
            log_costs = np.log(np.abs(costs) + 1e-10)
            if np.all(np.isfinite(log_costs)):
                # Simple linear regression
                A = np.vstack([iterations, np.ones(len(iterations))]).T
                rate, _ = np.linalg.lstsq(A, log_costs, rcond=None)[0]
                return -rate  # Negative because we want positive rate for convergence

        return None


class OptimizationDiagnostics:
    """Comprehensive optimization diagnostics and reporting.

    Tracks:
    - Iteration data (parameters, cost, gradients)
    - Convergence metrics
    - Memory usage
    - Timing information
    - Numerical stability indicators
    """

    def __init__(self, enable_plotting: bool = False, verbosity: int = 1):
        """Initialize diagnostics system.

        Parameters
        ----------
        enable_plotting : bool
            Whether to enable real-time plotting (requires matplotlib)
        verbosity : int
            Verbosity level:
            - 0: Minimal diagnostics (no condition number computation)
            - 1: Normal (cheap 1-norm condition estimate, O(nm))
            - 2: Detailed (full SVD condition number, O(mn²))
        """
        self.iteration_data: list[dict[str, Any]] = []
        self.convergence_monitor = ConvergenceMonitor()
        self.start_time: float | None = None
        self.enable_plotting = enable_plotting
        self.verbosity = verbosity

        # Problem detection
        self.warnings_issued: list[str] = []
        self.numerical_issues: list[str] = []

        # Performance metrics
        self.function_eval_count = 0
        self.jacobian_eval_count = 0

        # Memory tracking
        self.peak_memory: float = 0.0
        self.initial_memory: float = self._get_memory_usage()

        # Problem metadata (initialized by start_optimization)
        self.n_params: int | None = None
        self.n_data: int | None = None
        self.method: str | None = None
        self.loss: str | None = None
        self.initial_params: np.ndarray | None = None
        self.problem_name: str | None = None

    def start_optimization(
        self,
        x0: np.ndarray | None = None,
        problem_name: str = "optimization",
        *,
        n_params: int | None = None,
        n_data: int | None = None,
        method: str | None = None,
        loss: str | None = None,
    ):
        """Initialize diagnostics for new optimization.

        Parameters
        ----------
        x0 : np.ndarray, optional
            Initial parameters (legacy API)
        problem_name : str
            Name for this optimization problem
        n_params : int, optional
            Number of parameters (new API from LeastSquares)
        n_data : int, optional
            Number of data points (new API from LeastSquares)
        method : str, optional
            Optimization method (new API from LeastSquares)
        loss : str, optional
            Loss function name (new API from LeastSquares)
        """
        self.problem_name = problem_name
        self.start_time = time.time()
        self.iteration_data = []
        self.warnings_issued = []
        self.numerical_issues = []
        self.function_eval_count = 0
        self.jacobian_eval_count = 0

        # Store problem metadata (new API)
        self.n_params = n_params
        self.n_data = n_data
        self.method = method
        self.loss = loss

        # Handle both legacy (x0) and new (n_params) API
        if x0 is not None:
            self.initial_params = x0.copy()
        elif n_params is not None:
            # Create placeholder for initial params when only n_params provided
            self.initial_params = np.zeros(n_params)
        else:
            self.initial_params = None

        self.initial_memory = self._get_memory_usage()

    def record_iteration(
        self,
        iteration: int,
        x: np.ndarray,
        cost: float,
        gradient: np.ndarray | None = None,
        jacobian: np.ndarray | None = None,
        step_size: float | None = None,
        method_info: dict | None = None,
    ):
        """Record data for current iteration.

        Parameters
        ----------
        iteration : int
            Iteration number
        x : np.ndarray
            Current parameters
        cost : float
            Current cost value
        gradient : np.ndarray, optional
            Current gradient
        jacobian : np.ndarray, optional
            Current Jacobian matrix
        step_size : float, optional
            Step size taken
        method_info : dict, optional
            Algorithm-specific information
        """
        current_time = time.time()
        elapsed = current_time - self.start_time if self.start_time else 0

        # Basic data
        data = {
            "iteration": iteration,
            "parameters": x.copy(),
            "cost": cost,
            "timestamp": current_time,
            "elapsed_time": elapsed,
            "memory_usage_mb": self._get_memory_usage(),
        }

        # Gradient information
        if gradient is not None:
            grad_norm = np.linalg.norm(gradient)
            data["gradient_norm"] = grad_norm
            data["gradient_max"] = np.max(np.abs(gradient))

            # Check for numerical issues
            if not np.all(np.isfinite(gradient)):
                self.numerical_issues.append(
                    f"Iteration {iteration}: Non-finite gradient"
                )
            elif grad_norm > 1e10:
                self.numerical_issues.append(
                    f"Iteration {iteration}: Extremely large gradient"
                )

        # Jacobian information
        if jacobian is not None:
            self.jacobian_eval_count += 1
            # Only compute condition number if verbosity > 0
            if self.verbosity > 0:
                try:
                    if self.verbosity >= 2:
                        # Full SVD condition number (expensive: O(mn²))
                        svd_vals = np.linalg.svdvals(jacobian)
                        condition_number = svd_vals[0] / (svd_vals[-1] + 1e-10)
                    else:
                        # Cheap 1-norm condition estimate (O(nm), ~50-90% faster)
                        # Uses matrix norms instead of full SVD decomposition
                        condition_number = np.linalg.cond(jacobian, p=1)
                    data["jacobian_condition"] = condition_number

                    if condition_number > 1e12:
                        self.numerical_issues.append(
                            f"Iteration {iteration}: Ill-conditioned Jacobian (cond={condition_number:.2e})"
                        )
                except (np.linalg.LinAlgError, ValueError):
                    data["jacobian_condition"] = np.inf

        # Step information
        if step_size is not None:
            data["step_size"] = step_size

        # Method-specific info
        if method_info:
            data["method_info"] = method_info

        self.iteration_data.append(data)
        self.function_eval_count += 1

        # Update convergence monitor
        self.convergence_monitor.update(cost, x, gradient, step_size)

        # Check for convergence issues
        self._check_convergence_health(iteration)

        # Update peak memory
        current_memory = self._get_memory_usage()
        self.peak_memory = max(self.peak_memory, current_memory)

    def record_event(self, event_type: str, data: dict[str, Any] | None = None):
        """Record a special event during optimization.

        Parameters
        ----------
        event_type : str
            Type of event (e.g., 'recovery_success', 'recovery_failed')
        data : dict, optional
            Additional event data
        """
        event = {
            "type": event_type,
            "iteration": len(self.iteration_data),
            "timestamp": time.time() - self.start_time if self.start_time else 0,
            "data": data or {},
        }

        # Store in warnings if it's a warning/error event
        if "failed" in event_type or "error" in event_type:
            self.warnings_issued.append(f"{event_type}: {data}")

    def _check_convergence_health(self, iteration: int):
        """Check for convergence problems and issue warnings.

        Parameters
        ----------
        iteration : int
            Current iteration number
        """
        # Only check every few iterations to avoid spam
        if iteration % 5 != 0 or iteration < 10:
            return

        # Check oscillation
        is_oscillating, osc_score = self.convergence_monitor.detect_oscillation()
        if is_oscillating and "oscillation" not in self.warnings_issued:
            warnings.warn(
                f"Optimization may be oscillating (score={osc_score:.2f}). "
                "Consider reducing step size or changing algorithm.",
                RuntimeWarning,
            )
            self.warnings_issued.append("oscillation")

        # Check stagnation
        is_stagnant, stag_score = self.convergence_monitor.detect_stagnation()
        if is_stagnant and "stagnation" not in self.warnings_issued:
            warnings.warn(
                f"Optimization may be stagnant (score={stag_score:.2f}). "
                "Consider relaxing tolerances or perturbing parameters.",
                RuntimeWarning,
            )
            self.warnings_issued.append("stagnation")

        # Check divergence
        is_diverging, div_score = self.convergence_monitor.detect_divergence()
        if is_diverging and "divergence" not in self.warnings_issued:
            warnings.warn(
                f"Optimization may be diverging (score={div_score:.2f}). "
                "Consider better initial guess or different algorithm.",
                RuntimeWarning,
            )
            self.warnings_issued.append("divergence")

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB.

        Returns
        -------
        memory_mb : float
            Memory usage in megabytes
        """
        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                return process.memory_info().rss / 1024 / 1024
            except (OSError, AttributeError):
                return 0.0
        return 0.0

    def get_summary_statistics(self) -> dict:
        """Get summary statistics for the optimization.

        Returns
        -------
        stats : dict
            Summary statistics
        """
        if not self.iteration_data:
            return {}

        costs = [d["cost"] for d in self.iteration_data]

        stats = {
            "total_iterations": len(self.iteration_data),
            "function_evaluations": self.function_eval_count,
            "jacobian_evaluations": self.jacobian_eval_count,
            "initial_cost": costs[0],
            "final_cost": costs[-1],
            "cost_reduction": costs[0] - costs[-1],
            "relative_cost_reduction": (costs[0] - costs[-1]) / (abs(costs[0]) + 1e-10),
            "min_cost": min(costs),
            "max_cost": max(costs),
        }

        # Timing
        if self.start_time and self.iteration_data:
            total_time = self.iteration_data[-1]["elapsed_time"]
            stats["total_time_seconds"] = total_time
            stats["time_per_iteration"] = total_time / len(self.iteration_data)

        # Memory
        stats["peak_memory_mb"] = self.peak_memory
        stats["memory_increase_mb"] = self.peak_memory - self.initial_memory

        # Convergence rate
        conv_rate = self.convergence_monitor.get_convergence_rate()
        if conv_rate is not None:
            stats["convergence_rate"] = conv_rate

        # Gradient info
        if any("gradient_norm" in d for d in self.iteration_data):
            grad_norms = [
                d["gradient_norm"] for d in self.iteration_data if "gradient_norm" in d
            ]
            stats["initial_gradient_norm"] = grad_norms[0]
            stats["final_gradient_norm"] = grad_norms[-1]
            stats["min_gradient_norm"] = min(grad_norms)

        # Condition number
        if any("jacobian_condition" in d for d in self.iteration_data):
            cond_numbers = [
                d["jacobian_condition"]
                for d in self.iteration_data
                if "jacobian_condition" in d
            ]
            stats["max_condition_number"] = max(cond_numbers)
            stats["mean_condition_number"] = np.mean(cond_numbers)

        # Problems detected
        stats["warnings_issued"] = self.warnings_issued.copy()
        stats["numerical_issues"] = len(self.numerical_issues)

        return stats

    def generate_report(self, verbose: bool = True) -> str:
        """Generate human-readable optimization report.

        Parameters
        ----------
        verbose : bool
            Whether to include detailed information

        Returns
        -------
        report : str
            Formatted report
        """
        stats = self.get_summary_statistics()

        if not stats:
            return "No optimization data available"

        lines = []
        lines.append("=" * 60)
        lines.append(f"Optimization Report: {getattr(self, 'problem_name', 'Unknown')}")
        lines.append("=" * 60)

        # Basic metrics
        lines.append("\n--- Performance Metrics ---")
        lines.append(f"Total iterations: {stats.get('total_iterations', 0)}")
        lines.append(f"Function evaluations: {stats.get('function_evaluations', 0)}")
        lines.append(f"Jacobian evaluations: {stats.get('jacobian_evaluations', 0)}")

        # Cost reduction
        lines.append("\n--- Cost Reduction ---")
        lines.append(f"Initial cost: {stats.get('initial_cost', 0):.6e}")
        lines.append(f"Final cost: {stats.get('final_cost', 0):.6e}")
        lines.append(f"Absolute reduction: {stats.get('cost_reduction', 0):.6e}")
        lines.append(
            f"Relative reduction: {stats.get('relative_cost_reduction', 0) * 100:.2f}%"
        )

        # Convergence
        lines.append("\n--- Convergence ---")
        if "convergence_rate" in stats:
            rate = stats["convergence_rate"]
            if rate > 0:
                lines.append(f"Convergence rate: {rate:.4f} (converging)")
            else:
                lines.append(f"Convergence rate: {rate:.4f} (diverging)")

        if "final_gradient_norm" in stats:
            lines.append(f"Initial gradient norm: {stats['initial_gradient_norm']:.6e}")
            lines.append(f"Final gradient norm: {stats['final_gradient_norm']:.6e}")

        # Numerical stability
        if "max_condition_number" in stats:
            lines.append("\n--- Numerical Stability ---")
            lines.append(f"Max condition number: {stats['max_condition_number']:.2e}")
            lines.append(f"Mean condition number: {stats['mean_condition_number']:.2e}")

        # Timing
        if "total_time_seconds" in stats:
            lines.append("\n--- Timing ---")
            lines.append(f"Total time: {stats['total_time_seconds']:.2f} seconds")
            lines.append(
                f"Time per iteration: {stats['time_per_iteration'] * 1000:.1f} ms"
            )

        # Memory
        lines.append("\n--- Memory Usage ---")
        lines.append(f"Peak memory: {stats.get('peak_memory_mb', 0):.1f} MB")
        lines.append(f"Memory increase: {stats.get('memory_increase_mb', 0):.1f} MB")

        # Problems
        if verbose:
            if stats.get("warnings_issued"):
                lines.append("\n--- Warnings ---")
                lines.extend(f"  • {warning}" for warning in stats["warnings_issued"])

            if self.numerical_issues:
                lines.append("\n--- Numerical Issues ---")
                lines.extend(
                    f"  • {issue}" for issue in self.numerical_issues[:5]
                )  # Show first 5
                if len(self.numerical_issues) > 5:
                    lines.append(f"  ... and {len(self.numerical_issues) - 5} more")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)

    def plot_convergence(self, save_path: str | None = None):
        """Plot convergence history.

        Parameters
        ----------
        save_path : str, optional
            Path to save plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not available, cannot plot convergence")
            return

        if not self.iteration_data:
            return

        _fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        iterations = [d["iteration"] for d in self.iteration_data]

        # Cost history
        costs = [d["cost"] for d in self.iteration_data]
        axes[0, 0].semilogy(iterations, costs, "b-")
        axes[0, 0].set_xlabel("Iteration")
        axes[0, 0].set_ylabel("Cost")
        axes[0, 0].set_title("Cost History")
        axes[0, 0].grid(True)

        # Gradient norm
        if any("gradient_norm" in d for d in self.iteration_data):
            grad_norms = [d.get("gradient_norm", np.nan) for d in self.iteration_data]
            axes[0, 1].semilogy(iterations, grad_norms, "r-")
            axes[0, 1].set_xlabel("Iteration")
            axes[0, 1].set_ylabel("Gradient Norm")
            axes[0, 1].set_title("Gradient Norm History")
            axes[0, 1].grid(True)

        # Step size
        if any("step_size" in d for d in self.iteration_data):
            step_sizes = [d.get("step_size", np.nan) for d in self.iteration_data]
            axes[1, 0].plot(iterations, step_sizes, "g-")
            axes[1, 0].set_xlabel("Iteration")
            axes[1, 0].set_ylabel("Step Size")
            axes[1, 0].set_title("Step Size History")
            axes[1, 0].grid(True)

        # Memory usage
        memory = [d["memory_usage_mb"] for d in self.iteration_data]
        axes[1, 1].plot(iterations, memory, "m-")
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_ylabel("Memory (MB)")
        axes[1, 1].set_title("Memory Usage")
        axes[1, 1].grid(True)

        plt.suptitle(
            f"Optimization Convergence: {getattr(self, 'problem_name', 'Unknown')}"
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


# Global diagnostics instance
_global_diagnostics: OptimizationDiagnostics | None = None


def get_diagnostics(verbosity: int | None = None) -> OptimizationDiagnostics:
    """Get global diagnostics instance.

    Parameters
    ----------
    verbosity : int, optional
        Verbosity level for new instance (0=minimal, 1=normal, 2=detailed).
        Only used when creating a new instance.

    Returns
    -------
    diagnostics : OptimizationDiagnostics
        Global diagnostics instance
    """
    global _global_diagnostics  # noqa: PLW0603
    if _global_diagnostics is None:
        _global_diagnostics = OptimizationDiagnostics(
            verbosity=verbosity if verbosity is not None else 1
        )
    return _global_diagnostics


def reset_diagnostics(verbosity: int = 1):
    """Reset global diagnostics.

    Parameters
    ----------
    verbosity : int
        Verbosity level (0=minimal, 1=normal, 2=detailed)
    """
    global _global_diagnostics  # noqa: PLW0603
    _global_diagnostics = OptimizationDiagnostics(verbosity=verbosity)
