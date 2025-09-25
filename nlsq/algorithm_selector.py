"""Automatic algorithm selection for NLSQ optimization.

This module analyzes problem characteristics and automatically selects
the best optimization algorithm and parameters.
"""

from collections.abc import Callable
from inspect import signature

import numpy as np

from nlsq.config import JAXConfig

_jax_config = JAXConfig()


class AlgorithmSelector:
    """Automatically select best algorithm based on problem characteristics.

    This class analyzes the optimization problem and recommends the best
    algorithm, loss function, and parameters based on:
    - Problem size and dimensionality
    - Data characteristics (outliers, noise, conditioning)
    - Memory constraints
    - Convergence requirements
    """

    def __init__(self):
        """Initialize algorithm selector."""
        # Algorithm characteristics
        self.algorithm_properties = {
            "trf": {
                "memory_factor": 2.5,  # Relative memory usage
                "speed": "fast",
                "robust": True,
                "handles_bounds": True,
                "good_for_large": True,
                "good_for_ill_conditioned": False,
            },
            "lm": {
                "memory_factor": 1.5,
                "speed": "medium",
                "robust": False,
                "handles_bounds": False,
                "good_for_large": False,
                "good_for_ill_conditioned": True,
            },
            "dogbox": {
                "memory_factor": 2.0,
                "speed": "medium",
                "robust": False,
                "handles_bounds": True,
                "good_for_large": False,
                "good_for_ill_conditioned": False,
            },
        }

        # Loss function characteristics
        self.loss_properties = {
            "linear": {"robust": False, "smooth": True},
            "huber": {"robust": True, "smooth": True},
            "soft_l1": {"robust": True, "smooth": True},
            "cauchy": {"robust": True, "smooth": True},
            "arctan": {"robust": True, "smooth": True},
        }

    def analyze_problem(
        self,
        f: Callable,
        xdata: np.ndarray,
        ydata: np.ndarray,
        p0: np.ndarray | None = None,
        bounds: tuple | None = None,
        memory_limit_gb: float | None = None,
    ) -> dict:
        """Analyze problem characteristics.

        Parameters
        ----------
        f : callable
            Model function to fit
        xdata : np.ndarray
            Independent variable data
        ydata : np.ndarray
            Dependent variable data
        p0 : np.ndarray, optional
            Initial parameter guess
        bounds : tuple, optional
            Parameter bounds
        memory_limit_gb : float, optional
            Memory constraint in GB

        Returns
        -------
        analysis : dict
            Problem characteristics and statistics
        """
        # Convert to numpy for analysis
        xdata = np.asarray(xdata)
        ydata = np.asarray(ydata)

        n_points = len(xdata) if xdata.ndim == 1 else xdata.shape[0]

        # Estimate number of parameters
        n_params = self._estimate_n_params(f, p0)

        # Basic statistics
        analysis = {
            "n_points": n_points,
            "n_params": n_params,
            "overdetermination_ratio": n_points / n_params if n_params > 0 else np.inf,
        }

        # Data characteristics
        analysis.update(self._analyze_data(xdata, ydata))

        # Problem size classification
        if n_points <= 100:
            analysis["size_class"] = "small"
        elif n_points < 10000:
            analysis["size_class"] = "medium"
        elif n_points < 1000000:
            analysis["size_class"] = "large"
        else:
            analysis["size_class"] = "very_large"

        # Conditioning estimate
        analysis["condition_estimate"] = self._estimate_conditioning(xdata, n_params)

        # Memory requirements
        if memory_limit_gb is not None:
            analysis["memory_constrained"] = self._check_memory_constraints(
                n_points, n_params, memory_limit_gb
            )
        else:
            analysis["memory_constrained"] = False

        # Bounds analysis
        analysis["has_bounds"] = bounds is not None and bounds != (-np.inf, np.inf)

        # Parameter scale analysis
        if p0 is not None:
            analysis["param_scale_range"] = np.ptp(np.log10(np.abs(p0) + 1e-10))
        else:
            analysis["param_scale_range"] = 0

        return analysis

    def _estimate_n_params(self, f: Callable, p0: np.ndarray | None) -> int:
        """Estimate number of parameters.

        Parameters
        ----------
        f : callable
            Model function
        p0 : np.ndarray, optional
            Initial guess

        Returns
        -------
        n_params : int
            Estimated number of parameters
        """
        if p0 is not None:
            return len(p0)

        try:
            sig = signature(f)
            # Count parameters excluding x
            return len(sig.parameters) - 1
        except Exception:
            # Default guess
            return 3

    def _analyze_data(self, xdata: np.ndarray, ydata: np.ndarray) -> dict:
        """Analyze data characteristics.

        Parameters
        ----------
        xdata : np.ndarray
            Independent variable
        ydata : np.ndarray
            Dependent variable

        Returns
        -------
        characteristics : dict
            Data characteristics
        """
        results = {}

        # Check for outliers using IQR method
        q1, q3 = np.percentile(ydata, [25, 75])
        iqr = q3 - q1
        outlier_bounds = (q1 - 3 * iqr, q3 + 3 * iqr)
        n_outliers = np.sum((ydata < outlier_bounds[0]) | (ydata > outlier_bounds[1]))
        results["outlier_fraction"] = n_outliers / len(ydata)
        results["has_outliers"] = results["outlier_fraction"] > 0.01

        # Check for noise level (using local variation)
        if len(ydata) > 10:
            # Estimate noise from differences
            diff = np.diff(ydata)
            noise_estimate = np.median(np.abs(diff))
            signal_range = np.ptp(ydata)
            results["snr_estimate"] = signal_range / (noise_estimate + 1e-10)
            results["is_noisy"] = results["snr_estimate"] < 10
        else:
            results["snr_estimate"] = np.inf
            results["is_noisy"] = False

        # Data range
        if xdata.ndim == 1:
            results["x_range"] = np.ptp(xdata)
            results["x_scale"] = np.log10(results["x_range"] + 1e-10)

            # Check for uniform spacing
            if len(xdata) > 2:
                spacing = np.diff(xdata)
                results["x_uniform"] = np.std(spacing) / np.mean(spacing) < 0.01
            else:
                results["x_uniform"] = True
        else:
            results["x_range"] = 0
            results["x_scale"] = 0
            results["x_uniform"] = False

        results["y_range"] = np.ptp(ydata)
        results["y_scale"] = np.log10(results["y_range"] + 1e-10)

        # Check for zeros or near-zeros
        results["has_zeros"] = np.any(np.abs(ydata) < 1e-10)

        return results

    def _estimate_conditioning(self, xdata: np.ndarray, n_params: int) -> float:
        """Estimate problem conditioning.

        Parameters
        ----------
        xdata : np.ndarray
            Independent variable
        n_params : int
            Number of parameters

        Returns
        -------
        condition_estimate : float
            Estimated condition number
        """
        if xdata.ndim > 1:
            # Multi-dimensional x
            return 1.0

        # Build vandermonde-like matrix for polynomial features
        n_features = min(n_params, 5)
        n_samples = min(len(xdata), 1000)  # Sample for large datasets

        if n_samples < n_features:
            return np.inf

        # Sample data
        indices = np.linspace(0, len(xdata) - 1, n_samples, dtype=int)
        x_sample = xdata[indices]

        # Normalize to [0, 1]
        x_min, x_max = x_sample.min(), x_sample.max()
        if x_max - x_min < 1e-10:
            return np.inf

        x_norm = (x_sample - x_min) / (x_max - x_min)

        # Build feature matrix
        X = np.vstack([x_norm**i for i in range(n_features)]).T

        try:
            cond = np.linalg.cond(X)
            return cond
        except Exception:
            return np.inf

    def _check_memory_constraints(
        self, n_points: int, n_params: int, memory_limit_gb: float
    ) -> bool:
        """Check if problem fits in memory limit.

        Parameters
        ----------
        n_points : int
            Number of data points
        n_params : int
            Number of parameters
        memory_limit_gb : float
            Memory limit in GB

        Returns
        -------
        constrained : bool
            Whether memory is constrained
        """
        # Estimate memory for Jacobian and working arrays
        memory_needed_gb = (8 * n_points * n_params * 3) / 1e9  # Factor of 3 for safety

        return memory_needed_gb > memory_limit_gb

    def select_algorithm(
        self, problem_analysis: dict, user_preferences: dict | None = None
    ) -> dict:
        """Select best algorithm based on problem analysis.

        Parameters
        ----------
        problem_analysis : dict
            Results from analyze_problem
        user_preferences : dict, optional
            User preferences (e.g., prioritize speed vs accuracy)

        Returns
        -------
        recommendations : dict
            Recommended algorithm and parameters
        """
        recommendations = {
            "algorithm": "trf",  # Default
            "loss": "linear",
            "use_bounds": problem_analysis.get("has_bounds", False),
            "max_nfev": None,
            "ftol": 1e-8,
            "xtol": 1e-8,
            "gtol": 1e-8,
            "x_scale": "jac",
            "tr_solver": None,
            "verbose": 0,
        }

        # Apply user preferences
        if user_preferences:
            if "prioritize" in user_preferences:
                priority = user_preferences["prioritize"]
                if priority == "speed":
                    recommendations["ftol"] = 1e-6
                    recommendations["xtol"] = 1e-6
                    recommendations["max_nfev"] = 100
                elif priority == "accuracy":
                    recommendations["ftol"] = 1e-10
                    recommendations["xtol"] = 1e-10
                    recommendations["gtol"] = 1e-10

        # Algorithm selection logic
        n_points = problem_analysis["n_points"]
        n_params = problem_analysis["n_params"]
        has_bounds = problem_analysis.get("has_bounds", False)

        # NOTE: Currently NLSQ only implements TRF algorithm
        # In the future, we can add LM and dogbox support
        # For now, always use TRF but adjust parameters based on problem
        recommendations["algorithm"] = "trf"

        # Adjust TRF parameters based on problem characteristics
        if problem_analysis.get("condition_estimate", 1) > 1e10:
            # Ill-conditioned: use more conservative tolerances
            recommendations["ftol"] = 1e-6
            recommendations["xtol"] = 1e-6

        if n_points > 100000:
            # Large problem: use iterative solver
            if n_points > 1000000:
                recommendations["tr_solver"] = "lsmr"  # Iterative solver

        elif n_params > 100:
            # Many parameters: TRF with iterative solver
            recommendations["tr_solver"] = "lsmr"

        # Loss function selection
        if problem_analysis.get("has_outliers", False):
            outlier_fraction = problem_analysis.get("outlier_fraction", 0)

            if outlier_fraction > 0.1:
                recommendations["loss"] = "cauchy"  # Very robust
            elif outlier_fraction > 0.05:
                recommendations["loss"] = "huber"  # Moderately robust
            elif outlier_fraction > 0.01:
                recommendations["loss"] = "soft_l1"  # Slightly robust
            else:
                recommendations["loss"] = "linear"

        # Adjust for noisy data
        if problem_analysis.get("is_noisy", False):
            # Relax tolerances for noisy data
            recommendations["ftol"] = max(recommendations["ftol"], 1e-6)
            recommendations["xtol"] = max(recommendations["xtol"], 1e-6)

        # Memory-constrained adjustments
        if problem_analysis.get("memory_constrained", False):
            recommendations["algorithm"] = "trf"
            recommendations["tr_solver"] = "lsmr"  # Memory-efficient iterative solver

        # Adjust max iterations based on problem size
        if recommendations["max_nfev"] is None:
            if n_points > 1000000:
                recommendations["max_nfev"] = 50
            elif n_points > 100000:
                recommendations["max_nfev"] = 100
            elif n_points > 10000:
                recommendations["max_nfev"] = 200
            else:
                recommendations["max_nfev"] = None  # No limit

        # X-scale recommendation
        if problem_analysis.get("param_scale_range", 0) > 3:
            # Parameters vary over many orders of magnitude
            recommendations["x_scale"] = "jac"
        else:
            recommendations["x_scale"] = 1.0

        return recommendations

    def get_algorithm_explanation(self, recommendations: dict) -> str:
        """Get human-readable explanation of algorithm choice.

        Parameters
        ----------
        recommendations : dict
            Algorithm recommendations

        Returns
        -------
        explanation : str
            Explanation of the choices
        """
        explanation = []

        # Algorithm choice
        alg = recommendations["algorithm"]
        if alg == "trf":
            explanation.append(
                "Trust Region Reflective (TRF) algorithm selected (currently the only supported algorithm in NLSQ)"
            )
        # Future support for other algorithms
        # elif alg == 'lm':
        #     explanation.append("Levenberg-Marquardt (LM) algorithm selected for good convergence properties")
        # elif alg == 'dogbox':
        #     explanation.append("Dogbox algorithm selected for bounded optimization")

        # Loss function
        loss = recommendations["loss"]
        if loss != "linear":
            explanation.append(f"Using {loss} loss function for outlier robustness")

        # Tolerances
        if recommendations["ftol"] >= 1e-6:
            explanation.append("Relaxed tolerances for faster convergence")
        elif recommendations["ftol"] <= 1e-10:
            explanation.append("Tight tolerances for high accuracy")

        # Solver
        if recommendations.get("tr_solver") == "lsmr":
            explanation.append("Using iterative solver for memory efficiency")

        # Iterations
        if recommendations["max_nfev"] is not None:
            explanation.append(f"Limited to {recommendations['max_nfev']} iterations")

        return "\n".join(explanation)


# Global selector instance
_algorithm_selector = AlgorithmSelector()


def auto_select_algorithm(
    f: Callable,
    xdata: np.ndarray,
    ydata: np.ndarray,
    p0: np.ndarray | None = None,
    bounds: tuple | None = None,
    memory_limit_gb: float | None = None,
    user_preferences: dict | None = None,
) -> dict:
    """Automatically select best optimization algorithm.

    Parameters
    ----------
    f : callable
        Model function
    xdata : np.ndarray
        Independent variable
    ydata : np.ndarray
        Dependent variable
    p0 : np.ndarray, optional
        Initial guess
    bounds : tuple, optional
        Parameter bounds
    memory_limit_gb : float, optional
        Memory limit
    user_preferences : dict, optional
        User preferences

    Returns
    -------
    recommendations : dict
        Algorithm recommendations
    """
    analysis = _algorithm_selector.analyze_problem(
        f, xdata, ydata, p0, bounds, memory_limit_gb
    )

    recommendations = _algorithm_selector.select_algorithm(analysis, user_preferences)

    return recommendations
