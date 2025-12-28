"""Recovery strategies for optimization failures.

This module provides automatic recovery mechanisms for handling
optimization failures with multiple retry strategies.
"""

import warnings
from collections.abc import Callable

import numpy as np

from nlsq.config import JAXConfig

_jax_config = JAXConfig()

import jax.numpy as jnp

from nlsq.stability.guard import NumericalStabilityGuard
from nlsq.utils.diagnostics import OptimizationDiagnostics


class OptimizationRecovery:
    """Automatic recovery from optimization failures.

    This class provides multiple recovery strategies for handling
    optimization failures including:
    - Parameter perturbation
    - Algorithm switching
    - Regularization adjustment
    - Problem reformulation
    - Multi-start optimization

    Attributes
    ----------
    max_retries : int
        Maximum number of recovery attempts
    strategies : list
        List of recovery strategies to try
    diagnostics : nlsq.diagnostics.OptimizationDiagnostics
        Diagnostics collector for monitoring
    stability_guard : nlsq.stability.NumericalStabilityGuard
        Numerical stability checker
    """

    def __init__(self, max_retries: int = 3, enable_diagnostics: bool = True):
        """Initialize recovery system.

        Parameters
        ----------
        max_retries : int
            Maximum recovery attempts
        enable_diagnostics : bool
            Enable diagnostic collection
        """
        self.max_retries = max_retries
        self.enable_diagnostics = enable_diagnostics

        if enable_diagnostics:
            self.diagnostics = OptimizationDiagnostics()

        self.stability_guard = NumericalStabilityGuard()

        # Recovery strategies in order of preference
        self.strategies = [
            self._perturb_parameters,
            self._switch_algorithm,
            self._adjust_regularization,
            self._reformulate_problem,
            self._multi_start,
        ]

        # Track recovery history
        self.recovery_history: list[dict] = []

    def recover_from_failure(
        self,
        failure_type: str,
        optimization_state: dict,
        optimization_func: Callable,
        **kwargs,
    ) -> tuple[bool, dict]:
        """Attempt recovery from optimization failure.

        Parameters
        ----------
        failure_type : str
            Type of failure ('convergence', 'numerical', 'memory', etc.)
        optimization_state : dict
            Current state of optimization
        optimization_func : callable
            Function to retry optimization
        **kwargs
            Additional arguments for optimization function

        Returns
        -------
        success : bool
            Whether recovery succeeded
        result : dict
            Recovered optimization result or error info
        """
        self.recovery_history.append(
            {
                "failure_type": failure_type,
                "iteration": optimization_state.get("iteration", 0),
                "cost": optimization_state.get("cost", np.inf),
            }
        )

        for retry in range(self.max_retries):
            for strategy in self.strategies:
                try:
                    # Apply recovery strategy
                    modified_state = strategy(failure_type, optimization_state, retry)

                    # Retry optimization with modified state
                    result = optimization_func(**modified_state, **kwargs)

                    # Check if recovery succeeded
                    if self._check_recovery_success(result):
                        if self.enable_diagnostics:
                            self.diagnostics.record_event(
                                "recovery_success",
                                {"strategy": strategy.__name__, "retry": retry},
                            )
                        return True, result

                except Exception as e:
                    warnings.warn(f"Recovery strategy {strategy.__name__} failed: {e}")
                    continue

        # All recovery attempts failed
        if self.enable_diagnostics:
            self.diagnostics.record_event(
                "recovery_failed", {"attempts": self.max_retries * len(self.strategies)}
            )

        return False, {"error": f"Recovery failed for {failure_type}"}

    def _perturb_parameters(self, failure_type: str, state: dict, retry: int) -> dict:
        """Perturb parameters to escape local minima.

        Parameters
        ----------
        failure_type : str
            Type of failure
        state : dict
            Current optimization state
        retry : int
            Retry attempt number

        Returns
        -------
        modified_state : dict
            State with perturbed parameters
        """
        modified_state = state.copy()
        params = state.get("params", state.get("x"))

        if params is not None:
            # Increase perturbation with each retry
            noise_scale = 0.01 * (2**retry)

            # Add Gaussian noise using JAX
            import jax.random as jr

            key = jr.PRNGKey(42 + retry * 2)  # Different seed for noise perturbation
            noise = jr.normal(key, shape=params.shape) * noise_scale

            # Scale noise by parameter magnitude
            param_scale = np.abs(params) + 1e-10
            scaled_noise = noise * param_scale

            modified_state["params"] = params + scaled_noise

            # Also try different initial guess if available
            if "p0" in state:
                modified_state["p0"] = params + scaled_noise

        return modified_state

    def _switch_algorithm(self, failure_type: str, state: dict, retry: int) -> dict:
        """Switch to different optimization algorithm.

        Parameters
        ----------
        failure_type : str
            Type of failure
        state : dict
            Current optimization state
        retry : int
            Retry attempt number

        Returns
        -------
        modified_state : dict
            State with different algorithm
        """
        modified_state = state.copy()
        current_method = state.get("method", "trf")

        # Algorithm switching strategy
        algorithm_chain = {
            "trf": ["lm", "dogbox"],
            "lm": ["trf", "dogbox"],
            "dogbox": ["trf", "lm"],
        }

        alternatives = algorithm_chain.get(current_method, ["trf"])

        if retry < len(alternatives):
            modified_state["method"] = alternatives[retry]

            # Adjust tolerances for new algorithm
            if alternatives[retry] == "lm":
                # LM is less robust but more accurate
                modified_state["ftol"] = 1e-10
                modified_state["xtol"] = 1e-10
            elif alternatives[retry] == "dogbox":
                # Dogbox for bounded problems
                modified_state["ftol"] = 1e-8
                modified_state["xtol"] = 1e-8

        return modified_state

    def _adjust_regularization(
        self, failure_type: str, state: dict, retry: int
    ) -> dict:
        """Adjust regularization parameters.

        Parameters
        ----------
        failure_type : str
            Type of failure
        state : dict
            Current optimization state
        retry : int
            Retry attempt number

        Returns
        -------
        modified_state : dict
            State with adjusted regularization
        """
        modified_state = state.copy()

        if failure_type in ["numerical", "ill_conditioned"]:
            # Increase regularization for numerical issues
            current_reg = state.get("regularization", 0)
            new_reg = max(1e-8, current_reg) * (10 ** (retry + 1))
            modified_state["regularization"] = new_reg

            # Also adjust trust region parameters if applicable
            if state.get("method") == "trf":
                modified_state["tr_solver"] = "lsmr"  # More stable
                modified_state["x_scale"] = "jac"  # Jacobian scaling

        # Adjust loss function for outliers
        if failure_type == "outliers" or state.get("has_outliers", False):
            loss_progression = ["linear", "soft_l1", "huber", "cauchy"]
            current_loss = state.get("loss", "linear")

            try:
                current_idx = loss_progression.index(current_loss)
                if current_idx < len(loss_progression) - 1:
                    modified_state["loss"] = loss_progression[current_idx + 1]
            except ValueError:
                modified_state["loss"] = "huber"

        return modified_state

    def _reformulate_problem(self, failure_type: str, state: dict, retry: int) -> dict:
        """Reformulate the optimization problem.

        Parameters
        ----------
        failure_type : str
            Type of failure
        state : dict
            Current optimization state
        retry : int
            Retry attempt number

        Returns
        -------
        modified_state : dict
            State with reformulated problem
        """
        modified_state = state.copy()

        # Scale variables for better conditioning
        if "xdata" in state and "ydata" in state:
            xdata = np.asarray(state["xdata"])
            ydata = np.asarray(state["ydata"])

            # Normalize data
            x_mean = np.mean(xdata, axis=0)
            x_std = np.std(xdata, axis=0) + 1e-10
            y_mean = np.mean(ydata)
            y_std = np.std(ydata) + 1e-10

            modified_state["xdata"] = (xdata - x_mean) / x_std
            modified_state["ydata"] = (ydata - y_mean) / y_std

            # Store transformation for later
            modified_state["data_transform"] = {
                "x_mean": x_mean,
                "x_std": x_std,
                "y_mean": y_mean,
                "y_std": y_std,
            }

        # Adjust bounds if present
        if "bounds" in state and state["bounds"] is not None:
            bounds = state["bounds"]
            if isinstance(bounds, tuple) and len(bounds) == 2:
                lower, upper = bounds
                # Relax bounds slightly
                bound_range = upper - lower
                relaxation = 0.01 * (2**retry) * bound_range
                modified_state["bounds"] = (lower - relaxation, upper + relaxation)

        return modified_state

    def _multi_start(self, failure_type: str, state: dict, retry: int) -> dict:
        """Multi-start optimization from different initial points.

        Parameters
        ----------
        failure_type : str
            Type of failure
        state : dict
            Current optimization state
        retry : int
            Retry attempt number

        Returns
        -------
        modified_state : dict
            State with new starting point
        """
        modified_state = state.copy()

        # Generate new starting point
        if "bounds" in state and state["bounds"] is not None:
            bounds = state["bounds"]
            if isinstance(bounds, tuple) and len(bounds) == 2:
                lower, upper = bounds
                n_params = len(state.get("p0", state.get("params", [])))

                if n_params > 0:
                    # Use JAX-compatible random sampling for better coverage
                    import jax.random as jr

                    # Create deterministic key based on retry count
                    key = jr.PRNGKey(42 + retry)
                    # Generate uniform random sample
                    sample = jr.uniform(key, shape=(n_params,))

                    # Scale to bounds
                    new_p0 = lower + sample * (upper - lower)
                    modified_state["p0"] = new_p0
        else:
            # Random initialization around current point
            current = state.get("p0", state.get("params"))
            if current is not None:
                # Use JAX random for diversity
                import jax.random as jr

                key = jr.PRNGKey(42 + retry * 3)  # Different seed for unbounded init
                scale = jnp.abs(current) + 1
                new_p0 = current + jr.normal(key, shape=current.shape) * scale
                modified_state["p0"] = new_p0

        return modified_state

    def _check_recovery_success(self, result: dict) -> bool:
        """Check if recovery was successful.

        Parameters
        ----------
        result : dict
            Optimization result

        Returns
        -------
        success : bool
            Whether recovery succeeded
        """
        # Check for explicit success flag
        if "success" in result:
            return result["success"]

        # Check for valid parameters
        if "x" in result or "params" in result:
            params = result.get("x", result.get("params"))
            if params is not None:
                # Check for NaN/Inf
                if not np.all(np.isfinite(params)):
                    return False

                # Check cost if available
                if "cost" in result:
                    cost = result["cost"]
                    if not np.isfinite(cost) or cost > 1e10:
                        return False

                return True

        return False
