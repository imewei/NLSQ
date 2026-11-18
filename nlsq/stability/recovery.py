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
from nlsq.utils.bounds_sampling import sample_in_bounds
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
    diagnostics : nlsq.utils.diagnostics.OptimizationDiagnostics
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

        self.diagnostics = None
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
            },
        )

        attempts = 0
        last_error: Exception | None = None
        for retry in range(self.max_retries):
            for strategy in self.strategies:
                try:
                    # Apply recovery strategy
                    modified_state = strategy(failure_type, optimization_state, retry)

                    # A strategy with nothing to propose returns the state
                    # untouched (shallow copy: same value objects); re-running
                    # it would just repeat the failed call.
                    if modified_state.keys() == optimization_state.keys() and all(
                        modified_state[k] is v for k, v in optimization_state.items()
                    ):
                        continue

                    # Retry optimization with modified state
                    attempts += 1
                    result = optimization_func(**modified_state, **kwargs)

                    # Check if recovery succeeded
                    start = next(
                        (
                            optimization_state[k]
                            for k in ("params", "x", "p0")
                            if optimization_state.get(k) is not None
                        ),
                        None,
                    )
                    if self._check_recovery_success(
                        result,
                        optimization_state.get("bounds"),
                        None
                        if start is None
                        else np.shape(np.atleast_1d(np.asarray(start))),
                    ):
                        if self.enable_diagnostics and self.diagnostics is not None:
                            self.diagnostics.record_event(
                                "recovery_success",
                                {"strategy": strategy.__name__, "retry": retry},
                            )
                        return True, result

                except Exception as e:
                    last_error = e
                    warnings.warn(f"Recovery strategy {strategy.__name__} failed: {e}")
                    continue

        # All recovery attempts failed
        if self.enable_diagnostics and self.diagnostics is not None:
            self.diagnostics.record_event(
                "recovery_failed",
                {"attempts": attempts},
            )

        failure = {"error": f"Recovery failed for {failure_type}"}
        if last_error is not None:
            failure["last_error"] = f"{type(last_error).__name__}: {last_error}"
        return False, failure

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
            State with perturbed parameters. The perturbed start is written to
            every present start key as a float ``np.ndarray`` (a list start
            comes back as an array, not in its original container); a scalar
            or 0-d start is written back as a Python float.
        """
        modified_state = state.copy()
        # Perturb the first existing non-None start key, in the order
        # params > x > p0 (the same keys that are written back below).
        params = next(
            (state[k] for k in ("params", "x", "p0") if state.get(k) is not None),
            None,
        )

        if params is not None:
            params = np.asarray(params, dtype=float)  # lists / JAX arrays
            # Increase perturbation with each retry
            noise_scale = 0.01 * (2**retry)

            # Add Gaussian noise using JAX
            import jax.random as jr

            key = jr.PRNGKey(42 + retry * 2)  # Different seed for noise perturbation
            noise = jr.normal(key, shape=params.shape) * noise_scale

            # Scale noise by parameter magnitude; the floor keeps a zero
            # parameter from being "perturbed" by ~1e-12.
            param_scale = np.maximum(np.abs(params), 1e-3)
            perturbed = np.asarray(params + noise * param_scale, dtype=float)

            # A perturbed start outside the bounds is rejected by the solver
            # ("x0 is infeasible") and the retry is wasted.
            bounds = state.get("bounds")
            if bounds is not None and len(bounds) == 2:
                lower, upper = (
                    np.broadcast_to(np.asarray(b, dtype=float), np.shape(params))
                    for b in bounds
                )
                perturbed = np.clip(perturbed, lower, upper)

            # Write only keys the state already has: a retry callback taking just
            # `x` would reject an unexpected `params`/`p0` keyword.
            for k in ("params", "x", "p0"):
                if k in state:
                    # a scalar start stays a scalar (float); anything else array
                    modified_state[k] = (
                        float(perturbed) if params.ndim == 0 else perturbed
                    )

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

        # NLSQ supports only "trf" and "lm" ("dogbox" is rejected by the
        # solver); "lm" cannot handle bounds, so a bounded problem has no
        # alternative and the state is returned unchanged.
        algorithm_chain = {"trf": ["lm"], "lm": ["trf"]}
        alternatives = algorithm_chain.get(current_method, ["trf"])
        bounds = state.get("bounds")
        if bounds is not None and any(
            np.any(np.isfinite(np.asarray(b, dtype=float))) for b in bounds
        ):
            alternatives = [m for m in alternatives if m != "lm"]

        if retry < len(alternatives):
            modified_state["method"] = alternatives[retry]

            # Adjust tolerances for new algorithm
            if alternatives[retry] == "lm":
                # LM is less robust but more accurate
                modified_state["ftol"] = 1e-10
                modified_state["xtol"] = 1e-10

        return modified_state

    def _adjust_regularization(
        self,
        failure_type: str,
        state: dict,
        retry: int,
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

        Notes
        -----
        Only the keys forwarded by the recovery lambda in
        ``CurveFit.curve_fit`` (``p0``/``params``, ``xdata``, ``ydata``,
        ``bounds``, ``method``, ``ftol``, ``xtol``, ``tr_solver``,
        ``x_scale``, ``loss``) reach the solver. The ``regularization``
        entry is not forwarded (the solver has no such parameter) and has no
        effect on the retry.
        """
        modified_state = state.copy()

        # Everything below except the loss progression is retry-independent:
        # re-proposing it on retry > 0 would repeat the identical solve.
        if failure_type in ["numerical", "ill_conditioned"] and retry == 0:
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
                # One step further down the progression per retry.
                next_idx = loss_progression.index(current_loss) + 1 + retry
                if next_idx < len(loss_progression):
                    modified_state["loss"] = loss_progression[next_idx]
            except ValueError:
                if retry == 0:
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

        # NOTE: data normalization was intentionally removed here. Normalizing
        # xdata/ydata changes the fitted parameters in a model-dependent way, and
        # there is no general inverse to map them back to physical scale. The
        # previous code stored a "data_transform" that was never applied, so a
        # successful reformulated fit returned parameters in normalized space —
        # i.e. silently wrong results. Conditioning is handled correctly inside
        # the solver (ParameterNormalizer / NumericalStabilityGuard) instead.
        # Bounds are user constraints and are never relaxed here: widening them
        # (or turning a one-sided bound into -inf) would silently drop e.g. a
        # positivity constraint.

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

        # Same precedence as the recovery callback in CurveFit.curve_fit.
        current = next(
            (state[k] for k in ("p0", "params", "x") if state.get(k) is not None),
            None,
        )
        new_p0 = None

        # Generate new starting point
        bounds = state.get("bounds")
        if bounds is not None:
            # Any (lower, upper) pair: tuple, list or 2-row array.
            if len(bounds) == 2 and current is not None and len(current) > 0:
                lower, upper = bounds
                n_params = len(current)

                # Use JAX-compatible random sampling for better coverage
                import jax.random as jr

                # Create deterministic key based on retry count
                key = jr.PRNGKey(42 + retry)
                # Generate uniform random sample
                sample = np.asarray(jr.uniform(key, shape=(n_params,)))

                # Scale to bounds; half-infinite dims sample around p0 instead
                # of producing inf.
                center = np.asarray(current, dtype=np.float64)
                new_p0 = sample_in_bounds(
                    sample,
                    np.broadcast_to(np.asarray(lower, dtype=float), (n_params,)),
                    np.broadcast_to(np.asarray(upper, dtype=float), (n_params,)),
                    center,
                    np.maximum(np.abs(center), 1.0),
                )
        elif current is not None:
            # Random initialization around current point
            import jax.random as jr

            key = jr.PRNGKey(42 + retry * 3)  # Different seed for unbounded init
            current = jnp.asarray(current)
            scale = jnp.abs(current) + 1
            new_p0 = current + jr.normal(key, shape=current.shape) * scale

        if new_p0 is not None:
            written = False
            for k in ("p0", "params", "x"):
                if k in state:
                    modified_state[k] = new_p0
                    written = True
            if not written:
                modified_state["p0"] = new_p0

        return modified_state

    def _check_recovery_success(
        self, result: dict, bounds=None, expected_shape=None
    ) -> bool:
        """Check if recovery was successful.

        Parameters
        ----------
        result : dict
            Optimization result (dict or object with attributes)
        bounds : tuple, optional
            Original ``(lower, upper)`` user bounds; a solution outside them is
            rejected. Each must be a scalar or have the parameter vector's
            length (a length that would broadcast differently is rejected).
        expected_shape : tuple, optional
            Shape of the start vector (params > x > p0); a returned parameter
            field must be 1-D, non-empty and of this shape.

        Returns
        -------
        success : bool
            Whether recovery succeeded
        """

        def _get_value(obj, key, default=None):
            """Get value from dict or object attribute."""
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        # An explicit False is unambiguous -- trust it immediately.
        success = _get_value(result, "success")
        if success is not None and not bool(success):
            return False

        # Validate parameters/cost whenever available. Even an explicit
        # success=True must not bypass this: a result like
        # {"success": True, "popt": [nan], "cost": nan} is not a valid
        # recovery just because the flag says so. Every parameter field
        # (popt/x/params) must be finite and inside the bounds; a supplied
        # cost must be finite on its own, with or without a parameter field.
        fields = [
            v
            for v in (_get_value(result, k) for k in ("popt", "x", "params"))
            if v is not None
        ]
        for field in fields:
            params = np.asarray(field, dtype=float)
            if params.ndim != 1 or params.size == 0:
                return False
            if expected_shape is not None and params.shape != tuple(expected_shape):
                return False
            if not np.all(np.isfinite(params)):
                return False
            if bounds is not None:
                lo, hi = (np.asarray(b, dtype=float) for b in bounds)
                for b in (lo, hi):  # scalar or length-n only
                    if b.ndim > 1 or (b.ndim == 1 and b.shape != params.shape):
                        return False
                if not np.all((params >= lo) & (params <= hi)):
                    return False

        cost = _get_value(result, "cost")
        if cost is not None and not np.isfinite(cost):
            return False

        if fields:
            return True

        # No parameters to validate against -- fall back to the explicit
        # success flag, if any was given.
        if success is not None:
            return bool(success)

        return False
