import warnings


class OptimizeResult(dict):
    """Optimization result container for NLSQ curve fitting operations.

    This class stores the complete results from nonlinear least squares optimization
    performed using JAX-accelerated algorithms. It extends dict to provide both
    dictionary-style and attribute-style access to optimization results.

    Core Attributes
    ---------------
    x : jax.numpy.ndarray or numpy.ndarray
        Optimized parameter vector containing the final fitted parameters.
        These represent the solution to the nonlinear least squares problem.

    success : bool
        Indicates whether the optimization terminated successfully. True means
        convergence criteria were satisfied within tolerance limits.

    status : int
        Numerical termination status code indicating why optimization stopped:

        - 1: Gradient convergence (||g||_inf < gtol)
        - 2: Step size convergence (||dx||/||x|| < xtol)
        - 3: Function value convergence (delta_f/f < ftol)
        - 0: Maximum iterations reached
        - -1: Evaluation limit exceeded
        - -3: Inner loop iteration limit (algorithm-specific)

    message : str
        Human-readable description of termination cause. Provides detailed
        information about convergence status or failure reasons.

    Objective Function Results
    ---------------------------
    fun : jax.numpy.ndarray
        Final residual vector f(x) at the solution. For curve fitting, these
        are the differences between model predictions and data points.

    cost : float
        Final cost function value: 0.5 * ||f(x)||² for standard least squares,
        or 0.5 * sum(ρ(f_i²/σ²)) for robust loss functions.

    jac : jax.numpy.ndarray
        Final Jacobian matrix J(x) with shape (m, n) where m is number of
        data points and n is number of parameters. Computed using JAX autodiff.

    grad : jax.numpy.ndarray
        Final gradient vector g = J^T * f with shape (n,). Used for
        convergence checking and parameter uncertainty estimation.

    Convergence Metrics
    -------------------
    optimality : float
        Final gradient norm ||g||_inf used for convergence assessment.
        Should be less than gtol for successful convergence.

    active_mask : numpy.ndarray
        Boolean mask indicating which parameters hit bounds (for bounded
        optimization). Shape (n,) with True for parameters at constraints.

    Iteration Statistics
    --------------------
    nfev : int
        Total number of objective function evaluations during optimization.
        Each evaluation computes residuals f(x) for given parameters.

    njev : int
        Total number of Jacobian evaluations. With JAX autodiff, this equals
        the number of combined function+gradient evaluations.

    nit : int
        Number of optimization iterations completed. Not always available
        for all algorithms.

    Algorithm-Specific Results
    ---------------------------
    pcov : jax.numpy.ndarray, optional
        Parameter covariance matrix with shape (n, n). Provides parameter
        uncertainty estimates. Available when uncertainty estimation is requested.
        Computed as: pcov = inv(J^T * J) * residual_variance

    active_mask : numpy.ndarray
        For bounded optimization, indicates which parameters are at bounds.

    all_times : dict, optional
        Detailed timing information for algorithm profiling. Contains timing
        data for different optimization phases (function evaluation, Jacobian
        computation, linear algebra operations, etc.).

    Usage Examples
    --------------
    Basic result access::

        import nlsq

        # Perform curve fitting
        result = nlsq.curve_fit(model_func, x_data, y_data, p0=initial_guess)

        # Access optimized parameters
        fitted_params = result.x

        # Check convergence
        if result.success:
            print(f"Optimization converged: {result.message}")
            print(f"Final cost: {result.cost}")
            print(f"Function evaluations: {result.nfev}")
        else:
            print(f"Optimization failed: {result.message}")

        # Parameter uncertainties (if covariance computed)
        if hasattr(result, 'pcov'):
            param_errors = jnp.sqrt(jnp.diag(result.pcov))
            print(f"Parameter uncertainties: {param_errors}")

    Advanced result inspection::

        # Examine residuals and fit quality
        final_residuals = result.fun
        rms_error = jnp.sqrt(jnp.mean(final_residuals**2))

        # Check gradient convergence
        gradient_norm = result.optimality
        print(f"Final gradient norm: {gradient_norm}")

        # Analyze Jacobian condition
        jacobian = result.jac
        condition_number = jnp.linalg.cond(jacobian)
        print(f"Jacobian condition number: {condition_number}")

        # For bounded problems, check active constraints
        if hasattr(result, 'active_mask'):
            constrained_params = jnp.where(result.active_mask)[0]
            print(f"Parameters at bounds: {constrained_params}")

    Dictionary Interface
    --------------------
    Since OptimizeResult inherits from dict, all attributes are also
    accessible via dictionary syntax::

        # Dictionary-style access
        parameters = result['x']
        success_flag = result['success']

        # List all available results
        print("Available results:", list(result.keys()))

    Integration with SciPy
    ----------------------
    This class maintains compatibility with scipy.optimize.OptimizeResult
    while adding JAX-specific features and NLSQ-specific results. It can
    be used interchangeably with SciPy optimization results in most contexts.

    Technical Notes
    ---------------
    - All JAX arrays are automatically converted to NumPy arrays for compatibility
    - Covariance matrices use double precision for numerical stability
    - Large dataset results may include memory management statistics
    - GPU timing results require explicit timing mode activation
    - Progress monitoring data is stored in algorithm-specific attributes
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return "\n".join(
                [k.rjust(m) + ": " + repr(v) for k, v in sorted(self.items())]
            )
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


def _check_unknown_options(unknown_options):
    """Warn if unknown solver options are provided.

    Parameters
    ----------
    unknown_options : dict
        Dictionary of options that were not recognized by the solver.
        If non-empty, a warning is issued listing the unknown option names.

    Warns
    -----
    OptimizeWarning
        If any unknown options are present.
    """
    if unknown_options:
        msg = ", ".join(map(str, unknown_options.keys()))
        # Stack level 4: this is called from _minimize_*, which is
        # called from another function in SciPy. Level 4 is the first
        # level in user code.
        warnings.warn(f"Unknown solver options: {msg}", OptimizeWarning, 4)


class OptimizeWarning(UserWarning):
    """Warning class for optimization-related issues.

    This warning is raised when non-critical issues are encountered during
    optimization, such as unknown solver options, convergence concerns, or
    numerical stability warnings that don't prevent the optimization from
    completing but should be brought to the user's attention.

    Common scenarios:
        - Unknown or deprecated solver options passed to optimizer
        - Convergence achieved but with warnings about numerical conditioning
        - Parameter bounds adjusted automatically
        - Automatic algorithm selection overrides

    Example:
        >>> import warnings
        >>> warnings.filterwarnings('error', category=OptimizeWarning)
        >>> # Now OptimizeWarning will raise an exception instead of warning

    See Also:
        nlsq.error_messages.OptimizationError : Exception for critical failures
    """
