"""Parameter normalization for improved optimization convergence.

This module provides automatic parameter scaling to address gradient signal
weakness caused by parameter scale imbalance. Parameters spanning many orders
of magnitude can cause slow convergence and numerical instability.

The ParameterNormalizer class supports multiple normalization strategies:
- Bounds-based: Normalize to [0, 1] using parameter bounds
- p0-based: Scale by initial parameter magnitudes
- None: Identity transform (no normalization)

The NormalizedModelWrapper wraps user model functions to work transparently
in normalized parameter space while maintaining JAX JIT compatibility.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp

__all__ = ["NormalizedModelWrapper", "ParameterNormalizer"]


class ParameterNormalizer:
    """Normalizes parameters to improve optimization convergence.

    This class handles automatic parameter scaling to address gradient signal
    weakness from parameter scale imbalance. It supports multiple strategies:

    - **bounds**: Normalize to [0, 1] using parameter bounds (lb, ub)
    - **p0**: Scale by initial parameter magnitudes
    - **auto**: Use bounds if provided, else p0-based
    - **none**: Identity transform (no normalization)

    The normalizer computes and stores the normalization Jacobian analytically,
    which is needed for transforming covariance matrices back to original space.

    Parameters
    ----------
    p0 : array_like
        Initial parameter guess of shape (n_params,)
    bounds : tuple of array_like, optional
        Parameter bounds as (lb, ub) where lb and ub are arrays of shape
        (n_params,). If None, p0-based scaling is used.
    strategy : str, default='auto'
        Normalization strategy. Options:

        - 'auto': Use bounds if provided, else p0-based
        - 'bounds': Normalize to [0, 1] using bounds
        - 'p0': Scale by initial parameter magnitudes
        - 'none': Identity transform (no normalization)

    Attributes
    ----------
    strategy : str
        Selected normalization strategy
    scales : jax.Array
        Scaling factors for each parameter (diagonal of Jacobian)
    offsets : jax.Array
        Offset for each parameter (used in bounds-based)
    original_bounds : tuple of jax.Array or None
        Original parameter bounds (lb, ub)
    normalization_jacobian : jax.Array
        Denormalization Jacobian matrix (diagonal) of shape (n_params, n_params).
        For covariance transform: Cov_orig = J @ Cov_norm @ J.T

    Examples
    --------
    Bounds-based normalization:

    >>> import jax.numpy as jnp
    >>> from nlsq.precision.parameter_normalizer import ParameterNormalizer
    >>> p0 = jnp.array([50.0, 0.5])
    >>> bounds = (jnp.array([10.0, 0.0]), jnp.array([100.0, 1.0]))
    >>> normalizer = ParameterNormalizer(p0, bounds, strategy='bounds')
    >>> normalized = normalizer.normalize(p0)
    >>> print(normalized)
    [0.444... 0.5]
    >>> denormalized = normalizer.denormalize(normalized)
    >>> print(jnp.allclose(denormalized, p0))
    True

    p0-based normalization:

    >>> p0 = jnp.array([1000.0, 1.0, 0.001])
    >>> normalizer = ParameterNormalizer(p0, bounds=None, strategy='p0')
    >>> normalized = normalizer.normalize(p0)
    >>> print(normalized)
    [1. 1. 1.]

    No normalization:

    >>> p0 = jnp.array([5.0, 15.0])
    >>> normalizer = ParameterNormalizer(p0, bounds=None, strategy='none')
    >>> normalized = normalizer.normalize(p0)
    >>> print(jnp.allclose(normalized, p0))
    True

    See Also
    --------
    NormalizedModelWrapper : Wraps model functions for normalized parameters
    HybridStreamingConfig : Configuration with normalization_strategy parameter

    Notes
    -----
    Implements Phase 0 (Parameter Normalization Setup) of the Adaptive
    Hybrid Streaming Optimizer specification.
    """

    def __init__(
        self,
        p0: jnp.ndarray,
        bounds: tuple[jnp.ndarray, jnp.ndarray] | None = None,
        strategy: str = "auto",
    ):
        """Initialize parameter normalizer.

        Parameters
        ----------
        p0 : array_like
            Initial parameter guess
        bounds : tuple of array_like, optional
            Parameter bounds as (lb, ub)
        strategy : str, default='auto'
            Normalization strategy
        """
        self.p0 = jnp.asarray(p0, dtype=jnp.float64)
        self.n_params = len(self.p0)
        self.original_bounds = bounds

        # Determine strategy
        if strategy == "auto":
            # Auto-select: use bounds if provided, else p0-based
            if bounds is not None:
                self.strategy = "bounds"
            else:
                self.strategy = "p0"
        else:
            self.strategy = strategy

        # Validate strategy
        valid_strategies = ("bounds", "p0", "none", "auto")
        if strategy not in valid_strategies:
            raise ValueError(
                f"strategy must be one of {valid_strategies}, got: {strategy}"
            )

        # Compute scales and offsets based on strategy
        self._compute_normalization_parameters()

        # Compute normalization Jacobian (denormalization Jacobian)
        self._normalization_jacobian = self._compute_jacobian()

    def _compute_normalization_parameters(self):
        """Compute scaling factors and offsets based on strategy."""
        if self.strategy == "bounds":
            # Bounds-based: normalize to [0, 1]
            if self.original_bounds is None:
                raise ValueError("bounds must be provided for bounds-based strategy")

            lb, ub = self.original_bounds
            lb = jnp.asarray(lb, dtype=jnp.float64)
            ub = jnp.asarray(ub, dtype=jnp.float64)

            # Scale = (ub - lb), offset = lb
            # Normalized: (params - lb) / (ub - lb)
            self.scales = ub - lb

            # Handle zero range (constant parameter)
            eps = jnp.finfo(jnp.float64).eps
            self.scales = jnp.where(
                jnp.abs(self.scales) < eps, jnp.ones_like(self.scales), self.scales
            )

            self.offsets = lb

        elif self.strategy == "p0":
            # p0-based: scale by parameter magnitudes
            # Normalized: params / |p0|
            abs_p0 = jnp.abs(self.p0)

            # Handle zero parameters with small epsilon
            eps = jnp.finfo(jnp.float64).eps * 10
            self.scales = jnp.where(abs_p0 < eps, jnp.ones_like(abs_p0), abs_p0)

            self.offsets = jnp.zeros(self.n_params, dtype=jnp.float64)

        elif self.strategy == "none":
            # No normalization: identity transform
            self.scales = jnp.ones(self.n_params, dtype=jnp.float64)
            self.offsets = jnp.zeros(self.n_params, dtype=jnp.float64)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _compute_jacobian(self) -> jnp.ndarray:
        """Compute denormalization Jacobian analytically.

        For our scaling operations, the Jacobian is diagonal with scales on diagonal.

        Returns
        -------
        jax.Array
            Denormalization Jacobian matrix of shape (n_params, n_params)
        """
        # Jacobian is diagonal matrix with scales on diagonal
        # This is d(denormalized)/d(normalized) = diag(scales)
        return jnp.diag(self.scales)

    @property
    def normalization_jacobian(self) -> jnp.ndarray:
        """Get the denormalization Jacobian matrix.

        This is the Jacobian of the denormalization transform, needed for
        transforming covariance matrices from normalized to original space:

            Cov_orig = J @ Cov_norm @ J.T

        Returns
        -------
        jax.Array
            Denormalization Jacobian matrix of shape (n_params, n_params).
            For our scaling, this is a diagonal matrix with scales on the diagonal.
        """
        return self._normalization_jacobian

    def normalize(self, params: jnp.ndarray) -> jnp.ndarray:
        """Normalize parameters to scaled space.

        Parameters
        ----------
        params : array_like
            Parameters in original space of shape (n_params,)

        Returns
        -------
        jax.Array
            Normalized parameters of shape (n_params,)
        """
        params = jnp.asarray(params, dtype=jnp.float64)

        # Apply normalization: (params - offset) / scale
        normalized = (params - self.offsets) / self.scales

        return normalized

    def denormalize(self, normalized_params: jnp.ndarray) -> jnp.ndarray:
        """Denormalize parameters back to original space.

        This is the exact inverse of normalize().

        Parameters
        ----------
        normalized_params : array_like
            Parameters in normalized space of shape (n_params,)

        Returns
        -------
        jax.Array
            Parameters in original space of shape (n_params,)
        """
        normalized_params = jnp.asarray(normalized_params, dtype=jnp.float64)

        # Apply denormalization: params = normalized * scale + offset
        params = normalized_params * self.scales + self.offsets

        return params

    def transform_bounds(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Transform bounds to normalized space.

        Returns
        -------
        lb_normalized : jax.Array
            Lower bounds in normalized space of shape (n_params,)
        ub_normalized : jax.Array
            Upper bounds in normalized space of shape (n_params,)

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from nlsq.precision.parameter_normalizer import ParameterNormalizer
        >>> p0 = jnp.array([50.0])
        >>> bounds = (jnp.array([10.0]), jnp.array([100.0]))
        >>> normalizer = ParameterNormalizer(p0, bounds, strategy='bounds')
        >>> lb_norm, ub_norm = normalizer.transform_bounds()
        >>> print(lb_norm, ub_norm)
        [0.] [1.]
        """
        if self.original_bounds is None:
            # No bounds: return infinite bounds in normalized space
            lb = jnp.full(self.n_params, -jnp.inf, dtype=jnp.float64)
            ub = jnp.full(self.n_params, jnp.inf, dtype=jnp.float64)
            return lb, ub

        lb, ub = self.original_bounds
        lb = jnp.asarray(lb, dtype=jnp.float64)
        ub = jnp.asarray(ub, dtype=jnp.float64)

        # Transform bounds using normalize()
        lb_normalized = self.normalize(lb)
        ub_normalized = self.normalize(ub)

        return lb_normalized, ub_normalized


class NormalizedModelWrapper:
    """Wraps user model function to work with normalized parameters.

    This wrapper allows optimization algorithms to work in normalized parameter
    space while the user model function operates in original parameter space.
    The wrapper is JAX JIT-compatible and preserves gradients correctly.

    Parameters
    ----------
    model_fn : callable
        User model function with signature: ``model_fn(x, *params) -> predictions``
    normalizer : ParameterNormalizer
        Parameter normalizer instance

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from nlsq.precision.parameter_normalizer import ParameterNormalizer, NormalizedModelWrapper
    >>> def model(x, a, b):
    ...     return a * x + b
    >>> p0 = jnp.array([5.0, 10.0])
    >>> normalizer = ParameterNormalizer(p0, bounds=None, strategy='p0')
    >>> wrapped_model = NormalizedModelWrapper(model, normalizer)
    >>> x = jnp.array([1.0, 2.0, 3.0])
    >>> normalized_params = normalizer.normalize(p0)
    >>> output = wrapped_model(x, *normalized_params)
    >>> print(output)
    [15. 20. 25.]

    JIT compilation:

    >>> import jax
    >>> @jax.jit
    ... def optimized_model(x, a_norm, b_norm):
    ...     return wrapped_model(x, a_norm, b_norm)
    >>> output = optimized_model(x, *normalized_params)

    See Also
    --------
    ParameterNormalizer : Handles parameter normalization
    """

    def __init__(
        self, model_fn: Callable[..., jnp.ndarray], normalizer: ParameterNormalizer
    ):
        """Initialize normalized model wrapper.

        Parameters
        ----------
        model_fn : callable
            User model function
        normalizer : ParameterNormalizer
            Parameter normalizer
        """
        self.model_fn = model_fn
        self.normalizer = normalizer

    def __call__(self, x: jnp.ndarray, *normalized_params: jnp.ndarray) -> jnp.ndarray:
        """Call wrapped model with normalized parameters.

        Parameters
        ----------
        x : array_like
            Independent variable data
        *normalized_params : array_like
            Normalized parameter values (unpacked)

        Returns
        -------
        jax.Array
            Model predictions
        """
        # Convert normalized parameters to array
        normalized_params_array = jnp.asarray(normalized_params, dtype=jnp.float64)

        # Denormalize parameters to original space
        original_params = self.normalizer.denormalize(normalized_params_array)

        # Call original model with denormalized parameters
        # Unpack parameters for model function
        return self.model_fn(x, *original_params)
