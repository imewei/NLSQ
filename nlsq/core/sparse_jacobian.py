"""Sparse Jacobian support for large-scale optimization.

This module provides sparse matrix support for Jacobian computations,
enabling efficient handling of problems with 20M+ data points.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from nlsq.constants import FINITE_DIFF_REL_STEP
from nlsq.utils.logging import get_logger

if TYPE_CHECKING:
    from nlsq.result import CurveFitResult

logger = get_logger(__name__)


def _is_points_last_2d(xdata: object, n_data: int) -> bool:
    """True for a 2-D NumPy/JAX array laid out (n_features, n_points)."""
    return (
        isinstance(xdata, np.ndarray | jax.Array)
        and xdata.ndim == 2
        and xdata.shape[-1] == n_data
    )


def _normalize_xdata(xdata: Any) -> Any:
    """Normalize a list/tuple ``xdata``; other types pass through unchanged.

    A flat sequence of scalars becomes a 1-D array. A sequence of equal-length
    sequences/arrays (coordinate form, e.g. ``[X, Y]``) becomes a list of
    ndarray coordinates, so every coordinate supports advanced indexing.
    """
    if not isinstance(xdata, list | tuple):
        return xdata
    if len(xdata) == 0:
        return np.asarray(xdata, dtype=float)
    if all(np.ndim(e) == 0 for e in xdata):
        return np.asarray(xdata)
    if any(np.ndim(e) == 0 for e in xdata):
        raise ValueError("xdata mixes scalars and sequences; cannot interpret it")
    coords = [np.asarray(e) for e in xdata]
    if len({len(c) for c in coords}) != 1:
        raise ValueError(
            "xdata coordinate arrays must have equal length, got "
            f"{[len(c) for c in coords]}"
        )
    return coords


def _autodiff_jacobian(
    func: Callable, xdata: Any, x0: np.ndarray, n_data: int
) -> np.ndarray | None:
    """Forward-mode Jacobian at ``x0`` as ``(n_data, n_params)``, or None.

    None when the model is not JAX-traceable (numpy-only, data-dependent
    branching) or returns an unexpected shape.
    """
    try:
        J = np.asarray(
            jax.jacfwd(lambda p: jnp.asarray(func(xdata, *p)))(jnp.asarray(x0))
        )
    except Exception:  # untraceable model
        return None
    return J if J.shape == (n_data, len(x0)) else None


def _reject_complex(what: str, arr: Any) -> None:
    """Raise for complex data: the sparse paths use real transposes (real-only rule)."""
    if np.iscomplexobj(arr):
        raise ValueError(f"complex {what} is not supported (real-valued fits only)")


def _check_threshold(threshold: float) -> float:
    """Return ``threshold`` if it is a finite number in ``[0, 1)``, else raise."""
    try:
        ok = 0.0 <= float(threshold) < 1.0  # NaN compares False
    except (TypeError, ValueError):
        ok = False
    if not ok:
        raise ValueError(
            f"sparsity threshold must be a finite number in [0, 1), got {threshold!r}"
        )
    return float(threshold)


def _check_positive_int(name: str, value: object, minimum: int = 1) -> int:
    """Return ``value`` as an int if it is an integer >= ``minimum``, else raise."""
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


class SparseJacobianComputer:
    """Compute and manage sparse Jacobians for large-scale problems.

    For many curve fitting problems, the Jacobian has a sparse structure
    where each data point only depends on a subset of parameters. This
    class exploits that structure to reduce memory usage by 10-100x.
    """

    def __init__(self, sparsity_threshold: float = 0.01):
        """Initialize sparse Jacobian computer.

        Parameters
        ----------
        sparsity_threshold : float
            Relative threshold in ``[0, 1)``: an element ``J[i, j]`` is considered
            zero when ``|J[i, j]| <= sparsity_threshold * max_i |J[i, j]|``
            (per-column scale, so the result does not depend on the units of
            the model or of individual parameters). All-zero columns stay
            all-zero. Default is 0.01.
        """
        self.sparsity_threshold = _check_threshold(sparsity_threshold)
        self._sparsity_pattern: np.ndarray | None = None
        self._sparse_indices: tuple | None = None

    def detect_sparsity_pattern(
        self,
        func: Callable,
        x0: np.ndarray,
        xdata_sample: np.ndarray | list,
        n_samples: int = 100,
        points_last: bool | None = None,
    ) -> tuple[np.ndarray, float]:
        """Detect sparsity pattern of Jacobian from sample evaluations.

        Parameters
        ----------
        func : Callable
            Function to evaluate
        x0 : np.ndarray
            Initial parameter values
        xdata_sample : np.ndarray or list
            Sample of x data points. Can be a single array for 1D problems,
            a list of arrays [X, Y] for 2D problems, or a 2D array with
            shape (k, N) for multi-dimensional coordinates.
        n_samples : int
            Number of samples to use for pattern detection
        points_last : bool, optional
            Layout of a 2-D array ``xdata_sample`` (``True``: ``(k, N)``);
            default: inferred from the shape.

        Returns
        -------
        pattern : np.ndarray
            Boolean array indicating non-zero elements
        sparsity : float
            Fraction of zero elements
        """
        # Integer p0 would make `.at[i].add(1e-8 * ...)` truncate to a zero step.
        x0 = np.asarray(x0, dtype=np.float64)
        n_params = len(x0)
        n_samples = _check_positive_int("n_samples", n_samples, minimum=0)
        xdata_sample = _normalize_xdata(xdata_sample)
        xdata_arr = np.asarray(xdata_sample)

        # Handle multi-dimensional xdata (e.g., [X, Y] for 2D fitting)
        xdata_sliced: list | np.ndarray  # Can be list or ndarray
        if isinstance(xdata_sample, list):
            # Get number of data points from first coordinate array
            n_data_points = len(xdata_sample[0])
            n_data = min(n_samples, n_data_points)
            # Slice each coordinate array
            xdata_sliced = [coord[:n_data] for coord in xdata_sample]
        elif xdata_arr.ndim == 2 and (
            points_last
            if points_last is not None
            else xdata_arr.shape[0] <= xdata_arr.shape[1]
        ):
            # shortcut: with more rows than columns a 2-D array is read as
            # points-first (N, k); pass a list of coordinates to be explicit.
            # 2D array with shape (k, N) - slice along N dimension
            n_data_points = xdata_arr.shape[1]
            n_data = min(n_samples, n_data_points)
            xdata_sliced = xdata_arr[:, :n_data]
        else:
            n_data = min(n_samples, len(xdata_arr))
            xdata_sliced = xdata_arr[:n_data]

        # Sample Jacobian at a few points to detect pattern
        pattern = np.zeros((n_data, n_params), dtype=bool)

        # Use finite differences to detect sparsity
        eps = FINITE_DIFF_REL_STEP
        f0 = np.atleast_1d(np.asarray(func(xdata_sliced, *x0)))
        if not np.all(np.isfinite(f0)):
            raise ValueError(
                "model output at x0 is not finite (NaN/inf); sparsity cannot be "
                "detected for an invalid model at the initial guess"
            )

        # Validate that function output matches the subsampled xdata size.
        # Closure-based model functions may ignore xdata and always return
        # full-size output, making sparsity detection inapplicable.
        if f0.shape[0] != n_data:
            raise ValueError(
                f"Function output size ({f0.shape[0]}) does not match "
                f"xdata sample size ({n_data}). The function may use "
                f"closure-captured data instead of xdata.",
            )

        # Exact derivative first: a finite-difference estimate (especially an
        # enlarged-step secant) can vanish where the derivative does not.
        J_ad = _autodiff_jacobian(func, xdata_sliced, x0, n_data) if n_data else None
        f0_scale = np.max(np.abs(f0), initial=0.0)
        for i in range(n_params if n_data > 0 else 0):
            if J_ad is not None and np.all(np.isfinite(J_ad[:, i])):
                jv_abs = np.abs(J_ad[:, i])
                pattern[:, i] = jv_abs > self.sparsity_threshold * jv_abs.max()
                continue
            # Untraceable model: one-sided finite difference. If the difference is
            # at the round-off level of f (e.g. 1e8 + a), it is not evidence of a
            # zero derivative, and an enlarged-step secant can vanish where the
            # derivative does not, so the column is reported dense. shortcut: use
            # jnp in the model to classify such a column exactly.
            step = eps * max(1.0, abs(float(x0[i])))
            x_perturb = jnp.asarray(x0).at[i].add(step)
            f_perturb = np.asarray(func(xdata_sliced, *x_perturb))
            if not np.all(np.isfinite(f_perturb)):
                raise ValueError(
                    f"model output is not finite (NaN/inf) after perturbing "
                    f"parameter {i} by {step:g}; the derivative cannot be "
                    "classified as zero or non-zero"
                )
            jac_abs = np.abs((f_perturb - f0) / step)
            if jac_abs.max(initial=0.0) > 100 * np.finfo(float).eps * f0_scale / step:
                pattern[:, i] = jac_abs > self.sparsity_threshold * jac_abs.max()
            else:
                logger.debug(
                    f"Jacobian column {i} unresolved by finite differences and the "
                    "model is not JAX-traceable; treating it as dense"
                )
                pattern[:, i] = True

        # Calculate sparsity (handle empty pattern)
        if pattern.size > 0:
            sparsity = 1.0 - np.sum(pattern) / pattern.size
        else:
            sparsity = 0.0  # No data means no sparsity information

        self._sparsity_pattern = pattern
        return pattern, sparsity

    def compute_sparse_jacobian(
        self,
        jac_func: Callable,
        x: np.ndarray,
        xdata: np.ndarray,
        ydata: np.ndarray,
        data_mask: np.ndarray | None = None,
        chunk_size: int = 10000,
        func: Callable | None = None,  # Add func parameter for finite diff fallback
    ) -> csr_matrix:
        """Compute Jacobian in sparse format with chunking.

        Parameters
        ----------
        jac_func : Callable
            Jacobian function
        x : np.ndarray
            Current parameter values
        xdata : np.ndarray
            Independent variable data
        ydata : np.ndarray
            Dependent variable data
        data_mask : np.ndarray, optional
            Mask for valid data points
        chunk_size : int
            Size of chunks for computation
        func : Callable, optional
            Original function for finite difference fallback

        Returns
        -------
        J_sparse : csr_matrix
            Sparse Jacobian matrix
        """
        chunk_size = _check_positive_int("chunk_size", chunk_size)
        xdata = _normalize_xdata(xdata)
        n_data = len(ydata)
        n_params = len(x)
        n_chunks = (n_data + chunk_size - 1) // chunk_size

        if data_mask is None:
            data_mask = np.ones(n_data, dtype=bool)
        elif len(data_mask) != n_data:
            raise ValueError(
                f"data_mask has {len(data_mask)} entries but ydata has {n_data}"
            )

        # Predictor and response must describe the same observations: a short
        # chunk would otherwise broadcast into invented rows.
        n_x = (
            len(xdata[0])
            if isinstance(xdata, list)
            else len(xdata)
            if isinstance(xdata, np.ndarray | jax.Array) and xdata.ndim == 1
            else n_data
        )
        if n_x != n_data:
            raise ValueError(f"xdata has {n_x} points but ydata has {n_data}")

        # Accumulate COO format data across chunks for fully vectorized construction
        all_rows: list[np.ndarray] = []
        all_cols: list[np.ndarray] = []
        all_values: list[np.ndarray] = []
        # Running per-column max |J| over finite entries (global reference scale)
        col_max = np.zeros(n_params)

        # Process in chunks to manage memory
        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min((chunk_idx + 1) * chunk_size, n_data)

            # Compute dense Jacobian for chunk. Multi-dimensional xdata (e.g.
            # [X, Y] for 2D fitting) must have each coordinate array sliced
            # individually -- slicing the outer tuple/list itself returns the
            # whole container unchanged (or empty once start exceeds its
            # length), pairing unsliced/empty coordinates with sliced
            # y_chunk/mask_chunk.
            if isinstance(xdata, list):
                x_chunk = [coord[start:end] for coord in xdata]
            elif _is_points_last_2d(xdata, n_data):
                # (k, N) ndarray: slice the points axis, not the features.
                x_chunk = xdata[:, start:end]
            elif hasattr(xdata, "__getitem__"):
                x_chunk = xdata[start:end]
            else:
                x_chunk = xdata
            y_chunk = ydata[start:end]
            mask_chunk = data_mask[start:end]

            # Convert to JAX arrays for computation
            x_jax = jnp.asarray(x)

            # Compute Jacobian for chunk (assuming jac_func returns dense)
            if callable(jac_func):
                J_chunk = jac_func(x_jax, x_chunk, y_chunk, mask_chunk, None)
            else:
                # Fallback to finite differences if no jac_func
                if func is None:
                    raise ValueError(
                        "func parameter required for finite difference fallback",
                    )
                J_chunk = self._finite_diff_jacobian(
                    func,
                    x,
                    x_chunk,
                    y_chunk,
                    mask_chunk,
                )

            # Convert to numpy if needed
            if hasattr(J_chunk, "block_until_ready"):
                J_chunk = np.array(J_chunk)

            # Vectorized sparse extraction: O(nnz) instead of O(nm).
            # Threshold is relative to the column max; the chunk-local max is
            # <= the global one, so this keeps a superset that the final
            # global filter below trims. Non-finite entries are always kept.
            J_chunk = np.asarray(J_chunk)
            _reject_complex("Jacobian", J_chunk)
            expected = (end - start, n_params)
            if J_chunk.shape != expected:
                raise ValueError(
                    f"Jacobian callback returned shape {J_chunk.shape}, expected "
                    f"{expected} (points in chunk, parameters)"
                )
            # Masked points do not contribute (idempotent for mask-aware callbacks
            # and consistent with the finite-difference branch).
            J_chunk = np.where(
                np.asarray(mask_chunk, dtype=bool)[:, None], J_chunk, 0.0
            )
            finite = np.isfinite(J_chunk)
            J_abs = np.where(finite, np.abs(J_chunk), 0.0)
            chunk_max = J_abs.max(axis=0, initial=0.0)
            col_max = np.maximum(col_max, chunk_max)
            mask = (J_abs > self.sparsity_threshold * chunk_max) | ~finite
            chunk_rows, chunk_cols = np.where(mask)
            chunk_values = J_chunk[chunk_rows, chunk_cols]

            # Adjust row indices for the full matrix offset
            chunk_rows = chunk_rows + start

            # Accumulate for batch construction
            all_rows.append(chunk_rows)
            all_cols.append(chunk_cols)
            all_values.append(chunk_values)

        # Build sparse matrix in one vectorized operation using COO format
        if all_rows:
            rows = np.concatenate(all_rows)
            cols = np.concatenate(all_cols)
            values = np.concatenate(all_values)
            keep = (np.abs(values) > self.sparsity_threshold * col_max[cols]) | ~(
                np.isfinite(values)
            )
            rows, cols, values = rows[keep], cols[keep], values[keep]
            J_sparse = coo_matrix((values, (rows, cols)), shape=(n_data, n_params))
        else:
            # Empty matrix case
            J_sparse = coo_matrix((n_data, n_params))

        # Convert to CSR format for efficient operations
        return J_sparse.tocsr()

    def _finite_diff_jacobian(
        self,
        func: Callable,
        x: np.ndarray,
        xdata: np.ndarray,
        ydata: np.ndarray,
        data_mask: np.ndarray,
        eps: float = FINITE_DIFF_REL_STEP,
    ) -> np.ndarray:
        """Compute Jacobian using finite differences as fallback.

        Parameters
        ----------
        func : Callable
            Function to differentiate
        x : np.ndarray
            Current parameter values
        xdata : np.ndarray
            Independent variable data
        ydata : np.ndarray
            Dependent variable data
        data_mask : np.ndarray
            Mask for valid data
        eps : float
            Finite difference step size

        Returns
        -------
        J : np.ndarray
            Dense Jacobian matrix for chunk
        """
        x = np.asarray(x, dtype=np.float64)  # integer x would zero the step
        n_data = len(ydata)
        n_params = len(x)
        J = np.zeros((n_data, n_params))

        # Exact derivative first: a difference quotient rounds away the effect of
        # a small step on a large model output (e.g. 1e10 + a * x).
        J_ad = _autodiff_jacobian(func, xdata, x, n_data)
        if J_ad is not None:
            _reject_complex("model derivative", J_ad)
            return np.where(np.asarray(data_mask, dtype=bool)[:, None], J_ad, 0.0)

        # Base function evaluation
        # Difference the model outputs directly: subtracting a large ``ydata``
        # first would round away the (small) change in the model.
        f0 = np.asarray(func(xdata, *x))
        _reject_complex("model output", f0)
        if f0.shape != (n_data,):
            raise ValueError(
                f"model returned shape {f0.shape} for {n_data} observations; "
                "xdata and ydata must describe the same points"
            )

        # Compute finite differences
        for j in range(n_params):
            # OPT-6: Use JAX functional update instead of copy + mutate
            step = eps * max(1.0, abs(float(x[j])))
            x_perturb = jnp.asarray(x).at[j].add(step)

            f_perturb = np.asarray(func(xdata, *x_perturb))

            J[:, j] = np.where(data_mask, (f_perturb - f0) / step, 0)

        return J

    def sparse_matrix_vector_product(
        self,
        J_sparse: csr_matrix,
        v: np.ndarray,
    ) -> np.ndarray:
        """Efficient sparse matrix-vector product.

        Parameters
        ----------
        J_sparse : csr_matrix
            Sparse Jacobian matrix
        v : np.ndarray
            Vector to multiply

        Returns
        -------
        result : np.ndarray
            J @ v
        """
        return J_sparse @ v

    def sparse_normal_equations(
        self,
        J_sparse: csr_matrix,
        f: np.ndarray,
    ) -> tuple[Callable, np.ndarray]:
        """Set up normal equations with sparse Jacobian.

        Solves (J^T @ J) @ p = -J^T @ f without forming J^T @ J explicitly.

        Parameters
        ----------
        J_sparse : csr_matrix
            Sparse Jacobian matrix
        f : np.ndarray
            Residual vector

        Returns
        -------
        matvec : callable
            Function that computes (J^T @ J) @ v
        rhs : np.ndarray
            Right-hand side -J^T @ f
        """

        _reject_complex("Jacobian", J_sparse)
        _reject_complex("residual", f)

        def matvec(v):
            """Compute (J^T @ J) @ v without forming J^T @ J."""
            Jv = J_sparse @ v
            return J_sparse.T @ Jv

        rhs = -J_sparse.T @ f

        return matvec, rhs

    def estimate_memory_usage(
        self,
        n_data: int,
        n_params: int,
        sparsity: float = 0.99,
    ) -> dict:
        """Estimate memory usage for sparse vs dense Jacobian.

        Parameters
        ----------
        n_data : int
            Number of data points
        n_params : int
            Number of parameters
        sparsity : float
            Fraction of zero elements (0-1)

        Returns
        -------
        memory_info : dict
            Memory usage estimates in GB
        """
        # Dense memory usage
        dense_bytes = n_data * n_params * 8  # 8 bytes per float64
        dense_gb = dense_bytes / (1024**3)

        # Sparse memory usage (CSR format)
        # Need to store: values, column indices, row pointers
        nnz = int(n_data * n_params * (1 - sparsity))
        sparse_bytes = nnz * 8  # values
        sparse_bytes += nnz * 4  # column indices (int32)
        sparse_bytes += (n_data + 1) * 4  # row pointers (int32)
        sparse_gb = sparse_bytes / (1024**3)

        # Memory savings
        savings = (dense_gb - sparse_gb) / dense_gb * 100 if dense_gb > 0 else 0.0

        return {
            "dense_gb": dense_gb,
            "sparse_gb": sparse_gb,
            "savings_percent": savings,
            "sparsity": sparsity,
            "nnz": nnz,
            "reduction_factor": dense_gb / sparse_gb if sparse_gb > 0 else float("inf"),
        }


class SparseOptimizer:
    """Optimizer that uses sparse Jacobians for large-scale problems.

    This optimizer automatically detects when sparse Jacobians would be
    beneficial and switches to sparse computations transparently.
    """

    def __init__(
        self,
        sparsity_threshold: float = 0.01,
        min_sparsity: float = 0.9,
        auto_detect: bool = True,
    ):
        """Initialize sparse optimizer.

        Parameters
        ----------
        sparsity_threshold : float
            Relative per-column threshold for considering elements as zero
            (see `SparseJacobianComputer`)
        min_sparsity : float
            Minimum sparsity level to use sparse methods
        auto_detect : bool
            Automatically detect and use sparsity
        """
        self.sparsity_threshold = sparsity_threshold
        self.min_sparsity = min_sparsity
        self.auto_detect = auto_detect
        self.sparse_computer = SparseJacobianComputer(sparsity_threshold)
        self._use_sparse = False
        self._detected_sparsity = 0.0

    def should_use_sparse(
        self,
        n_data: int,
        n_params: int,
        force_check: bool = False,
    ) -> bool:
        """Determine if sparse methods should be used.

        Parameters
        ----------
        n_data : int
            Number of data points
        n_params : int
            Number of parameters
        force_check : bool
            Force sparsity detection even if auto_detect is False

        Returns
        -------
        use_sparse : bool
            Whether to use sparse methods
        """
        # Heuristic: use sparse for large problems
        problem_size = n_data * n_params

        if problem_size < 1e6:  # Less than 1M elements
            return False

        if not self.auto_detect and not force_check:
            # For very large problems, assume sparse is beneficial
            return problem_size > 1e8  # More than 100M elements

        # Auto-detect based on problem characteristics
        # Many curve fitting problems have local parameter influence
        expected_sparsity = 1.0 - min(10.0 / n_params, 1.0)

        return expected_sparsity >= self.min_sparsity

    def optimize_with_sparsity(
        self,
        func: Callable,
        x0: np.ndarray,
        xdata: np.ndarray,
        ydata: np.ndarray,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray] | CurveFitResult:
        """Optimize using sparse Jacobian methods.

        Parameters
        ----------
        func : Callable
            Objective function
        x0 : np.ndarray
            Initial parameters
        xdata : np.ndarray
            Independent variable data
        ydata : np.ndarray
            Dependent variable data
        **kwargs
            Additional optimization parameters

        Returns
        -------
        result : dict
            Optimization result
        """
        n_data = len(ydata)
        n_params = len(x0)

        # Check if sparse methods should be used
        self._use_sparse = self.should_use_sparse(n_data, n_params)

        if self._use_sparse:
            logger.info(
                f"Using sparse Jacobian methods for {n_data}×{n_params} problem",
            )

            # Detect sparsity pattern from samples
            sample_size = min(1000, n_data)
            sample_indices = np.random.default_rng(0).choice(
                n_data, sample_size, replace=False
            )
            logger.info(
                f"Sparsity detection samples {sample_size} of {n_data} points "
                "(seeded random subset)",
            )
            xdata = _normalize_xdata(xdata)
            if isinstance(xdata, list):
                xdata_sample: Any = [c[sample_indices] for c in xdata]
            elif _is_points_last_2d(xdata, n_data):
                xdata_sample = xdata[:, sample_indices]
            else:
                xdata_sample = xdata[sample_indices]
            _pattern, sparsity = self.sparse_computer.detect_sparsity_pattern(
                func,
                x0,
                xdata_sample,
                sample_size,
            )

            self._detected_sparsity = sparsity
            logger.info(f"Detected sparsity: {sparsity:.1%}")

            # Estimate memory savings
            memory_info = self.sparse_computer.estimate_memory_usage(
                n_data,
                n_params,
                sparsity,
            )
            logger.info(f"Memory savings: {memory_info['savings_percent']:.1f}%")
            logger.info(
                f"Dense: {memory_info['dense_gb']:.2f}GB → Sparse: {memory_info['sparse_gb']:.2f}GB",
            )

            # Use sparse methods if beneficial
            if sparsity >= self.min_sparsity:
                return self._optimize_sparse(func, x0, xdata, ydata, **kwargs)

        # Fall back to standard dense optimization
        logger.info(f"Using standard dense methods for {n_data}×{n_params} problem")
        from nlsq.core.minpack import curve_fit

        return curve_fit(func, xdata, ydata, p0=x0, **kwargs)

    def _optimize_sparse(
        self,
        func: Callable,
        x0: np.ndarray,
        xdata: np.ndarray,
        ydata: np.ndarray,
        **kwargs,
    ):
        """Internal sparse optimization implementation.

        Not implemented: this would integrate with the existing TRF optimizer
        using sparse matrix operations throughout, but that integration
        doesn't exist yet. Raises rather than silently returning `x0`
        unchanged as a fake "success" result -- returning a placeholder here
        previously meant a caller with a genuinely sparse problem
        (sparsity >= min_sparsity) got back an unoptimized `x0` with
        `success=True` and no indication the fit never ran.
        """
        raise NotImplementedError(
            "SparseOptimizer's sparse optimization path is not implemented. "
            "Use nlsq.core.minpack.curve_fit() directly (dense), or set "
            "min_sparsity=1.1 / auto_detect=False with a low problem-size "
            "threshold to avoid ever triggering this path via "
            "optimize_with_sparsity().",
        )


def detect_jacobian_sparsity(
    func: Callable,
    x0: np.ndarray,
    xdata_sample: np.ndarray | list,
    threshold: float = 0.01,
    n_samples: int = 100,
    points_last: bool | None = None,
) -> tuple[float, dict]:
    """Detect and analyze Jacobian sparsity for a given problem.

    Parameters
    ----------
    func : Callable
        Objective function
    x0 : np.ndarray
        Initial parameters
    xdata_sample : np.ndarray or list
        Sample of x data. Can be a single array for 1D problems,
        a list of arrays [X, Y] for 2D problems, or a 2D array
        with shape (k, N) for multi-dimensional coordinates.
    threshold : float
        Relative per-column threshold for zero elements (see
        `SparseJacobianComputer`)
    n_samples : int, default 100
        Number of leading points of ``xdata_sample`` that are evaluated. When
        ``xdata_sample`` holds more points, the rest are ignored and the
        truncation is logged.
    points_last : bool, optional
        Layout of a 2-D array ``xdata_sample``: ``True`` for ``(k, N)``,
        ``False`` for ``(N, k)``. Default: inferred from the shape.

    Returns
    -------
    sparsity : float
        Fraction of zero elements
    info : dict
        Additional sparsity information
    """
    # Get number of data points, handling multi-dimensional xdata
    xdata_sample = _normalize_xdata(xdata_sample)
    xdata_arr = np.asarray(xdata_sample)
    if isinstance(xdata_sample, list):
        n_data_points = len(xdata_sample[0])
    elif xdata_arr.ndim == 2 and (
        points_last
        if points_last is not None
        else xdata_arr.shape[0] <= xdata_arr.shape[1]
    ):
        # 2D array with shape (k, N) - N is number of data points
        n_data_points = xdata_arr.shape[1]
    else:
        n_data_points = len(xdata_arr)

    n_samples = _check_positive_int("n_samples", n_samples)
    if n_data_points == 0:
        raise ValueError("xdata_sample is empty; no points to detect sparsity on")
    computer = SparseJacobianComputer(threshold)
    if n_samples < n_data_points:
        logger.info(
            f"Jacobian sparsity detection evaluates the first {n_samples} of "
            f"{n_data_points} supplied points",
        )
    pattern, sparsity = computer.detect_sparsity_pattern(
        func,
        x0,
        xdata_sample,
        min(n_samples, n_data_points),
        points_last=points_last,
    )

    # Analyze pattern
    _n_data, _n_params = pattern.shape
    nnz_per_row = np.sum(pattern, axis=1)
    nnz_per_col = np.sum(pattern, axis=0)

    info = {
        "sparsity": sparsity,
        "nnz": np.sum(pattern),
        "avg_nnz_per_row": np.mean(nnz_per_row),
        "avg_nnz_per_col": np.mean(nnz_per_col) if nnz_per_col.size else 0.0,
        "max_nnz_per_row": np.max(nnz_per_row),
        "max_nnz_per_col": np.max(nnz_per_col, initial=0),
        "pattern_shape": pattern.shape,
        "memory_reduction": sparsity * 100,
    }

    return sparsity, info


def detect_sparsity_at_p0(
    func: Callable,
    p0: np.ndarray,
    xdata: np.ndarray,
    n_residuals: int,
    threshold: float = 0.01,
    sample_size: int = 100,
) -> tuple[float, bool]:
    """Detect sparsity at p0 initialization for automatic sparse solver selection.

    This function computes the Jacobian at the initial parameter guess p0 and
    calculates the sparsity ratio. The result is not cached.

    Parameters
    ----------
    func : Callable
        Model function f(x, \\*params) -> residuals
    p0 : np.ndarray
        Initial parameter guess
    xdata : np.ndarray
        Independent variable data
    n_residuals : int
        Number of residuals (data points)
    threshold : float, optional
        Relative per-column threshold for considering elements as zero
        (default: 0.01)
    sample_size : int, optional
        Number of data points to sample for detection (default: 100)
        Using a sample speeds up detection for large datasets

    Returns
    -------
    sparsity_ratio : float
        Fraction of zero elements in Jacobian (0.0 = dense, 1.0 = completely sparse)
        Example: 0.9 means 90% of Jacobian elements are zero
    is_sparse : bool
        Whether the problem is considered sparse (sparsity_ratio > 0.5)

    Notes
    -----
    Detection strategy:
    - Samples up to `sample_size` data points for efficiency
    - Uses finite differences to compute Jacobian at p0
    - Considers elements with \\|J[i,j]\\| <= threshold * max_i \\|J[i,j]\\| as zero
    - Samples are evenly spaced over the whole data range (logged at DEBUG)

    For auto-selection to activate sparse solver, both conditions must be met:
    - sparsity_ratio > 0.5 (more than 50% zeros)
    - n_residuals > 10000 (problem size threshold)

    Examples
    --------
    >>> def model(x, a, b):
    ...     # Each parameter affects different data regions (sparse)
    ...     return jnp.where(x < 0.5, a, b)
    >>> p0 = np.array([1.0, 2.0])
    >>> xdata = np.linspace(0, 1, 1000)
    >>> sparsity_ratio, is_sparse = detect_sparsity_at_p0(
    ...     model, p0, xdata, n_residuals=1000
    ... )
    >>> print(f"Sparsity: {sparsity_ratio:.1%}, Is sparse: {is_sparse}")
    Sparsity: 50.0%, Is sparse: False
    """
    sample_size = _check_positive_int("sample_size", sample_size)
    _check_threshold(threshold)

    # Sample data for efficient detection
    # Handle multi-dimensional xdata (e.g., [X, Y] for 2D fitting)
    xdata = _normalize_xdata(xdata)
    xdata_arr = np.asarray(xdata)
    # Layout of a 2-D array, decided once and passed to the detector as well
    # An unambiguous residual-count match beats the shape heuristic.
    points_last = xdata_arr.ndim == 2 and (
        _is_points_last_2d(xdata_arr, n_residuals)
        or (
            xdata_arr.shape[0] != n_residuals
            and xdata_arr.shape[0] <= xdata_arr.shape[1]
        )
    )

    if isinstance(xdata, list):
        # xdata is a list of coordinate arrays (e.g., [X, Y] for 2D problems)
        # Get the number of data points from the first coordinate array
        n_data_points = len(xdata[0])
        actual_sample_size = min(sample_size, n_residuals, n_data_points)
        if actual_sample_size < n_data_points:
            sample_indices = np.linspace(
                0,
                n_data_points - 1,
                actual_sample_size,
                dtype=int,
            )
            xdata_sample = [coord[sample_indices] for coord in xdata]
        else:
            xdata_sample = xdata
    elif xdata_arr.ndim == 2 and points_last:
        # xdata is a 2D array with shape (k, N) where k is number of coords
        # and N is number of data points (e.g., shape (2, 40000) for X,Y)
        n_data_points = xdata_arr.shape[1]
        actual_sample_size = min(sample_size, n_residuals, n_data_points)
        if actual_sample_size < n_data_points:
            sample_indices = np.linspace(
                0,
                n_data_points - 1,
                actual_sample_size,
                dtype=int,
            )
            xdata_sample = xdata_arr[:, sample_indices]
        else:
            xdata_sample = xdata
    else:
        # xdata is a single 1D array
        n_data_points = len(xdata_arr)
        actual_sample_size = min(sample_size, n_residuals, n_data_points)
        if actual_sample_size < n_data_points:
            sample_indices = np.linspace(
                0,
                n_data_points - 1,
                actual_sample_size,
                dtype=int,
            )
            xdata_sample = xdata_arr[sample_indices]
        else:
            xdata_sample = xdata

    # Use existing detection function
    sparsity_ratio, _info = detect_jacobian_sparsity(
        func,
        p0,
        xdata_sample,
        threshold=threshold,
        n_samples=actual_sample_size,
        points_last=points_last if xdata_arr.ndim == 2 else None,
    )

    # Determine if sparse based on sparsity threshold
    is_sparse = sparsity_ratio > 0.5

    logger.debug(
        f"Sparsity detection at p0 on {actual_sample_size} of {n_data_points} "
        f"points: {sparsity_ratio:.1%} ({'sparse' if is_sparse else 'dense'})",
    )

    return sparsity_ratio, is_sparse
