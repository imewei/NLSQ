"""DataPreprocessor component for CurveFit decomposition.

Handles input validation, array conversion, data masking, and padding
for curve fitting operations. This component is extracted from the
CurveFit class as part of the God class decomposition.

Reference: specs/017-curve-fit-decomposition/spec.md FR-001
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from nlsq.interfaces.orchestration_protocol import PreprocessedData

if TYPE_CHECKING:
    from collections.abc import Callable

    from nlsq.types import ArrayLike


class DataPreprocessor:
    """Preprocessor for curve fitting input data.

    Handles:
    1. Input validation (type checking, finiteness)
    2. Array conversion (numpy/list to JAX)
    3. Length consistency checking
    4. Data masking for invalid points
    5. NaN/Inf handling via nan_policy

    Example:
        >>> preprocessor = DataPreprocessor()
        >>> data = preprocessor.preprocess(
        ...     f=my_model,
        ...     xdata=x_values,
        ...     ydata=y_values,
        ...     sigma=uncertainties,
        ...     check_finite=True,
        ... )
        >>> print(f"Valid points: {data.n_points}")
    """

    def preprocess(
        self,
        f: Callable[..., ArrayLike],
        xdata: ArrayLike,
        ydata: ArrayLike,
        *,
        sigma: ArrayLike | None = None,
        absolute_sigma: bool = False,
        check_finite: bool = True,
        nan_policy: str = "raise",
        stability_check: bool = False,
    ) -> PreprocessedData:
        """Validate and preprocess input data for curve fitting.

        Args:
            f: Model function to fit (used for parameter count detection)
            xdata: Independent variable data
            ydata: Dependent variable data (observations)
            sigma: Uncertainty/weights for observations
            absolute_sigma: If True, sigma is absolute; else relative
            check_finite: If True, raise on NaN/Inf values
            nan_policy: How to handle NaN: 'raise', 'omit', or 'propagate'
            stability_check: If True, run additional stability checks

        Returns:
            PreprocessedData with validated, converted arrays

        Raises:
            ValueError: If inputs are invalid (wrong shape, non-finite, etc.)
            TypeError: If inputs have wrong types
        """
        # Step 1: Convert to arrays
        xdata_arr, ydata_arr = self._convert_to_arrays(xdata, ydata, check_finite)

        # Step 2: Validate data is not empty
        if ydata_arr.size == 0:
            msg = "`ydata` must not be empty!"
            raise ValueError(msg)

        # Step 3: Validate length consistency
        m, xdims = self._validate_lengths(xdata_arr, ydata_arr)

        # Step 4: Handle NaN values based on policy
        has_nans_removed = False
        has_infs_removed = False

        if nan_policy == "omit":
            xdata_arr, ydata_arr, sigma, mask, has_nans_removed, has_infs_removed = (
                self._handle_nan_omit(xdata_arr, ydata_arr, sigma, xdims)
            )
            m = len(ydata_arr)
        else:
            mask = np.ones(m, dtype=bool)

        # Step 5: Validate sigma if provided
        sigma_arr = (
            self._validate_sigma(sigma, ydata_arr.shape) if sigma is not None else None
        )

        # Step 6: Convert to JAX arrays
        jnp_xdata = jnp.asarray(xdata_arr)
        jnp_ydata = jnp.asarray(ydata_arr)
        jnp_sigma = jnp.asarray(sigma_arr) if sigma_arr is not None else None
        jnp_mask = jnp.asarray(mask)

        return PreprocessedData(
            xdata=jnp_xdata,
            ydata=jnp_ydata,
            sigma=jnp_sigma,
            mask=jnp_mask,
            n_points=m,
            is_padded=False,
            original_length=m,
            has_nans_removed=has_nans_removed,
            has_infs_removed=has_infs_removed,
        )

    def _convert_to_arrays(
        self,
        xdata: ArrayLike,
        ydata: ArrayLike,
        check_finite: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert inputs to numpy arrays and validate finiteness.

        Args:
            xdata: Independent variable data
            ydata: Dependent variable data
            check_finite: Whether to check for finite values

        Returns:
            Tuple of (xdata_array, ydata_array)

        Raises:
            ValueError: If check_finite=True and data contains NaN/Inf
        """
        # Convert ydata
        if check_finite:
            ydata_arr = np.asarray_chkfinite(ydata, float)
        else:
            ydata_arr = np.asarray(ydata, float)

        # Convert xdata
        if hasattr(xdata, "__array__") or isinstance(
            xdata, (list, tuple, np.ndarray, jnp.ndarray)
        ):
            if check_finite:
                xdata_arr = np.asarray_chkfinite(xdata, float)
            else:
                xdata_arr = np.asarray(xdata, float)
        else:
            msg = "X needs arrays"
            raise ValueError(msg)

        return xdata_arr, ydata_arr

    def _validate_lengths(
        self, xdata: np.ndarray, ydata: np.ndarray
    ) -> tuple[int, int]:
        """Validate that X and Y data lengths match.

        Args:
            xdata: X data array
            ydata: Y data array

        Returns:
            Tuple of (data_length, x_dimensions)

        Raises:
            ValueError: If X and Y lengths don't match
        """
        m = len(ydata)
        xdims = xdata.ndim
        xlen = len(xdata) if xdims == 1 else len(xdata[0])

        if xlen != m:
            msg = "X and Y data lengths dont match"
            raise ValueError(msg)

        return m, xdims

    def _handle_nan_omit(
        self,
        xdata: np.ndarray,
        ydata: np.ndarray,
        sigma: ArrayLike | None,
        xdims: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, bool, bool]:
        """Handle NaN values by omitting them from data.

        Args:
            xdata: X data array
            ydata: Y data array
            sigma: Sigma array or None
            xdims: Dimensionality of xdata

        Returns:
            Tuple of (xdata, ydata, sigma, mask, has_nans_removed, has_infs_removed)
        """
        # Find valid indices (not NaN or Inf)
        y_valid = np.isfinite(ydata)

        if xdims == 1:
            x_valid = np.isfinite(xdata)
        else:
            # For 2D xdata, check all rows
            x_valid = np.all(np.isfinite(xdata), axis=0)

        valid_mask = y_valid & x_valid

        # Track what was removed
        has_nans = bool(np.any(np.isnan(ydata)) or np.any(np.isnan(xdata)))
        has_infs = bool(np.any(np.isinf(ydata)) or np.any(np.isinf(xdata)))

        # Filter data
        ydata_clean = ydata[valid_mask]

        if xdims == 1:
            xdata_clean = xdata[valid_mask]
        else:
            xdata_clean = xdata[:, valid_mask]

        sigma_clean = None
        if sigma is not None:
            sigma_arr = np.asarray(sigma)
            if sigma_arr.ndim == 1:
                sigma_clean = sigma_arr[valid_mask]
            else:
                # 2D covariance matrix - need to extract submatrix
                sigma_clean = sigma_arr[np.ix_(valid_mask, valid_mask)]

        # Create mask for clean data (all True since we filtered)
        mask = np.ones(len(ydata_clean), dtype=bool)

        return xdata_clean, ydata_clean, sigma_clean, mask, has_nans, has_infs

    def validate_sigma(
        self,
        sigma: ArrayLike | None,
        ydata_shape: tuple[int, ...],
    ) -> np.ndarray | None:
        """Validate and convert sigma to appropriate format.

        Public interface matching DataPreprocessorProtocol.

        Args:
            sigma: Input sigma (1D for diagonal, 2D for full covariance)
            ydata_shape: Shape of ydata for compatibility check

        Returns:
            Validated numpy array or None

        Raises:
            ValueError: If sigma shape is incompatible with ydata
        """
        return self._validate_sigma(sigma, ydata_shape)

    def _validate_sigma(
        self,
        sigma: ArrayLike | None,
        ydata_shape: tuple[int, ...],
    ) -> np.ndarray | None:
        """Validate and convert sigma to appropriate format.

        Args:
            sigma: Input sigma (1D for diagonal, 2D for full covariance)
            ydata_shape: Shape of ydata for compatibility check

        Returns:
            Validated numpy array or None

        Raises:
            ValueError: If sigma shape is incompatible with ydata
        """
        if sigma is None:
            return None

        sigma_arr = np.asarray(sigma, dtype=float)
        n = ydata_shape[0]

        if sigma_arr.ndim == 1:
            if len(sigma_arr) != n:
                msg = f"Sigma length ({len(sigma_arr)}) must match ydata length ({n})"
                raise ValueError(msg)
        elif sigma_arr.ndim == 2:
            if sigma_arr.shape != (n, n):
                msg = f"Sigma shape {sigma_arr.shape} must be ({n}, {n})"
                raise ValueError(msg)
        else:
            msg = f"Sigma must be 1D or 2D, got {sigma_arr.ndim}D"
            raise ValueError(msg)

        return sigma_arr
