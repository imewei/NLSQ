"""Method selector for global optimization.

Selects between CMA-ES and multi-start optimization based on problem
characteristics and available dependencies.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np

from nlsq.global_optimization.cmaes_config import is_evosax_available

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

__all__ = ["MethodSelector"]

logger = logging.getLogger(__name__)

# Default threshold for scale ratio to prefer CMA-ES
# 1000x (3 orders of magnitude) difference in parameter scales
DEFAULT_SCALE_THRESHOLD = 1000.0

MethodType = Literal["cmaes", "multi-start", "auto"] | None


class MethodSelector:
    """Select optimization method based on problem characteristics.

    The selector analyzes the parameter bounds to compute a scale ratio,
    which indicates how many orders of magnitude separate the parameter
    scales. CMA-ES is preferred for multi-scale problems (high scale ratio)
    when evosax is available.

    Parameters
    ----------
    scale_threshold : float, optional
        Threshold for scale ratio above which CMA-ES is preferred.
        Default is 1000.0 (3 orders of magnitude).

    Attributes
    ----------
    scale_threshold : float
        The scale ratio threshold for CMA-ES preference.

    Examples
    --------
    >>> from nlsq.global_optimization import MethodSelector
    >>> import numpy as np
    >>>
    >>> selector = MethodSelector()
    >>> lower = np.array([1e2, 1e-5, 0.5])
    >>> upper = np.array([1e6, 1e-1, 3.0])
    >>>
    >>> method = selector.select('auto', lower, upper)
    >>> print(f"Selected method: {method}")
    """

    def __init__(self, scale_threshold: float = DEFAULT_SCALE_THRESHOLD) -> None:
        """Initialize MethodSelector.

        Parameters
        ----------
        scale_threshold : float, optional
            Threshold for scale ratio above which CMA-ES is preferred.
        """
        self.scale_threshold = scale_threshold

    def compute_scale_ratio(
        self, lower_bounds: ArrayLike, upper_bounds: ArrayLike
    ) -> float:
        """Compute the scale ratio from parameter bounds.

        The scale ratio is the ratio of the largest parameter range to
        the smallest, indicating how many orders of magnitude separate
        the parameter scales.

        Parameters
        ----------
        lower_bounds : ArrayLike
            Lower bounds for parameters.
        upper_bounds : ArrayLike
            Upper bounds for parameters.

        Returns
        -------
        float
            Scale ratio (>= 1.0). Higher values indicate more diverse scales.
        """
        lower = np.asarray(lower_bounds)
        upper = np.asarray(upper_bounds)

        # Compute ranges
        ranges = upper - lower

        # Filter out zero-width ranges (fixed parameters)
        nonzero_ranges = ranges[ranges > 0]

        if len(nonzero_ranges) == 0:
            return 1.0

        # Ratio of max to min range
        max_range = np.max(nonzero_ranges)
        min_range = np.min(nonzero_ranges)

        if min_range == 0:
            return float("inf")

        return float(max_range / min_range)

    def select(
        self,
        requested_method: MethodType,
        lower_bounds: ArrayLike,
        upper_bounds: ArrayLike,
    ) -> Literal["cmaes", "multi-start"]:
        """Select the optimization method to use.

        Parameters
        ----------
        requested_method : {'cmaes', 'multi-start', 'auto'} | None
            Requested method. If 'auto' or None, selection is based on
            scale ratio and evosax availability.
        lower_bounds : ArrayLike
            Lower bounds for parameters.
        upper_bounds : ArrayLike
            Upper bounds for parameters.

        Returns
        -------
        Literal['cmaes', 'multi-start']
            The selected optimization method.

        Notes
        -----
        Selection logic:

        1. If 'cmaes' requested and evosax available -> 'cmaes'
        2. If 'cmaes' requested but evosax unavailable -> 'multi-start' (with warning)
        3. If 'multi-start' requested -> 'multi-start'
        4. If 'auto' or None:

           - If scale_ratio > threshold and evosax available -> 'cmaes'
           - Otherwise -> 'multi-start'
        """
        evosax_available = is_evosax_available()

        # Explicit CMA-ES request
        if requested_method == "cmaes":
            if evosax_available:
                logger.debug("CMA-ES requested and evosax available")
                return "cmaes"
            else:
                logger.info(
                    "CMA-ES requested but evosax not installed. "
                    "Falling back to multi-start optimization. "
                    "Install evosax with: pip install 'nlsq[global]'"
                )
                return "multi-start"

        # Explicit multi-start request
        if requested_method == "multi-start":
            logger.debug("Multi-start explicitly requested")
            return "multi-start"

        # Auto selection (or None)
        scale_ratio = self.compute_scale_ratio(lower_bounds, upper_bounds)
        logger.debug(f"Auto method selection: scale_ratio={scale_ratio:.2f}")

        if scale_ratio > self.scale_threshold:
            if evosax_available:
                logger.info(
                    f"Scale ratio {scale_ratio:.0f}x exceeds threshold "
                    f"({self.scale_threshold:.0f}x). Using CMA-ES for "
                    "multi-scale optimization."
                )
                return "cmaes"
            else:
                logger.info(
                    f"Scale ratio {scale_ratio:.0f}x suggests CMA-ES would be "
                    "beneficial, but evosax not installed. Using multi-start. "
                    "Install evosax with: pip install 'nlsq[global]'"
                )
                return "multi-start"
        else:
            logger.debug(
                f"Scale ratio {scale_ratio:.0f}x below threshold "
                f"({self.scale_threshold:.0f}x). Using multi-start."
            )
            return "multi-start"
