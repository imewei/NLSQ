"""DEPRECATED: Use nlsq.result instead.

This module is maintained for backward compatibility only.
Import from nlsq.result instead:

    # Old (deprecated):
    from nlsq.core._optimize import OptimizeResult, OptimizeWarning

    # New (recommended):
    from nlsq.result import OptimizeResult, OptimizeWarning

This shim will be removed in v0.5.0.
"""

import warnings

# Re-export from new canonical location
from nlsq.result import OptimizeResult, OptimizeWarning, _check_unknown_options

# Issue deprecation warning on import
warnings.warn(
    "Importing from nlsq.core._optimize is deprecated. "
    "Use 'from nlsq.result import OptimizeResult, OptimizeWarning' instead. "
    "This module will be removed in v0.5.0.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["OptimizeResult", "OptimizeWarning", "_check_unknown_options"]
