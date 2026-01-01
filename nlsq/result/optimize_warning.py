"""Optimization warning types for NLSQ.

This module provides the OptimizeWarning class, which is used when non-critical
issues are encountered during optimization.

This is the canonical location for OptimizeWarning. For backward compatibility,
the class is also re-exported from nlsq.core._optimize (deprecated).
"""

import warnings


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
