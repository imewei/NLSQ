"""Optimization warning types for NLSQ.

This module provides the OptimizeWarning class, which is used when non-critical
issues are encountered during optimization.

This is the canonical location for OptimizeWarning.
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

    Examples
    --------
    >>> import warnings
    >>> warnings.filterwarnings('error', category=OptimizeWarning)
    >>> # Now OptimizeWarning will raise an exception instead of warning

    See Also
    --------
    nlsq.utils.error_messages : Exceptions for critical failures
    """


def _check_unknown_options(unknown_options, stacklevel: int = 2):
    """Warn if unknown solver options are provided.

    Parameters
    ----------
    unknown_options : dict
        Dictionary of options that were not recognized by the solver.
        If non-empty, a warning is issued listing the unknown option names.
    stacklevel : int, optional
        ``warnings.warn`` stack level; the default attributes the warning to
        the caller of this function.

    Warns
    -----
    OptimizeWarning
        If any unknown options are present.
    """
    if unknown_options:
        msg = ", ".join(map(str, unknown_options.keys()))
        warnings.warn(
            f"Unknown solver options: {msg}", OptimizeWarning, stacklevel=stacklevel
        )
