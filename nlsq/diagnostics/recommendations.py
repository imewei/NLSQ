"""Recommendation text mappings for diagnostic issue codes.

This module provides human-readable, actionable recommendations for each
diagnostic issue code. Recommendations are designed to help users understand
and resolve issues detected during model health analysis.

Issue Code Format
-----------------
Issue codes follow the pattern: CATEGORY-NNN

Categories:
- IDENT: Identifiability issues
- CORR: Correlation issues
- GRAD: Gradient issues
- COND: Conditioning issues
- SENS: Parameter sensitivity issues
- CONV: Convergence issues

Usage
-----
>>> from nlsq.diagnostics.recommendations import RECOMMENDATIONS
>>> recommendation = RECOMMENDATIONS.get("IDENT-001", "No recommendation available")
>>> print(recommendation)
"""


class Recommendations(dict[str, str]):
    """Mapping of issue codes to recommendation text."""


# Mapping of issue codes to recommendation text
RECOMMENDATIONS = Recommendations(
    {
        # Identifiability issues
        "IDENT-001": (
            "Structural unidentifiability detected: The Jacobian matrix is rank-deficient, "
            "meaning some parameters cannot be uniquely determined from the data. "
            "Consider: (1) Reparameterizing the model to reduce the number of parameters, "
            "(2) Adding constraints between parameters, or (3) Collecting additional data "
            "that provides information about the unidentifiable parameters."
        ),
        "IDENT-002": (
            "Practical unidentifiability detected: The Fisher Information Matrix has a "
            "very high condition number, indicating that some parameter combinations are "
            "poorly determined by the data. Consider: (1) Increasing the amount of data, "
            "(2) Improving the signal-to-noise ratio, (3) Sampling at more informative "
            "experimental conditions, or (4) Using regularization techniques."
        ),
        # Correlation issues
        "CORR-001": (
            "Highly correlated parameters detected: Some parameters are strongly correlated, "
            "which can lead to large uncertainties and unstable fits. Consider: "
            "(1) Combining correlated parameters into a single effective parameter, "
            "(2) Fixing one of the correlated parameters to a known value, or "
            "(3) Collecting data that better distinguishes between the correlated effects."
        ),
        # Gradient issues
        "GRAD-001": (
            "Vanishing gradients detected: Gradient magnitudes became very small during "
            "optimization while the cost function was still significant. This may indicate: "
            "(1) A flat region in the cost landscape, (2) Poor parameter scaling, or "
            "(3) Numerical precision issues. Consider: (1) Rescaling parameters to similar "
            "magnitudes, (2) Using tighter bounds, or (3) Trying different initial guesses."
        ),
        "GRAD-002": (
            "Gradient imbalance detected: The gradient magnitudes for different parameters "
            "differ by many orders of magnitude. This can slow convergence and cause "
            "numerical issues. Consider: (1) Normalizing or rescaling parameters, "
            "(2) Using parameter transformations (e.g., log-scale for rate constants), or "
            "(3) Applying preconditioning to balance parameter sensitivities."
        ),
        "GRAD-003": (
            "Gradient stagnation detected: The gradient norm remained nearly constant for "
            "multiple consecutive iterations. This may indicate: (1) Convergence to a "
            "local minimum, (2) A saddle point, or (3) Numerical precision limits. "
            "Consider: (1) Trying different initial guesses, (2) Using a global optimization "
            "method first, or (3) Checking for model implementation issues."
        ),
        # Conditioning issues
        "COND-001": (
            "Ill-conditioned Jacobian detected: The Jacobian matrix has a high condition "
            "number, which can lead to numerical instability and unreliable parameter "
            "estimates. Consider: (1) Rescaling the data or parameters, (2) Simplifying "
            "the model, or (3) Using regularization to improve conditioning."
        ),
        # Parameter sensitivity issues
        "SENS-001": (
            "Wide parameter sensitivity spectrum detected: The eigenvalue spectrum of the "
            "Fisher Information Matrix spans many orders of magnitude, indicating that some "
            "parameter combinations are well-determined (stiff) while others are "
            "poorly-determined. This is common in complex nonlinear models with many "
            "parameters. Consider: (1) Focusing on predictions rather than "
            "individual parameter values, (2) Reparameterizing along stiff directions, "
            "or (3) Using ensemble methods that account for parameter uncertainty."
        ),
        "SENS-002": (
            "Low effective dimensionality detected: The model has fewer well-determined "
            "parameter combinations than total parameters. This suggests the model may "
            "be overparameterized for the available data. Consider: (1) Reducing the "
            "number of parameters, (2) Collecting more informative data, or "
            "(3) Accepting that only certain parameter combinations can be determined."
        ),
        # Convergence issues
        "CONV-001": (
            "Slow convergence detected: The optimization required many iterations to "
            "converge. Consider: (1) Providing better initial parameter guesses, "
            "(2) Rescaling parameters for better conditioning, or (3) Using a more "
            "aggressive optimization strategy."
        ),
        "CONV-002": (
            "Convergence to bounds detected: One or more parameters converged to their "
            "boundary values. This may indicate: (1) Bounds are too restrictive, "
            "(2) The true parameter values lie outside the specified bounds, or "
            "(3) The model is inappropriate for the data. Consider: (1) Widening the "
            "bounds, (2) Checking the model formulation, or (3) Examining the data "
            "for outliers or systematic errors."
        ),
    }
)


def get_recommendation(code: str) -> str:
    """Get the recommendation text for an issue code.

    Parameters
    ----------
    code : str
        The issue code (e.g., "IDENT-001", "GRAD-002").

    Returns
    -------
    str
        The recommendation text, or a default message if the code is not found.

    Examples
    --------
    >>> get_recommendation("IDENT-001")  # doctest: +ELLIPSIS
    'Structural unidentifiability detected: ...'
    >>> get_recommendation("UNKNOWN-999")
    'No specific recommendation available for this issue.'
    """
    return RECOMMENDATIONS.get(
        code, "No specific recommendation available for this issue."
    )
