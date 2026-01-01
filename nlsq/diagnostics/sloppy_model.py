"""Sloppy model analysis for nonlinear least squares models.

This module provides the SloppyModelAnalyzer class for analyzing
eigenvalue spectra to identify stiff vs sloppy parameter directions.

Sloppy models are common in biological and chemical kinetics where:
- Eigenvalues of the FIM span many orders of magnitude
- Some parameter combinations are well-determined (stiff)
- Others are poorly-determined (sloppy)

The analyzer generates actionable issues:
- SLOPPY-001: Sloppy model behavior detected
- SLOPPY-002: Low effective dimensionality

References
----------
- Gutenkunst et al. (2007) "Universally Sloppy Parameter Sensitivities
  in Systems Biology Models"
- Transtrum et al. (2010) "Perspective: Sloppiness and emergent theories
  in physics, biology, and beyond"
"""

import time

import numpy as np

from nlsq.diagnostics.recommendations import get_recommendation
from nlsq.diagnostics.types import (
    DiagnosticsConfig,
    HealthStatus,
    IssueCategory,
    IssueSeverity,
    ModelHealthIssue,
    SloppyModelReport,
)


class SloppyModelAnalyzer:
    """Analyzer for sloppy model characteristics from Jacobian matrices.

    This class analyzes the eigenvalue spectrum of the Fisher Information
    Matrix (FIM) to identify stiff vs sloppy parameter directions. It
    detects sloppy behavior when eigenvalues span many orders of magnitude.

    Parameters
    ----------
    config : DiagnosticsConfig | None
        Configuration containing thresholds for sloppy analysis.
        If None, uses default configuration.

    Attributes
    ----------
    config : DiagnosticsConfig
        Configuration for the analyzer.

    Notes
    -----
    Sloppy detection logic:

    - A model is considered "sloppy" when its eigenvalue range exceeds
      a threshold derived from the config's sloppy_threshold.
    - The sloppy_threshold (default 1e-6) defines when a direction is
      classified as sloppy: eigenvalue < threshold * max_eigenvalue.
    - The overall is_sloppy flag is set when eigenvalue range (in orders
      of magnitude) is significant enough to cause practical issues.

    Examples
    --------
    >>> import numpy as np
    >>> from nlsq.diagnostics import DiagnosticsConfig
    >>> from nlsq.diagnostics.sloppy_model import SloppyModelAnalyzer
    >>> config = DiagnosticsConfig()
    >>> analyzer = SloppyModelAnalyzer(config)
    >>> J = np.random.randn(100, 3)  # 100 data points, 3 parameters
    >>> report = analyzer.analyze(J)
    >>> print(report.is_sloppy)
    False
    """

    # Default threshold for sloppy model detection: 2.0 orders of magnitude
    # This is a conservative threshold that identifies models where some
    # parameter combinations are 100x less well-determined than others.
    # More strict analysis uses 6+ orders of magnitude per the sloppy model literature.
    DEFAULT_SLOPPY_DETECTION_ORDERS = 2.0

    # Threshold for stiff direction classification: 10% of max eigenvalue
    STIFF_RATIO_THRESHOLD = 0.1

    # Threshold for sloppy direction classification: 1% of max eigenvalue
    SLOPPY_RATIO_THRESHOLD = 0.01

    def __init__(self, config: DiagnosticsConfig | None = None) -> None:
        """Initialize the sloppy model analyzer.

        Parameters
        ----------
        config : DiagnosticsConfig | None
            Configuration containing analysis thresholds.
            If None, uses default configuration.
        """
        self.config = config if config is not None else DiagnosticsConfig()

    def analyze(self, jacobian: np.ndarray) -> SloppyModelReport:
        """Analyze sloppy model characteristics from a Jacobian matrix.

        Computes the Fisher Information Matrix (FIM) as J.T @ J and
        analyzes its eigenvalue spectrum for sloppy behavior.

        Parameters
        ----------
        jacobian : np.ndarray
            Jacobian matrix of shape (n_data, n_params).

        Returns
        -------
        SloppyModelReport
            Report containing analysis results and any detected issues.

        Notes
        -----
        The analysis includes:

        1. FIM computation: FIM = J.T @ J
        2. Eigenvalue decomposition for spectrum analysis
        3. Eigenvalue range computation (log10 ratio)
        4. Stiff/sloppy direction classification
        5. Effective dimensionality using participation ratio
        6. Issue detection based on thresholds
        """
        start_time = time.perf_counter()

        # Validate input
        validation_result = self._validate_jacobian(jacobian, start_time)
        if validation_result is not None:
            return validation_result

        n_params = jacobian.shape[1]

        # Compute FIM
        fim = self._compute_fim(jacobian)

        # Analyze FIM eigenvalue spectrum
        return self._analyze_eigenvalue_spectrum(fim, n_params, start_time)

    def analyze_from_fim(self, fim: np.ndarray) -> SloppyModelReport:
        """Analyze sloppy model characteristics from a pre-computed FIM.

        Parameters
        ----------
        fim : np.ndarray
            Fisher Information Matrix of shape (n_params, n_params).

        Returns
        -------
        SloppyModelReport
            Report containing analysis results and any detected issues.
        """
        start_time = time.perf_counter()

        # Validate FIM
        if fim.ndim != 2 or fim.shape[0] != fim.shape[1]:
            return SloppyModelReport(
                available=False,
                error_message="FIM must be a square matrix",
                is_sloppy=False,
                eigenvalues=np.array([]),
                eigenvectors=None,
                eigenvalue_range=0.0,
                effective_dimensionality=0.0,
                stiff_indices=[],
                sloppy_indices=[],
                issues=[],
                health_status=HealthStatus.CRITICAL,
                computation_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        n_params = fim.shape[0]
        return self._analyze_eigenvalue_spectrum(fim, n_params, start_time)

    def _validate_jacobian(
        self, jacobian: np.ndarray, start_time: float
    ) -> SloppyModelReport | None:
        """Validate the Jacobian matrix.

        Parameters
        ----------
        jacobian : np.ndarray
            Jacobian matrix to validate.
        start_time : float
            Start time for timing computation.

        Returns
        -------
        SloppyModelReport | None
            Error report if validation fails, None otherwise.
        """
        # Check for empty Jacobian
        if jacobian.size == 0:
            return SloppyModelReport(
                available=False,
                error_message="Empty Jacobian matrix",
                is_sloppy=False,
                eigenvalues=np.array([]),
                eigenvectors=None,
                eigenvalue_range=0.0,
                effective_dimensionality=0.0,
                stiff_indices=[],
                sloppy_indices=[],
                issues=[],
                health_status=HealthStatus.CRITICAL,
                computation_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Check dimensions
        if jacobian.ndim != 2:
            return SloppyModelReport(
                available=False,
                error_message=f"Jacobian must be 2D, got {jacobian.ndim}D",
                is_sloppy=False,
                eigenvalues=np.array([]),
                eigenvectors=None,
                eigenvalue_range=0.0,
                effective_dimensionality=0.0,
                stiff_indices=[],
                sloppy_indices=[],
                issues=[],
                health_status=HealthStatus.CRITICAL,
                computation_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Check for NaN
        if np.any(np.isnan(jacobian)):
            return SloppyModelReport(
                available=False,
                error_message="Jacobian contains NaN values",
                is_sloppy=False,
                eigenvalues=np.array([]),
                eigenvectors=None,
                eigenvalue_range=0.0,
                effective_dimensionality=0.0,
                stiff_indices=[],
                sloppy_indices=[],
                issues=[],
                health_status=HealthStatus.CRITICAL,
                computation_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Check for Inf
        if np.any(np.isinf(jacobian)):
            return SloppyModelReport(
                available=False,
                error_message="Jacobian contains Inf values",
                is_sloppy=False,
                eigenvalues=np.array([]),
                eigenvectors=None,
                eigenvalue_range=0.0,
                effective_dimensionality=0.0,
                stiff_indices=[],
                sloppy_indices=[],
                issues=[],
                health_status=HealthStatus.CRITICAL,
                computation_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        return None

    def _compute_fim(self, jacobian: np.ndarray) -> np.ndarray:
        """Compute the Fisher Information Matrix.

        Parameters
        ----------
        jacobian : np.ndarray
            Jacobian matrix of shape (n_data, n_params).

        Returns
        -------
        np.ndarray
            Fisher Information Matrix of shape (n_params, n_params).
        """
        return jacobian.T @ jacobian

    def _analyze_eigenvalue_spectrum(
        self, fim: np.ndarray, n_params: int, start_time: float
    ) -> SloppyModelReport:
        """Analyze the eigenvalue spectrum of the FIM.

        Parameters
        ----------
        fim : np.ndarray
            Fisher Information Matrix.
        n_params : int
            Number of parameters.
        start_time : float
            Start time for timing computation.

        Returns
        -------
        SloppyModelReport
            Analysis results.
        """
        issues: list[ModelHealthIssue] = []
        health_status = HealthStatus.HEALTHY

        # Compute eigenvalue decomposition
        try:
            eigenvalues, eigenvectors = self._compute_eigendecomposition(fim)
        except Exception as e:
            # Graceful degradation on eigenvalue computation failure
            computation_time = (time.perf_counter() - start_time) * 1000
            return SloppyModelReport(
                available=False,
                error_message=f"Eigenvalue computation failed: {e!s}",
                is_sloppy=False,
                eigenvalues=np.array([]),
                eigenvectors=None,
                eigenvalue_range=0.0,
                effective_dimensionality=0.0,
                stiff_indices=[],
                sloppy_indices=[],
                issues=[],
                health_status=HealthStatus.CRITICAL,
                computation_time_ms=computation_time,
            )

        # Compute eigenvalue range (log10 ratio of max to min non-zero eigenvalue)
        eigenvalue_range = self._compute_eigenvalue_range(eigenvalues)

        # Compute sloppy detection threshold
        # The sloppy_threshold config parameter (default 1e-6) is used for direction
        # classification. For overall sloppy detection, we use a threshold that
        # identifies significant sloppiness (at least 2 orders of magnitude).
        sloppy_threshold_log = -np.log10(self.config.sloppy_threshold)
        sloppy_detection_threshold = max(
            sloppy_threshold_log / 3.0,
            self.DEFAULT_SLOPPY_DETECTION_ORDERS
        )

        # Check for sloppy model behavior (SLOPPY-001)
        # Use Python bool() to ensure is_sloppy is a native bool, not np.bool_
        is_sloppy = bool(eigenvalue_range > sloppy_detection_threshold)

        # Classify stiff vs sloppy directions
        # Pass is_sloppy to use adaptive thresholds when model is sloppy
        stiff_indices, sloppy_indices = self._classify_directions(
            eigenvalues, is_sloppy
        )

        # Compute effective dimensionality using participation ratio
        effective_dimensionality = self._compute_effective_dimensionality(eigenvalues)

        if is_sloppy:
            issue = self._create_sloppy_001_issue(
                eigenvalue_range, sloppy_detection_threshold
            )
            issues.append(issue)
            health_status = HealthStatus.WARNING

        # Check for low effective dimensionality (SLOPPY-002)
        # Threshold: effective_dim < n_params / 2
        if effective_dimensionality < n_params / 2.0:
            issue = self._create_sloppy_002_issue(
                effective_dimensionality, n_params
            )
            issues.append(issue)
            # Keep as INFO since this is informational, not necessarily a problem
            # health_status already set by SLOPPY-001 if applicable

        computation_time = (time.perf_counter() - start_time) * 1000

        return SloppyModelReport(
            available=True,
            is_sloppy=is_sloppy,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            eigenvalue_range=float(eigenvalue_range),
            effective_dimensionality=float(effective_dimensionality),
            stiff_indices=stiff_indices,
            sloppy_indices=sloppy_indices,
            issues=issues,
            health_status=health_status,
            computation_time_ms=computation_time,
        )

    def _compute_eigendecomposition(
        self, fim: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute eigenvalue decomposition of the FIM.

        Uses SVD for numerical stability since FIM = J.T @ J is symmetric
        positive semi-definite. The singular values are the eigenvalues,
        and V are the eigenvectors.

        Parameters
        ----------
        fim : np.ndarray
            Fisher Information Matrix.

        Returns
        -------
        eigenvalues : np.ndarray
            Eigenvalues sorted in descending order.
        eigenvectors : np.ndarray
            Eigenvectors as columns, corresponding to sorted eigenvalues.
        """
        try:
            # Try to use NLSQ's SVD fallback for robustness
            from nlsq.stability.svd_fallback import compute_svd_with_fallback

            # compute_svd_with_fallback returns (U, s, V) where V is already transposed
            # For symmetric matrix, U and V are the same (eigenvectors)
            _, singular_values, V = compute_svd_with_fallback(fim)
            eigenvalues = np.asarray(singular_values)
            eigenvectors = np.asarray(V)

            # Sort in descending order (should already be sorted by SVD)
            sort_idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[sort_idx]
            eigenvectors = eigenvectors[:, sort_idx]

        except ImportError:
            # Fallback to numpy SVD if module not available
            _, singular_values, Vt = np.linalg.svd(fim)
            eigenvalues = singular_values
            eigenvectors = Vt.T  # Columns are eigenvectors

            # Sort in descending order (should already be sorted by SVD)
            sort_idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[sort_idx]
            eigenvectors = eigenvectors[:, sort_idx]

        # Ensure eigenvalues are non-negative (numerical precision may give tiny negatives)
        eigenvalues = np.maximum(eigenvalues, 0.0)

        return eigenvalues, eigenvectors

    def _compute_eigenvalue_range(self, eigenvalues: np.ndarray) -> float:
        """Compute the eigenvalue range as log10(max/min).

        Parameters
        ----------
        eigenvalues : np.ndarray
            Eigenvalues sorted in descending order.

        Returns
        -------
        float
            Log10 range of eigenvalues (orders of magnitude).
            Returns 0.0 if all eigenvalues are zero or only one non-zero eigenvalue.
        """
        # Filter to non-zero eigenvalues
        nonzero_eigenvalues = eigenvalues[eigenvalues > 0]

        if len(nonzero_eigenvalues) <= 1:
            # Single eigenvalue or all zeros - no range
            return 0.0

        max_eigenvalue = np.max(nonzero_eigenvalues)
        min_eigenvalue = np.min(nonzero_eigenvalues)

        # Compute log10 ratio
        if min_eigenvalue > 0:
            return float(np.log10(max_eigenvalue / min_eigenvalue))
        else:
            return float("inf")

    def _classify_directions(
        self, eigenvalues: np.ndarray, is_sloppy: bool
    ) -> tuple[list[int], list[int]]:
        """Classify directions as stiff or sloppy based on eigenvalues.

        Stiff directions have eigenvalues close to the maximum.
        Sloppy directions have eigenvalues much smaller than the maximum.

        Parameters
        ----------
        eigenvalues : np.ndarray
            Eigenvalues sorted in descending order.
        is_sloppy : bool
            Whether the model is overall sloppy.

        Returns
        -------
        stiff_indices : list[int]
            Indices of stiff (well-determined) directions.
        sloppy_indices : list[int]
            Indices of sloppy (poorly-determined) directions.
        """
        if len(eigenvalues) == 0:
            return [], []

        max_eigenvalue = np.max(eigenvalues)

        if max_eigenvalue <= 0:
            # All zero eigenvalues - all are sloppy
            return [], list(range(len(eigenvalues)))

        # For sloppy models, use adaptive thresholds based on eigenvalue distribution
        # For non-sloppy models, use fixed thresholds
        if is_sloppy:
            # In sloppy models, classify directions relative to the spectrum
            # Sloppy: bottom 1/3 of eigenvalues (by value, not count)
            # Stiff: top 1/3 of eigenvalues
            # Use geometric mean of sqrt of ratio as threshold
            # For a 3-order range, sqrt = 1.5 orders, so threshold = 10^(-1.5) ~ 0.03
            eigenvalue_range = self._compute_eigenvalue_range(eigenvalues)
            if eigenvalue_range > 0:
                # Threshold based on eigenvalue range
                # For 3-order range: sloppy < 10^(-1.5) * max ~ 3% of max
                # For 6-order range: sloppy < 10^(-3) * max ~ 0.1% of max
                sloppy_threshold = 10.0 ** (-eigenvalue_range / 2.0)
                stiff_threshold = 10.0 ** (-eigenvalue_range / 4.0)
            else:
                sloppy_threshold = self.SLOPPY_RATIO_THRESHOLD
                stiff_threshold = self.STIFF_RATIO_THRESHOLD
        else:
            # Non-sloppy: use config-based thresholds
            sloppy_threshold = self.config.sloppy_threshold
            stiff_threshold = self.STIFF_RATIO_THRESHOLD

        stiff_indices = []
        sloppy_indices = []

        for i, eigenvalue in enumerate(eigenvalues):
            ratio = eigenvalue / max_eigenvalue

            if ratio < sloppy_threshold:
                sloppy_indices.append(i)
            elif ratio >= stiff_threshold:
                stiff_indices.append(i)
            # Intermediate eigenvalues are neither

        return stiff_indices, sloppy_indices

    def _compute_effective_dimensionality(self, eigenvalues: np.ndarray) -> float:
        """Compute effective dimensionality using participation ratio.

        The participation ratio is defined as:
            effective_dim = (sum(eigenvalues))^2 / sum(eigenvalues^2)

        This equals n for n equal eigenvalues, and 1 when one eigenvalue dominates.

        Parameters
        ----------
        eigenvalues : np.ndarray
            Eigenvalues of the FIM.

        Returns
        -------
        float
            Effective dimensionality in [0, n_params].
        """
        # Filter to positive eigenvalues to avoid numerical issues
        positive_eigenvalues = eigenvalues[eigenvalues > 0]

        if len(positive_eigenvalues) == 0:
            return 0.0

        sum_eigenvalues = np.sum(positive_eigenvalues)
        sum_eigenvalues_squared = np.sum(positive_eigenvalues ** 2)

        if sum_eigenvalues_squared <= 0:
            return 0.0

        # Participation ratio
        effective_dim = (sum_eigenvalues ** 2) / sum_eigenvalues_squared

        return float(effective_dim)

    def _create_sloppy_001_issue(
        self, eigenvalue_range: float, threshold: float
    ) -> ModelHealthIssue:
        """Create SLOPPY-001 issue for sloppy model detection.

        Parameters
        ----------
        eigenvalue_range : float
            Log10 range of eigenvalues.
        threshold : float
            Threshold used for sloppy detection (in orders of magnitude).

        Returns
        -------
        ModelHealthIssue
            Issue describing sloppy model behavior.
        """
        return ModelHealthIssue(
            category=IssueCategory.SLOPPY,
            severity=IssueSeverity.WARNING,
            code="SLOPPY-001",
            message=(
                f"Sloppy model detected: eigenvalue spectrum spans "
                f"{eigenvalue_range:.1f} orders of magnitude. "
                "Some parameter combinations are poorly determined."
            ),
            affected_parameters=None,
            details={
                "eigenvalue_range": eigenvalue_range,
                "threshold_orders_of_magnitude": threshold,
            },
            recommendation=get_recommendation("SLOPPY-001"),
        )

    def _create_sloppy_002_issue(
        self, effective_dimensionality: float, n_params: int
    ) -> ModelHealthIssue:
        """Create SLOPPY-002 issue for low effective dimensionality.

        Parameters
        ----------
        effective_dimensionality : float
            Computed effective dimensionality.
        n_params : int
            Total number of parameters.

        Returns
        -------
        ModelHealthIssue
            Issue describing low effective dimensionality.
        """
        return ModelHealthIssue(
            category=IssueCategory.SLOPPY,
            severity=IssueSeverity.INFO,
            code="SLOPPY-002",
            message=(
                f"Low effective dimensionality: {effective_dimensionality:.1f} "
                f"out of {n_params} parameters. "
                "The model may be overparameterized for the available data."
            ),
            affected_parameters=None,
            details={
                "effective_dimensionality": effective_dimensionality,
                "n_params": n_params,
                "ratio": effective_dimensionality / n_params if n_params > 0 else 0.0,
            },
            recommendation=get_recommendation("SLOPPY-002"),
        )
