"""Identifiability analysis for nonlinear least squares models.

This module provides the IdentifiabilityAnalyzer class for analyzing
parameter identifiability from the Jacobian matrix. It computes:

- Fisher Information Matrix (FIM) from the Jacobian
- Condition number and numerical rank for identifiability assessment
- Parameter correlation matrix
- Detection of highly correlated parameter pairs

The analyzer generates actionable issues:
- IDENT-001: Structural unidentifiability (rank-deficient FIM)
- IDENT-002: Practical unidentifiability (ill-conditioned FIM)
- CORR-001: Highly correlated parameters
"""

import time

import numpy as np

from nlsq.diagnostics.recommendations import get_recommendation
from nlsq.diagnostics.types import (
    DiagnosticsConfig,
    HealthStatus,
    IdentifiabilityReport,
    IssueCategory,
    IssueSeverity,
    ModelHealthIssue,
)


class IdentifiabilityAnalyzer:
    """Analyzer for parameter identifiability from Jacobian matrices.

    This class analyzes the Fisher Information Matrix (FIM) derived from
    the Jacobian to assess parameter identifiability. It detects both
    structural unidentifiability (rank deficiency) and practical
    unidentifiability (ill-conditioning).

    Parameters
    ----------
    config : DiagnosticsConfig
        Configuration containing thresholds for identifiability analysis.

    Attributes
    ----------
    config : DiagnosticsConfig
        Configuration for the analyzer.

    Examples
    --------
    >>> import numpy as np
    >>> from nlsq.diagnostics import DiagnosticsConfig
    >>> from nlsq.diagnostics.identifiability import IdentifiabilityAnalyzer
    >>> config = DiagnosticsConfig()
    >>> analyzer = IdentifiabilityAnalyzer(config)
    >>> J = np.random.randn(100, 3)  # 100 data points, 3 parameters
    >>> report = analyzer.analyze(J)
    >>> print(report.health_status)
    HealthStatus.HEALTHY
    """

    def __init__(self, config: DiagnosticsConfig) -> None:
        """Initialize the identifiability analyzer.

        Parameters
        ----------
        config : DiagnosticsConfig
            Configuration containing analysis thresholds.
        """
        self.config = config

    def analyze(self, jacobian: np.ndarray) -> IdentifiabilityReport:
        """Analyze identifiability from a Jacobian matrix.

        Computes the Fisher Information Matrix (FIM) as J.T @ J and
        analyzes it for identifiability issues.

        Parameters
        ----------
        jacobian : np.ndarray
            Jacobian matrix of shape (n_data, n_params).

        Returns
        -------
        IdentifiabilityReport
            Report containing analysis results and any detected issues.

        Notes
        -----
        The analysis includes:

        1. FIM computation: FIM = J.T @ J
        2. SVD of FIM for condition number and rank
        3. Correlation matrix extraction
        4. Issue detection based on thresholds
        """
        start_time = time.perf_counter()

        # Validate input
        validation_result = self._validate_jacobian(jacobian)
        if validation_result is not None:
            validation_result.computation_time_ms = (
                time.perf_counter() - start_time
            ) * 1000
            return validation_result

        n_params = jacobian.shape[1]

        # Compute FIM
        fim = self._compute_fim(jacobian)

        # Analyze FIM
        return self._analyze_fim(fim, n_params, start_time)

    def analyze_from_fim(self, fim: np.ndarray) -> IdentifiabilityReport:
        """Analyze identifiability from a pre-computed FIM.

        Parameters
        ----------
        fim : np.ndarray
            Fisher Information Matrix of shape (n_params, n_params).

        Returns
        -------
        IdentifiabilityReport
            Report containing analysis results and any detected issues.
        """
        start_time = time.perf_counter()

        # Validate FIM
        if fim.ndim != 2 or fim.shape[0] != fim.shape[1]:
            return IdentifiabilityReport(
                available=False,
                error_message="FIM must be a square matrix",
                n_params=fim.shape[0] if fim.ndim == 2 else 0,
            )

        n_params = fim.shape[0]
        return self._analyze_fim(fim, n_params, start_time)

    def _validate_jacobian(self, jacobian: np.ndarray) -> IdentifiabilityReport | None:
        """Validate the Jacobian matrix.

        Parameters
        ----------
        jacobian : np.ndarray
            Jacobian matrix to validate.

        Returns
        -------
        IdentifiabilityReport | None
            Error report if validation fails, None otherwise.
        """
        # Check for empty Jacobian
        if jacobian.size == 0:
            return IdentifiabilityReport(
                available=False,
                error_message="Empty Jacobian matrix",
                n_params=jacobian.shape[1] if jacobian.ndim == 2 else 0,
            )

        # Check dimensions
        if jacobian.ndim != 2:
            return IdentifiabilityReport(
                available=False,
                error_message=f"Jacobian must be 2D, got {jacobian.ndim}D",
                n_params=0,
            )

        # Check for NaN
        if np.any(np.isnan(jacobian)):
            return IdentifiabilityReport(
                available=False,
                error_message="Jacobian contains NaN values",
                n_params=jacobian.shape[1],
            )

        # Check for Inf
        if np.any(np.isinf(jacobian)):
            return IdentifiabilityReport(
                available=False,
                error_message="Jacobian contains Inf values",
                n_params=jacobian.shape[1],
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

    def _analyze_fim(
        self, fim: np.ndarray, n_params: int, start_time: float
    ) -> IdentifiabilityReport:
        """Analyze the Fisher Information Matrix.

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
        IdentifiabilityReport
            Analysis results.
        """
        issues: list[ModelHealthIssue] = []
        health_status = HealthStatus.HEALTHY

        # Compute SVD for condition number and rank
        try:
            svd_result = self._compute_svd(fim)
        except Exception as e:
            # Graceful degradation on SVD failure
            computation_time = (time.perf_counter() - start_time) * 1000
            return IdentifiabilityReport(
                available=False,
                error_message=f"SVD computation failed: {e!s}",
                condition_number=float("inf"),
                numerical_rank=0,
                n_params=n_params,
                correlation_matrix=None,
                highly_correlated_pairs=[],
                issues=[],
                health_status=HealthStatus.CRITICAL,
                computation_time_ms=computation_time,
            )

        _, condition_number, numerical_rank = svd_result

        # Compute correlation matrix
        correlation_matrix = self._compute_correlation_matrix(fim)

        # Detect highly correlated pairs
        highly_correlated_pairs = self._detect_highly_correlated_pairs(
            correlation_matrix
        )

        # Check for structural unidentifiability (IDENT-001)
        if numerical_rank < n_params:
            issue = self._create_ident_001_issue(numerical_rank, n_params)
            issues.append(issue)
            health_status = HealthStatus.CRITICAL

        # Check for practical unidentifiability (IDENT-002)
        if condition_number > self.config.condition_threshold:
            issue = self._create_ident_002_issue(condition_number)
            issues.append(issue)
            if health_status != HealthStatus.CRITICAL:
                health_status = HealthStatus.WARNING

        # Check for high correlations (CORR-001)
        if highly_correlated_pairs:
            issue = self._create_corr_001_issue(highly_correlated_pairs)
            issues.append(issue)
            if health_status != HealthStatus.CRITICAL:
                health_status = HealthStatus.WARNING

        computation_time = (time.perf_counter() - start_time) * 1000

        return IdentifiabilityReport(
            available=True,
            condition_number=condition_number,
            numerical_rank=numerical_rank,
            n_params=n_params,
            correlation_matrix=correlation_matrix,
            highly_correlated_pairs=highly_correlated_pairs,
            issues=issues,
            health_status=health_status,
            computation_time_ms=computation_time,
        )

    def _compute_svd(self, fim: np.ndarray) -> tuple[np.ndarray, float, int]:
        """Compute SVD of the FIM for condition number and rank.

        Uses the existing nlsq.stability.svd_fallback module for robust
        SVD computation with GPU/CPU fallback.

        Parameters
        ----------
        fim : np.ndarray
            Fisher Information Matrix.

        Returns
        -------
        singular_values : np.ndarray
            Singular values of the FIM.
        condition_number : float
            Condition number (ratio of largest to smallest singular value).
        numerical_rank : int
            Numerical rank based on singular value tolerance.
        """
        try:
            # Try to use NLSQ's SVD fallback for robustness
            from nlsq.stability.svd_fallback import compute_svd_with_fallback

            # compute_svd_with_fallback returns (U, s, V)
            _, singular_values, _ = compute_svd_with_fallback(fim)
            singular_values = np.asarray(singular_values)
        except ImportError:
            # Fallback to numpy SVD if module not available
            singular_values = np.linalg.svd(fim, compute_uv=False)

        # Handle edge case of all-zero FIM
        max_sv = np.max(singular_values)
        min_sv_nonzero = singular_values[singular_values > 0]

        if max_sv == 0 or len(min_sv_nonzero) == 0:
            condition_number = float("inf")
            numerical_rank = 0
        else:
            min_sv = np.min(min_sv_nonzero)
            condition_number = float(max_sv / min_sv)

            # Compute numerical rank with relative tolerance
            tol = max_sv * max(fim.shape) * np.finfo(fim.dtype).eps
            numerical_rank = int(np.sum(singular_values > tol))

        return singular_values, condition_number, numerical_rank

    def _compute_correlation_matrix(self, fim: np.ndarray) -> np.ndarray | None:
        """Compute the correlation matrix from the FIM.

        The correlation matrix is derived from the covariance matrix,
        which is the inverse of the FIM.

        Parameters
        ----------
        fim : np.ndarray
            Fisher Information Matrix.

        Returns
        -------
        np.ndarray | None
            Correlation matrix, or None if computation fails.
        """
        try:
            # Get diagonal elements for normalization
            diag = np.diag(fim)

            # Handle zero or near-zero diagonal elements
            if np.any(diag <= 0):
                # FIM should be positive semi-definite, but handle edge cases
                return None

            # Compute correlation directly from FIM
            # correlation[i,j] = FIM[i,j] / sqrt(FIM[i,i] * FIM[j,j])
            sqrt_diag = np.sqrt(diag)
            correlation = fim / np.outer(sqrt_diag, sqrt_diag)

            # Clip to [-1, 1] to handle numerical precision issues
            correlation = np.clip(correlation, -1.0, 1.0)

            return correlation

        except Exception:
            return None

    def _detect_highly_correlated_pairs(
        self, correlation_matrix: np.ndarray | None
    ) -> list[tuple[int, int, float]]:
        """Detect pairs of parameters with high correlation.

        Parameters
        ----------
        correlation_matrix : np.ndarray | None
            Parameter correlation matrix.

        Returns
        -------
        list[tuple[int, int, float]]
            List of (param_i, param_j, correlation) for highly correlated pairs.
        """
        if correlation_matrix is None:
            return []

        pairs = []
        n = correlation_matrix.shape[0]
        threshold = self.config.correlation_threshold

        for i in range(n):
            for j in range(i + 1, n):
                corr = abs(correlation_matrix[i, j])
                if corr >= threshold:
                    pairs.append((i, j, float(correlation_matrix[i, j])))

        return pairs

    def _create_ident_001_issue(
        self, numerical_rank: int, n_params: int
    ) -> ModelHealthIssue:
        """Create IDENT-001 issue for structural unidentifiability.

        Parameters
        ----------
        numerical_rank : int
            Numerical rank of the FIM.
        n_params : int
            Total number of parameters.

        Returns
        -------
        ModelHealthIssue
            Issue describing structural unidentifiability.
        """
        n_unidentifiable = n_params - numerical_rank
        return ModelHealthIssue(
            category=IssueCategory.IDENTIFIABILITY,
            severity=IssueSeverity.CRITICAL,
            code="IDENT-001",
            message=(
                f"Structural unidentifiability: FIM has rank {numerical_rank} "
                f"but model has {n_params} parameters. "
                f"{n_unidentifiable} parameter combination(s) cannot be uniquely determined."
            ),
            affected_parameters=None,  # Would need more analysis to determine which
            details={
                "numerical_rank": numerical_rank,
                "n_params": n_params,
                "n_unidentifiable": n_unidentifiable,
            },
            recommendation=get_recommendation("IDENT-001"),
        )

    def _create_ident_002_issue(self, condition_number: float) -> ModelHealthIssue:
        """Create IDENT-002 issue for practical unidentifiability.

        Parameters
        ----------
        condition_number : float
            Condition number of the FIM.

        Returns
        -------
        ModelHealthIssue
            Issue describing practical unidentifiability.
        """
        return ModelHealthIssue(
            category=IssueCategory.IDENTIFIABILITY,
            severity=IssueSeverity.WARNING,
            code="IDENT-002",
            message=(
                f"Practical unidentifiability: FIM condition number is {condition_number:.2e}, "
                f"exceeding threshold {self.config.condition_threshold:.2e}. "
                "Some parameter combinations may be poorly determined."
            ),
            affected_parameters=None,
            details={
                "condition_number": condition_number,
                "threshold": self.config.condition_threshold,
            },
            recommendation=get_recommendation("IDENT-002"),
        )

    def _create_corr_001_issue(
        self, correlated_pairs: list[tuple[int, int, float]]
    ) -> ModelHealthIssue:
        """Create CORR-001 issue for highly correlated parameters.

        Parameters
        ----------
        correlated_pairs : list[tuple[int, int, float]]
            List of (param_i, param_j, correlation) pairs.

        Returns
        -------
        ModelHealthIssue
            Issue describing high correlation.
        """
        # Collect all affected parameter indices
        affected = set()
        for i, j, _ in correlated_pairs:
            affected.add(i)
            affected.add(j)

        # Format pairs for message
        pair_strs = [f"({i}, {j}): {corr:.3f}" for i, j, corr in correlated_pairs]
        pairs_text = ", ".join(pair_strs[:5])  # Limit to first 5 pairs
        if len(correlated_pairs) > 5:
            pairs_text += f", ... ({len(correlated_pairs) - 5} more)"

        return ModelHealthIssue(
            category=IssueCategory.CORRELATION,
            severity=IssueSeverity.WARNING,
            code="CORR-001",
            message=(
                f"Highly correlated parameters detected: {pairs_text}. "
                f"Threshold: {self.config.correlation_threshold:.2f}"
            ),
            affected_parameters=tuple(sorted(affected)),
            details={
                "correlated_pairs": correlated_pairs,
                "threshold": self.config.correlation_threshold,
                "n_pairs": len(correlated_pairs),
            },
            recommendation=get_recommendation("CORR-001"),
        )
