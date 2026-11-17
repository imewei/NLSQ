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

import logging
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
    _validate_fim,
)

_logger = logging.getLogger(__name__)


def _equilibrate_columns(jacobian: np.ndarray) -> np.ndarray:
    """Scale each Jacobian column to unit 2-norm (zero columns stay zero)."""
    jacobian = _as_float64(jacobian)
    cmax = np.max(np.abs(jacobian), axis=0)
    jn = jacobian / np.where(cmax > 0, cmax, 1.0)  # pre-scale: avoids norm underflow
    norms = np.linalg.norm(jn, axis=0)
    return jn / np.where(norms > 0, norms, 1.0)


def _column_log_scale(jacobian: np.ndarray) -> np.ndarray:
    """log of the per-column scale applied by :func:`_equilibrate_columns`.

    Kept in log space so columns near the float limits neither over- nor
    underflow (zero columns get scale 1, i.e. log 0).
    """
    jacobian = _as_float64(jacobian)
    cmax = np.max(np.abs(jacobian), axis=0)
    cmax = np.where(cmax > 0, cmax, 1.0)
    norms = np.linalg.norm(jacobian / cmax, axis=0)
    return np.log(cmax) + np.log(np.where(norms > 0, norms, 1.0))


def _directions_to_original(vectors: np.ndarray, log_scale: np.ndarray) -> np.ndarray:
    """Map unit directions from equilibrated coordinates (q = scale * p) back to p.

    ``p_i = v_i / scale_i`` is formed in log space and max-normalized per column
    before the L2 normalization, so extreme scales cannot overflow to inf/NaN or
    underflow to a zero vector; components far below the largest one in the unit
    result simply round to zero (correct for a unit vector).
    """
    with np.errstate(divide="ignore"):
        t = np.log(np.abs(vectors)) - log_scale[:, None]
    m = np.max(t, axis=0)
    ok = np.isfinite(m)
    out = np.sign(vectors) * np.exp(t - np.where(ok, m, 0.0))
    out[:, ~ok] = 0.0
    norms = np.linalg.norm(out, axis=0)
    return out / np.where(norms > 0, norms, 1.0)


def _as_float64(a: np.ndarray) -> np.ndarray:
    """At least float64: float32/float16/integer inputs are upcast.

    The rank tolerances scale with ``max(m, n) * eps``: with float32's eps that
    is ~0.1 * s_max for a million-row Jacobian, which reports false rank
    deficiency. A diagnostic should not be limited by the storage precision of
    the Jacobian. Jacobians are analyzed as stored: entries are treated as exact,
    so rounding noise already present in a float32-computed Jacobian is not
    modeled; a FIM instead gets the storage-precision cutoff
    (:func:`_storage_eps`), since forming/storing the normal matrix rounds it.
    Limits: complex Jacobians/FIMs are rejected by the validators
    (unavailable analysis) because the SVD/eigen code assumes real input; a
    longdouble (float128) input is not converted and ``np.linalg`` raises
    ``TypeError``, which the analyzers report as an unavailable analysis.
    """
    a = np.asarray(a)
    return np.asarray(a, dtype=np.result_type(a.dtype, np.float64))


def _storage_eps(a: np.ndarray) -> float:
    """Round-off level of the *supplied* matrix: its own dtype's eps, never below float64's.

    The decompositions run in float64 (see :func:`_as_float64`), but a float32
    Jacobian/FIM already carries eps32-level noise from its storage, so rank and
    null-space tolerances must be taken from the storage dtype before the upcast.
    """
    eps64 = float(np.finfo(np.float64).eps)
    dt = np.asarray(a).dtype
    return max(
        float(np.finfo(dt).eps) if np.issubdtype(dt, np.floating) else 0.0, eps64
    )


def _host_svd(
    matrix: np.ndarray, full_matrices: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Singular values and V (not transposed) computed on the host in float64.

    JAX would silently run a float64 input in float32 when ``jax_enable_x64`` is
    off while the tolerances here assume float64; NumPy keeps them consistent
    (as ``core/covariance.py`` does). A tall matrix is reduced to its R factor
    first (same ``s`` and ``V``, no m x n ``U``). Only if LAPACK fails to
    converge does it use the robust fallback.
    """
    matrix = _as_float64(matrix)
    m, n = matrix.shape
    try:
        if not full_matrices and m > 2 * n:
            matrix = np.linalg.qr(matrix, mode="r")
        _, s, vh = np.linalg.svd(matrix, full_matrices=full_matrices)
        return s, vh.T
    except np.linalg.LinAlgError:
        from nlsq.stability.svd_fallback import compute_svd_with_fallback

        _, s, v = compute_svd_with_fallback(matrix, full_matrices=full_matrices)
        return np.asarray(s), np.asarray(v)


def _host_eigh(fim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (descending, negative round-off clamped to 0) and eigenvectors.

    A FIM is symmetric positive semi-definite, but validation accepts slightly
    negative eigenvalues from rounding; singular values would turn those into
    positive ones, so symmetric matrices use a symmetric eigensolve instead.
    """
    fim = _as_float64(fim)
    w, v = np.linalg.eigh((fim + fim.T) / 2)
    order = np.argsort(w)[::-1]
    return np.maximum(w[order], 0.0), v[:, order]


def _fim_scale(fim: np.ndarray) -> np.ndarray:
    """Per-parameter scale sqrt(|diag(F)|); 1 where diag is zero/non-finite."""
    d = np.sqrt(np.abs(np.diag(fim)))
    return np.where((d > 0) & np.isfinite(d), d, 1.0)


def _correlation_with_null_limit(
    covariance: np.ndarray, null_basis: np.ndarray, bound: float
) -> np.ndarray | None:
    """Correlation of a pseudo-inverse covariance, with null directions restored.

    The pseudo-inverse drops the null directions that drive the strongest
    correlations (and flips the sign of a trade-off pair).  Taking the limit
    ``cov = pinv + P / lam`` as ``lam -> 0`` (``P`` = projector onto the null
    space): two parameters that both load on the null space get
    ``P_ij / sqrt(P_ii P_jj)``, a parameter pair with exactly one such member
    is uncorrelated in the limit, and the rest use the pseudo-inverse.

    ``bound`` is the error bound on a null-vector component for the precision
    of the decomposition actually used (``~ n*eps*sigma_max/sigma_gap``, from
    the caller's rank tolerance): a component above it is a resolved loading,
    one at or below it cannot be told apart from an exact zero and is treated
    as structurally zero. That limit is inherent: a real loading smaller than
    the decomposition's resolution is reported as uncorrelated.
    """
    n = covariance.shape[0]
    if null_basis.shape[1] >= n:
        return None  # nothing is determined (e.g. all-zero Jacobian)
    if null_basis.shape[1]:
        proj = null_basis @ null_basis.T
        pd = np.diag(proj)
        involved = np.max(np.abs(null_basis), axis=1) > bound
    else:
        proj, pd = np.zeros((n, n)), np.zeros(n)
        involved = np.zeros(n, dtype=bool)
    free = ~involved
    diag = np.diag(covariance)
    if np.any(diag[free] <= 0):
        return None
    corr = np.zeros((n, n))
    if free.any():
        sd = np.sqrt(diag[free])
        corr[np.ix_(free, free)] = covariance[np.ix_(free, free)] / np.outer(sd, sd)
    if involved.any():
        sp = np.sqrt(pd[involved])
        corr[np.ix_(involved, involved)] = proj[np.ix_(involved, involved)] / np.outer(
            sp, sp
        )
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


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

        # Compute condition number/rank/correlation from SVD(jacobian) directly
        # rather than forming FIM = J.T @ J first. Forming the normal equations
        # squares the condition number of J before it's even analyzed (a
        # near-degenerate model with cond(J) ~ 1e10 reports cond(FIM) ~ 1e21
        # analytically, but explicit J.T @ J + SVD can lose 5+ orders of
        # magnitude of that to float64 rounding -- exactly in the
        # ill-conditioned regime this diagnostic exists to flag).
        return self._analyze_jacobian_via_svd(jacobian, n_params, start_time)

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
        validation_result = _validate_fim(fim)
        if validation_result is not None:
            return IdentifiabilityReport(
                available=False,
                error_message=validation_result,
                n_params=fim.shape[0] if fim.ndim == 2 else 0,
                health_status=HealthStatus.CRITICAL,
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
                health_status=HealthStatus.CRITICAL,
            )

        # Check dimensions
        if jacobian.ndim != 2:
            return IdentifiabilityReport(
                available=False,
                error_message=f"Jacobian must be 2D, got {jacobian.ndim}D",
                n_params=0,
                health_status=HealthStatus.CRITICAL,
            )

        # Complex input would need Hermitian (conjugate) handling: not supported
        if np.iscomplexobj(jacobian):
            return IdentifiabilityReport(
                available=False,
                error_message="Complex Jacobians are not supported (real input required)",
                n_params=jacobian.shape[1],
                health_status=HealthStatus.CRITICAL,
            )

        # Check for NaN
        if np.any(np.isnan(jacobian)):
            return IdentifiabilityReport(
                available=False,
                error_message="Jacobian contains NaN values",
                n_params=jacobian.shape[1],
                health_status=HealthStatus.CRITICAL,
            )

        # Check for Inf
        if np.any(np.isinf(jacobian)):
            return IdentifiabilityReport(
                available=False,
                error_message="Jacobian contains Inf values",
                n_params=jacobian.shape[1],
                health_status=HealthStatus.CRITICAL,
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

    def _compute_svd_from_jacobian(
        self,
        jacobian: np.ndarray,
    ) -> tuple[np.ndarray, float, int, np.ndarray, float]:
        """Compute FIM eigenstructure directly from SVD(jacobian).

        FIM = J.T @ J has eigenvalues s**2 and eigenvectors V, where
        J = U @ diag(s) @ V.T is the SVD of J -- this is analytically
        exact and avoids ever forming or inverting J.T @ J.

        Parameters
        ----------
        jacobian : np.ndarray
            Jacobian matrix of shape (n_data, n_params).

        Returns
        -------
        s : np.ndarray
            Singular values of J (not squared).
        condition_number : float
            cond(FIM) = (s_max / s_min)**2.
        numerical_rank : int
            Numerical rank of J (equals the rank of FIM).
        v : np.ndarray
            Right singular vectors of J, shape (n_params, k) -- used to
            build the correlation matrix without forming FIM.
        tol : float
            Rank tolerance used above -- returned so the correlation-matrix
            computation can reuse it instead of recomputing with a
            different shape reference (which previously let the two
            disagree about which singular values count as "zero").
        """
        # Column-equilibrate so cond/rank/correlation do not depend on the units
        # of x, y or the parameters (correlation is invariant to this scaling).
        jacobian = _equilibrate_columns(jacobian)
        # With fewer observations than parameters the thin SVD drops the right
        # nullspace; request full V and pad s with the n - m zeros.
        wide = jacobian.shape[0] < jacobian.shape[1]
        s, v = _host_svd(jacobian, full_matrices=wide)
        if wide:
            s = np.concatenate([s, np.zeros(jacobian.shape[1] - s.size, s.dtype)])

        max_s = float(np.max(s)) if s.size else 0.0
        nonzero_s = s[s > 0]
        # Precision of the decomposition actually used.
        # The Jacobian is analyzed as stored (entries treated as exact, solved in
        # float64): rounding already present in a float32 Jacobian is not modeled.
        tol = max_s * max(jacobian.shape) * np.finfo(s.dtype).eps

        if max_s == 0 or len(nonzero_s) == 0:
            condition_number = float("inf")
            numerical_rank = 0
        else:
            # Use the smallest singular value overall: dropping exact zeros
            # would report a finite cond for a rank-deficient Jacobian.
            min_s = float(np.min(s))
            ratio = max_s / min_s if min_s > 0 else float("inf")
            condition_number = ratio * ratio
            numerical_rank = int(np.sum(s > tol))

        return s, condition_number, numerical_rank, v, tol

    def _compute_correlation_matrix_from_svd(
        self,
        s: np.ndarray,
        v: np.ndarray,
        tol: float,
    ) -> np.ndarray | None:
        """Compute the correlation matrix directly from SVD(jacobian).

        Builds the FIM pseudo-inverse (covariance) as V @ diag(1/s**2) @ V.T
        using the Jacobian's own SVD, rather than calling ``pinv`` on an
        explicitly-formed ``J.T @ J`` -- this avoids ever squaring J's
        condition number in an actual matrix product.

        Parameters
        ----------
        s : np.ndarray
            Singular values of the Jacobian (not squared).
        v : np.ndarray
            Right singular vectors of the Jacobian, shape (n_params, k).
        tol : float
            Rank tolerance from ``_compute_svd_from_jacobian`` -- reused
            here so a singular value at/near the rank cutoff can't be
            declared "absent" for `numerical_rank` while still being
            inverted into the reported covariance/correlation.

        Returns
        -------
        np.ndarray | None
            Correlation matrix, or None if computation fails.
        """
        try:
            if s.size == 0:
                return None
            # Correlation is invariant to a global scale of J; dividing by the
            # largest singular value keeps 1/s**2 finite for tiny-scale Jacobians.
            s_max = float(np.max(s))
            s_n = s / s_max if s_max > 0 else s
            safe_s = np.where(s_n > 0, s_n, 1.0)
            inv_s2 = np.where(s > tol, 1.0 / safe_s**2, 0.0)
            covariance = (v * inv_s2) @ v.T
            kept = s[s > tol]
            s_gap = float(np.min(kept)) if kept.size else s_max
            return _correlation_with_null_limit(covariance, v[:, s <= tol], tol / s_gap)

        except Exception:
            _logger.warning("Correlation matrix computation failed", exc_info=True)
            return None

    def _analyze_jacobian_via_svd(
        self,
        jacobian: np.ndarray,
        n_params: int,
        start_time: float,
    ) -> IdentifiabilityReport:
        """Analyze identifiability directly from SVD(jacobian).

        Mirrors ``_analyze_fim`` but never forms FIM = J.T @ J.

        Parameters
        ----------
        jacobian : np.ndarray
            Jacobian matrix of shape (n_data, n_params).
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

        try:
            s, condition_number, numerical_rank, v, tol = (
                self._compute_svd_from_jacobian(
                    jacobian,
                )
            )
        except Exception as e:
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

        correlation_matrix = self._compute_correlation_matrix_from_svd(s, v, tol)

        highly_correlated_pairs = self._detect_highly_correlated_pairs(
            correlation_matrix,
        )

        if numerical_rank < n_params:
            issue = self._create_ident_001_issue(numerical_rank, n_params)
            issues.append(issue)
            health_status = HealthStatus.CRITICAL

        if condition_number > self.config.condition_threshold:
            issue = self._create_ident_002_issue(condition_number)
            issues.append(issue)
            if health_status != HealthStatus.CRITICAL:
                health_status = HealthStatus.WARNING

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

    def _analyze_fim(
        self,
        fim: np.ndarray,
        n_params: int,
        start_time: float,
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
            correlation_matrix,
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
        # Unit-invariant: analyze D^-1/2 F D^-1/2 with D = diag(F).
        storage_eps = _storage_eps(fim)
        fim = _as_float64(fim)
        d = _fim_scale(fim)
        fim = fim / np.outer(d, d)
        singular_values, _ = _host_eigh(fim)

        # Handle edge case of all-zero FIM
        max_sv = np.max(singular_values)
        min_sv_nonzero = singular_values[singular_values > 0]

        if max_sv == 0 or len(min_sv_nonzero) == 0:
            condition_number = float("inf")
            numerical_rank = 0
        else:
            # Smallest singular value overall: exact zeros mean rank deficiency.
            min_sv = np.min(singular_values)
            condition_number = float(max_sv / min_sv) if min_sv > 0 else float("inf")

            # Compute numerical rank with relative tolerance
            tol = max_sv * max(fim.shape) * storage_eps
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
            # Parameter-estimate correlation comes from the covariance matrix
            # (inverse of the FIM), not the raw FIM - normalizing the FIM
            # directly gives cosine similarity of Jacobian columns, a
            # different quantity that diverges from parameter correlation
            # exactly in the ill-conditioned regime this diagnostic targets.
            # Use pinv: the FIM can be near-singular for poorly identifiable
            # models, where a plain inverse would raise/blow up.
            # Equilibrate by diag(FIM) first: corr(inv(F)) == corr(inv(D^-1 F D^-1)),
            # but pinv's rank cutoff no longer depends on the columns' units.
            storage_eps = _storage_eps(fim)
            fim = _as_float64(fim)
            d = np.sqrt(np.abs(np.diag(fim)))
            if not np.all(np.isfinite(d)):
                return None
            d = np.where(d > 0, d, 1.0)  # zero column: leave it unscaled
            scaled = fim / np.outer(d, d)
            w, u = np.linalg.eigh((scaled + scaled.T) / 2)
            keep = w > max(float(np.max(w)), 0.0) * len(w) * storage_eps
            covariance = (u[:, keep] / w[keep]) @ u[:, keep].T
            tol = max(float(np.max(w)), 0.0) * len(w) * storage_eps
            w_gap = float(np.min(w[keep])) if keep.any() else float(np.max(w))
            return _correlation_with_null_limit(
                covariance, u[:, ~keep], tol / w_gap if w_gap > 0 else np.inf
            )

        except Exception:
            _logger.warning("Correlation matrix computation failed", exc_info=True)
            return None

    def _detect_highly_correlated_pairs(
        self,
        correlation_matrix: np.ndarray | None,
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
        self,
        numerical_rank: int,
        n_params: int,
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
        self,
        correlated_pairs: list[tuple[int, int, float]],
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
