"""Edge-path tests for common_scipy helpers."""

from __future__ import annotations

import numpy as np
import pytest

from nlsq.common_scipy import (
    intersect_trust_region,
    solve_lsq_trust_region,
    solve_trust_region_2d,
)


def test_intersect_trust_region_errors() -> None:
    """intersect_trust_region should reject zero steps and outside points."""
    x = np.array([1.0, 0.0])
    s = np.array([0.0, 0.0])
    with pytest.raises(ValueError, match="`s` is zero"):
        intersect_trust_region(x, s, 1.0)

    s = np.array([1.0, 0.0])
    with pytest.raises(ValueError, match="not within the trust region"):
        intersect_trust_region(np.array([2.0, 0.0]), s, 1.0)


def test_solve_lsq_trust_region_full_rank_returns_gn_step() -> None:
    """Full-rank case should return Gauss-Newton step when within Delta."""
    n = 1
    m = 1
    uf = np.array([1.0])
    s = np.array([2.0])
    V = np.array([[1.0]])
    p, alpha, n_iter = solve_lsq_trust_region(n, m, uf, s, V, Delta=10.0)

    assert n_iter == 0
    assert alpha == 0.0
    assert np.isfinite(p).all()


def test_solve_lsq_trust_region_iterative_path() -> None:
    """Under-determined case should iterate and return finite parameters."""
    n = 2
    m = 1
    uf = np.array([1.0, 0.5])
    s = np.array([1.0, 0.2])
    V = np.eye(2)

    p, alpha, n_iter = solve_lsq_trust_region(n, m, uf, s, V, Delta=0.5, max_iter=3)

    assert n_iter >= 1
    assert alpha >= 0.0
    assert np.isfinite(p).all()


def test_solve_trust_region_2d_fallback_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """solve_trust_region_2d should handle LinAlgError and use root solver path."""
    from nlsq import common_scipy

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise common_scipy.LinAlgError("bad")

    monkeypatch.setattr(common_scipy, "cho_factor", _raise)

    B = np.array([[0.0, 1.0], [1.0, 0.0]])
    g = np.array([1.0, -1.0])
    p, newton_step = solve_trust_region_2d(B, g, Delta=1.0)

    assert newton_step is False
    assert np.isfinite(p).all()
