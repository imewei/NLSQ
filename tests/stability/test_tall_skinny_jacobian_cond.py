"""Regression tests: condition-number estimation on tall-skinny Jacobians.

Historical context:
  ``NumericalStabilityGuard.check_and_fix_jacobian`` skipped the
  condition-number SVD only when the Jacobian exceeded
  ``max_jacobian_elements_for_svd`` (10M *elements*). That bounds the element
  count, not the SVD workspace: XLA's ``svdvals`` on an ``(m, n)`` input
  allocates an ``m x m`` scratch buffer, so a ``(1_200_000, 7)`` Jacobian --
  8.4M elements, comfortably under the skip threshold -- asked for ~11.5 TB and
  raised ``RESOURCE_EXHAUSTED``.

  The exception was caught and the condition number fell back to ``inf``, which
  then tripped the ``condition_number > condition_threshold`` branch, so a
  *perfectly conditioned* Jacobian (true cond ~1.003) was reported as
  ill-conditioned and silently diagonal-regularized. The failure band was
  bounded on both sides: crossing the 10M-element threshold skipped the SVD
  entirely and behaved correctly, so a larger Jacobian worked while a smaller
  one did not.

  Resolution: the singular values are taken from the ``R`` factor of a QR
  decomposition. ``J = QR`` with orthonormal ``Q``, so ``R`` carries exactly
  the singular values of ``J`` while the workspace stays ``O(m * n)``. Wide
  Jacobians are transposed first -- they hit the identical blowup (a
  ``(7, 300_000)`` input asks for 720 GB) and singular values are invariant
  under transposition.

  Backend note: the OOM is CPU-only. cuSOLVER handles these shapes, so on a
  GPU host the pre-fix code passes these tests; CI runs CPU-only, where it
  does not.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.stability.guard import NumericalStabilityGuard

# Memory-intensive: these allocate up to 8.4M-element float64 arrays, well over
# the 100K-element threshold that requires serial execution (CLAUDE.md).
pytestmark = [pytest.mark.serial, pytest.mark.stability]


def _well_conditioned(shape, seed=0):
    return jnp.asarray(np.random.default_rng(seed).normal(size=shape))


class TestTallSkinnyConditionNumber:
    """A tall-skinny Jacobian must not be mistaken for an ill-conditioned one."""

    @pytest.mark.parametrize("shape", [(50_000, 7), (200_000, 12)])
    def test_matches_direct_svd(self, shape):
        """QR-derived singular values agree with a direct SVD.

        Only the (200_000, 12) case reproduces the historical CPU OOM; the
        (50_000, 7) case is a plain QR-vs-SVD correctness check.
        """
        J = _well_conditioned(shape)
        guard = NumericalStabilityGuard()

        _, issues = guard.check_and_fix_jacobian(J)

        sv = np.linalg.svd(np.asarray(J), compute_uv=False)
        expected = float(sv[0] / sv[-1])
        assert issues["condition_number"] == pytest.approx(expected, rel=1e-5)

    def test_large_tall_skinny_is_not_falsely_regularized(self):
        """The shape that used to OOM must report a finite cond and no fix-up.

        8.4M elements stays under ``max_jacobian_elements_for_svd`` (10M), so
        this exercises the SVD path rather than the large-Jacobian skip path.
        Before the QR change this raised RESOURCE_EXHAUSTED internally, fell
        back to ``condition_number = inf``, and regularized ``J`` in place.
        """
        guard = NumericalStabilityGuard()
        J = _well_conditioned((1_200_000, 7))
        assert J.size < guard.max_jacobian_elements_for_svd

        J_fixed, issues = guard.check_and_fix_jacobian(J)

        assert issues["svd_skipped"] is False
        assert np.isfinite(issues["condition_number"])
        assert issues["condition_number"] < 10.0
        assert issues["regularized"] is False
        np.testing.assert_array_equal(np.asarray(J_fixed), np.asarray(J))

    def test_genuinely_singular_tall_jacobian_still_detected(self):
        """A structurally rank-deficient column must still report inf.

        Guards the fix against over-correcting: pinning a parameter (collapsing
        its bounds to zero width) leaves an all-zero Jacobian column, and that
        must keep being flagged and regularized.
        """
        guard = NumericalStabilityGuard()
        J = np.array(_well_conditioned((50_000, 7)))
        J[:, 3] = 0.0  # rank-deficient by construction

        with pytest.warns(UserWarning, match="Ill-conditioned Jacobian"):
            _, issues = guard.check_and_fix_jacobian(jnp.asarray(J))

        assert not np.isfinite(issues["condition_number"])
        assert issues["regularized"] is True


class TestWideAndSquareConditionNumber:
    """The wide (m < n) path had the identical blowup, transposed."""

    def test_wide_jacobian_is_not_falsely_regularized(self):
        """A wide Jacobian must not OOM into a spurious `inf`.

        ``svdvals`` on a ``(7, 300_000)`` input asks for 720 GB on the CPU
        backend -- 2.1M elements, under ``max_jacobian_elements_for_svd`` -- so
        this reaches the SVD path and used to fail exactly like the tall case.
        """
        guard = NumericalStabilityGuard()
        J = _well_conditioned((7, 300_000))
        assert J.size < guard.max_jacobian_elements_for_svd

        J_fixed, issues = guard.check_and_fix_jacobian(J)

        assert issues["svd_skipped"] is False
        assert issues["svd_failed"] is False
        assert np.isfinite(issues["condition_number"])
        assert issues["regularized"] is False
        np.testing.assert_array_equal(np.asarray(J_fixed), np.asarray(J))

    @pytest.mark.parametrize("shape", [(9, 9), (5, 40)])
    def test_square_and_wide_match_direct_svd(self, shape):
        """Pins the m >= n orientation choice against an accidental flip."""
        J = _well_conditioned(shape)
        guard = NumericalStabilityGuard()

        _, issues = guard.check_and_fix_jacobian(J)

        sv = np.linalg.svd(np.asarray(J), compute_uv=False)
        assert issues["condition_number"] == pytest.approx(
            float(sv[0] / sv[-1]), rel=1e-5
        )


class TestSvdFailureIsNotIllConditioning:
    """A failed measurement must not masquerade as a measured `inf`."""

    def test_svd_failure_does_not_regularize(self, monkeypatch):
        """An exception in the SVD must leave J untouched, not perturb it."""
        guard = NumericalStabilityGuard()
        J = _well_conditioned((2_000, 7))

        def _boom(*_args, **_kwargs):
            raise RuntimeError("RESOURCE_EXHAUSTED: simulated")

        monkeypatch.setattr(jnp.linalg, "qr", _boom)

        with pytest.warns(UserWarning, match="Could not compute SVD"):
            J_fixed, issues = guard.check_and_fix_jacobian(J)

        assert issues["svd_failed"] is True
        assert issues["condition_number"] is None
        assert issues["is_ill_conditioned"] is False
        assert issues["regularized"] is False
        np.testing.assert_array_equal(np.asarray(J_fixed), np.asarray(J))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
