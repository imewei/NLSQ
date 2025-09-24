"""Tests for common_scipy module."""

import unittest

import numpy as np

from nlsq.common_scipy import (
    check_termination,
    evaluate_quadratic,
    find_active_constraints,
    in_bounds,
    make_strictly_feasible,
    print_header_nonlinear,
    print_iteration_nonlinear,
    solve_trust_region_2d,
    step_size_to_bound,
    update_tr_radius,
)


class TestCommonScipy(unittest.TestCase):
    """Test common scipy utility functions."""

    def test_in_bounds(self):
        """Test in_bounds function."""
        # Test with scalar bounds - all in bounds
        x = np.array([0.5, 1.5, 1.9])
        lb = 0.0
        ub = 2.0

        result = in_bounds(x, lb, ub)
        self.assertTrue(result)  # All elements are in bounds

        # Test with scalar bounds - not all in bounds
        x = np.array([0.5, 1.5, 2.5])
        result = in_bounds(x, lb, ub)
        self.assertFalse(result)  # Not all elements are in bounds

        # Test with array bounds
        x = np.array([0.5, 1.5, 3.5])
        lb = np.array([0.0, 1.0, 3.0])
        ub = np.array([1.0, 2.0, 4.0])

        result = in_bounds(x, lb, ub)
        self.assertTrue(result)  # All elements within their respective bounds

        # Test with array bounds - out of bounds
        x = np.array([0.5, 2.5, 3.5])  # Second element exceeds its upper bound
        result = in_bounds(x, lb, ub)
        self.assertFalse(result)

        # Test with infinite bounds
        x = np.array([0.5, 1.5, 2.5])
        lb = -np.inf
        ub = np.inf

        result = in_bounds(x, lb, ub)
        self.assertTrue(result)  # All elements are in bounds

    def test_make_strictly_feasible(self):
        """Test make_strictly_feasible function."""
        # Test with array bounds (function expects arrays, not scalars)
        x = np.array([0.0, 1.0, 2.0])
        lb = np.array([0.0, 0.0, 0.0])
        ub = np.array([2.0, 2.0, 2.0])

        # Should push away from boundaries
        x_feasible = make_strictly_feasible(x, lb, ub)
        self.assertTrue(np.all(x_feasible > lb))
        self.assertTrue(np.all(x_feasible < ub))

        # Test with array bounds
        x = np.array([0.0, 1.0, 3.0])
        lb = np.array([0.0, 0.5, 2.0])
        ub = np.array([1.0, 1.5, 4.0])

        x_feasible = make_strictly_feasible(x, lb, ub)
        self.assertTrue(np.all(x_feasible > lb))
        self.assertTrue(np.all(x_feasible < ub))

        # Test with infinite bounds (as arrays)
        x = np.array([0.0, 1.0, 2.0])
        n = len(x)
        lb = np.full(n, -np.inf)
        ub = np.full(n, np.inf)

        x_feasible = make_strictly_feasible(x, lb, ub)
        np.testing.assert_array_equal(x_feasible, x)  # Should be unchanged

    def test_check_termination(self):
        """Test check_termination function."""
        # Test ftol termination
        status = check_termination(
            dF=0.5, F=10.0, dx_norm=1.0, x_norm=5.0, ratio=0.5, ftol=0.1, xtol=1e-8
        )
        self.assertEqual(status, 2)  # ftol satisfied

        # Test xtol termination
        status = check_termination(
            dF=0.1, F=10.0, dx_norm=1e-10, x_norm=1.0, ratio=0.5, ftol=1e-8, xtol=1e-8
        )
        self.assertEqual(status, 3)  # xtol satisfied

        # Test both satisfied
        status = check_termination(
            dF=0.5, F=10.0, dx_norm=1e-10, x_norm=1.0, ratio=0.5, ftol=0.1, xtol=1e-8
        )
        self.assertEqual(status, 4)  # both satisfied

        # Test none satisfied
        status = check_termination(
            dF=1e-10, F=10.0, dx_norm=1.0, x_norm=1.0, ratio=0.5, ftol=1e-12, xtol=1e-12
        )
        self.assertIsNone(status)

    def test_evaluate_quadratic(self):
        """Test evaluate_quadratic function."""
        # Simple 2D case
        J = np.array([[1.0, 0.0], [0.0, 2.0]])
        g = np.array([1.0, 2.0])
        s = np.array([0.5, 0.5])

        value = evaluate_quadratic(J, g, s)
        # f(s) = 0.5 * s^T * J^T * J * s + g^T * s
        # = 0.5 * [0.5, 0.5] * [[1, 0], [0, 4]] * [0.5, 0.5] + [1, 2] * [0.5, 0.5]
        # = 0.5 * [0.5, 0.5] * [0.5, 2] + 1.5
        # = 0.5 * (0.25 + 1) + 1.5 = 0.625 + 1.5 = 2.125
        expected = 0.5 * np.dot(s, np.dot(J.T @ J, s)) + np.dot(g, s)
        self.assertAlmostEqual(value, expected, places=10)

    def test_step_size_to_bound(self):
        """Test step_size_to_bound function."""
        x = np.array([0.5, 0.5])
        s = np.array([1.0, 1.0])
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        alpha, hits = step_size_to_bound(x, s, lb, ub)

        # Should hit upper bound at alpha = 0.5
        self.assertAlmostEqual(alpha, 0.5)
        np.testing.assert_array_equal(hits, np.array([True, True]))

    def test_find_active_constraints(self):
        """Test find_active_constraints function."""
        x = np.array([0.0, 0.5, 1.0])
        lb = np.array([0.0, 0.0, 0.0])
        ub = np.array([1.0, 1.0, 1.0])

        active = find_active_constraints(x, lb, ub, rtol=1e-10)

        # First element at lower bound (-1), second not at bound (0), third at upper bound (1)
        expected = np.array([-1, 0, 1])
        np.testing.assert_array_equal(active, expected)

    def test_solve_trust_region_2d(self):
        """Test solve_trust_region_2d function."""
        # Simple 2x2 system
        B = np.array([[2.0, 0.0], [0.0, 2.0]])
        g = np.array([1.0, 1.0])
        Delta = 0.5

        p, newton_step = solve_trust_region_2d(B, g, Delta)

        # Should return step within trust region
        self.assertTrue(np.linalg.norm(p) <= Delta + 1e-10)

        # For this case, unconstrained minimum is at p = -B^{-1} g = [-0.5, -0.5]
        # which has norm sqrt(0.5) ≈ 0.707 > 0.5, so NOT a Newton step
        # (the step had to be constrained to the trust region boundary)
        self.assertFalse(newton_step)

    def test_update_tr_radius(self):
        """Test update_tr_radius function."""
        # Test good step (ratio > 0.75)
        Delta_new, ratio = update_tr_radius(
            Delta=1.0,
            actual_reduction=0.9,
            predicted_reduction=1.0,
            step_norm=1.0,
            bound_hit=True,
        )
        self.assertEqual(ratio, 0.9)
        self.assertEqual(Delta_new, 2.0)  # Should double

        # Test bad step (ratio < 0.25)
        Delta_new, ratio = update_tr_radius(
            Delta=1.0,
            actual_reduction=0.1,
            predicted_reduction=1.0,
            step_norm=0.5,
            bound_hit=False,
        )
        self.assertEqual(ratio, 0.1)
        self.assertEqual(Delta_new, 0.125)  # 0.25 * step_norm

        # Test medium step
        Delta_new, ratio = update_tr_radius(
            Delta=1.0,
            actual_reduction=0.5,
            predicted_reduction=1.0,
            step_norm=0.8,
            bound_hit=False,
        )
        self.assertEqual(ratio, 0.5)
        self.assertEqual(Delta_new, 1.0)  # Unchanged

    def test_print_functions(self):
        """Test print functions (just ensure they run without error)."""
        # Test print_header_nonlinear
        print_header_nonlinear()  # Should not raise

        # Test print_iteration_nonlinear
        print_iteration_nonlinear(
            iteration=1,
            nfev=10,
            cost=1.5,
            cost_reduction=0.5,
            step_norm=0.1,
            optimality=0.01,
        )  # Should not raise


if __name__ == "__main__":
    unittest.main()
