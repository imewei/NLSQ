"""Comprehensive tests for nlsq._optimize module.

This test suite covers:
- OptimizeResult class functionality
- Dictionary and attribute access
- Special methods (__repr__, __dir__, __getattr__)
- _check_unknown_options function
- OptimizeWarning class
"""

import unittest
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.core._optimize import OptimizeResult, OptimizeWarning, _check_unknown_options


class TestOptimizeResultBasic(unittest.TestCase):
    """Tests for basic OptimizeResult functionality."""

    def test_empty_initialization(self):
        """Test creating an empty OptimizeResult."""
        result = OptimizeResult()

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 0)

    def test_initialization_with_kwargs(self):
        """Test creating OptimizeResult with keyword arguments."""
        result = OptimizeResult(x=np.array([1.0, 2.0]), success=True, status=1)

        self.assertEqual(len(result), 3)
        np.testing.assert_array_equal(result["x"], np.array([1.0, 2.0]))
        self.assertTrue(result["success"])
        self.assertEqual(result["status"], 1)

    def test_dict_style_access(self):
        """Test dictionary-style access."""
        result = OptimizeResult()
        result["x"] = np.array([1.0, 2.0, 3.0])
        result["cost"] = 0.5
        result["success"] = True

        self.assertEqual(len(result), 3)
        np.testing.assert_array_equal(result["x"], np.array([1.0, 2.0, 3.0]))
        self.assertEqual(result["cost"], 0.5)
        self.assertTrue(result["success"])

    def test_attribute_style_access(self):
        """Test attribute-style access via __getattr__."""
        result = OptimizeResult()
        result["x"] = np.array([1.0, 2.0, 3.0])
        result["cost"] = 0.5
        result["success"] = True

        # Access as attributes
        np.testing.assert_array_equal(result.x, np.array([1.0, 2.0, 3.0]))
        self.assertEqual(result.cost, 0.5)
        self.assertTrue(result.success)

    def test_attribute_assignment(self):
        """Test attribute assignment via __setattr__."""
        result = OptimizeResult()

        # Assign as attributes
        result.x = np.array([1.0, 2.0])
        result.success = True
        result.message = "Optimization terminated successfully."

        # Verify stored as dict items
        self.assertIn("x", result)
        self.assertIn("success", result)
        self.assertIn("message", result)
        np.testing.assert_array_equal(result["x"], np.array([1.0, 2.0]))
        self.assertTrue(result["success"])

    def test_attribute_deletion(self):
        """Test attribute deletion via __delattr__."""
        result = OptimizeResult()
        result.x = np.array([1.0, 2.0])
        result.success = True

        # Delete attribute
        del result.success

        self.assertNotIn("success", result)
        self.assertIn("x", result)

    def test_missing_attribute_error(self):
        """Test AttributeError for missing attributes."""
        result = OptimizeResult()
        result.x = np.array([1.0, 2.0])

        with self.assertRaises(AttributeError):
            _ = result.nonexistent_attribute

    def test_keys_values_items(self):
        """Test dict methods (keys, values, items)."""
        result = OptimizeResult(x=np.array([1.0]), cost=0.5, success=True)

        self.assertEqual(set(result.keys()), {"x", "cost", "success"})
        self.assertIn(0.5, result.values())
        self.assertIn(True, result.values())

        items = dict(result.items())
        self.assertEqual(items["cost"], 0.5)
        self.assertTrue(items["success"])


class TestOptimizeResultRepr(unittest.TestCase):
    """Tests for OptimizeResult.__repr__ method."""

    def test_repr_empty(self):
        """Test __repr__ with empty result."""
        result = OptimizeResult()

        repr_str = repr(result)

        self.assertEqual(repr_str, "OptimizeResult()")

    def test_repr_with_items(self):
        """Test __repr__ with items."""
        result = OptimizeResult(x=np.array([1.0, 2.0]), success=True, cost=0.5)

        repr_str = repr(result)

        # Should contain key-value pairs
        self.assertIn("x", repr_str)
        self.assertIn("success", repr_str)
        self.assertIn("cost", repr_str)
        self.assertIn(":", repr_str)

    def test_repr_sorted_keys(self):
        """Test that __repr__ shows sorted keys."""
        result = OptimizeResult(z=3, a=1, m=2)

        repr_str = repr(result)

        # Keys should be sorted: a, m, z
        a_idx = repr_str.find("a")
        m_idx = repr_str.find("m")
        z_idx = repr_str.find("z")

        self.assertLess(a_idx, m_idx)
        self.assertLess(m_idx, z_idx)

    def test_repr_multiline(self):
        """Test that __repr__ produces multiline output."""
        result = OptimizeResult(x=np.array([1.0]), success=True, status=1)

        repr_str = repr(result)

        # Should have multiple lines
        lines = repr_str.split("\n")
        self.assertGreater(len(lines), 1)


class TestOptimizeResultDir(unittest.TestCase):
    """Tests for OptimizeResult.__dir__ method."""

    def test_dir_empty(self):
        """Test __dir__ with empty result."""
        result = OptimizeResult()

        dir_list = dir(result)

        self.assertEqual(dir_list, [])

    def test_dir_with_items(self):
        """Test __dir__ with items."""
        result = OptimizeResult(x=np.array([1.0]), success=True, cost=0.5, message="OK")

        dir_list = dir(result)

        self.assertEqual(set(dir_list), {"x", "success", "cost", "message"})

    def test_dir_after_addition(self):
        """Test __dir__ after adding items."""
        result = OptimizeResult(x=np.array([1.0]))

        dir_list_1 = dir(result)
        self.assertEqual(set(dir_list_1), {"x"})

        result.success = True
        result.cost = 0.5

        dir_list_2 = dir(result)
        self.assertEqual(set(dir_list_2), {"x", "success", "cost"})

    def test_dir_after_deletion(self):
        """Test __dir__ after deleting items."""
        result = OptimizeResult(x=np.array([1.0]), success=True, cost=0.5)

        del result.cost

        dir_list = dir(result)
        self.assertEqual(set(dir_list), {"x", "success"})


class TestOptimizeResultComplex(unittest.TestCase):
    """Tests for complex OptimizeResult scenarios."""

    def test_realistic_optimization_result(self):
        """Test with realistic optimization result data."""
        result = OptimizeResult(
            x=np.array([2.0, 3.0, 1.5]),
            success=True,
            status=1,
            message="Optimization terminated successfully.",
            fun=np.array([0.01, -0.02, 0.015, -0.01]),
            cost=0.000275,
            jac=np.random.rand(4, 3),
            grad=np.array([1e-8, -2e-8, 1.5e-8]),
            optimality=2e-8,
            nfev=25,
            njev=25,
            nit=10,
        )

        # Verify all attributes accessible
        self.assertEqual(len(result.x), 3)
        self.assertTrue(result.success)
        self.assertEqual(result.status, 1)
        self.assertEqual(result.nfev, 25)
        self.assertAlmostEqual(result.cost, 0.000275)

    def test_jax_array_storage(self):
        """Test that JAX arrays can be stored."""
        result = OptimizeResult()

        # Store JAX arrays
        result.x = jnp.array([1.0, 2.0, 3.0])
        result.fun = jnp.array([0.1, -0.2, 0.15])

        # Should be accessible
        self.assertEqual(len(result.x), 3)
        self.assertEqual(len(result.fun), 3)

    def test_mixed_types(self):
        """Test storing mixed types (arrays, scalars, strings, bools)."""
        result = OptimizeResult(
            x=np.array([1.0, 2.0]),
            cost=0.5,
            success=True,
            message="Converged",
            nfev=10,
            optimality=1e-8,
        )

        # All types should be accessible
        self.assertIsInstance(result.x, np.ndarray)
        self.assertIsInstance(result.cost, float)
        self.assertIsInstance(result.success, bool)
        self.assertIsInstance(result.message, str)
        self.assertIsInstance(result.nfev, int)
        self.assertIsInstance(result.optimality, float)

    def test_update_existing_attributes(self):
        """Test updating existing attributes."""
        result = OptimizeResult(x=np.array([1.0, 2.0]), success=False)

        # Update attributes
        result.x = np.array([2.0, 3.0, 4.0])
        result.success = True

        np.testing.assert_array_equal(result.x, np.array([2.0, 3.0, 4.0]))
        self.assertTrue(result.success)

    def test_optional_attributes(self):
        """Test that optional attributes can be added."""
        result = OptimizeResult(x=np.array([1.0, 2.0]), success=True)

        # Add optional attributes
        result.pcov = np.eye(2)
        result.active_mask = np.array([False, True])
        result.all_times = {"total": 1.5, "jac": 0.8}

        self.assertIn("pcov", result)
        self.assertIn("active_mask", result)
        self.assertIn("all_times", result)


class TestCheckUnknownOptions(unittest.TestCase):
    """Tests for _check_unknown_options function."""

    def test_empty_options(self):
        """Test with no unknown options."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            _check_unknown_options({})

            # No warnings should be issued
            self.assertEqual(len(w), 0)

    def test_single_unknown_option(self):
        """Test with single unknown option."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            _check_unknown_options({"unknown_param": 123})

            # Should issue a warning
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, OptimizeWarning))
            self.assertIn("unknown_param", str(w[0].message))
            self.assertIn("Unknown solver options", str(w[0].message))

    def test_multiple_unknown_options(self):
        """Test with multiple unknown options."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            _check_unknown_options(
                {"bad_option1": 1, "bad_option2": 2, "bad_option3": 3}
            )

            # Should issue a warning
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, OptimizeWarning))
            msg = str(w[0].message)
            self.assertIn("Unknown solver options", msg)
            # All option names should be in message
            self.assertIn("bad_option1", msg)
            self.assertIn("bad_option2", msg)
            self.assertIn("bad_option3", msg)

    def test_warning_stacklevel(self):
        """Test that warning uses correct stacklevel."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            _check_unknown_options({"unknown": 1})

            # Should have stacklevel=4
            self.assertEqual(len(w), 1)
            # The warning should be issued (testing that stacklevel doesn't break it)
            self.assertIsInstance(w[0].message, Warning)


class TestOptimizeWarning(unittest.TestCase):
    """Tests for OptimizeWarning class."""

    def test_is_user_warning_subclass(self):
        """Test that OptimizeWarning is a UserWarning subclass."""
        self.assertTrue(issubclass(OptimizeWarning, UserWarning))

    def test_can_be_raised(self):
        """Test that OptimizeWarning can be raised."""
        with self.assertRaises(OptimizeWarning):
            raise OptimizeWarning("Test warning message")

    def test_can_be_caught_as_user_warning(self):
        """Test that OptimizeWarning can be caught as UserWarning."""
        with self.assertRaises(UserWarning):
            raise OptimizeWarning("Test warning message")

    def test_warning_message(self):
        """Test that warning message is preserved."""
        msg = "Custom optimization warning"

        with self.assertRaises(OptimizeWarning) as ctx:
            raise OptimizeWarning(msg)

        self.assertIn(msg, str(ctx.exception))

    def test_warnings_module_integration(self):
        """Test integration with warnings module."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            warnings.warn("Test message", OptimizeWarning)

            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, OptimizeWarning))
            self.assertIn("Test message", str(w[0].message))


class TestOptimizeResultEdgeCases(unittest.TestCase):
    """Tests for edge cases in OptimizeResult."""

    def test_none_values(self):
        """Test storing None values."""
        result = OptimizeResult(x=None, success=None)

        self.assertIsNone(result.x)
        self.assertIsNone(result["success"])

    def test_empty_arrays(self):
        """Test storing empty arrays."""
        result = OptimizeResult(x=np.array([]), fun=np.array([]))

        self.assertEqual(len(result.x), 0)
        self.assertEqual(len(result.fun), 0)

    def test_large_number_of_attributes(self):
        """Test with many attributes."""
        result = OptimizeResult()

        # Add many attributes
        for i in range(50):
            result[f"param_{i}"] = i

        self.assertEqual(len(result), 50)
        self.assertEqual(result.param_0, 0)
        self.assertEqual(result.param_49, 49)

    def test_special_characters_in_keys(self):
        """Test keys with special characters."""
        result = OptimizeResult()

        result["param_1"] = 1
        result["param-2"] = 2
        result["param.3"] = 3

        # Dict access should work
        self.assertEqual(result["param_1"], 1)
        self.assertEqual(result["param-2"], 2)
        self.assertEqual(result["param.3"], 3)

    def test_iteration(self):
        """Test iteration over OptimizeResult."""
        result = OptimizeResult(a=1, b=2, c=3)

        keys = list(result)
        self.assertEqual(set(keys), {"a", "b", "c"})

    def test_in_operator(self):
        """Test 'in' operator."""
        result = OptimizeResult(x=np.array([1.0]), success=True)

        self.assertIn("x", result)
        self.assertIn("success", result)
        self.assertNotIn("nonexistent", result)

    def test_get_method(self):
        """Test dict.get() method."""
        result = OptimizeResult(x=np.array([1.0]), success=True)

        self.assertTrue(result.get("success"))
        self.assertIsNone(result.get("nonexistent"))
        self.assertEqual(result.get("nonexistent", "default"), "default")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
