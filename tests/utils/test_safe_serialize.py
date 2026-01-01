"""Tests for nlsq.utils.safe_serialize module.

Tests for the safe JSON-based serialization that replaces pickle for
checkpoint data, addressing CWE-502 (Deserialization of Untrusted Data).
"""

import math

import numpy as np
import pytest

from nlsq.utils.safe_serialize import (
    SafeSerializationError,
    safe_dumps,
    safe_loads,
)


class TestBasicTypes:
    """Tests for basic Python type serialization."""

    def test_string(self):
        """Test string serialization."""
        data = "hello world"
        assert safe_loads(safe_dumps(data)) == data

    def test_int(self):
        """Test integer serialization."""
        data = 42
        assert safe_loads(safe_dumps(data)) == data

    def test_negative_int(self):
        """Test negative integer serialization."""
        data = -123
        assert safe_loads(safe_dumps(data)) == data

    def test_large_int(self):
        """Test large integer serialization."""
        data = 10**18
        assert safe_loads(safe_dumps(data)) == data

    def test_float(self):
        """Test float serialization."""
        data = 3.14159
        result = safe_loads(safe_dumps(data))
        assert abs(result - data) < 1e-10

    def test_float_nan(self):
        """Test NaN float serialization."""
        data = float("nan")
        result = safe_loads(safe_dumps(data))
        assert math.isnan(result)

    def test_float_inf(self):
        """Test positive infinity serialization."""
        data = float("inf")
        result = safe_loads(safe_dumps(data))
        assert math.isinf(result) and result > 0

    def test_float_neg_inf(self):
        """Test negative infinity serialization."""
        data = float("-inf")
        result = safe_loads(safe_dumps(data))
        assert math.isinf(result) and result < 0

    def test_bool_true(self):
        """Test True serialization."""
        data = True
        assert safe_loads(safe_dumps(data)) is True

    def test_bool_false(self):
        """Test False serialization."""
        data = False
        assert safe_loads(safe_dumps(data)) is False

    def test_none(self):
        """Test None serialization."""
        data = None
        assert safe_loads(safe_dumps(data)) is None


class TestContainerTypes:
    """Tests for container type serialization."""

    def test_empty_list(self):
        """Test empty list serialization."""
        data = []
        assert safe_loads(safe_dumps(data)) == data

    def test_list_of_ints(self):
        """Test list of integers serialization."""
        data = [1, 2, 3, 4, 5]
        assert safe_loads(safe_dumps(data)) == data

    def test_list_of_mixed(self):
        """Test list with mixed types serialization."""
        data = [1, "two", 3.0, True, None]
        assert safe_loads(safe_dumps(data)) == data

    def test_nested_list(self):
        """Test nested list serialization."""
        data = [[1, 2], [3, 4], [5, 6]]
        assert safe_loads(safe_dumps(data)) == data

    def test_empty_dict(self):
        """Test empty dict serialization."""
        data = {}
        assert safe_loads(safe_dumps(data)) == data

    def test_simple_dict(self):
        """Test simple dict serialization."""
        data = {"a": 1, "b": 2, "c": 3}
        assert safe_loads(safe_dumps(data)) == data

    def test_nested_dict(self):
        """Test nested dict serialization."""
        data = {"outer": {"inner": {"deep": 42}}}
        assert safe_loads(safe_dumps(data)) == data

    def test_dict_with_lists(self):
        """Test dict containing lists serialization."""
        data = {"items": [1, 2, 3], "values": [4.0, 5.0, 6.0]}
        assert safe_loads(safe_dumps(data)) == data

    def test_tuple_converted_to_list(self):
        """Test tuple is converted to tuple (not list)."""
        data = (1, 2, 3)
        result = safe_loads(safe_dumps(data))
        assert result == (1, 2, 3)
        assert isinstance(result, tuple)

    def test_nested_tuple(self):
        """Test nested tuple serialization."""
        data = ((1, 2), (3, 4))
        result = safe_loads(safe_dumps(data))
        assert result == ((1, 2), (3, 4))
        assert isinstance(result, tuple)
        assert isinstance(result[0], tuple)


class TestNumpyTypes:
    """Tests for numpy type serialization."""

    def test_numpy_int32(self):
        """Test numpy int32 scalar serialization."""
        data = np.int32(42)
        result = safe_loads(safe_dumps(data))
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_int64(self):
        """Test numpy int64 scalar serialization."""
        data = np.int64(1234567890)
        result = safe_loads(safe_dumps(data))
        assert result == 1234567890
        assert isinstance(result, int)

    def test_numpy_float32(self):
        """Test numpy float32 scalar serialization."""
        data = np.float32(3.14)
        result = safe_loads(safe_dumps(data))
        assert abs(result - 3.14) < 1e-5
        assert isinstance(result, float)

    def test_numpy_float64(self):
        """Test numpy float64 scalar serialization."""
        data = np.float64(3.14159265358979)
        result = safe_loads(safe_dumps(data))
        assert abs(result - 3.14159265358979) < 1e-10
        assert isinstance(result, float)

    def test_small_numpy_array(self):
        """Test small numpy array serialization."""
        data = np.array([1, 2, 3, 4, 5])
        result = safe_loads(safe_dumps(data))
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, data)

    def test_numpy_2d_array(self):
        """Test 2D numpy array serialization."""
        data = np.array([[1, 2], [3, 4]])
        result = safe_loads(safe_dumps(data))
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result, data)

    def test_numpy_float_array(self):
        """Test numpy float array serialization."""
        data = np.array([1.1, 2.2, 3.3], dtype=np.float64)
        result = safe_loads(safe_dumps(data))
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, data)

    def test_large_array_rejected(self):
        """Test that large numpy arrays are rejected."""
        data = np.ones(1001)
        with pytest.raises(SafeSerializationError, match="too large"):
            safe_dumps(data)


class TestPhaseHistoryScenarios:
    """Tests for phase_history-like data structures from adaptive_hybrid.py."""

    def test_phase_0_record(self):
        """Test Phase 0 normalization setup record."""
        data = {
            "phase": 0,
            "name": "normalization_setup",
            "strategy": "log_magnitude",
            "timestamp": 1234567890.123,
            "normalized_params_shape": (3,),
            "has_bounds": True,
        }
        result = safe_loads(safe_dumps(data))
        assert result["phase"] == 0
        assert result["name"] == "normalization_setup"
        assert result["strategy"] == "log_magnitude"
        assert result["timestamp"] == 1234567890.123
        assert result["normalized_params_shape"] == (3,)
        assert result["has_bounds"] is True

    def test_phase_1_record(self):
        """Test Phase 1 L-BFGS warmup record."""
        data = {
            "phase": 1,
            "name": "lbfgs_warmup",
            "iterations": 50,
            "final_loss": 0.0012345,
            "best_loss": 0.0010234,
            "switch_reason": "Gradient norm below tolerance",
            "timestamp": 1234567891.456,
            "lr_mode": "adaptive",
            "relative_loss": 0.00123,
        }
        result = safe_loads(safe_dumps(data))
        assert result["phase"] == 1
        assert result["iterations"] == 50
        assert abs(result["final_loss"] - 0.0012345) < 1e-10

    def test_phase_2_record(self):
        """Test Phase 2 Gauss-Newton record."""
        data = {
            "phase": 2,
            "name": "gauss_newton",
            "iterations": 15,
            "final_cost": 0.0005678,
            "best_cost": 0.0005432,
            "convergence_reason": "Cost change below tolerance",
            "gradient_norm": 1e-8,
            "timestamp": 1234567892.789,
        }
        result = safe_loads(safe_dumps(data))
        assert result["phase"] == 2
        assert result["convergence_reason"] == "Cost change below tolerance"

    def test_phase_history_list(self):
        """Test full phase_history list structure."""
        data = [
            {"phase": 0, "name": "setup", "timestamp": 1.0},
            {"phase": 1, "name": "warmup", "timestamp": 2.0, "iterations": 100},
            {"phase": 2, "name": "refinement", "timestamp": 3.0, "iterations": 20},
            {"phase": 3, "name": "finalize", "timestamp": 4.0, "duration": 0.5},
        ]
        result = safe_loads(safe_dumps(data))
        assert len(result) == 4
        assert result[0]["phase"] == 0
        assert result[-1]["phase"] == 3


class TestTournamentCheckpointScenarios:
    """Tests for tournament_checkpoint-like data structures."""

    def test_round_history(self):
        """Test tournament round history list."""
        data = [
            {"round": 0, "survivors": 16, "eliminated": 0},
            {"round": 1, "survivors": 8, "eliminated": 8},
            {"round": 2, "survivors": 4, "eliminated": 4},
        ]
        result = safe_loads(safe_dumps(data))
        assert len(result) == 3
        assert result[1]["survivors"] == 8

    def test_scalar_values(self):
        """Test scalar checkpoint values."""
        data = {
            "current_round": 2,
            "total_batches_evaluated": 150,
            "numerical_failures": 3,
            "n_candidates": 16,
            "n_params": 5,
        }
        result = safe_loads(safe_dumps(data))
        assert result["current_round"] == 2
        assert result["n_params"] == 5


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_string(self):
        """Test empty string serialization."""
        data = ""
        assert safe_loads(safe_dumps(data)) == ""

    def test_unicode_string(self):
        """Test unicode string serialization."""
        data = "Hello, 世界! 🎉"
        assert safe_loads(safe_dumps(data)) == data

    def test_deeply_nested_structure(self):
        """Test deeply nested structure serialization."""
        data = {"a": {"b": {"c": {"d": {"e": {"f": 42}}}}}}
        result = safe_loads(safe_dumps(data))
        assert result["a"]["b"]["c"]["d"]["e"]["f"] == 42

    def test_list_with_none(self):
        """Test list containing None values."""
        data = [1, None, 2, None, 3]
        assert safe_loads(safe_dumps(data)) == data

    def test_dict_with_none_values(self):
        """Test dict with None values."""
        data = {"key1": None, "key2": "value"}
        assert safe_loads(safe_dumps(data)) == data

    def test_numeric_string_keys_preserved(self):
        """Test that dict keys are converted to strings."""
        data = {1: "one", 2: "two"}
        result = safe_loads(safe_dumps(data))
        # Keys converted to strings
        assert result["1"] == "one"
        assert result["2"] == "two"


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_json_bytes(self):
        """Test that invalid JSON bytes raise error."""
        with pytest.raises(SafeSerializationError, match="Deserialization failed"):
            safe_loads(b"not valid json")

    def test_invalid_utf8_bytes(self):
        """Test that invalid UTF-8 bytes raise error."""
        with pytest.raises(SafeSerializationError, match="Deserialization failed"):
            safe_loads(b"\xff\xfe")

    def test_unsupported_type_rejected(self):
        """Test that unsupported types are rejected."""

        class CustomClass:
            pass

        with pytest.raises(SafeSerializationError, match="Cannot safely serialize"):
            safe_dumps(CustomClass())

    def test_function_rejected(self):
        """Test that functions are rejected."""
        with pytest.raises(SafeSerializationError, match="Cannot safely serialize"):
            safe_dumps(lambda x: x)

    def test_set_rejected(self):
        """Test that sets are rejected."""
        with pytest.raises(SafeSerializationError, match="Cannot safely serialize"):
            safe_dumps({1, 2, 3})


class TestRoundTrip:
    """Tests for complete round-trip serialization."""

    def test_complex_nested_structure(self):
        """Test complex nested structure round-trip."""
        data = {
            "config": {
                "name": "test",
                "values": [1, 2, 3],
                "nested": {"deep": True},
            },
            "results": [
                {"iteration": 1, "loss": 0.5, "params": (1.0, 2.0, 3.0)},
                {"iteration": 2, "loss": 0.3, "params": (1.1, 2.1, 3.1)},
            ],
            "metadata": None,
        }
        result = safe_loads(safe_dumps(data))

        assert result["config"]["name"] == "test"
        assert result["config"]["values"] == [1, 2, 3]
        assert result["results"][0]["params"] == (1.0, 2.0, 3.0)
        assert result["metadata"] is None

    def test_bytes_output_type(self):
        """Test that safe_dumps returns bytes."""
        data = {"test": 123}
        result = safe_dumps(data)
        assert isinstance(result, bytes)

    def test_bytes_input_type(self):
        """Test that safe_loads accepts bytes."""
        data = {"test": 123}
        serialized = safe_dumps(data)
        result = safe_loads(serialized)
        assert result == data
