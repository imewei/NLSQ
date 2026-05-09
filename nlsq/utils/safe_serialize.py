"""Safe serialization utilities for checkpoint data.

This module provides JSON-based serialization as a secure alternative to pickle
for checkpointing internal optimizer state. It addresses CWE-502 (Deserialization
of Untrusted Data) by using JSON which cannot execute arbitrary code.

The serialization handles:
- Standard JSON types (str, int, float, bool, None, list, dict)
- Python tuples (converted to lists with metadata)
- NumPy scalar types (converted to Python primitives)

For binary data or numpy arrays, use the HDF5 storage directly instead of
serializing through this module.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Any

import numpy as np

__all__ = ["SafeSerializationError", "safe_dumps", "safe_loads"]

# Hard cap on array elements during deserialization — prevents memory-exhaustion
# DoS via attacker-controlled shape values in untrusted JSON (e.g. [999999999, 999999999]).
_MAX_DESERIALIZE_ELEMENTS = 100_000_000  # 100M elements (~800 MB at float64)


class SafeSerializationError(Exception):
    """Exception raised when serialization/deserialization fails."""

    pass


def _serialize_dict_key(key: Any) -> str:
    """Convert a dict key to a collision-resistant string for JSON serialization."""
    if isinstance(key, str):
        return key
    if isinstance(key, int):
        return f"__int_key__{key}"
    if isinstance(key, float):
        return f"__float_key__{key}"
    return f"__key_{type(key).__name__}__{key}"


def _deserialize_dict_key(key: str) -> str | int | float:
    """Restore original dict key type from serialized string form."""
    if key.startswith("__int_key__"):
        try:
            return int(key[len("__int_key__") :])
        except ValueError as e:
            raise SafeSerializationError(
                f"Malformed integer key in serialized data: {key!r}"
            ) from e
    if key.startswith("__float_key__"):
        try:
            return float(key[len("__float_key__") :])
        except ValueError as e:
            raise SafeSerializationError(
                f"Malformed float key in serialized data: {key!r}"
            ) from e
    return key


def _convert_to_serializable(obj: Any) -> Any:
    """Convert object to JSON-serializable form.

    Parameters
    ----------
    obj : Any
        Object to convert.

    Returns
    -------
    Any
        JSON-serializable representation.

    Raises
    ------
    SafeSerializationError
        If the object cannot be safely serialized.
    """
    # Handle None
    if obj is None:
        return None

    # Handle basic types
    if isinstance(obj, (str, bool)):
        # bool must come before int since bool is subclass of int
        return obj

    # Handle numeric types (including numpy scalars)
    if isinstance(obj, (int, np.integer)):
        return int(obj)

    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        # Handle special float values
        if np.isnan(value):
            return {"__nlsq_type__": "float", "value": "nan"}
        if np.isinf(value):
            return {"__nlsq_type__": "float", "value": "inf" if value > 0 else "-inf"}
        return value

    # Handle tuples (convert to list with marker)
    if isinstance(obj, tuple):
        return {
            "__nlsq_type__": "tuple",
            "value": [_convert_to_serializable(item) for item in obj],
        }

    # Handle lists and deques (deque serializes as list)
    if isinstance(obj, (list, deque)):
        return [_convert_to_serializable(item) for item in obj]

    # Handle dicts
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            str_key = _serialize_dict_key(key)
            if str_key in result:
                raise SafeSerializationError(
                    f"Key collision during serialization: {key!r} maps to "
                    f"existing key {str_key!r}"
                )
            result[str_key] = _convert_to_serializable(value)
        return result

    # Handle numpy arrays (small ones only - large arrays should use HDF5)
    if isinstance(obj, np.ndarray):
        if obj.size > 10_000:
            raise SafeSerializationError(
                f"NumPy array too large for JSON serialization ({obj.size} elements). "
                "Use HDF5 storage for large arrays."
            )
        return {
            "__nlsq_type__": "ndarray",
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "data": obj.tolist(),
        }

    # Handle JAX arrays (convert to numpy first, then serialize)
    if type(obj).__module__.startswith("jax"):
        return _convert_to_serializable(np.asarray(obj))

    # Reject unknown types
    raise SafeSerializationError(
        f"Cannot safely serialize object of type {type(obj).__name__}. "
        "Only basic types (str, int, float, bool, None, list, dict, tuple) "
        "and small numpy arrays are supported."
    )


def _deserialize_ndarray(obj: dict) -> np.ndarray:
    """Reconstruct a numpy array from its serialized dict representation."""
    try:
        dtype = np.dtype(obj["dtype"])
        data = obj["data"]
        shape = tuple(obj["shape"])
    except KeyError as e:
        raise SafeSerializationError(
            f"Malformed ndarray in serialized data: missing key {e}"
        ) from e
    # Only allow numeric dtypes — object dtype arrays can hold arbitrary Python
    # objects, defeating the security guarantee of this module
    if dtype.kind not in ("b", "i", "u", "f", "c"):
        raise SafeSerializationError(
            f"Refusing to deserialize array with non-numeric dtype "
            f"{dtype!r}. Only numeric dtypes are allowed."
        )
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    if n_elements > _MAX_DESERIALIZE_ELEMENTS:
        raise SafeSerializationError(
            f"Refusing to deserialize array with {n_elements:,} elements "
            f"(limit: {_MAX_DESERIALIZE_ELEMENTS:,}). "
            "Possible DoS via attacker-controlled shape."
        )
    return np.array(data, dtype=dtype).reshape(shape)


def _convert_from_serializable(obj: Any) -> Any:
    """Convert JSON-deserialized object back to Python types.

    Parameters
    ----------
    obj : Any
        JSON-deserialized object.

    Returns
    -------
    Any
        Reconstructed Python object.
    """
    # Handle None
    if obj is None:
        return None

    # Handle basic types
    if isinstance(obj, (str, bool, int, float)):
        return obj

    # Handle lists
    if isinstance(obj, list):
        return [_convert_from_serializable(item) for item in obj]

    # Handle dicts (may contain type markers)
    if isinstance(obj, dict):
        # Check for NLSQ type marker (namespaced key only — the legacy "__type__" fallback
        # was removed because any user dict with that key would incorrectly trigger type
        # dispatch, and attacker-controlled values could reach unbounded np.reshape).
        if "__nlsq_type__" in obj:
            type_name = obj["__nlsq_type__"]

            if type_name == "tuple":
                return tuple(_convert_from_serializable(item) for item in obj["value"])

            if type_name == "float":
                value = obj["value"]
                if value == "nan":
                    return float("nan")
                if value == "inf":
                    return float("inf")
                if value == "-inf":
                    return float("-inf")
                raise SafeSerializationError(
                    f"Unknown float value in serialized data: {value!r}"
                )

            if type_name == "ndarray":
                return _deserialize_ndarray(obj)

        # Regular dict — restore non-string keys from prefixed encoding
        return {
            _deserialize_dict_key(key): _convert_from_serializable(value)
            for key, value in obj.items()
        }

    return obj


def safe_dumps(obj: Any) -> bytes:
    """Serialize object to bytes using JSON.

    This is a secure alternative to pickle.dumps() that cannot execute
    arbitrary code during deserialization.

    Parameters
    ----------
    obj : Any
        Object to serialize. Must be JSON-serializable or contain only
        basic Python types, tuples, and small numpy arrays.

    Returns
    -------
    bytes
        UTF-8 encoded JSON bytes.

    Raises
    ------
    SafeSerializationError
        If the object cannot be safely serialized.

    Examples
    --------
    >>> data = {"phase": 1, "cost": 0.5, "timestamp": 1234567890.0}
    >>> serialized = safe_dumps(data)
    >>> isinstance(serialized, bytes)
    True
    """
    try:
        serializable = _convert_to_serializable(obj)
        return json.dumps(serializable, separators=(",", ":")).encode("utf-8")
    except RecursionError as e:
        raise SafeSerializationError(
            "Serialization failed: circular reference detected"
        ) from e
    except (TypeError, ValueError) as e:
        raise SafeSerializationError(f"Serialization failed: {e}") from e


def safe_loads(data: bytes) -> Any:
    """Deserialize bytes to object using JSON.

    This is a secure alternative to pickle.loads() that cannot execute
    arbitrary code during deserialization.

    Parameters
    ----------
    data : bytes
        UTF-8 encoded JSON bytes from safe_dumps().

    Returns
    -------
    Any
        Deserialized Python object.

    Raises
    ------
    SafeSerializationError
        If the data cannot be safely deserialized.

    Examples
    --------
    >>> data = {"phase": 1, "cost": 0.5}
    >>> serialized = safe_dumps(data)
    >>> restored = safe_loads(serialized)
    >>> restored == data
    True
    """
    try:
        parsed = json.loads(data.decode("utf-8"))
        return _convert_from_serializable(parsed)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise SafeSerializationError(f"Deserialization failed: {e}") from e
    except RecursionError as e:
        raise SafeSerializationError(
            "Deserialization failed: input is too deeply nested "
            "(possible DoS via deeply nested JSON)"
        ) from e
