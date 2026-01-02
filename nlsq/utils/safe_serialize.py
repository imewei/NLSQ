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
from typing import Any

import numpy as np

__all__ = ["SafeSerializationError", "safe_dumps", "safe_loads"]


class SafeSerializationError(Exception):
    """Exception raised when serialization/deserialization fails."""

    pass


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
            return {"__type__": "float", "value": "nan"}
        if np.isinf(value):
            return {"__type__": "float", "value": "inf" if value > 0 else "-inf"}
        return value

    # Handle tuples (convert to list with marker)
    if isinstance(obj, tuple):
        return {
            "__type__": "tuple",
            "value": [_convert_to_serializable(item) for item in obj],
        }

    # Handle lists
    if isinstance(obj, list):
        return [_convert_to_serializable(item) for item in obj]

    # Handle dicts
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            # JSON requires string keys
            str_key = str(key) if not isinstance(key, str) else key
            result[str_key] = _convert_to_serializable(value)
        return result

    # Handle numpy arrays (small ones only - large arrays should use HDF5)
    if isinstance(obj, np.ndarray):
        if obj.size > 1000:
            raise SafeSerializationError(
                f"NumPy array too large for JSON serialization ({obj.size} elements). "
                "Use HDF5 storage for large arrays."
            )
        return {
            "__type__": "ndarray",
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "data": obj.tolist(),
        }

    # Reject unknown types
    raise SafeSerializationError(
        f"Cannot safely serialize object of type {type(obj).__name__}. "
        "Only basic types (str, int, float, bool, None, list, dict, tuple) "
        "and small numpy arrays are supported."
    )


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
        # Check for type markers
        if "__type__" in obj:
            type_name = obj["__type__"]

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

            if type_name == "ndarray":
                dtype = np.dtype(obj["dtype"])
                data = obj["data"]
                shape = tuple(obj["shape"])
                return np.array(data, dtype=dtype).reshape(shape)

        # Regular dict
        return {key: _convert_from_serializable(value) for key, value in obj.items()}

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
