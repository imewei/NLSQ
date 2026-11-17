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


# Every reserved key-marker prefix. A literal string key that happens to
# start with one of these must be escaped (via _STR_KEY_PREFIX) so it can't
# be misread as a typed key on deserialization.
_KEY_PREFIXES = (
    "__bool_key__",
    "__int_key__",
    "__float_key__",
    "__str_key__",
)
_STR_KEY_PREFIX = "__str_key__"


def _serialize_dict_key(key: Any) -> str:
    """Convert a dict key to a collision-resistant string for JSON serialization."""
    # bool must come before int since bool is a subclass of int
    if isinstance(key, (bool, np.bool_)):
        return f"__bool_key__{bool(key)}"
    if isinstance(key, str):
        # Escape a literal string that would otherwise be misread as a
        # typed-key marker on deserialization (e.g. the string "__int_key__7").
        if key.startswith(_KEY_PREFIXES):
            return f"{_STR_KEY_PREFIX}{key}"
        return key
    if isinstance(key, (int, np.integer)):
        return f"__int_key__{int(key)}"
    if isinstance(key, (float, np.floating)):
        _float_to_serializable(key)
        return f"__float_key__{float(key)}"
    # Any other key type cannot be restored on deserialization — fail loud
    # rather than silently producing a mangled, unrestorable string key.
    raise SafeSerializationError(
        f"Cannot safely serialize dict key of type {type(key).__name__}. "
        "Only str, int, float, and bool keys are supported.",
    )


def _deserialize_dict_key(key: str) -> str | int | float | bool:
    """Restore original dict key type from serialized string form."""
    # Escaped string keys take priority: strip exactly one escape layer and
    # return the remainder as a literal string, whatever it looks like.
    if key.startswith(_STR_KEY_PREFIX):
        return key[len(_STR_KEY_PREFIX) :]
    if key.startswith("__bool_key__"):
        value = key[len("__bool_key__") :]
        if value not in ("True", "False"):
            raise SafeSerializationError(
                f"Malformed boolean key in serialized data: {key!r}",
            )
        return value == "True"
    if key.startswith("__int_key__"):
        text = key[len("__int_key__") :]
        try:
            int_value = int(text)
        except ValueError as e:
            raise SafeSerializationError(
                f"Malformed integer key in serialized data: {key!r}",
            ) from e
        if str(int_value) != text:  # reject '01', '+1', ' 1' (alias canonical keys)
            raise SafeSerializationError(
                f"Non-canonical integer key in serialized data: {key!r}",
            )
        return int_value
    if key.startswith("__float_key__"):
        try:
            return float(key[len("__float_key__") :])
        except ValueError as e:
            raise SafeSerializationError(
                f"Malformed float key in serialized data: {key!r}",
            ) from e
    return key


def _convert_dict_to_serializable(obj: dict) -> dict:
    """Convert a dict to its JSON-serializable form.

    Raises
    ------
    SafeSerializationError
        On key collision after string-encoding, or if the dict uses the
        reserved "__nlsq_type__" marker key (which would be misread as an
        internal type marker on deserialization).
    """
    result = {}
    for key, value in obj.items():
        str_key = _serialize_dict_key(key)
        if str_key in result:
            raise SafeSerializationError(
                f"Key collision during serialization: {key!r} maps to "
                f"existing key {str_key!r}",
            )
        result[str_key] = _convert_to_serializable(value)
    if "__nlsq_type__" in result:
        raise SafeSerializationError(
            "Cannot serialize dict containing reserved key '__nlsq_type__'; "
            "this key is reserved for internal type markers.",
        )
    if _is_legacy_type_payload(result):
        raise SafeSerializationError(
            "Cannot serialize a dict that looks like the legacy '__type__' marker "
            "format; it would be rejected on load. Rename the '__type__' key.",
        )
    return result


def _sanitize_array_data(data: Any) -> Any:
    """Recursively wrap NaN/Inf floats in nested list data from ndarray.tolist().

    Mirrors the scalar float handling in ``_convert_to_serializable`` so that
    NaN/Inf values inside arrays round-trip through standard JSON instead of
    emitting non-standard ``NaN``/``Infinity`` tokens.
    """
    if isinstance(data, list):
        return [_sanitize_array_data(item) for item in data]
    if isinstance(data, complex):
        return {
            "__nlsq_type__": "complex",
            "real": _sanitize_array_data(data.real),
            "imag": _sanitize_array_data(data.imag),
        }
    if isinstance(data, float):
        if np.isnan(data):
            return {"__nlsq_type__": "float", "value": "nan"}
        if np.isinf(data):
            return {"__nlsq_type__": "float", "value": "inf" if data > 0 else "-inf"}
    return data


def _marker_part(part: Any) -> float:
    """Real or imaginary part of a complex marker: a plain number or float marker.

    Validated without recursion, so a marker cannot make the loader walk an
    arbitrarily large nested payload before being rejected.
    """
    if isinstance(part, (int, float)):
        return part
    if (
        isinstance(part, dict)
        and part.keys() == {"__nlsq_type__", "value"}
        and part["__nlsq_type__"] == "float"
    ):
        value = (
            _FLOAT_MARKERS.get(part["value"])
            if isinstance(part["value"], str)
            else None
        )
        if value is not None:
            return value
    raise SafeSerializationError(
        "Complex marker parts must be real numbers or float markers"
    )


_FLOAT_MARKERS = {"nan": float("nan"), "inf": float("inf"), "-inf": float("-inf")}


def _unsanitize_array_data(data: Any) -> Any:
    """Reverse ``_sanitize_array_data``, restoring NaN/Inf floats in array data."""
    if isinstance(data, list):
        return [_unsanitize_array_data(item) for item in data]
    if isinstance(data, dict):
        if data.get("__nlsq_type__") == "complex":
            return complex(
                _marker_part(data.get("real")), _marker_part(data.get("imag"))
            )
        if data.get("__nlsq_type__") == "float":
            value = data.get("value")
            if value == "nan":
                return float("nan")
            if value == "inf":
                return float("inf")
            if value == "-inf":
                return float("-inf")
            raise SafeSerializationError(
                f"Unknown float value in serialized array data: {value!r}",
            )
        raise SafeSerializationError(
            f"Unrecognized type marker in serialized array data: "
            f"{data.get('__nlsq_type__')!r}",
        )
    return data


def _float_to_serializable(obj: Any) -> Any:
    """Float scalar -> JSON value; non-finite as marker, extended precision rejected."""
    value = float(obj)
    if (
        isinstance(obj, np.floating)
        and obj.itemsize > 8
        and np.isfinite(obj)
        and value != obj
    ):
        # extended precision would be silently rounded to float64
        raise SafeSerializationError(
            f"Cannot serialize {type(obj).__name__} without losing precision"
        )
    if np.isnan(value):
        return {"__nlsq_type__": "float", "value": "nan"}
    if np.isinf(value):
        return {"__nlsq_type__": "float", "value": "inf" if value > 0 else "-inf"}
    return value


def _complex_to_serializable(obj: Any) -> Any:
    """Complex scalar -> the marker used for complex array elements."""
    if isinstance(obj, np.complexfloating) and obj.itemsize > 16:
        raise SafeSerializationError(
            f"Cannot serialize {type(obj).__name__} without losing precision"
        )
    return _sanitize_array_data(complex(obj))


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
    if isinstance(obj, np.bool_):
        return bool(obj)

    # Handle numeric types (including numpy scalars)
    if isinstance(obj, (int, np.integer)):
        return int(obj)

    if isinstance(obj, (float, np.floating)):
        return _float_to_serializable(obj)

    if isinstance(obj, (complex, np.complexfloating)):
        return _complex_to_serializable(obj)

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
        return _convert_dict_to_serializable(obj)

    # MaskedArray is an ndarray subclass; tolist() would drop the mask silently
    if isinstance(obj, np.ma.MaskedArray):
        raise SafeSerializationError(
            "masked arrays are not supported; pass np.ma.filled(...) / the mask separately",
        )

    # Handle numpy arrays (small ones only - large arrays should use HDF5)
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind not in ("b", "i", "u", "f", "c"):
            raise SafeSerializationError(
                f"Cannot serialize array with non-numeric dtype {obj.dtype!r}; "
                "only numeric and bool dtypes round-trip.",
            )
        if obj.size > 10_000:
            raise SafeSerializationError(
                f"NumPy array too large for JSON serialization ({obj.size} elements). "
                "Use HDF5 storage for large arrays.",
            )
        return {
            "__nlsq_type__": "ndarray",
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "data": _sanitize_array_data(obj.tolist()),
        }

    # Handle JAX arrays (convert to numpy first, then serialize)
    if type(obj).__module__.startswith("jax"):
        return _convert_to_serializable(np.asarray(obj))

    # Reject unknown types
    raise SafeSerializationError(
        f"Cannot safely serialize object of type {type(obj).__name__}. "
        "Only basic types (str, int, float, bool, None, list, dict, tuple) "
        "and small numpy arrays are supported.",
    )


_MAX_NDIM = 64


def _check_nested_structure(data: Any, shape: tuple, limit: int) -> None:
    """Require ``data`` to be nested lists exactly matching ``shape``.

    Walks level by level and stops at the first mismatch, so no container or
    leaf beyond what the declared shape (and ``limit`` nodes per level) allows
    is traversed or copied; empty-shape arrays cannot smuggle uncounted
    containers.
    """
    if len(shape) > _MAX_NDIM:
        raise SafeSerializationError(f"Array has too many dimensions ({len(shape)})")
    level = [data]
    for dim in shape:
        if len(level) * dim > limit:
            raise SafeSerializationError(
                f"Array data is larger than the limit ({limit:,} nodes per level)"
            )
        nxt: list = []
        for node in level:
            if not isinstance(node, list) or len(node) != dim:
                raise SafeSerializationError(
                    f"Malformed ndarray: data does not match shape {shape}"
                )
            nxt.extend(node)
        level = nxt
    if any(isinstance(node, list) for node in level):
        raise SafeSerializationError(
            f"Malformed ndarray: data is nested deeper than shape {shape}"
        )


def _deserialize_ndarray(obj: dict) -> np.ndarray:
    """Reconstruct a numpy array from its serialized dict representation."""
    try:
        dtype = np.dtype(obj["dtype"])
        raw_data = obj["data"]
        shape = tuple(obj["shape"])
    except (KeyError, TypeError, ValueError) as e:
        raise SafeSerializationError(
            f"Malformed ndarray in serialized data: {e!r}",
        ) from e
    # Only allow numeric dtypes — object dtype arrays can hold arbitrary Python
    # objects, defeating the security guarantee of this module. Reject this
    # BEFORE unsanitizing: a rejected object-dtype array's "data" may itself
    # contain legitimate dict elements that happen to look like the
    # NaN/Inf sentinel shape, which _unsanitize_array_data must not be given
    # a chance to misinterpret.
    if dtype.kind not in ("b", "i", "u", "f", "c"):
        raise SafeSerializationError(
            f"Refusing to deserialize array with non-numeric dtype "
            f"{dtype!r}. Only numeric dtypes are allowed.",
        )
    # Validate each dimension is a non-negative integer. A zero dimension
    # makes n_elements=0 which bypasses the limit check while allowing
    # an arbitrarily large logical shape through reshape.
    for i, dim in enumerate(shape):
        if not isinstance(dim, int) or dim < 0:
            raise SafeSerializationError(
                f"Array shape[{i}]={dim!r} must be a non-negative integer",
            )
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    if n_elements > _MAX_DESERIALIZE_ELEMENTS:
        raise SafeSerializationError(
            f"Refusing to deserialize array with {n_elements:,} elements "
            f"(limit: {_MAX_DESERIALIZE_ELEMENTS:,}). "
            "Possible DoS via attacker-controlled shape.",
        )
    # Bound the ACTUAL payload before allocating anything from it: the declared
    # shape alone says nothing about what "data" really holds.
    _check_nested_structure(raw_data, shape, _MAX_DESERIALIZE_ELEMENTS)
    data = _unsanitize_array_data(raw_data)
    # np.array(["False"], dtype=bool) is True: only accept real JSON booleans.
    if dtype.kind == "b" and not all(
        isinstance(v, bool) for v in np.asarray(data, dtype=object).ravel()
    ):
        raise SafeSerializationError("Boolean array data must contain only booleans")
    try:
        return np.array(data, dtype=dtype).reshape(shape)
    except (ValueError, TypeError, OverflowError) as e:
        raise SafeSerializationError(
            f"Malformed ndarray in serialized data: {e}",
        ) from e


def _is_legacy_type_payload(obj: dict) -> bool:
    return obj.get("__type__") in ("ndarray", "tuple", "float") and (
        "data" in obj or "value" in obj
    )


def _reject_legacy_type_payload(obj: dict) -> None:
    """Fail loudly on the removed legacy ``__type__`` format.

    Such a payload would otherwise restore as a plain dict where an
    array/tuple/float was written.
    """
    if _is_legacy_type_payload(obj):
        raise SafeSerializationError(
            "Legacy '__type__' payload is no longer supported; re-create the "
            "checkpoint with this version of NLSQ"
        )


def _tuple_from_marker(obj: dict) -> tuple:
    if not isinstance(obj.get("value"), list):
        raise SafeSerializationError("Malformed tuple marker: 'value' must be a list")
    return tuple(_convert_from_serializable(item) for item in obj["value"])


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
                return _tuple_from_marker(obj)

            if type_name == "complex":
                return _unsanitize_array_data(obj)

            if type_name == "float":
                value = obj["value"]
                if value == "nan":
                    return float("nan")
                if value == "inf":
                    return float("inf")
                if value == "-inf":
                    return float("-inf")
                raise SafeSerializationError(
                    f"Unknown float value in serialized data: {value!r}",
                )

            if type_name == "ndarray":
                return _deserialize_ndarray(obj)

            raise SafeSerializationError(
                f"Unrecognized __nlsq_type__ tag in serialized data: {type_name!r}",
            )

        _reject_legacy_type_payload(obj)

        # Regular dict — restore non-string keys from prefixed encoding
        result: dict = {}
        for key, value in obj.items():
            decoded = _deserialize_dict_key(key)
            if decoded in result:
                raise SafeSerializationError(
                    f"Duplicate dict key after decoding in serialized data: {key!r}",
                )
            result[decoded] = _convert_from_serializable(value)
        return result

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
            "Serialization failed: circular reference detected",
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
    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        AttributeError,
    ) as e:
        raise SafeSerializationError(f"Deserialization failed: {e!r}") from e
    except RecursionError as e:
        raise SafeSerializationError(
            "Deserialization failed: input is too deeply nested "
            "(possible DoS via deeply nested JSON)",
        ) from e
