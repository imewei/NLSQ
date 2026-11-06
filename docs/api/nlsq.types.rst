nlsq.types module
=================

.. automodule:: nlsq.types
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``types`` module defines type annotations and type aliases used throughout NLSQ.

Type Aliases
------------

This module provides type aliases for:

- **Array types**: ``ArrayLike``, ``JAXArray``, ``FloatArray``
- **Function types**: ``ModelFunction``, ``JacobianFunction``, ``CallbackFunction``, ``LossFunction``
- **Option types**: ``BoundsTuple``, ``MethodLiteral``
- **Diagnostic dictionaries** (``TypedDict``): ``CheckpointInfo``, ``CommonError``, ``AggregateStats``, ``StreamingDiagnostics``

``OptimizeResult`` is documented in :doc:`nlsq.result`.

Example Usage
-------------

.. code-block:: python

   import jax.numpy as jnp

   from nlsq.types import ArrayLike, ModelFunction


   # Type-annotated function
   def my_model(x: ArrayLike, a: float, b: float) -> ArrayLike:
       return a * jnp.exp(-b * x)


   # Using ModelFunction type alias
   model: ModelFunction = my_model

Benefits
--------

Using type annotations from this module:

- **Improved IDE support** with autocomplete
- **Type checking** with mypy/pyright
- **Better documentation** for function signatures
- **Clearer code** with explicit types

Type Definitions
----------------

.. code-block:: python

   from collections.abc import Callable
   from typing import TypeAlias

   import numpy as np

   FloatArray: TypeAlias = np.ndarray

   # Runtime definition; static checkers see np.ndarray | jnp.ndarray | list | tuple
   ModelFunction: TypeAlias = Callable[..., "ArrayLike"]  # f(x, *params) -> y_pred
   JacobianFunction: TypeAlias = Callable[..., "ArrayLike"]  # jac(x, *params) -> J

See Also
--------

- :doc:`nlsq.result` - Result containers
- :doc:`nlsq.config` - Configuration types
