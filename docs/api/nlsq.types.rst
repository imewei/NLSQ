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

- **Array types**: ``Array``, ``ArrayLike``
- **Function types**: ``ModelFunction``, ``ResidualFunction``
- **Result types**: ``OptimizeResult``
- **Configuration types**: ``Config``, ``MemoryConfig``

Example Usage
-------------

.. code-block:: python

   from nlsq.types import Array, ModelFunction
   import jax.numpy as jnp


   # Type-annotated function
   def my_model(x: Array, a: float, b: float) -> Array:
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

   from typing import Callable
   import jax.numpy as jnp

   # Common type aliases
   Array = jnp.ndarray
   ArrayLike = Union[Array, np.ndarray, Sequence[float]]

   ModelFunction = Callable[[Array, ...], Array]
   ResidualFunction = Callable[[Array], Array]

See Also
--------

- :doc:`nlsq.result` - Result containers
- :doc:`nlsq.config` - Configuration types
