JAX and Automatic Differentiation
==================================

NLSQ uses JAX for GPU acceleration and automatic Jacobian computation.
This guide explains how it works and why it matters for curve fitting.

What is JAX?
------------

JAX is a numerical computing library that provides:

1. **NumPy-compatible API**: Write code like you would with NumPy
2. **Automatic differentiation**: Compute derivatives automatically
3. **JIT compilation**: Compile Python to optimized machine code
4. **GPU/TPU acceleration**: Run on accelerators with no code changes

Why JAX for Curve Fitting?
--------------------------

**Traditional approach** (SciPy):

- Compute Jacobian using finite differences
- Each partial derivative requires a function evaluation
- For m parameters: 2m extra function calls per iteration
- Numerically approximate (subject to step size errors)

**JAX approach** (NLSQ):

- Compute exact Jacobian via automatic differentiation
- Single backward pass computes all derivatives
- No extra function evaluations
- Analytically exact (machine precision)

.. code-block:: python

   # You write this
   def model(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   # JAX automatically computes
   # ∂f/∂a = exp(-b * x)
   # ∂f/∂b = -a * x * exp(-b * x)
   # ∂f/∂c = 1

Automatic Differentiation
-------------------------

AD is not:
- Symbolic differentiation (like Mathematica)
- Numerical differentiation (finite differences)

AD is:
- Algorithmic transformation of code
- Tracks derivatives through computations
- Exact to machine precision

Two Modes
~~~~~~~~~

**Forward mode**: Propagate derivatives forward through computation

.. code-block:: text

   Input: x     →     f₁(x)     →     f₂(f₁(x))     →     Output
          ∂x/∂x=1 → ∂f₁/∂x     →     ∂f₂/∂x        →     ∂y/∂x

Good for few inputs, many outputs.

**Reverse mode** (backpropagation): Propagate derivatives backward

.. code-block:: text

   Input: x     →     f₁(x)     →     f₂(f₁(x))     →     Output
          ∂y/∂x ←     ∂y/∂f₁   ←     ∂y/∂f₂=1     ←     ∂y/∂y=1

Good for many inputs, few outputs (like gradients in optimization).

NLSQ uses reverse mode to efficiently compute the Jacobian.

JIT Compilation
---------------

JAX's Just-In-Time compiler transforms Python to optimized XLA code:

.. code-block:: python

   @jax.jit
   def model(x, a, b):
       return a * jnp.exp(-b * x)


   # First call: compile (slower)
   y1 = model(x, 1.0, 0.5)

   # Subsequent calls: run compiled code (fast!)
   y2 = model(x, 2.0, 0.3)

Benefits:

1. **Operator fusion**: Combine multiple operations
2. **Memory optimization**: Reduce intermediate allocations
3. **Parallelization**: Utilize all CPU cores or GPU threads
4. **Constant folding**: Pre-compute static values

Why Use jax.numpy?
------------------

JAX operations must be traced to enable AD and JIT:

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp


   # This WON'T work with JAX
   def bad_model(x, a, b):
       return a * np.exp(-b * x)  # NumPy exp can't be traced


   # This WORKS with JAX
   def good_model(x, a, b):
       return a * jnp.exp(-b * x)  # JAX exp is traceable

Rule: Use ``jax.numpy`` for any math inside model functions.

GPU Acceleration
----------------

JAX automatically uses GPU when available:

.. code-block:: python

   import jax

   # Check available devices
   print(jax.devices())  # [cuda(id=0)] or [cpu()]

   # Data automatically moves to GPU
   x = jnp.array([1, 2, 3])  # Lives on GPU if available

   # Computations run on GPU
   y = jnp.exp(x)  # Computed on GPU

No code changes needed - same code runs on CPU or GPU.

Pure Functions
--------------

JAX requires **pure functions** - no side effects:

.. code-block:: python

   # BAD: Side effects
   counter = 0


   def bad_model(x, a, b):
       global counter
       counter += 1  # Side effect!
       return a * jnp.exp(-b * x)


   # GOOD: Pure function
   def good_model(x, a, b):
       return a * jnp.exp(-b * x)  # No side effects

Why? JAX may:
- Cache and reuse results
- Execute operations in different order
- Run computations in parallel

Common Gotchas
--------------

1. **Dynamic shapes**

   .. code-block:: python

      # BAD: Shape depends on values
      def bad(x, a):
          if a > 0:  # Python control flow on traced value
              return x[:10]
          return x


      # GOOD: Use jnp.where for conditionals
      def good(x, a):
          return jnp.where(a > 0, x * 2, x)

2. **In-place mutation**

   .. code-block:: python

      # BAD: Mutating arrays
      def bad(x):
          x[0] = 0  # JAX arrays are immutable!
          return x


      # GOOD: Create new array
      def good(x):
          return x.at[0].set(0)

3. **Random numbers**

   .. code-block:: python

      # BAD: NumPy random
      def bad():
          return np.random.randn()  # Not reproducible in JAX


      # GOOD: JAX random with key
      def good(key):
          return jax.random.normal(key)

Performance Tips
----------------

1. **Warm up JIT**

   .. code-block:: python

      # First call compiles (slow)
      _ = model(x_small, *p0)

      # Subsequent calls are fast
      result = model(x_large, *p0)

2. **Batch similar computations**

   .. code-block:: python

      # Use vmap for vectorization
      batched_model = jax.vmap(model, in_axes=(0, None, None))
      results = batched_model(x_batch, a, b)

3. **Use float32 for larger datasets**

   .. code-block:: python

      x = x.astype(jnp.float32)  # Half the memory, faster

Summary
-------

JAX enables NLSQ's key features:

- **Automatic Jacobians**: Exact derivatives, no manual math
- **JIT compilation**: Fast execution after first call
- **GPU acceleration**: Same code, massive speedups
- **Numerical precision**: IEEE 754 exact derivatives

Just remember to use ``jax.numpy`` in your model functions!

See Also
--------

- :doc:`gpu_architecture` - GPU acceleration details
- :doc:`how_fitting_works` - Overall fitting process
- `JAX Documentation <https://jax.readthedocs.io/>`_
