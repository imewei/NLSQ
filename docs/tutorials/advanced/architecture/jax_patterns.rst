JAX Patterns
============

NLSQ is built on JAX for GPU acceleration and automatic differentiation.
Understanding JAX patterns helps you write efficient custom code.

JIT Compilation
---------------

Just-In-Time (JIT) compilation converts Python to optimized XLA code:

.. code-block:: python

   import jax
   import jax.numpy as jnp


   # Without JIT - interpreted Python (slow)
   def slow_model(x, a, b):
       return a * jnp.exp(-b * x)


   # With JIT - compiled to XLA (fast)
   @jax.jit
   def fast_model(x, a, b):
       return a * jnp.exp(-b * x)


   # First call: compilation (~1s)
   # Subsequent calls: execution (~1ms)

**NLSQ handles JIT automatically** - model functions are compiled internally.

Automatic Differentiation
-------------------------

JAX computes exact gradients automatically:

.. code-block:: python

   import jax


   def loss(params, x, y):
       a, b = params
       y_pred = a * jnp.exp(-b * x)
       return jnp.sum((y_pred - y) ** 2)


   # Gradient function
   grad_loss = jax.grad(loss)
   gradients = grad_loss(params, x, y)


   # Jacobian (multiple outputs)
   def residuals(params, x, y):
       a, b = params
       return a * jnp.exp(-b * x) - y


   jacobian = jax.jacrev(residuals)(params, x, y)

Forward vs Reverse Mode
-----------------------

JAX supports both differentiation modes:

.. code-block:: python

   # Forward mode: efficient for n_params < n_outputs
   J_fwd = jax.jacfwd(func)(params)

   # Reverse mode: efficient for n_params > n_outputs
   J_rev = jax.jacrev(func)(params)

**NLSQ auto-selects** based on dimensions:

- Few parameters, many data points → reverse mode
- Many parameters, few outputs → forward mode

Pure Functions
--------------

JAX requires pure functions (no side effects):

.. code-block:: python

   # Wrong - side effect (modifies global)
   results = []


   def bad_model(x, a, b):
       results.append(a)  # Side effect!
       return a * jnp.exp(-b * x)


   # Correct - pure function
   def good_model(x, a, b):
       return a * jnp.exp(-b * x)

Array Operations
----------------

Use JAX array operations, not Python loops:

.. code-block:: python

   # Wrong - Python loop (slow, not JIT-able)
   def slow_sum(x):
       total = 0
       for xi in x:
           total += xi
       return total


   # Correct - JAX array operation (fast)
   def fast_sum(x):
       return jnp.sum(x)


   # Wrong - Python conditional (not JIT-able)
   def bad_clip(x, a, b):
       if a < 0:
           return x * b
       return x * a


   # Correct - JAX conditional
   def good_clip(x, a, b):
       return jnp.where(a < 0, x * b, x * a)

Type Conversion
---------------

JAX arrays and NumPy arrays interoperate:

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp

   # NumPy to JAX (explicit)
   x_np = np.array([1, 2, 3])
   x_jax = jnp.array(x_np)

   # JAX to NumPy (explicit)
   x_back = np.array(x_jax)

   # NLSQ handles conversion automatically

Device Placement
----------------

JAX manages CPU/GPU placement:

.. code-block:: python

   import jax

   # Check default device
   print(jax.default_backend())  # 'cpu' or 'gpu'

   # Force CPU
   with jax.default_device(jax.devices("cpu")[0]):
       result = model(x, a, b)

   # Arrays on specific device
   x_gpu = jax.device_put(x, jax.devices("gpu")[0])

Vmap for Batching
-----------------

Vectorize functions over batch dimensions:

.. code-block:: python

   import jax


   def fit_single(x, y, p0):
       # Fit one dataset
       return popt


   # Vectorize over multiple datasets
   fit_batch = jax.vmap(fit_single)

   # Fit 100 datasets in parallel
   all_popt = fit_batch(x_batch, y_batch, p0_batch)

Random Numbers
--------------

JAX uses explicit PRNG keys:

.. code-block:: python

   import jax.random as random

   # Create key
   key = random.PRNGKey(42)

   # Split for multiple uses
   key1, key2 = random.split(key)

   # Generate random numbers
   noise = random.normal(key1, shape=(100,))

Common Pitfalls
---------------

**1. Traced values in shapes:**

.. code-block:: python

   # Wrong - shape depends on value
   def bad_func(x, n):
       return x[:n]  # n is traced, can't use for shape


   # Correct - static shape
   def good_func(x, n):
       mask = jnp.arange(len(x)) < n
       return jnp.where(mask, x, 0)

**2. In-place operations:**

.. code-block:: python

   # Wrong - in-place modification
   def bad_func(x):
       x[0] = 0  # JAX arrays are immutable
       return x


   # Correct - functional update
   def good_func(x):
       return x.at[0].set(0)

**3. Python control flow:**

.. code-block:: python

   # Wrong - Python if (not JIT-able)
   def bad_func(x, a):
       if a > 0:
           return x * a
       return x


   # Correct - JAX control flow
   def good_func(x, a):
       return jax.lax.cond(a > 0, lambda: x * a, lambda: x)

NLSQ JIT Caching
----------------

NLSQ caches JIT compilations:

.. code-block:: python

   # Persistent cache location
   # ~/.cache/nlsq/jax_cache

   # First fit: JIT compilation (~1-5s)
   popt1, pcov1 = fit(model, x1, y1, p0=[...])

   # Second fit: cached (~0.1s)
   popt2, pcov2 = fit(model, x2, y2, p0=[...])

   # Disable caching (for debugging)
   import os

   os.environ["NLSQ_DISABLE_PERSISTENT_CACHE"] = "1"

Next Steps
----------

- :doc:`../core_apis/index` - Using core API classes
- :doc:`../performance/index` - Performance tuning
