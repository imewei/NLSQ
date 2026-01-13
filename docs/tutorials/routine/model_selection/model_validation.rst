Model Validation
================

Before fitting, verify your model is correct. This prevents debugging
issues that come from model bugs rather than fitting problems.

Quick Validation Checklist
--------------------------

1. Model returns correct shape
2. Model uses ``jax.numpy``
3. Parameters affect output as expected
4. Model handles edge cases
5. Initial guess produces reasonable output

Step 1: Check Output Shape
--------------------------

The model should return the same shape as input ``x``:

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp


   def my_model(x, a, b):
       return a * jnp.exp(-b * x)


   # Test
   x_test = np.linspace(0, 10, 50)
   y_test = my_model(x_test, 2.0, 0.5)

   print(f"Input shape:  {x_test.shape}")
   print(f"Output shape: {y_test.shape}")
   assert x_test.shape == y_test.shape, "Shape mismatch!"

Step 2: Verify JAX Compatibility
--------------------------------

Ensure the model works with JAX's automatic differentiation:

.. code-block:: python

   import jax


   def my_model(x, a, b):
       return a * jnp.exp(-b * x)


   # Test gradient computation
   def loss(params):
       a, b = params
       y_pred = my_model(x_test, a, b)
       return jnp.sum((y_pred - y_test) ** 2)


   # This should work without errors
   grad_fn = jax.grad(loss)
   grads = grad_fn(jnp.array([2.0, 0.5]))
   print(f"Gradients: {grads}")

If this fails, your model likely uses incompatible operations.

Step 3: Parameter Sensitivity
-----------------------------

Verify each parameter affects the output:

.. code-block:: python

   import matplotlib.pyplot as plt

   x = np.linspace(0, 10, 100)

   # Vary parameter 'a'
   plt.figure(figsize=(12, 4))

   plt.subplot(1, 2, 1)
   for a in [1, 2, 3, 4]:
       y = my_model(x, a, b=0.5)
       plt.plot(x, y, label=f"a={a}")
   plt.legend()
   plt.title("Effect of parameter a")

   # Vary parameter 'b'
   plt.subplot(1, 2, 2)
   for b in [0.2, 0.5, 1.0, 2.0]:
       y = my_model(x, a=2.0, b=b)
       plt.plot(x, y, label=f"b={b}")
   plt.legend()
   plt.title("Effect of parameter b")

   plt.tight_layout()
   plt.show()

Step 4: Edge Cases
------------------

Test boundary conditions:

.. code-block:: python

   # Test at x = 0
   y_zero = my_model(0.0, 2.0, 0.5)
   print(f"y(0) = {y_zero}")  # Should be defined

   # Test large x
   y_large = my_model(100.0, 2.0, 0.5)
   print(f"y(100) = {y_large}")  # Should not overflow

   # Test edge parameter values
   y_edge = my_model(x_test, 0.0, 0.5)  # a = 0
   print(f"y with a=0: {y_edge[:3]}")  # Should be all zeros

Step 5: Compare with Data
-------------------------

Plot model output with initial guess against data:

.. code-block:: python

   import matplotlib.pyplot as plt

   # Your data
   x_data = ...
   y_data = ...

   # Initial guess
   p0 = [2.0, 0.5]

   # Model prediction with initial guess
   y_init = my_model(x_data, *p0)

   # Plot
   plt.figure(figsize=(10, 6))
   plt.scatter(x_data, y_data, label="Data", alpha=0.5)
   plt.plot(x_data, y_init, "r-", label="Initial guess", linewidth=2)
   plt.xlabel("x")
   plt.ylabel("y")
   plt.legend()
   plt.title("Data vs Initial Guess")
   plt.show()

The initial guess curve should be roughly similar to the data. If it's
completely wrong, adjust ``p0`` or check the model.

Common Model Bugs
-----------------

**Using numpy instead of jax.numpy:**

.. code-block:: python

   # Bug: uses numpy
   import numpy as np


   def bad_model(x, a, b):
       return a * np.exp(-b * x)  # Won't work with JAX


   # Fix: use jax.numpy
   import jax.numpy as jnp


   def good_model(x, a, b):
       return a * jnp.exp(-b * x)

**Forgetting return statement:**

.. code-block:: python

   # Bug: no return
   def bad_model(x, a, b):
       y = a * jnp.exp(-b * x)
       # Missing return!


   # Fix
   def good_model(x, a, b):
       return a * jnp.exp(-b * x)

**Parameter order mismatch:**

.. code-block:: python

   # Model expects (x, a, b)
   def model(x, a, b):
       return a * jnp.exp(-b * x)


   # Bug: p0 order doesn't match
   p0 = [0.5, 2.0]  # Should be [a, b] = [2.0, 0.5]

**Division by zero:**

.. code-block:: python

   # Bug: divides by zero when x=c
   def bad_model(x, a, b, c):
       return a / (x - c) + b


   # Fix: add small epsilon
   def good_model(x, a, b, c):
       return a / (x - c + 1e-10) + b

Complete Validation Script
--------------------------

.. code-block:: python

   import numpy as np
   import jax
   import jax.numpy as jnp
   from nlsq import fit


   def validate_model(model, x_data, y_data, p0, param_names=None):
       """Validate a model before fitting."""

       print("=== Model Validation ===\n")

       # 1. Shape check
       y_test = model(x_data, *p0)
       print(f"1. Shape check: input={x_data.shape}, output={y_test.shape}")
       assert x_data.shape == y_test.shape, "FAIL: Shape mismatch"
       print("   PASS\n")

       # 2. JAX gradient check
       def loss(params):
           return jnp.sum((model(x_data, *params) - y_data) ** 2)

       try:
           grads = jax.grad(loss)(jnp.array(p0))
           print(f"2. JAX gradient check: {grads}")
           print("   PASS\n")
       except Exception as e:
           print(f"   FAIL: {e}\n")

       # 3. Parameter influence
       y_base = model(x_data, *p0)
       print("3. Parameter influence:")
       for i, p in enumerate(p0):
           p_mod = list(p0)
           p_mod[i] = p * 1.1  # 10% change
           y_mod = model(x_data, *p_mod)
           diff = jnp.abs(y_mod - y_base).mean()
           name = param_names[i] if param_names else f"p{i}"
           status = "OK" if diff > 1e-10 else "WARNING: no effect"
           print(f"   {name}: avg change = {diff:.2e} ({status})")
       print()

       # 4. Quick fit test
       print("4. Quick fit test:")
       try:
           popt, pcov = fit(model, x_data, y_data, p0=p0, max_nfev=50)
           print(f"   Converged to: {popt}")
           print("   PASS\n")
       except Exception as e:
           print(f"   FAIL: {e}\n")

       print("=== Validation Complete ===")


   # Usage
   def my_model(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   x = np.linspace(0, 10, 50)
   y = 2.5 * np.exp(-0.5 * x) + 0.3 + 0.1 * np.random.randn(len(x))

   validate_model(
       my_model, x, y, p0=[2.0, 0.4, 0.0], param_names=["amplitude", "decay", "offset"]
   )

Next Steps
----------

- :doc:`../data_handling/index` - Prepare your data for fitting
- :doc:`../three_workflows/auto_workflow` - Start fitting
