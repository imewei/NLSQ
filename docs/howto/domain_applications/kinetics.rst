Kinetics Applications
=====================

Curve fitting for enzyme kinetics, binding, and chemical reactions.

Enzyme Kinetics
---------------

Michaelis-Menten
~~~~~~~~~~~~~~~~

.. code-block:: python

   import jax.numpy as jnp
   from nlsq import fit


   def michaelis_menten(S, Vmax, Km):
       """Michaelis-Menten enzyme kinetics."""
       return Vmax * S / (Km + S)


   # Fit
   popt, pcov = fit(
       michaelis_menten,
       substrate_conc,
       velocity,
       p0=[100, 10],
       bounds=([0, 0], [1000, 1000]),
   )

Hill Equation
~~~~~~~~~~~~~

.. code-block:: python

   def hill_equation(S, Vmax, K, n):
       """Hill equation (cooperative binding)."""
       return Vmax * S**n / (K**n + S**n)

Substrate Inhibition
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def substrate_inhibition(S, Vmax, Km, Ki):
       """Substrate inhibition model."""
       return Vmax * S / (Km + S * (1 + S / Ki))

Binding Isotherms
-----------------

Langmuir Isotherm
~~~~~~~~~~~~~~~~~

.. code-block:: python

   def langmuir(C, Qmax, Kd):
       """Langmuir binding isotherm."""
       return Qmax * C / (Kd + C)

Two-Site Langmuir
~~~~~~~~~~~~~~~~~

.. code-block:: python

   def two_site_langmuir(C, Q1, K1, Q2, K2):
       """Two-site Langmuir (heterogeneous binding)."""
       return Q1 * C / (K1 + C) + Q2 * C / (K2 + C)

Freundlich Isotherm
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def freundlich(C, Kf, n):
       """Freundlich isotherm (heterogeneous surfaces)."""
       return Kf * C ** (1 / n)

Dose-Response
-------------

4-Parameter Logistic
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def four_parameter_logistic(x, bottom, top, EC50, hill):
       """4PL dose-response curve."""
       return bottom + (top - bottom) / (1 + (EC50 / x) ** hill)


   # IC50/EC50 determination
   popt, pcov = fit(
       four_parameter_logistic,
       dose,
       response,
       p0=[0, 100, 10, 1],
       bounds=([0, 0, 0.001, 0.1], [50, 200, 1000, 10]),
       preset="quality",
   )

5-Parameter Logistic
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def five_parameter_logistic(x, bottom, top, EC50, hill, asymmetry):
       """5PL asymmetric dose-response."""
       return bottom + (top - bottom) / (1 + (EC50 / x) ** hill) ** asymmetry

Chemical Kinetics
-----------------

First-Order Decay
~~~~~~~~~~~~~~~~~

.. code-block:: python

   def first_order(t, A0, k, offset):
       """First-order reaction: A → products."""
       return A0 * jnp.exp(-k * t) + offset

Second-Order Kinetics
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def second_order_equal(t, A0, k):
       """Second-order reaction with equal concentrations."""
       return A0 / (1 + k * A0 * t)

Consecutive Reactions
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def consecutive_reactions(t, A0, k1, k2):
       """A → B → C consecutive first-order reactions.
       Returns concentration of B."""
       return A0 * k1 / (k2 - k1) * (jnp.exp(-k1 * t) - jnp.exp(-k2 * t))

Tips for Kinetics Fitting
-------------------------

1. **Transform for linear analysis first**:

   - Lineweaver-Burk, Eadie-Hofstee for enzymes
   - Linear plots help estimate initial guesses

2. **Use weighted fitting** when variance changes with concentration:

   .. code-block:: python

      sigma = np.sqrt(response)  # Poisson-like errors
      popt, pcov = fit(model, x, y, sigma=sigma, absolute_sigma=True)

3. **Report with proper statistics**:

   - 95% confidence intervals
   - R² and RMSE
   - Residual analysis

4. **For competitive models**, compare with AIC/BIC:

   .. code-block:: python

      result1 = fit(michaelis_menten, S, v)
      result2 = fit(substrate_inhibition, S, v)

      print(f"MM: AIC={result1.aic:.2f}")
      print(f"SI: AIC={result2.aic:.2f}")

See Also
--------

- :doc:`/howto/choose_model` - Model selection
- :doc:`/tutorials/02_understanding_results` - Result interpretation
