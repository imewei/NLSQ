Materials Science Applications
==============================

Curve fitting for mechanical testing, thermal analysis, and relaxation.

Stress-Strain Analysis
----------------------

Power Law Hardening
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import jax.numpy as jnp
   from nlsq import fit


   def power_law_hardening(strain, K, n):
       """Power law strain hardening: sigma = K * epsilon^n."""
       return K * strain**n


   # Fit
   popt, pcov = fit(
       power_law_hardening,
       strain_data,
       stress_data,
       p0=[500, 0.2],
       bounds=([0, 0], [2000, 1]),
   )

Voce Hardening
~~~~~~~~~~~~~~

.. code-block:: python

   def voce_hardening(strain, sigma_y, sigma_sat, theta):
       """Voce hardening law."""
       return sigma_sat - (sigma_sat - sigma_y) * jnp.exp(-theta * strain / sigma_sat)

Hollomon Equation
~~~~~~~~~~~~~~~~~

.. code-block:: python

   def hollomon(strain, K, n):
       """Hollomon equation for true stress-strain."""
       return K * strain**n

Thermal Analysis
----------------

Arrhenius Rate
~~~~~~~~~~~~~~

.. code-block:: python

   def arrhenius(T, A, Ea):
       """Arrhenius rate constant: k = A * exp(-Ea/RT)."""
       R = 8.314  # J/(mol·K)
       return A * jnp.exp(-Ea / (R * T))


   # Thermal decomposition fitting
   popt, pcov = fit(
       arrhenius,
       temperature_K,
       rate_constant,
       p0=[1e13, 100000],
       bounds=([1e8, 10000], [1e20, 500000]),
   )

Kissinger Peak
~~~~~~~~~~~~~~

.. code-block:: python

   def kissinger_peak(T, Tm, Ea, A):
       """Kissinger peak shape for DSC."""
       R = 8.314
       x = Ea / (R * T)
       xm = Ea / (R * Tm)
       return A * jnp.exp(xm - x - jnp.exp(xm - x))

Glass Transition
~~~~~~~~~~~~~~~~

.. code-block:: python

   def wlf_viscosity(T, eta_g, C1, C2, Tg):
       """Williams-Landel-Ferry equation for viscosity near Tg."""
       return eta_g * jnp.exp(-C1 * (T - Tg) / (C2 + T - Tg))

Relaxation Phenomena
--------------------

Kohlrausch-Williams-Watts
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def kww_relaxation(t, amplitude, tau, beta):
       """Stretched exponential relaxation."""
       return amplitude * jnp.exp(-((t / tau) ** beta))

Cole-Cole
~~~~~~~~~

.. code-block:: python

   def cole_cole(omega, eps_s, eps_inf, tau, alpha):
       """Cole-Cole dielectric relaxation.
       Returns complex permittivity (real part)."""
       denom = 1 + (1j * omega * tau) ** (1 - alpha)
       eps = eps_inf + (eps_s - eps_inf) / denom
       return jnp.real(eps)

Havriliak-Negami
~~~~~~~~~~~~~~~~

.. code-block:: python

   def havriliak_negami(omega, eps_s, eps_inf, tau, alpha, beta):
       """Havriliak-Negami dielectric relaxation."""
       denom = (1 + (1j * omega * tau) ** (1 - alpha)) ** beta
       eps = eps_inf + (eps_s - eps_inf) / denom
       return jnp.real(eps)

Creep Analysis
--------------

Power Law Creep
~~~~~~~~~~~~~~~

.. code-block:: python

   def power_law_creep(t, epsilon_0, A, n):
       """Power law creep: epsilon = epsilon_0 + A * t^n."""
       return epsilon_0 + A * t**n

Burgers Model
~~~~~~~~~~~~~

.. code-block:: python

   def burgers_model(t, sigma, E1, eta1, E2, eta2):
       """Burgers viscoelastic model."""
       # Maxwell arm
       maxwell = sigma * t / eta1
       # Kelvin-Voigt arm
       tau2 = eta2 / E2
       kelvin = (sigma / E2) * (1 - jnp.exp(-t / tau2))
       # Instantaneous elastic
       elastic = sigma / E1
       return elastic + maxwell + kelvin

Tips for Materials Fitting
--------------------------

1. **Use appropriate units**:

   - SI units for consistency
   - Document units in code comments

2. **Consider measurement uncertainty**:

   .. code-block:: python

      # Weight by measurement precision
      sigma = measurement_uncertainty
      popt, pcov = fit(model, x, y, sigma=sigma, absolute_sigma=True)

3. **For rate parameters**, use log-transform if spanning orders of magnitude

4. **For temperature-dependent data**, use 1/T for Arrhenius analysis

5. **Validate with standards**: Test on known materials first

See Also
--------

- :doc:`/howto/choose_model` - Model selection
- :doc:`/tutorials/03_fitting_with_bounds` - Using bounds
