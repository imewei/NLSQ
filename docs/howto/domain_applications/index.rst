Domain-Specific Applications
=============================

Ready-to-use fitting workflows for specific scientific domains.

.. toctree::
   :maxdepth: 1

   spectroscopy
   imaging
   kinetics
   materials

Overview
--------

These guides provide complete workflows for common curve fitting applications
in specific scientific domains. Each guide includes:

- Common model functions
- Recommended parameter bounds
- Initial guess estimation strategies
- Domain-specific tips and pitfalls

Spectroscopy
~~~~~~~~~~~~

:doc:`spectroscopy`

- Peak fitting (Gaussian, Lorentzian, Voigt)
- Fluorescence lifetime analysis
- Absorption spectroscopy
- Raman/IR spectra deconvolution

Imaging
~~~~~~~

:doc:`imaging`

- 2D Gaussian fitting (PSF characterization)
- Spot detection and quantification
- Background estimation
- Multi-spot fitting

Kinetics
~~~~~~~~

:doc:`kinetics`

- Michaelis-Menten enzyme kinetics
- Binding isotherms (Langmuir, Hill)
- Pharmacokinetics (1/2-compartment models)
- Chemical reaction kinetics

Materials Science
~~~~~~~~~~~~~~~~~

:doc:`materials`

- Stress-strain curves
- Thermal analysis (DSC, TGA)
- Relaxation phenomena
- Creep and fatigue

General Tips
------------

1. **Start with known physics**: Use models derived from theory
2. **Validate with standards**: Test on known samples first
3. **Document your workflow**: Keep records of fitting parameters
4. **Report uncertainties**: Always include parameter errors

See Also
--------

- :doc:`/howto/choose_model` - Model selection guide
- :doc:`/tutorials/index` - General tutorials
