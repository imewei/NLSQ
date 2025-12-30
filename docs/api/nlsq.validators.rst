nlsq.validators module
=======================

.. currentmodule:: nlsq.utils.validators

.. automodule:: nlsq.utils.validators
   :noindex:

Overview
--------

The ``nlsq.validators`` module provides comprehensive input validation for curve fitting
and optimization functions. It catches common errors early, provides helpful error messages,
and ensures data quality before expensive optimization operations begin.

**New in version 0.1.1**: Complete validation system with fast mode and extensive checks.

**New in version 0.3.1**: Added security-focused validation (array size limits, bounds checking,
parameter validation).

Key Features
------------

- **Comprehensive input validation** for all curve fitting parameters
- **Security validation** (v0.3.1): Array size limits, bounds numeric range, parameter values
- **Early error detection** with clear, actionable error messages
- **Data quality checks** for outliers, duplicates, and degenerate cases
- **Fast mode** to skip expensive checks for performance-critical code
- **Function signature analysis** to detect parameter mismatches
- **Automatic type conversion** with warnings
- **Bounds validation** including initial guess checking
- **Decorator support** for automatic validation

Classes
-------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   InputValidator

Functions
---------

.. autosummary::
   :toctree: generated/

   validate_inputs

Usage Examples
--------------

Basic Validation for curve_fit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Validate inputs before curve fitting:

.. code-block:: python

    from nlsq.validators import InputValidator
    import numpy as np

    # Create validator
    validator = InputValidator(fast_mode=False)


    # Define model and data
    def model(x, a, b):
        return a * np.exp(-b * x)


    x = np.linspace(0, 10, 100)
    y = 2.5 * np.exp(-0.5 * x) + np.random.normal(0, 0.1, 100)
    p0 = [2.0, 0.4]

    # Validate inputs
    errors, warnings, x_clean, y_clean = validator.validate_curve_fit_inputs(
        model, x, y, p0=p0, bounds=([0, 0], [10, 10])
    )

    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Validation passed!")

    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")

Fast Mode Validation
~~~~~~~~~~~~~~~~~~~~

Skip expensive checks for performance:

.. code-block:: python

    from nlsq.validators import InputValidator

    # Fast mode skips function callable tests and data quality checks
    fast_validator = InputValidator(fast_mode=True)

    errors, warnings, x_clean, y_clean = fast_validator.validate_curve_fit_inputs(
        model, x, y, p0=p0
    )

    # Much faster, suitable for production code with trusted inputs

Decorator-Based Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Automatically validate function inputs:

.. code-block:: python

    from nlsq.validators import validate_inputs
    import numpy as np


    @validate_inputs(validation_type="curve_fit")
    def my_curve_fit(f, xdata, ydata, p0=None, **kwargs):
        """Custom curve fit with automatic validation."""
        # Inputs are automatically validated before this code runs
        # Invalid inputs raise ValueError with detailed message
        # xdata and ydata are converted to numpy arrays

        # Your fitting logic here
        return optimize(f, xdata, ydata, p0, **kwargs)


    # Use it - validation happens automatically
    try:
        result = my_curve_fit(model, x, y, p0=[1.0, 0.5])
    except ValueError as e:
        print(f"Validation failed: {e}")

Validation for least_squares
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Validate least squares inputs:

.. code-block:: python

    from nlsq.validators import InputValidator

    validator = InputValidator()


    # Define residual function
    def residual(params):
        a, b = params
        return y_data - (a * np.exp(-b * x_data))


    x0 = np.array([2.0, 0.5])
    bounds = ([0, 0], [10, 10])

    # Validate
    errors, warnings, x0_clean = validator.validate_least_squares_inputs(
        residual,
        x0,
        bounds=bounds,
        method="trf",
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        max_nfev=1000,
    )

    if errors:
        raise ValueError(f"Validation failed: {'; '.join(errors)}")

Detailed Validation Checks
~~~~~~~~~~~~~~~~~~~~~~~~~~~

See what validation detects:

.. code-block:: python

    from nlsq.validators import InputValidator
    import numpy as np

    validator = InputValidator(fast_mode=False)

    # Problematic data
    x = np.array([1.0, 2.0, np.nan, 4.0])  # Contains NaN
    y = np.array([1.0, 2.0, 3.0])  # Wrong length
    p0 = [1.0, 2.0, 3.0]  # Wrong number of params


    def model(x, a, b):
        return a * x + b


    errors, warnings, _, _ = validator.validate_curve_fit_inputs(model, x, y, p0=p0)

    # Errors will include:
    # - "xdata contains 1 NaN or Inf values"
    # - "xdata (4 points) and ydata (3 points) must have same length"
    # - "Initial guess p0 has 3 parameters, but function expects 2"

Data Quality Warnings
~~~~~~~~~~~~~~~~~~~~~

Detect potential data quality issues:

.. code-block:: python

    from nlsq.validators import InputValidator
    import numpy as np

    validator = InputValidator(fast_mode=False)

    # Data with quality issues
    x = np.array([1, 2, 3, 3, 3, 4, 5, 100])  # Duplicates and outlier
    y = np.array([1, 2, 3, 3.1, 2.9, 4, 5, 200])  # Outlier


    def linear(x, a, b):
        return a * x + b


    errors, warnings, _, _ = validator.validate_curve_fit_inputs(
        linear, x, y, p0=[1.0, 0.0]
    )

    # Warnings will include:
    # - "xdata contains 3 duplicate values"
    # - "ydata may contain 1 outliers - consider using robust loss function"

Bounds Validation
~~~~~~~~~~~~~~~~~

Check parameter bounds:

.. code-block:: python

    from nlsq.validators import InputValidator
    import numpy as np

    validator = InputValidator()

    x = np.linspace(0, 10, 50)
    y = 2 * x + 1 + np.random.randn(50) * 0.5


    def linear(x, a, b):
        return a * x + b


    # Invalid bounds
    bad_bounds = ([0, 0], [0, 10])  # Lower >= Upper for first param
    p0 = [2.0, 1.0]

    errors, warnings, _, _ = validator.validate_curve_fit_inputs(
        linear, x, y, p0=p0, bounds=bad_bounds
    )

    # Errors: "Lower bounds must be less than upper bounds"

    # Initial guess outside bounds
    bounds = ([0, 0], [10, 10])
    p0_out = [-1.0, 1.0]  # First param outside bounds

    errors, warnings, _, _ = validator.validate_curve_fit_inputs(
        linear, x, y, p0=p0_out, bounds=bounds
    )

    # Warnings: "Initial guess p0 is outside bounds"

Sigma Validation
~~~~~~~~~~~~~~~~

Validate uncertainty parameters:

.. code-block:: python

    from nlsq.validators import InputValidator
    import numpy as np

    validator = InputValidator()

    x = np.linspace(0, 10, 50)
    y = 2 * x + 1 + np.random.randn(50) * 0.5

    # Invalid sigma - wrong shape
    sigma_bad = np.ones(40)  # Wrong length

    errors, warnings, _, _ = validator.validate_curve_fit_inputs(
        lambda x, a, b: a * x + b, x, y, p0=[1, 0], sigma=sigma_bad
    )

    # Error: "sigma must have same shape as ydata"

    # Invalid sigma - negative values
    sigma_neg = np.ones(50)
    sigma_neg[10] = -0.5

    errors, warnings, _, _ = validator.validate_curve_fit_inputs(
        lambda x, a, b: a * x + b, x, y, p0=[1, 0], sigma=sigma_neg
    )

    # Error: "sigma values must be positive"

Degenerate Data Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~

Detect problematic data patterns:

.. code-block:: python

    from nlsq.validators import InputValidator
    import numpy as np

    validator = InputValidator()

    # All x values identical
    x_const = np.ones(100)
    y = np.random.randn(100)

    errors, warnings, _, _ = validator.validate_curve_fit_inputs(
        lambda x, a, b: a * x + b, x_const, y, p0=[1, 0]
    )

    # Error: "All x values are identical - cannot fit"

    # Very small range
    x_small_range = np.linspace(1.0, 1.0000000001, 100)  # Range ~ 1e-9

    errors, warnings, _, _ = validator.validate_curve_fit_inputs(
        lambda x, a, b: a * x + b, x_small_range, y, p0=[1, 0]
    )

    # Warning: "x data range is very small (1.00e-09) - consider rescaling"

Tolerance Validation
~~~~~~~~~~~~~~~~~~~~

Validate convergence tolerances:

.. code-block:: python

    from nlsq.validators import InputValidator

    validator = InputValidator()


    def residual(x):
        return x**2


    # Very small tolerances
    errors, warnings, x0_clean = validator.validate_least_squares_inputs(
        residual, x0=[1.0], ftol=1e-16, xtol=1e-17, gtol=1e-8  # Too small  # Too small
    )

    # Warnings:
    # - "ftol=1e-16 is very small, may not converge"
    # - "xtol=1e-17 is very small, may not converge"

Custom Validation Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build custom validation logic:

.. code-block:: python

    from nlsq.validators import InputValidator
    import numpy as np


    class CustomValidator(InputValidator):
        """Extended validator with custom checks."""

        def validate_my_data(self, x, y, **kwargs):
            """Custom validation pipeline."""
            # Use parent class methods
            errors, warnings, x, y = self.validate_curve_fit_inputs(
                kwargs["f"], x, y, p0=kwargs.get("p0"), bounds=kwargs.get("bounds")
            )

            # Add custom checks
            if np.std(y) < 0.01:
                warnings.append(
                    "y data has very low variance - may indicate measurement issue"
                )

            if len(x) < 10:
                errors.append("Need at least 10 data points for reliable fit")

            return errors, warnings, x, y


    # Use it
    custom_validator = CustomValidator()
    errors, warnings, x, y = custom_validator.validate_my_data(
        x_data, y_data, f=model_func, p0=initial_guess
    )

Security Validation (v0.3.1)
-----------------------------

The validator includes security-focused checks to prevent resource exhaustion
and detect malformed inputs early:

.. code-block:: python

   from nlsq.validators import InputValidator, validate_security_constraints

   validator = InputValidator()

   # Security checks run automatically and early in the validation pipeline
   errors, warnings, x, y = validator.validate_curve_fit_inputs(model, x, y, p0=p0)

   # Or call directly
   errors, warnings = validate_security_constraints(x, y, n_params=3)

**Array Size Limits**:

- Maximum 10 billion (10^10) data points
- Maximum 100 billion (10^11) Jacobian elements
- Detects integer overflow in size calculations
- Memory estimation warnings (>10GB, >100GB)

.. code-block:: python

   # Example: Excessive data size detection
   x_huge = np.ones(15_000_000_000)  # 15 billion points
   # Error: "Data size exceeds maximum allowed (10B points)"

**Bounds Numeric Range**:

- Warns for extreme bound values (absolute value > 1e100)
- Errors for NaN in bounds

.. code-block:: python

   # Example: Extreme bounds detection
   bounds = ([0, 0], [1e200, 10])  # Extremely large upper bound
   # Warning: "Bounds contain extreme values (>1e100)"

**Parameter Value Validation**:

- Warns for extreme initial parameters (abs(p0) > 1e50)
- Errors for NaN/Inf in initial parameters

.. code-block:: python

   # Example: Invalid parameter detection
   p0 = [1.0, np.nan, 0.5]
   # Error: "Initial parameters contain NaN/Inf values"

**Early Fail-Fast**:

Security validation runs before expensive operations to fail fast on
malformed input that could cause denial-of-service or numerical instability.

Validation Check Reference
---------------------------

The validator performs these checks:

**Array Validation**:

- Convert to numpy arrays
- Check dimensions (at least 1D)
- Validate array lengths match
- Check for tuple xdata (multi-dimensional fitting)

**Finite Values**:

- Detect NaN values
- Detect Inf values
- Report counts of bad values

**Data Shapes**:

- Minimum 2 data points
- More data points than parameters
- Matching xdata/ydata lengths

**Initial Guess (p0)**:

- Correct number of parameters
- All finite values
- Length matches function signature

**Bounds**:

- 2-tuple format (lower, upper)
- Correct lengths
- Lower < Upper for all parameters
- Initial guess within bounds
- Method compatibility (LM doesn't support bounds)

**Sigma**:

- Correct shape (matches ydata)
- All positive values
- All finite values

**Tolerances**:

- All positive (ftol, xtol, gtol)
- Not too small (<1e-15)

**Method**:

- Valid method name ('trf', 'lm', 'dogbox')

**Function Callable** (unless fast_mode=True):

- Function can be evaluated
- Output shape matches expected

**Data Quality** (unless fast_mode=True):

- Check for duplicate x values
- Detect potential outliers (3 IQR rule)
- Degenerate x values (all identical, tiny range)
- Degenerate y values (all identical, tiny range)

Performance Impact
------------------

**Fast mode** (fast_mode=True):

- Skips: Function callable test, data quality checks
- Speedup: ~30-50% faster validation
- Use for: Production code with trusted inputs

**Full mode** (fast_mode=False):

- All checks enabled
- Recommended for: Interactive use, debugging, untrusted inputs

**Typical validation overhead**:

- Small datasets (<1000 points): <1ms
- Medium datasets (1000-100K points): 1-10ms
- Large datasets (>100K points): 10-100ms

**Recommendation**: Use fast mode in tight loops, full mode for user-facing APIs

Integration Examples
--------------------

With curve_fit
~~~~~~~~~~~~~~

.. code-block:: python

    from nlsq import curve_fit
    from nlsq.validators import InputValidator


    def safe_curve_fit(f, xdata, ydata, **kwargs):
        """curve_fit with validation."""
        validator = InputValidator(fast_mode=kwargs.pop("fast_validation", False))

        errors, warnings, x, y = validator.validate_curve_fit_inputs(
            f,
            xdata,
            ydata,
            p0=kwargs.get("p0"),
            bounds=kwargs.get("bounds"),
            sigma=kwargs.get("sigma"),
        )

        if errors:
            raise ValueError(f"Validation failed: {'; '.join(errors)}")

        # Show warnings
        for warning in warnings:
            import warnings

            warnings.warn(warning)

        return curve_fit(f, x, y, **kwargs)

With least_squares
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from nlsq import least_squares
    from nlsq.validators import InputValidator


    def safe_least_squares(fun, x0, **kwargs):
        """least_squares with validation."""
        validator = InputValidator()

        errors, warnings, x0 = validator.validate_least_squares_inputs(
            fun,
            x0,
            bounds=kwargs.get("bounds"),
            method=kwargs.get("method", "trf"),
            ftol=kwargs.get("ftol", 1e-8),
            xtol=kwargs.get("xtol", 1e-8),
            gtol=kwargs.get("gtol", 1e-8),
            max_nfev=kwargs.get("max_nfev"),
        )

        if errors:
            raise ValueError(f"Validation failed: {'; '.join(errors)}")

        for warning in warnings:
            import warnings

            warnings.warn(warning)

        return least_squares(fun, x0, **kwargs)

See Also
--------

- :doc:`nlsq.minpack` : Main curve fitting API
- :doc:`nlsq.least_squares` : Least squares solver
- :doc:`../howto/troubleshooting` : Troubleshooting guide
