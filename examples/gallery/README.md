# NLSQ Example Gallery

Comprehensive collection of real-world curve fitting examples across scientific domains.

## Overview

This gallery demonstrates NLSQ's capabilities through practical, domain-specific examples. Each example includes:
- **Background**: Scientific context and problem description
- **Data**: Realistic data generation or loading
- **Fitting**: NLSQ curve fitting with appropriate parameters
- **Visualization**: Plots showing data, fits, and residuals
- **Analysis**: Statistical analysis and interpretation

## Available Examples

### 🔬 Physics (3 examples)

#### 1. Radioactive Decay (`physics/radioactive_decay.py`)
- **Application**: Nuclear physics, carbon dating
- **Model**: Exponential decay N(t) = N₀ exp(-λt)
- **Features**:
  - Half-life calculation from decay constant
  - Uncertainty propagation
  - Weighted least squares fitting
  - χ² goodness-of-fit analysis
- **Dataset**: Carbon-14 decay (half-life: 5,730 years)

#### 2. Damped Oscillation (`physics/damped_oscillation.py`)
- **Application**: Mechanical systems, resonance
- **Model**: Damped harmonic oscillator
- **Features**:
  - Quality factor determination
  - Decay envelope fitting
  - Frequency analysis
- **Dataset**: Spring-mass-damper system

#### 3. Spectroscopy Peaks (`physics/spectroscopy_peaks.py`)
- **Application**: Optical spectroscopy, atomic physics
- **Model**: Multi-peak Gaussian/Lorentzian
- **Features**:
  - Peak deconvolution
  - Line width analysis
  - Wavelength calibration
- **Dataset**: Hydrogen emission spectrum

### ⚙️ Engineering (3 examples)

#### 4. Sensor Calibration (`engineering/sensor_calibration.py`)
- **Application**: Instrumentation, metrology
- **Model**: Polynomial or rational function
- **Features**:
  - Non-linear calibration curves
  - Inverse function calculation
  - Residual analysis
- **Dataset**: Temperature sensor calibration

#### 5. System Identification (`engineering/system_identification.py`)
- **Application**: Control systems, signal processing
- **Model**: Transfer function (rational polynomial)
- **Features**:
  - Frequency response fitting
  - Pole-zero analysis
  - Bode plot visualization
- **Dataset**: RLC circuit frequency response

#### 6. Materials Characterization (`engineering/materials_characterization.py`)
- **Application**: Materials science, mechanical testing
- **Model**: Stress-strain relationships
- **Features**:
  - Elastic modulus extraction
  - Yield point determination
  - Material property analysis
- **Dataset**: Stress-strain curves for metals

### 🧬 Biology (3 examples)

#### 7. Growth Curves (`biology/growth_curves.py`)
- **Application**: Microbiology, cell culture
- **Model**: Logistic growth (Gompertz, Richards)
- **Features**:
  - Lag time estimation
  - Maximum growth rate
  - Carrying capacity
  - Strain comparison
- **Dataset**: Bacterial growth (E. coli)

#### 8. Enzyme Kinetics (`biology/enzyme_kinetics.py`)
- **Application**: Biochemistry, drug discovery
- **Model**: Michaelis-Menten equation
- **Features**:
  - Km and Vmax determination
  - Competitive inhibition analysis
  - Lineweaver-Burk plots
- **Dataset**: Enzyme-substrate reaction rates

#### 9. Dose-Response Curves (`biology/dose_response.py`)
- **Application**: Pharmacology, toxicology
- **Model**: Hill equation (sigmoid)
- **Features**:
  - EC50 (half-maximal effective concentration)
  - Hill coefficient
  - Efficacy and potency analysis
- **Dataset**: Drug concentration vs. response

### 🧪 Chemistry (2 examples)

#### 10. Reaction Kinetics (`chemistry/reaction_kinetics.py`)
- **Application**: Chemical kinetics, catalysis
- **Model**: Integrated rate laws (0th, 1st, 2nd order)
- **Features**:
  - Rate constant determination
  - Reaction order identification
  - Arrhenius activation energy
- **Dataset**: Chemical reaction time courses

#### 11. Titration Curves (`chemistry/titration_curves.py`)
- **Application**: Analytical chemistry, pH analysis
- **Model**: Henderson-Hasselbalch equation
- **Features**:
  - pKa determination
  - Equivalence point identification
  - Buffer capacity analysis
- **Dataset**: Acid-base titration

## Usage

### Running Examples

Each example is a self-contained Python script:

```bash
# Run a specific example
python examples/gallery/physics/radioactive_decay.py

# Output includes:
# - Statistical results (fitted parameters, uncertainties)
# - Goodness-of-fit metrics
# - Visualization (saved as PNG + displayed)
```

### Using in Your Code

Examples demonstrate best practices that you can adapt:

```python
from nlsq import curve_fit
import jax.numpy as jnp


# 1. Define your model (use jax.numpy)
def model(x, a, b, c):
    return a * jnp.exp(-b * x) + c


# 2. Fit with uncertainty quantification
result = curve_fit(model, xdata, ydata, sigma=uncertainties, absolute_sigma=True)

# 3. Analyze with new result features
print(f"R² = {result.r_squared:.4f}")
result.plot(show_residuals=True)
result.summary()

# 4. Propagate uncertainties to derived quantities
popt, pcov = result.popt, result.pcov
perr = np.sqrt(np.diag(pcov))
```

## Key Features Demonstrated

### 🎯 Fitting Techniques

- **Weighted Least Squares**: Using measurement uncertainties
- **Bounded Optimization**: Physical constraints on parameters
- **Multi-parameter Fits**: Complex models (4-10 parameters)
- **Initial Guess Strategies**: Domain-specific p0 estimation

### 📊 Statistical Analysis

- **Uncertainty Quantification**: Parameter errors from covariance
- **Error Propagation**: Derived quantity uncertainties
- **Goodness-of-Fit**: χ², R², residual analysis
- **Model Comparison**: AIC, BIC for model selection

### 📈 Visualization

- **Data + Fit**: Scatter plots with fitted curves
- **Residual Plots**: Normalized residuals with bounds
- **Log-scale Plots**: Semi-log for exponentials
- **Multi-panel Figures**: Comprehensive result visualization

### 🔧 NLSQ-Specific Features

- **Automatic p0**: `p0='auto'` for common models
- **GPU Acceleration**: JAX backend for large datasets
- **Enhanced Results**: `.plot()`, `.summary()`, `.confidence_intervals()`
- **Callbacks**: Progress monitoring, early stopping

## Example Output

Typical output from running an example:

```
======================================================================
RADIOACTIVE DECAY: CARBON-14 HALF-LIFE DETERMINATION
======================================================================

Fitted Parameters:
  N0 = 1002.34 ± 12.45 counts/min
  λ  = 1.209e-04 ± 3.456e-06 yr⁻¹

Derived Half-Life:
  t₁/₂ = 5732 ± 82 years
  True value: 5730 years
  Error: 2 years (0.0%)

  ✅ Within 1σ uncertainty: True

Parameter Correlation:
  ρ(N0, λ) = -0.8234

Goodness of Fit:
  χ² = 28.45
  χ²/dof = 1.02 (expect ≈ 1.0 for good fit)

✅ Plot saved as 'radioactive_decay.png'
======================================================================
```

## File Format

Examples are currently provided as Python scripts (`.py`). To use interactively:

**Option 1: Run as script**
```bash
python examples/gallery/physics/radioactive_decay.py
```

**Option 2: Convert to notebook**
```bash
# Using jupytext (install: pip install jupytext)
jupytext --to notebook examples/gallery/physics/radioactive_decay.py

# Or use nbconvert
jupyter nbconvert --to notebook --execute radioactive_decay.py
```

**Option 3: Copy code to notebook**
- Open the `.py` file
- Copy sections to Jupyter notebook cells
- Run interactively

## Requirements

All examples require:
- `nlsq` (this package)
- `numpy` (data handling)
- `jax` (model functions)
- `matplotlib` (visualization)
- `scipy` (for statistical functions in some examples)

Install with:
```bash
pip install nlsq numpy matplotlib scipy
```

## Gallery Statistics

| Domain | Examples | Total Lines | Avg. Complexity |
|--------|----------|-------------|-----------------|
| Physics | 3 | ~1,200 | High (multi-peak fitting) |
| Engineering | 3 | ~1,500 | Medium (calibration) |
| Biology | 3 | ~1,600 | Medium (kinetics) |
| Chemistry | 2 | ~1,000 | Medium (kinetics) |
| **Total** | **11** | **~5,300** | **Varies** |

## Contributing

To add a new example:

1. Choose appropriate domain subdirectory
2. Follow naming convention: `descriptive_name.py`
3. Include sections:
   - Docstring with description and key concepts
   - Model definition with JAX
   - Data generation/loading
   - NLSQ fitting with error handling
   - Statistical analysis
   - Visualization (4-panel plots recommended)
   - Summary output
4. Test thoroughly
5. Update this README

## References

Each example includes scientific references and typical applications. See individual files for details.

## License

Same as NLSQ package (MIT License).

---

**Last Updated**: 2025-10-08
**NLSQ Version**: v0.1.1
**Total Examples**: 11 across 4 domains
**Status**: Complete Python scripts (conversion to .ipynb optional)
