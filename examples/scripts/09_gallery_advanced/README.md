# 09 - Advanced Gallery (fit() API + workflow System)

This gallery contains the same examples as `04_gallery/` but updated to use NLSQ's
v0.6.3 3-workflow system:

- **`fit()` API**: Unified entry point with automatic memory-based strategy selection
- **Three Workflows**: `'auto'`, `'auto_global'`, `'hpc'` for all use cases
- **Global Optimization**: Multi-start and CMA-ES for robust parameter estimation

## Relationship to 04_gallery/

| Original (04_gallery/)           | Advanced (09_gallery_advanced/)           |
|----------------------------------|-------------------------------------------|
| Uses `curve_fit()` directly      | Uses `fit()` with workflows               |
| Single local optimization        | Multi-start global optimization           |
| Manual parameter tuning          | Automatic workflow and memory management  |
| Works well with good guesses     | Robust even with poor initial guesses     |

## Examples by Domain

### Biology (3 examples)

| File | Description | Key Features |
|------|-------------|--------------|
| `biology/dose_response.py` | EC50/IC50 determination | Hill equation, 4PL model, multi-drug comparison |
| `biology/enzyme_kinetics.py` | Michaelis-Menten kinetics | K_M and V_max, competitive inhibition |
| `biology/growth_curves.py` | Bacterial growth | Logistic growth, doubling time, growth phases |

### Chemistry (2 examples)

| File | Description | Key Features |
|------|-------------|--------------|
| `chemistry/reaction_kinetics.py` | Rate constant determination | 1st/2nd order kinetics, half-life, model selection |
| `chemistry/titration_curves.py` | pKa determination | Henderson-Hasselbalch, buffer capacity, diprotic acids |

### Engineering (3 examples)

| File | Description | Key Features |
|------|-------------|--------------|
| `engineering/materials_characterization.py` | Stress-strain analysis | Young's modulus, yield strength, strain hardening |
| `engineering/sensor_calibration.py` | Nonlinear sensor calibration | Polynomial models, model comparison, uncertainty |
| `engineering/system_identification.py` | Transfer function fitting | Time constant, step response, rise/settling time |

### Physics (3 examples)

| File | Description | Key Features |
|------|-------------|--------------|
| `physics/damped_oscillation.py` | Damping coefficient extraction | Quality factor, phase space, FFT analysis |
| `physics/radioactive_decay.py` | Half-life determination | Exponential decay, uncertainty propagation |
| `physics/spectroscopy_peaks.py` | Multi-peak deconvolution | Gaussian/Lorentzian, background subtraction |

## API Patterns Demonstrated

### Basic fit() with workflow='auto' (default)

```python
from nlsq import fit

# Local optimization with automatic memory management
popt, pcov = fit(
    model_function,
    xdata,
    ydata,
    p0=initial_guess,
    bounds=(lower, upper),
    workflow="auto",  # Default - local optimization
)
```

### Global Optimization with workflow='auto_global'

```python
from nlsq import fit

# Global optimization with 20 multi-starts
popt, pcov = fit(
    model_function,
    xdata,
    ydata,
    p0=initial_guess,
    bounds=(lower, upper),
    workflow="auto_global",  # Multi-start or CMA-ES
    n_starts=20,
)
```

### Custom Multi-Start Configuration

```python
from nlsq import fit

# Full control over multi-start optimization
popt, pcov = fit(
    model_function,
    xdata,
    ydata,
    p0=initial_guess,
    bounds=(lower, upper),
    workflow="auto_global",
    n_starts=15,
    sampler="lhs",  # Latin Hypercube Sampling
)
```

## Comparison: curve_fit() vs fit()

### Using curve_fit() (Section 04)

```python
from nlsq import curve_fit

# Single-start local optimization
popt, pcov = curve_fit(
    model_function,
    xdata,
    ydata,
    p0=initial_guess,
    bounds=(lower, upper),
)
```

### Using fit() (Section 09)

```python
from nlsq import fit

# Multi-start with automatic workflow selection
popt, pcov = fit(
    model_function,
    xdata,
    ydata,
    p0=initial_guess,
    bounds=(lower, upper),
    workflow="auto_global",  # Automatic multi-start
    n_starts=5,
)
```

## When to Use Each Workflow

| Workflow | Description | Best For |
|----------|-------------|----------|
| `'auto'` | Local optimization | Well-conditioned problems, known good initial guess |
| `'auto_global'` | Multi-start or CMA-ES | Most applications, multimodal problems, uncertain initial guess |
| `'hpc'` | `auto_global` + checkpointing | Long-running HPC jobs, fault tolerance |

## Global Optimization is Critical For

- **Spectroscopy peak fitting**: Multi-peak models have many local minima
- **Dose-response curves**: EC50/IC50 estimation with uncertain range
- **Multi-parameter models**: >5 parameters with correlations
- **Noisy data**: When initial guess quality is uncertain

## Related Sections

- **Section 07**: `07_global_optimization/` - Detailed global optimization tutorials
- **Section 08**: `08_workflow_system/` - Full workflow system documentation
- **Section 04**: `04_gallery/` - Original examples using curve_fit()

## Running Examples

```bash
# Run a single example
python examples/scripts/09_gallery_advanced/biology/dose_response.py

# Run all biology examples
for f in examples/scripts/09_gallery_advanced/biology/*.py; do python "$f"; done

# Run all examples in the gallery
for d in biology chemistry engineering physics; do
    for f in examples/scripts/09_gallery_advanced/$d/*.py; do
        python "$f"
    done
done
```

## Output

Each example:
1. Prints detailed analysis to stdout
2. Saves a multi-panel figure to `figures/<example_name>/fig_01.png`
3. Demonstrates multiple fit() API usage patterns
