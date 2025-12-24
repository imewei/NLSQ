# NLSQ CLI Command Examples

This directory demonstrates the NLSQ command-line interface (CLI) for running
curve fitting workflows from YAML configuration files.

## Prerequisites

```bash
# Install NLSQ (includes CLI)
pip install nlsq

# Verify installation
nlsq --version
nlsq info
```

## Directory Structure

```
10_cli-commands/
├── README.md                     # This file
├── data/                         # Sample data files (generated)
│   ├── radioactive_decay.csv
│   ├── enzyme_kinetics.csv
│   └── reaction_kinetics.csv
├── workflows/                    # YAML workflow configurations
│   ├── 01_radioactive_decay.yaml
│   ├── 02_enzyme_kinetics.yaml
│   ├── 03_reaction_kinetics.yaml
│   └── 04_custom_model.yaml
├── models/                       # Custom model definitions
│   └── custom_models.py
├── output/                       # Results directory
├── generate_data.py              # Generate sample data files
├── run_cli_examples.sh           # Shell script demonstrations
└── run_cli_examples.py           # Python demonstrations
```

## Quick Start

```bash
# 1. Generate sample data files
python generate_data.py

# 2. Display system info
nlsq info

# 3. Run a single fit
nlsq fit workflows/01_radioactive_decay.yaml

# 4. Run batch fitting
nlsq batch workflows/*.yaml --summary output/batch_summary.json

# 5. Output to stdout (for piping)
nlsq fit workflows/01_radioactive_decay.yaml --stdout | jq '.popt'
```

## CLI Commands

### `nlsq info`

Display system information, JAX backend status, and available builtin models.

```bash
nlsq info           # Basic info
nlsq info -v        # Verbose with model docstrings
```

### `nlsq fit`

Execute a single fitting workflow from a YAML configuration file.

```bash
# Basic usage
nlsq fit workflow.yaml

# Override output file
nlsq fit workflow.yaml --output results/custom_output.json

# Output JSON to stdout (for piping)
nlsq fit workflow.yaml --stdout

# Verbose mode
nlsq fit workflow.yaml -v
```

### `nlsq batch`

Run multiple workflows in parallel with aggregate summary.

```bash
# Multiple explicit files
nlsq batch w1.yaml w2.yaml w3.yaml

# Shell glob expansion
nlsq batch workflows/*.yaml

# With summary file
nlsq batch workflows/*.yaml --summary batch_results.json

# Limit parallel workers
nlsq batch workflows/*.yaml --workers 2

# Verbose mode
nlsq batch workflows/*.yaml -v
```

## Workflow YAML Format

### Minimal Example

```yaml
data:
  input_file: "data/experiment.csv"
  columns:
    x: 0
    y: 1

model:
  type: builtin
  name: exponential_decay
  auto_p0: true

export:
  results_file: "output/results.json"
```

### Complete Example

See `workflows/01_radioactive_decay.yaml` for a fully documented example.

## Builtin Models

| Name               | Function                        | Parameters       |
|--------------------|---------------------------------|------------------|
| `linear`           | `a*x + b`                       | a, b             |
| `exponential_decay`| `a*exp(-b*x) + c`              | a, b, c          |
| `exponential_growth`| `a*exp(b*x) + c`               | a, b, c          |
| `gaussian`         | `amp*exp(-(x-mu)²/(2σ²))`      | amp, mu, sigma   |
| `sigmoid`          | `L/(1+exp(-k*(x-x0))) + b`     | L, x0, k, b      |
| `power_law`        | `a*x^b`                         | a, b             |

## Custom Models

Create a Python file with your model function:

```python
# models/my_model.py
import jax.numpy as jnp

def my_model(x, a, b, c):
    return a * jnp.exp(-b * x) + c

def estimate_p0(xdata, ydata):
    """Optional: auto-estimate initial parameters."""
    return [ydata.max(), 0.1, ydata.min()]

def bounds():
    """Optional: default parameter bounds."""
    return ([0, 0, -1], [10, 1, 1])
```

Reference in YAML:

```yaml
model:
  type: custom
  path: models/my_model.py
  function: my_model
```

## Data Formats

The CLI supports multiple data formats:

- **ASCII** (`.txt`, `.dat`): Whitespace-delimited
- **CSV** (`.csv`): Comma-separated values
- **NPZ** (`.npz`): NumPy compressed archives
- **HDF5** (`.h5`, `.hdf5`): Hierarchical Data Format

## Output Format

Results are exported as JSON:

```json
{
  "popt": [1000.5, 0.000121],
  "pcov": [[12.3, -0.001], [-0.001, 1.5e-10]],
  "uncertainties": [3.51, 1.22e-05],
  "success": true,
  "nfev": 42,
  "r_squared": 0.9987,
  "rmse": 15.3
}
```

## Batch Summary

When using `nlsq batch --summary`, a JSON summary is generated:

```json
{
  "total": 4,
  "succeeded": 4,
  "failed": 0,
  "duration_seconds": 2.5,
  "successes": [...],
  "failures": []
}
```

## Integration with Shell Scripts

```bash
#!/bin/bash
# Example: Batch processing with error handling

nlsq batch workflows/*.yaml --summary summary.json
if [ $? -eq 0 ]; then
    echo "All fits succeeded"
else
    echo "Some fits failed - check summary.json"
fi
```

## Integration with Python

```python
import subprocess
import json

# Run CLI and capture output
result = subprocess.run(
    ["nlsq", "fit", "workflow.yaml", "--stdout"],
    capture_output=True,
    text=True
)

if result.returncode == 0:
    fit_result = json.loads(result.stdout)
    print(f"Fitted parameters: {fit_result['popt']}")
```

## See Also

- [CLI Reference Documentation](https://nlsq.readthedocs.io/en/latest/user_guide/cli_reference.html)
- [YAML Configuration Guide](https://nlsq.readthedocs.io/en/latest/user_guide/yaml_configuration.html)
- [Workflow Template](../../../workflow_config_template.yaml)
