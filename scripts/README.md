# NLSQ Scripts

Utility scripts for maintaining the NLSQ project.

## Available Scripts

### `convert_examples.py`
Bidirectional conversion utility for examples:
- Convert Python scripts to Jupyter notebooks
- Convert Jupyter notebooks to Python scripts
- Supports single files or entire directories

**Usage:**
```bash
# Convert notebook to Python script
python scripts/convert_examples.py notebook-to-script examples/notebooks/01_getting_started/nlsq_quickstart.ipynb

# Convert script to notebook
python scripts/convert_examples.py script-to-notebook examples/scripts/01_getting_started/nlsq_quickstart.py

# Convert all notebooks in a directory
python scripts/convert_examples.py notebook-to-script examples/notebooks/01_getting_started/

# Convert all scripts in a directory
python scripts/convert_examples.py script-to-notebook examples/scripts/01_getting_started/
```

### `dismiss_codeql_false_positives.sh`
Dismiss false positive security alerts from CodeQL analysis.

**Usage:**
```bash
./scripts/dismiss_codeql_false_positives.sh
```

## Maintenance

These scripts help maintain consistency between notebook and script versions of examples.
Always ensure both formats stay synchronized when updating examples.
