#!/bin/bash
# post-link.sh — pip-install runtime deps that are not on conda-forge
# Runs automatically after `conda install nlsq` in the target environment.
set -euo pipefail

# $PREFIX is the conda environment prefix (set by conda at link time)
PYTHON="${PREFIX}/bin/python"

echo "[nlsq post-link] Installing pip-only runtime dependencies..."
"${PYTHON}" -m pip install --no-input --quiet \
    "evosax>=0.2.0,<0.3.0" \
    "PySide6>=6.10.0" \
    "pyqtgraph>=0.14.0"
echo "[nlsq post-link] Done."
