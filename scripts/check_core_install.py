"""Smoke check for a core-only (no extras) NLSQ install.

Run with the interpreter of a venv that has only the built wheel installed:
    MPLBACKEND=Agg .venv-test/bin/python scripts/check_core_install.py
Exit code 0 means every check passed; 1 means at least one failed.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys

# Must be set before matplotlib is imported anywhere; do not rely on the caller.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

# Packages that live in extras / dev groups and must not leak into core deps.
NON_CORE = ("PySide6", "pyqtgraph", "pytest", "ruff", "sphinx", "lmfit", "jaxopt")
TRUE_A, TRUE_B = 2.5, 1.3


def _model(x, a, b):
    import jax.numpy as jnp

    return a * jnp.exp(-b * x)


def _data(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 4.0, n)
    y = TRUE_A * np.exp(-TRUE_B * x) + 0.01 * rng.standard_normal(n)
    return x, y


def _close(popt) -> bool:
    popt = np.asarray(popt, dtype=float)
    return bool(
        np.all(np.isfinite(popt))
        and math.isclose(popt[0], TRUE_A, rel_tol=1e-1)
        and math.isclose(popt[1], TRUE_B, rel_tol=1e-1)
    )


def check_no_extras() -> str | None:
    found = [p for p in NON_CORE if importlib.util.find_spec(p) is not None]
    return f"non-core packages importable: {', '.join(found)}" if found else None


def check_curve_fit() -> str | None:
    import nlsq

    x, y = _data(200)
    popt, _ = nlsq.curve_fit(_model, x, y, p0=[1.0, 1.0])
    return None if _close(popt) else f"curve_fit popt off: {popt}"


def check_chunked() -> str | None:
    import nlsq
    from nlsq.streaming import LargeDatasetFitter, LDMemoryConfig

    x, y = _data(20_000)
    # min_chunk_size floors the chunk size, so these bounds force >= 10 chunks.
    config = LDMemoryConfig(min_chunk_size=1000, max_chunk_size=2000)
    result = LargeDatasetFitter(config=config).fit(_model, x, y, p0=[1.0, 1.0])
    if result["n_chunks"] < 2:
        return f"LargeDatasetFitter used {result['n_chunks']} chunk(s), expected >= 2"
    if not _close(result["popt"]):
        return f"LargeDatasetFitter popt off: {result['popt']}"

    # auto_size_detection=False, otherwise < 1M points go to plain curve_fit.
    popt, _ = nlsq.curve_fit_large(
        _model, x, y, p0=[1.0, 1.0], auto_size_detection=False, chunk_size=2000
    )
    return None if _close(popt) else f"curve_fit_large popt off: {popt}"


def check_plot() -> str | None:
    import matplotlib.pyplot as plt

    import nlsq

    x, y = _data(200)
    result = nlsq.curve_fit(_model, x, y, p0=[1.0, 1.0])
    if not callable(getattr(result, "plot", None)):
        return "result.plot is not callable"
    result.plot()
    plt.close("all")
    return None


def main() -> int:
    failures = []
    for check in (check_no_extras, check_curve_fit, check_chunked, check_plot):
        try:
            msg = check()
        except Exception as exc:  # report every failing check, not just the first
            msg = f"{type(exc).__name__}: {exc}"
        status = "FAIL" if msg else "ok"
        print(f"[{status}] {check.__name__}" + (f": {msg}" if msg else ""))
        if msg:
            failures.append(check.__name__)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
