# SciPy default method (`lm`) spot check, CPU, `exp_decay` (2026-11-14)

Why: the 2026-11-13 library comparison (`2026-11-13-library-comparison-1k-100M.md`) runs SciPy with
`curve_fit(method="trf")` to match NLSQ's algorithm. It does not measure SciPy's default for unbounded problems,
`method="lm"` (MINPACK). This run does, for the `exp_decay` problem only.

Setup: same machine as the 2026-11-13 run (i9-13900H, CPU only, `JAX_PLATFORMS=cpu`), project `.venv`, one
process, all four variants measured back to back, warm median (7 repeats up to 100K points, 5 at 1M, 3 at 10M),
tolerance 1e-8 for `ftol`, `xtol` and `gtol`, at most 1000 evaluations (`maxfev=4000` for `lm`, which counts
function evaluations). Data from `benchmarks.compare.problems`, seed 0. Load average at the end of the run: 4.6
(1 minute). One run, no confidence intervals. This is not the full harness (no fresh process per case, no
accuracy gate), so it is a spot check, not a replacement baseline.

Run from the repository root with `PYTHONPATH=. .venv/bin/python <script>`:

```python
import os, time, statistics as st

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, nlsq
from scipy.optimize import curve_fit as sp
from benchmarks.compare.problems import PROBLEMS
import jax.numpy as jnp

P = PROBLEMS["exp_decay"]
TOL = 1e-8
f_np = lambda x, a, b, c: P.model(np, x, a, b, c)
f_jx = lambda x, a, b, c: P.model(jnp, x, a, b, c)


def med(fn, rep):
    fn()
    t = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return st.median(t)


for n, rep in ((1000, 7), (10000, 7), (100000, 7), (1000000, 5), (10000000, 3)):
    x, y = P.data(n)
    r = {}
    r["scipy lm (default)"] = med(
        lambda: sp(
            f_np,
            x,
            y,
            p0=P.p0,
            method="lm",
            ftol=TOL,
            xtol=TOL,
            gtol=TOL,
            maxfev=1000 * 4,
        ),
        rep,
    )
    r["scipy trf"] = med(
        lambda: sp(
            f_np,
            x,
            y,
            p0=P.p0,
            method="trf",
            ftol=TOL,
            xtol=TOL,
            gtol=TOL,
            max_nfev=1000,
        ),
        rep,
    )
    if n <= 1000000:
        r["nlsq curve_fit"] = med(
            lambda: nlsq.curve_fit(
                f_jx, x, y, p0=P.p0, ftol=TOL, xtol=TOL, gtol=TOL, max_nfev=1000
            ),
            rep,
        )
    if n >= 1000000:
        r["nlsq curve_fit_large"] = med(
            lambda: nlsq.curve_fit_large(
                f_jx,
                x,
                y,
                p0=P.p0,
                method="exact_streaming",
                ftol=TOL,
                xtol=TOL,
                gtol=TOL,
                max_nfev=1000,
            ),
            rep,
        )
    print(
        f"{n:>10,}", "  ".join(f"{k}: {v*1e3:.1f} ms" for k, v in r.items()), flush=True
    )
```

Output (milliseconds, warm median):

| Points | SciPy `lm` (default) | SciPy `trf` | NLSQ `curve_fit` | NLSQ `curve_fit_large` |
|---|---|---|---|---|
| 1,000 | 0.3 | 1.0 | 3.6 | not run |
| 10,000 | 1.7 | 2.8 | 5.2 | not run |
| 100,000 | 18.3 | 67.5 | 55.1 | not run |
| 1,000,000 | 212.7 | 651.7 | 373.9 | 93.8 |
| 10,000,000 | 2823.4 | 5321.4 | not run | 670.7 |

Reading: against SciPy's default `lm`, SciPy was faster than NLSQ `curve_fit` at every size from 1K to 1M points
(by 12x at 1K, 3.1x at 10K, 3.0x at 100K and 1.8x at 1M). NLSQ `curve_fit_large` was faster than SciPy `lm` at
1M points (2.3x) and 10M points (4.2x). SciPy `trf` timings differ from the 2026-11-13 run (for example 651.7 ms
against 1.14 s at 1M points): absolute times drift between sessions on this machine, so compare within a row.
