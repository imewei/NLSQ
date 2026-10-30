"""cProfile of one warm NLSQ fit: exp_decay, n=1000.

Prints the top 15 functions by cumulative time. Use it to find where small-fit
overhead goes; for timing comparisons use benchmarks.compare.run_library_comparison.

Run: uv run python benchmarks/small_fit_overhead.py
"""

import cProfile
import pstats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.compare.adapters import ADAPTERS
from benchmarks.compare.problems import PROBLEMS

N = 1000
WARMUP = 3


def main() -> None:
    problem = PROBLEMS["exp_decay"]
    x, y = problem.data(N)
    run = ADAPTERS["nlsq"].prepare(problem, x, y, "end_to_end")
    for _ in range(WARMUP):  # compile and fill caches so the profile is warm
        run()
    prof = cProfile.Profile()
    prof.enable()
    out = run()
    prof.disable()
    assert out.converged, out
    pstats.Stats(prof).sort_stats("cumulative").print_stats(15)


if __name__ == "__main__":
    main()
