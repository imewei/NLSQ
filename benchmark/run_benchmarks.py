#!/usr/bin/env python3
"""
Command-line interface for running NLSQ benchmarks.

Examples
--------
Run standard benchmarks:
    python run_benchmarks.py

Run with custom output directory:
    python run_benchmarks.py --output ./my_results

Run quick benchmarks (small sizes, fewer repeats):
    python run_benchmarks.py --quick

Run only specific problems:
    python run_benchmarks.py --problems exponential gaussian

Run without SciPy comparison:
    python run_benchmarks.py --no-scipy
"""

import argparse
import sys
from pathlib import Path

# Add benchmark directory to path for imports
benchmark_dir = Path(__file__).parent
sys.path.insert(0, str(benchmark_dir))

from benchmark_suite import (
    BenchmarkConfig,
    BenchmarkSuite,
    ExponentialDecayProblem,
    GaussianProblem,
    PolynomialProblem,
    SinusoidalProblem,
)


def main():
    """Main entry point for benchmark CLI."""
    parser = argparse.ArgumentParser(
        description="Run NLSQ performance benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              Run standard benchmarks
  %(prog)s --quick                      Run quick benchmarks
  %(prog)s --output ./results           Use custom output directory
  %(prog)s --problems exponential       Run only exponential problem
  %(prog)s --no-scipy                   Skip SciPy comparison
  %(prog)s --sizes 100 1000 10000       Use custom problem sizes
        """,
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./benchmark_results",
        help="Output directory for results (default: ./benchmark_results)",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmarks (small sizes, fewer repeats)",
    )

    parser.add_argument(
        "--problems",
        nargs="+",
        choices=["exponential", "gaussian", "polynomial", "sinusoidal", "all"],
        default=["all"],
        help="Problems to benchmark (default: all)",
    )

    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        help="Problem sizes to test (default: 100 1000 10000 100000)",
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["trf", "lm"],
        default=["trf", "lm"],
        help="Optimization methods to benchmark (default: trf lm)",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of repeats per benchmark (default: 5)",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup runs (default: 1)",
    )

    parser.add_argument(
        "--no-scipy",
        action="store_true",
        help="Skip SciPy comparison",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Configure benchmarks
    if args.quick:
        problem_sizes = [100, 1000, 10000]
        n_repeats = 3
        warmup_runs = 1
    else:
        problem_sizes = args.sizes or [100, 1000, 10000, 100000]
        n_repeats = args.repeats
        warmup_runs = args.warmup

    config = BenchmarkConfig(
        name="custom",
        problem_sizes=problem_sizes,
        n_repeats=n_repeats,
        warmup_runs=warmup_runs,
        methods=args.methods,
        backends=["cpu"],
        compare_scipy=not args.no_scipy,
    )

    # Create suite
    suite = BenchmarkSuite(config)

    # Add problems
    problem_map = {
        "exponential": ExponentialDecayProblem,
        "gaussian": GaussianProblem,
        "polynomial": PolynomialProblem,
        "sinusoidal": SinusoidalProblem,
    }

    if "all" in args.problems:
        problems_to_add = problem_map.keys()
    else:
        problems_to_add = args.problems

    for problem_name in problems_to_add:
        suite.add_problem(problem_map[problem_name]())
        if args.verbose:
            print(f"Added problem: {problem_name}")

    # Run benchmarks
    print("\nRunning NLSQ Benchmarks...")
    print("=" * 80)
    print("Configuration:")
    print(f"  Problem sizes: {problem_sizes}")
    print(f"  Methods: {args.methods}")
    print(f"  Repeats: {n_repeats}")
    print(f"  Warmup runs: {warmup_runs}")
    print(f"  Compare SciPy: {not args.no_scipy}")
    print(f"  Output: {args.output}")
    print("=" * 80)
    print()

    suite.run_benchmarks(verbose=True)

    # Generate and print report
    print("\n")
    report = suite.generate_report()
    print(report)

    # Save results
    print(f"\nSaving results to {args.output}...")
    suite.save_results(args.output)
    print("Done!")

    # Print quick access info
    output_path = Path(args.output)
    print("\nResults saved:")
    print(f"  Text report: {output_path / 'benchmark_report.txt'}")
    print(f"  CSV data: {output_path / 'benchmark_results.csv'}")
    print(f"  Dashboard: {output_path / 'dashboard/dashboard.html'}")


if __name__ == "__main__":
    main()
