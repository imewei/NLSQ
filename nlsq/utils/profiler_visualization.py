"""
Performance Profiling Visualization
====================================

Visualization and dashboard tools for NLSQ performance profiling data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend for CI/headless environments
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from nlsq.utils.profiler import PerformanceProfiler


class ProfilerVisualization:
    """
    Visualization tools for performance profiling data.

    Examples
    --------
    >>> from nlsq.utils.profiler import get_global_profiler
    >>> from nlsq.utils.profiler_visualization import ProfilerVisualization
    >>> profiler = get_global_profiler()
    >>> viz = ProfilerVisualization(profiler)
    >>> fig = viz.plot_timing_comparison(["test1", "test2"])
    >>> viz.save_html_report("report.html")
    """

    def __init__(self, profiler: PerformanceProfiler):
        """
        Initialize visualization tools.

        Parameters
        ----------
        profiler : PerformanceProfiler
            Profiler instance to visualize
        """
        self.profiler = profiler

    def plot_timing_series(
        self, name: str = "default", figsize: tuple[float, float] = (10, 6)
    ) -> Figure | None:
        """
        Plot timing metrics across runs.

        Parameters
        ----------
        name : str
            Profile name to visualize
        figsize : tuple
            Figure size (width, height)

        Returns
        -------
        fig : Figure or None
            Matplotlib figure or None if matplotlib not available
        """
        if not HAS_MATPLOTLIB:
            return None

        metrics_list = self.profiler.get_metrics(name)
        if not metrics_list:
            return None

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f"Performance Metrics: {name}", fontsize=14, fontweight="bold")

        runs = list(range(1, len(metrics_list) + 1))

        # Total time
        total_times = [m.total_time for m in metrics_list]
        axes[0, 0].plot(runs, total_times, "o-", linewidth=2, markersize=6)
        axes[0, 0].set_xlabel("Run Number")
        axes[0, 0].set_ylabel("Total Time (s)")
        axes[0, 0].set_title("Total Time per Run")
        axes[0, 0].grid(True, alpha=0.3)

        # Optimization time
        opt_times = [m.optimization_time for m in metrics_list]
        axes[0, 1].plot(
            runs, opt_times, "o-", color="orange", linewidth=2, markersize=6
        )
        axes[0, 1].set_xlabel("Run Number")
        axes[0, 1].set_ylabel("Optimization Time (s)")
        axes[0, 1].set_title("Optimization Time per Run")
        axes[0, 1].grid(True, alpha=0.3)

        # Iterations
        iterations = [m.n_iterations for m in metrics_list]
        axes[1, 0].plot(
            runs, iterations, "o-", color="green", linewidth=2, markersize=6
        )
        axes[1, 0].set_xlabel("Run Number")
        axes[1, 0].set_ylabel("Iterations")
        axes[1, 0].set_title("Iterations per Run")
        axes[1, 0].grid(True, alpha=0.3)

        # Success indicators
        successes = [1 if m.success else 0 for m in metrics_list]
        axes[1, 1].bar(
            runs, successes, color=["green" if s else "red" for s in successes]
        )
        axes[1, 1].set_xlabel("Run Number")
        axes[1, 1].set_ylabel("Success (1) / Failure (0)")
        axes[1, 1].set_title("Success Rate")
        axes[1, 1].set_ylim(-0.1, 1.1)
        axes[1, 1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        return fig

    def plot_timing_comparison(
        self, names: list[str], figsize: tuple[float, float] = (10, 6)
    ) -> Figure | None:
        """
        Compare timing metrics across multiple profiles.

        Parameters
        ----------
        names : list of str
            Profile names to compare
        figsize : tuple
            Figure size (width, height)

        Returns
        -------
        fig : Figure or None
            Matplotlib figure or None if matplotlib not available
        """
        if not HAS_MATPLOTLIB:
            return None

        summaries = {}
        for name in names:
            summary = self.profiler.get_summary(name)
            if summary:
                summaries[name] = summary

        if not summaries:
            return None

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle("Profile Comparison", fontsize=14, fontweight="bold")

        x_pos = np.arange(len(summaries))
        labels = list(summaries.keys())

        # Total time comparison
        total_times = [s["total_time"]["mean"] for s in summaries.values()]
        total_stds = [s["total_time"]["std"] for s in summaries.values()]
        axes[0].bar(x_pos, total_times, yerr=total_stds, capsize=5, alpha=0.7)
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(labels, rotation=45, ha="right")
        axes[0].set_ylabel("Total Time (s)")
        axes[0].set_title("Mean Total Time")
        axes[0].grid(True, alpha=0.3, axis="y")

        # Iterations comparison
        iterations = [s["iterations"]["mean"] for s in summaries.values()]
        axes[1].bar(x_pos, iterations, alpha=0.7, color="green")
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(labels, rotation=45, ha="right")
        axes[1].set_ylabel("Iterations")
        axes[1].set_title("Mean Iterations")
        axes[1].grid(True, alpha=0.3, axis="y")

        # Success rate comparison
        success_rates = [s["success_rate"] * 100 for s in summaries.values()]
        bars = axes[2].bar(x_pos, success_rates, alpha=0.7, color="orange")
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(labels, rotation=45, ha="right")
        axes[2].set_ylabel("Success Rate (%)")
        axes[2].set_title("Success Rate")
        axes[2].set_ylim(0, 105)
        axes[2].grid(True, alpha=0.3, axis="y")

        # Color bars based on success rate
        for bar, rate in zip(bars, success_rates, strict=False):
            if rate < 50:
                bar.set_color("red")
            elif rate < 90:
                bar.set_color("orange")
            else:
                bar.set_color("green")

        plt.tight_layout()
        return fig

    def plot_timing_distribution(
        self, name: str = "default", figsize: tuple[float, float] = (10, 6)
    ) -> Figure | None:
        """
        Plot distribution of timing metrics.

        Parameters
        ----------
        name : str
            Profile name to visualize
        figsize : tuple
            Figure size (width, height)

        Returns
        -------
        fig : Figure or None
            Matplotlib figure or None if matplotlib not available
        """
        if not HAS_MATPLOTLIB:
            return None

        metrics_list = self.profiler.get_metrics(name)
        if not metrics_list or len(metrics_list) < 2:
            return None

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f"Timing Distribution: {name}", fontsize=14, fontweight="bold")

        # Total time distribution
        total_times = [m.total_time for m in metrics_list]
        axes[0].hist(
            total_times,
            bins=min(20, len(metrics_list) // 2 + 1),
            alpha=0.7,
            edgecolor="black",
        )
        axes[0].axvline(
            np.mean(total_times), color="red", linestyle="--", linewidth=2, label="Mean"
        )
        axes[0].axvline(
            np.median(total_times),
            color="green",
            linestyle="--",
            linewidth=2,
            label="Median",
        )
        axes[0].set_xlabel("Total Time (s)")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Total Time Distribution")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis="y")

        # Iterations distribution
        iterations = [m.n_iterations for m in metrics_list]
        axes[1].hist(
            iterations,
            bins=min(20, len(metrics_list) // 2 + 1),
            alpha=0.7,
            color="green",
            edgecolor="black",
        )
        axes[1].axvline(
            np.mean(iterations), color="red", linestyle="--", linewidth=2, label="Mean"
        )
        axes[1].axvline(
            np.median(iterations),
            color="darkgreen",
            linestyle="--",
            linewidth=2,
            label="Median",
        )
        axes[1].set_xlabel("Iterations")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Iterations Distribution")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        return fig

    def generate_html_report(self, output_path: str | Path | None = None) -> str:
        """
        Generate HTML report with all profiling data.

        Parameters
        ----------
        output_path : str or Path, optional
            Path to save HTML file. If None, returns HTML string.

        Returns
        -------
        html : str
            HTML report content
        """
        profiles = self.profiler.export_to_dict()

        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<meta charset='utf-8'>",
            "<title>NLSQ Performance Profile Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }",
            "h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }",
            "h2 { color: #34495e; margin-top: 30px; }",
            ".profile-section { background: white; padding: 20px; margin: 20px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
            ".summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }",
            ".metric { background: #ecf0f1; padding: 15px; border-radius: 5px; border-left: 4px solid #3498db; }",
            ".metric-label { font-weight: bold; color: #7f8c8d; font-size: 0.9em; }",
            ".metric-value { font-size: 1.5em; color: #2c3e50; margin-top: 5px; }",
            "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
            "th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }",
            "th { background-color: #3498db; color: white; }",
            "tr:nth-child(even) { background-color: #f9f9f9; }",
            ".success { color: green; font-weight: bold; }",
            ".failure { color: red; font-weight: bold; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>NLSQ Performance Profile Report</h1>",
        ]

        for name, metrics_list in profiles.items():
            summary = self.profiler.get_summary(name)

            html_parts.extend(
                [
                    "<div class='profile-section'>",
                    f"<h2>Profile: {name}</h2>",
                    "<div class='summary'>",
                    f"<div class='metric'><div class='metric-label'>Total Runs</div><div class='metric-value'>{summary['n_runs']}</div></div>",
                    f"<div class='metric'><div class='metric-label'>Success Rate</div><div class='metric-value'>{summary['success_rate']:.1%}</div></div>",
                    f"<div class='metric'><div class='metric-label'>Mean Time</div><div class='metric-value'>{summary['total_time']['mean']:.3f}s</div></div>",
                    f"<div class='metric'><div class='metric-label'>Mean Iterations</div><div class='metric-value'>{summary['iterations']['mean']:.1f}</div></div>",
                    "</div>",
                ]
            )

            # Detailed table
            html_parts.append("<table>")
            html_parts.append(
                "<tr><th>Run</th><th>Total Time (s)</th><th>Optimization (s)</th><th>Iterations</th><th>Function Evals</th><th>Success</th></tr>"
            )

            for i, metrics in enumerate(metrics_list, 1):
                success_class = "success" if metrics["success"] else "failure"
                success_text = "✓" if metrics["success"] else "✗"
                html_parts.append(
                    f"<tr>"
                    f"<td>{i}</td>"
                    f"<td>{metrics['total_time']:.3f}</td>"
                    f"<td>{metrics['optimization_time']:.3f}</td>"
                    f"<td>{metrics['n_iterations']}</td>"
                    f"<td>{metrics['n_function_evals']}</td>"
                    f"<td class='{success_class}'>{success_text}</td>"
                    f"</tr>"
                )

            html_parts.append("</table>")
            html_parts.append("</div>")

        html_parts.extend(["</body>", "</html>"])

        html_content = "\n".join(html_parts)

        if output_path:
            output_path = Path(output_path)
            output_path.write_text(html_content, encoding="utf-8")

        return html_content

    def export_json(self, output_path: str | Path) -> None:
        """
        Export profiling data to JSON.

        Parameters
        ----------
        output_path : str or Path
            Path to save JSON file
        """
        data = self.profiler.export_to_dict()
        output_path = Path(output_path)
        output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def export_csv(self, name: str, output_path: str | Path) -> None:
        """
        Export profile data to CSV.

        Parameters
        ----------
        name : str
            Profile name to export
        output_path : str or Path
            Path to save CSV file
        """
        metrics_list = self.profiler.get_metrics(name)
        if not metrics_list:
            return

        output_path = Path(output_path)

        # CSV header
        headers = [
            "run",
            "total_time",
            "jit_compile_time",
            "optimization_time",
            "jacobian_time",
            "n_iterations",
            "n_function_evals",
            "n_jacobian_evals",
            "n_data_points",
            "n_parameters",
            "final_cost",
            "initial_cost",
            "cost_reduction",
            "final_gradient_norm",
            "success",
            "method",
            "backend",
        ]

        lines = [",".join(headers)]

        for i, m in enumerate(metrics_list, 1):
            row = [
                str(i),
                f"{m.total_time:.6f}",
                f"{m.jit_compile_time:.6f}",
                f"{m.optimization_time:.6f}",
                f"{m.jacobian_time:.6f}",
                str(m.n_iterations),
                str(m.n_function_evals),
                str(m.n_jacobian_evals),
                str(m.n_data_points),
                str(m.n_parameters),
                f"{m.final_cost:.6f}",
                f"{m.initial_cost:.6f}",
                f"{m.cost_reduction:.6f}",
                f"{m.final_gradient_norm:.6f}",
                str(m.success),
                m.method,
                m.backend,
            ]
            lines.append(",".join(row))

        output_path.write_text("\n".join(lines), encoding="utf-8")


class ProfilingDashboard:
    """
    Interactive dashboard for profiling data.

    Examples
    --------
    >>> from nlsq.utils.profiler import get_global_profiler
    >>> from nlsq.utils.profiler_visualization import ProfilingDashboard
    >>> profiler = get_global_profiler()
    >>> dashboard = ProfilingDashboard(profiler)
    >>> dashboard.add_profile("test1")
    >>> dashboard.add_profile("test2")
    >>> dashboard.generate_comparison_report()
    """

    def __init__(self, profiler: PerformanceProfiler):
        """
        Initialize dashboard.

        Parameters
        ----------
        profiler : PerformanceProfiler
            Profiler instance to visualize
        """
        self.profiler = profiler
        self.viz = ProfilerVisualization(profiler)
        self.tracked_profiles: list[str] = []

    def add_profile(self, name: str) -> None:
        """
        Add a profile to track in the dashboard.

        Parameters
        ----------
        name : str
            Profile name to track
        """
        if name not in self.tracked_profiles:
            self.tracked_profiles.append(name)

    def remove_profile(self, name: str) -> None:
        """
        Remove a profile from tracking.

        Parameters
        ----------
        name : str
            Profile name to remove
        """
        if name in self.tracked_profiles:
            self.tracked_profiles.remove(name)

    def generate_comparison_report(self) -> str:
        """
        Generate comparison report for tracked profiles.

        Returns
        -------
        report : str
            Formatted comparison report
        """
        if not self.tracked_profiles:
            return "No profiles tracked in dashboard."

        lines = [
            "=" * 80,
            "NLSQ Profiling Dashboard - Comparison Report",
            "=" * 80,
            "",
        ]

        summaries = {}
        for name in self.tracked_profiles:
            summary = self.profiler.get_summary(name)
            if summary:
                summaries[name] = summary

        if not summaries:
            return "No profiling data available for tracked profiles."

        # Overall comparison table
        lines.extend(
            [
                "Profile Summary:",
                "-" * 80,
                f"{'Profile':<20} {'Runs':<8} {'Success%':<12} {'Mean Time':<15} {'Mean Iter':<12}",
                "-" * 80,
            ]
        )

        for name, summary in summaries.items():
            lines.append(
                f"{name:<20} {summary['n_runs']:<8} "
                f"{summary['success_rate'] * 100:<12.1f} "
                f"{summary['total_time']['mean']:<15.3f} "
                f"{summary['iterations']['mean']:<12.1f}"
            )

        lines.append("-" * 80)

        # Best performers
        if len(summaries) > 1:
            lines.extend(["", "Best Performers:", "-" * 80])

            fastest = min(summaries.items(), key=lambda x: x[1]["total_time"]["mean"])
            lines.append(
                f"  Fastest:         {fastest[0]} ({fastest[1]['total_time']['mean']:.3f}s)"
            )

            most_reliable = max(summaries.items(), key=lambda x: x[1]["success_rate"])
            lines.append(
                f"  Most Reliable:   {most_reliable[0]} ({most_reliable[1]['success_rate'] * 100:.1f}%)"
            )

            fewest_iter = min(
                summaries.items(), key=lambda x: x[1]["iterations"]["mean"]
            )
            lines.append(
                f"  Fewest Iters:    {fewest_iter[0]} ({fewest_iter[1]['iterations']['mean']:.1f})"
            )

        lines.append("=" * 80)

        return "\n".join(lines)

    def plot_all_comparisons(
        self, figsize: tuple[float, float] = (15, 10)
    ) -> Figure | None:
        """
        Generate comprehensive comparison plots.

        Parameters
        ----------
        figsize : tuple
            Figure size (width, height)

        Returns
        -------
        fig : Figure or None
            Matplotlib figure or None if matplotlib not available
        """
        if not HAS_MATPLOTLIB or not self.tracked_profiles:
            return None

        return self.viz.plot_timing_comparison(self.tracked_profiles, figsize=figsize)

    def save_dashboard(self, output_dir: str | Path) -> None:
        """
        Save complete dashboard to directory.

        Parameters
        ----------
        output_dir : str or Path
            Directory to save dashboard files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save HTML report
        html_path = output_dir / "dashboard.html"
        self.viz.generate_html_report(html_path)

        # Save JSON export
        json_path = output_dir / "profiles.json"
        self.viz.export_json(json_path)

        # Save comparison report
        report_path = output_dir / "comparison_report.txt"
        report_path.write_text(self.generate_comparison_report(), encoding="utf-8")

        # Save plots if matplotlib available
        if HAS_MATPLOTLIB:
            for name in self.tracked_profiles:
                fig = self.viz.plot_timing_series(name)
                if fig:
                    fig.savefig(
                        output_dir / f"{name}_series.png", dpi=150, bbox_inches="tight"
                    )
                    plt.close(fig)

            fig = self.plot_all_comparisons()
            if fig:
                fig.savefig(output_dir / "comparison.png", dpi=150, bbox_inches="tight")
                plt.close(fig)


__all__ = [
    "ProfilerVisualization",
    "ProfilingDashboard",
]
