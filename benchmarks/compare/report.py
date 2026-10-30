"""Record schema, environment capture, ranking and Markdown rendering."""

import importlib.metadata as md
import json
import os
import platform
from itertools import groupby
from pathlib import Path

RECORD_FIELDS: tuple[str, ...] = (
    "library",
    "mode",
    "problem",
    "n",
    "batch",
    "backend",
    "device",
    "status",
    "reason",
    "converged",
    "cold_s",
    "warm_median_s",
    "warm_iqr_s",
    "nfev",
    "cost",
    "cost_gap",
    "max_param_error_vs_truth",
)

PACKAGES = ("numpy", "scipy", "jax", "jaxlib", "nlsq", "lmfit", "jaxopt", "optimistix")


def _version(dist: str) -> str:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return "not installed"


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def environment() -> dict:
    import jax

    return {
        "packages": {p: _version(p) for p in PACKAGES},
        "backend": jax.default_backend(),
        "devices": ", ".join(sorted({d.device_kind for d in jax.devices()})),
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
    }


def write_json(records: list[dict], path: Path) -> None:
    path.write_text(json.dumps(records, indent=2) + "\n")


def rank_case(records: list[dict]) -> list[dict]:
    """Measured, optimum-reaching records, fastest warm median first."""
    ok = [r for r in records if r["status"] == "ok"]
    return sorted(ok, key=lambda r: r["warm_median_s"])


def _fmt(v, spec: str = ".3g") -> str:
    return "-" if v is None else format(v, spec)


def _case_key(r: dict):
    return (r["problem"], r["n"], r.get("batch"))


def _table(rows: list[dict]) -> list[str]:
    out = [
        "| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev "
        "| cost gap | max param err | lib converged |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    out += [
        f"| {r['library']} | {r['mode']} | {_fmt(r['cold_s'])} "
        f"| {_fmt(r['warm_median_s'])} | {_fmt(r['warm_iqr_s'])} "
        f"| {_fmt(r['nfev'], 'd')} | {_fmt(r['cost_gap'])} "
        f"| {_fmt(r['max_param_error_vs_truth'])} | {r['converged']} |"
        for r in rows
    ]
    return out


def to_markdown(records: list[dict], env: dict) -> str:
    from .adapters import ADAPTERS

    backend = env.get("backend", "unknown")
    lines = ["# Library comparison", ""]
    lines += [
        f"- Backend: `{backend}`; devices: {env.get('devices', 'unknown')}",
        f"- CPU: {env.get('cpu', 'unknown')} ({env.get('cpu_count', '?')} logical)",
        "",
        "Caveats:",
        "",
        "- Stopping criteria are analogous, not identical: every library gets the "
        "same tolerance and iteration budget for each criterion it exposes. A "
        "library is ranked only if its final cost matches the best cost in the "
        "case (relative 1e-6); its own `converged` flag is reported but does not "
        "decide that. A ranked library can still show `lib converged` = False: it "
        "reached the optimum but ran out its iteration budget, so its time "
        "includes the unused iterations.",
        "- lmfit wraps SciPy (`leastsq` / `least_squares`); its timings include "
        "lmfit's own overhead on top of SciPy.",
        '- JAXopt: README states "JAXopt is no longer maintained nor developed".',
        "- Cold = first call in a fresh process after a trivial backend warm-up "
        "(includes JIT compilation); warm = median and IQR of repeated calls.",
    ]
    if backend == "cpu":
        lines.append("- GPU was not measured: this run used the CPU backend.")
    lines += ["", "## Results", ""]
    for (problem, n, batch), group in groupby(
        sorted(records, key=lambda r: (r["problem"], r["n"], r.get("batch") or 0)),
        key=_case_key,
    ):
        grp = list(group)
        title = f"{problem}, n={n}" + (f", batch={batch}" if batch else "")
        ranked = rank_case(grp)
        lines += [f"### {title}", ""]
        lines += _table(ranked) if ranked else ["No library reached the optimum."]
        lines.append("")

    lines += ["## Capability matrix", ""]
    lines += [
        "| library | bounds | sigma/weights | covariance | vmappable | maintained | note |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, a in ADAPTERS.items():
        c = a.capabilities
        yn = lambda b: "yes" if b else "no"  # noqa: E731
        lines.append(
            f"| {name} | {yn(c.bounds)} | {yn(c.sigma_weights)} | "
            f"{yn(c.covariance)} | {yn(c.vmappable)} | {yn(c.maintained)} | {c.note} |"
        )

    lines += ["", "## Unavailable, unsupported or excluded", ""]
    excluded = [r for r in records if r["status"] != "ok"]
    if not excluded:
        lines.append("None.")
    for r in excluded:
        detail = r["reason"] or (
            f"final cost {_fmt(r['cost_gap'])} above the best (relative)"
            if r["status"] == "did_not_reach_optimum"
            else ""
        )
        lines.append(
            f"- {r['library']} ({r['mode']}) on {r['problem']} n={r['n']}: "
            f"{r['status']}. {detail}"
        )

    lines += ["", "## Package versions", ""]
    lines += [f"- {k}: {v}" for k, v in env.get("packages", {}).items()]
    return "\n".join(lines) + "\n"
