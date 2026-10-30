"""Guard the split between runtime dependencies, the gui extra, and dev/bench groups."""

import re
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"

RUNTIME = {
    "numpy",
    "scipy",
    "jax",
    "jaxlib",
    "optax",
    "optimistix",
    "matplotlib",
    "psutil",
    "tqdm",
    "h5py",
    "xxhash",
    "pyyaml",
    "evosax",
}


def _load_pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _dist_name(requirement: str) -> str:
    """Return the normalised distribution name of a PEP 508 requirement."""
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    assert match, requirement
    return match.group(1).lower().replace("_", "-")


def _all_requirements(data: dict) -> list[str]:
    project = data["project"]
    reqs = list(project["dependencies"])
    for extra in project.get("optional-dependencies", {}).values():
        reqs += extra
    for group in data.get("dependency-groups", {}).values():
        reqs += [r for r in group if isinstance(r, str)]
    return reqs


def test_core_dependencies_are_exactly_the_runtime_set():
    deps = _load_pyproject()["project"]["dependencies"]
    assert {_dist_name(r) for r in deps} == RUNTIME


def test_gui_extra_declares_toolkit():
    gui = _load_pyproject()["project"]["optional-dependencies"]["gui"]
    assert {_dist_name(r) for r in gui} == {"pyside6", "pyqtgraph"}
    assert all(">=" in r for r in gui)


def test_dev_group_has_test_lint_docs_build_tools():
    dev = _load_pyproject()["dependency-groups"]["dev"]
    names = {_dist_name(r) for r in dev if isinstance(r, str)}
    required = {
        "pytest",
        "ruff",
        "mypy",
        "sphinx",
        "furo",
        "build",
        "twine",
        "hypothesis",
    }
    assert required <= names


def test_bench_group_has_comparison_libraries():
    bench = _load_pyproject()["dependency-groups"]["bench"]
    assert {_dist_name(r) for r in bench} == {"lmfit", "jaxopt"}


def test_no_upper_caps():
    for req in _all_requirements(_load_pyproject()):
        spec = req.split(";", 1)[0]
        assert "<" not in spec, req


REPO = PYPROJECT.parent
INSTALL_REFERENCE_FILES = [
    "README.md",
    "CONTRIBUTING.md",
    "Makefile",
    ".readthedocs.yaml",
    "docs/requirements.txt",
    "docs/tutorials/routine/getting_started/installation.rst",
    *sorted(
        str(p.relative_to(REPO)) for p in (REPO / ".github" / "workflows").glob("*.yml")
    ),
]
# The extra must start with a letter so shell classes such as `[0-9]+` do not match.
EXTRA_REF = re.compile(r"(?:nlsq|\.)\[([A-Za-z][A-Za-z0-9_,.-]*)\]")


def test_install_references_name_existing_extras():
    extras = {
        _dist_name(name)
        for name in _load_pyproject()["project"]["optional-dependencies"]
    }
    for rel in INSTALL_REFERENCE_FILES:
        text = (REPO / rel).read_text(encoding="utf-8")
        for match in EXTRA_REF.finditer(text):
            for name in match.group(1).split(","):
                assert _dist_name(name) in extras, f"{rel}: unknown extra {name!r}"
    assert "nlsq[gui]" in (REPO / "README.md").read_text(encoding="utf-8")


def test_docs_requirements_cover_docs_packages():
    text = (REPO / "docs" / "requirements.txt").read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines()]
    names = {_dist_name(ln) for ln in lines if ln and not ln.startswith(("#", "-"))}
    required = {
        "sphinx",
        "furo",
        "sphinx-copybutton",
        "sphinx-autodoc-typehints",
        "sphinx-design",
        "myst-parser",
    }
    assert required <= names
    # autodoc imports nlsq.gui_qt; fail_on_warning breaks without PySide6.
    assert ".[gui]" in "\n".join(lines)
