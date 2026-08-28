"""Regression tests for scripts/notebooks/io_patch.py's sandbox containment.

io_patch.patch_savefig() redirects matplotlib's savefig() into
NLSQ_OUTPUT_DIR/artifacts/<script_name>/ so example scripts/notebooks can't
write figures outside the intended output tree. These tests pin the
containment invariant directly, since io_patch.py otherwise has no
automated coverage (scripts/ is excluded from both pytest's default
testpaths and bandit's scan).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts" / "notebooks"))


@pytest.fixture
def savefig_spy(monkeypatch, tmp_path):
    """Patch matplotlib.pyplot.savefig, call io_patch.patch_savefig(), and
    return a list that captures every resolved path the patched savefig()
    is invoked with — without touching the filesystem via a real save.
    """
    import matplotlib.figure
    import matplotlib.pyplot as plt

    calls: list[Path] = []

    def _spy(fname, *args, **kwargs):
        calls.append(Path(fname))

    # io_patch.patch_savefig() patches matplotlib.figure.Figure.savefig via
    # a raw class-attribute assignment (that's the production behavior
    # being tested — io_patch.py is meant to monkeypatch globally for the
    # life of a script-runner subprocess), not through pytest's monkeypatch
    # fixture, so it's never undone on its own. Register it with
    # monkeypatch here (before patch_savefig() reassigns it) so monkeypatch
    # restores the true original at teardown regardless — otherwise this
    # leaks a global Figure.savefig patch that silently redirects any
    # OTHER test's fig.savefig() call to this test's tmp_path sandbox
    # whenever pytest-xdist schedules it on the same worker afterward.
    monkeypatch.setattr(
        matplotlib.figure.Figure, "savefig", matplotlib.figure.Figure.savefig
    )
    monkeypatch.setattr(plt, "savefig", _spy)
    monkeypatch.setenv("NLSQ_OUTPUT_DIR", str(tmp_path))

    yield calls, tmp_path

    # io_patch caches the original savefig at import time; reload so the
    # next test gets a clean, unpatched module instead of chained patches.
    if "io_patch" in sys.modules:
        del sys.modules["io_patch"]


def _resolved(calls: list[Path]) -> Path:
    assert len(calls) == 1, f"expected exactly one savefig call, got {calls}"
    # .resolve() matters here: Path.is_relative_to() is a lexical/parts
    # comparison, not filesystem resolution — an unresolved ".../myscript/.."
    # lexically "starts with" ".../myscript" and would pass containment
    # checks while actually opening one directory up on disk. Resolving
    # first makes the assertion match what actually gets written.
    return calls[0].resolve()


def _expected_safe_script_name(script_name: str) -> str:
    """Mirror io_patch._safe_name()'s sanitization for the per-script dir
    name, so tests assert against the *intended* sandbox (artifacts/<script>/)
    rather than the looser artifacts/ root — a path that only escapes its
    own script's subdirectory (e.g. "myscript/..") is still "under
    artifacts/" lexically/by resolve(), so checking against artifacts/
    alone would silently accept that partial escape.
    """
    name = Path(script_name).name
    return name if name not in ("", "..", ".") else "unknown"


class TestPatchSavefigContainment:
    """Every fname/NLSQ_CURRENT_SCRIPT combination must resolve under
    NLSQ_OUTPUT_DIR/artifacts/<sanitized-script-name>/, never outside it —
    not even one level up into a sibling script's directory or artifacts/.
    """

    def _run(self, savefig_spy, monkeypatch, script_name: str, fname: str) -> Path:
        import io_patch

        calls, tmp_path = savefig_spy
        monkeypatch.setenv("NLSQ_CURRENT_SCRIPT", script_name)
        io_patch.patch_savefig()

        import matplotlib.pyplot as plt

        plt.savefig(fname)
        resolved = _resolved(calls)
        safe_script = _expected_safe_script_name(script_name)
        sandbox_root = (tmp_path / "artifacts" / safe_script).resolve()
        assert resolved.is_relative_to(sandbox_root), (
            f"{fname=} {script_name=} escaped its per-script sandbox "
            f"{sandbox_root}: got {resolved}"
        )
        return resolved

    def test_normal_relative_path(self, savefig_spy, monkeypatch):
        resolved = self._run(savefig_spy, monkeypatch, "myscript", "figures/fig1.png")
        assert resolved.name == "fig1.png"
        assert "myscript" in resolved.parts

    def test_relative_traversal_is_stripped(self, savefig_spy, monkeypatch):
        resolved = self._run(
            savefig_spy, monkeypatch, "myscript", "../../../etc/evil.png"
        )
        assert resolved.name == "evil.png"

    def test_pure_dotdot_fname_does_not_escape(self, savefig_spy, monkeypatch):
        # Regression: a naive token filter that falls back to Path(fname).name
        # reintroduces ".." here, since Path("..").name == ".." (not "").
        self._run(savefig_spy, monkeypatch, "myscript", "..")

    def test_multi_level_dotdot_fname_does_not_escape(self, savefig_spy, monkeypatch):
        self._run(savefig_spy, monkeypatch, "myscript", "../../../..")

    def test_absolute_path_is_contained(self, savefig_spy, monkeypatch):
        resolved = self._run(savefig_spy, monkeypatch, "myscript", "/etc/passwd")
        assert resolved.name == "passwd"

    def test_dot_fname(self, savefig_spy, monkeypatch):
        self._run(savefig_spy, monkeypatch, "myscript", ".")

    def test_malicious_script_name_is_sanitized(self, savefig_spy, monkeypatch):
        # NLSQ_CURRENT_SCRIPT is attacker/env-controlled too, not just fname.
        self._run(savefig_spy, monkeypatch, "../../evil_script", "fig.png")

    def test_pure_dotdot_script_name_does_not_escape(self, savefig_spy, monkeypatch):
        self._run(savefig_spy, monkeypatch, "..", "fig.png")

    def test_empty_script_name_falls_back(self, savefig_spy, monkeypatch):
        self._run(savefig_spy, monkeypatch, "", "fig.png")
