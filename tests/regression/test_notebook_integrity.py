"""Static integrity checks for every example notebook.

These tests only read the ``.ipynb`` JSON (no kernel is started), so they are
fast and need no ``serial`` marker. They guard against regressions that the
execution test in ``test_notebooks.py`` cannot see:

- source lists whose newlines were lost, so headings run into the following
  text and a whole code cell collapses into one comment line;
- relative markdown links that point at files which do not exist.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat", reason="nbformat not installed")

NB_ROOT = Path(__file__).resolve().parents[2] / "examples" / "notebooks"
NOTEBOOKS = sorted(NB_ROOT.rglob("*.ipynb"))

_LINK_RE = re.compile(r"\]\(([^)\s]+)\)")
_EXTERNAL_RE = re.compile(r"^(?:[a-zA-Z][a-zA-Z0-9+.-]*:|#)")


def _source(cell: dict) -> str:
    src = cell["source"]
    return src if isinstance(src, str) else "".join(src)


def _rel(path: Path) -> str:
    return str(path.relative_to(NB_ROOT))


def test_notebooks_found():
    assert NOTEBOOKS, f"no notebooks found under {NB_ROOT}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=_rel)
def test_notebook_is_valid_and_has_code(path: Path):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    assert any(c.cell_type == "code" for c in nb.cells), "no code cell"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=_rel)
def test_no_code_cell_collapsed_to_a_comment(path: Path):
    """A long single-line cell with no executable code lost its newlines.

    Multi-line all-comment cells (a deliberately commented-out solution) are fine.
    """
    nb = json.loads(path.read_text(encoding="utf-8"))
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        text = _source(cell)
        body = [
            line
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        assert not (len(text) > 120 and not body and len(text.splitlines()) <= 1), (
            f"{_rel(path)} cell {i}: {len(text)} chars but no executable line "
            "(newlines lost?)"
        )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=_rel)
def test_no_glued_markdown_heading(path: Path):
    """A single-entry heading cell holding bold text and a rule is a run-on."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "markdown":
            continue
        for entry in cell["source"] if isinstance(cell["source"], list) else []:
            glued = (
                re.match(r"#{1,6} ", entry)
                and len(entry) > 200
                and "**" in entry
                and "---" in entry
            )
            assert not glued, (
                f"{_rel(path)} cell {i}: heading line looks glued: {entry[:80]!r}"
            )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=_rel)
def test_relative_markdown_links_resolve(path: Path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    dead = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "markdown":
            continue
        for target in _LINK_RE.findall(_source(cell)):
            if _EXTERNAL_RE.match(target):
                continue
            file_part = target.split("#", 1)[0]
            if file_part and not (path.parent / file_part).resolve().exists():
                dead.append(f"cell {i}: {target}")
    assert not dead, f"{_rel(path)} has dead relative links: {dead}"
