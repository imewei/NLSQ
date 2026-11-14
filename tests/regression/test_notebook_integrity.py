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
    # nbformat.read() would silently fill in missing cell ids; validate the raw JSON.
    raw = json.loads(path.read_text(encoding="utf-8"))
    nbformat.validate(nbformat.from_dict(raw))
    ids = [c.get("id") for c in raw["cells"]]
    assert all(ids) and len(set(ids)) == len(ids), "cells need unique ids"
    assert any(c["cell_type"] == "code" for c in raw["cells"]), "no code cell"


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


def _cells(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=_rel)
def test_code_cells_have_no_stored_outputs(path: Path):
    for i, cell in enumerate(_cells(path)):
        if cell["cell_type"] == "code":
            assert cell.get("outputs") == [], f"{_rel(path)} cell {i} has outputs"
            assert cell.get("execution_count") is None, (
                f"{_rel(path)} cell {i} has an execution_count"
            )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=_rel)
def test_no_empty_cells(path: Path):
    for i, cell in enumerate(_cells(path)):
        assert _source(cell).strip(), f"{_rel(path)} cell {i} is empty"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=_rel)
def test_long_markdown_cells_keep_their_newlines(path: Path):
    """A long markdown cell that is one line in one entry lost its newlines."""
    for i, cell in enumerate(_cells(path)):
        if cell["cell_type"] != "markdown" or len(_source(cell)) <= 300:
            continue
        src = cell["source"]
        assert "\n" in _source(cell) or (isinstance(src, list) and len(src) >= 2), (
            f"{_rel(path)} cell {i}: {len(_source(cell))} chars on a single line"
        )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=_rel)
def test_random_numbers_are_seeded(path: Path):
    code = "\n".join(_source(c) for c in _cells(path) if c["cell_type"] == "code")
    if "np.random." in code or "default_rng(" in code:
        assert "seed(" in code or re.search(r"default_rng\(\s*\d", code), (
            f"{_rel(path)} draws random numbers without a seed"
        )
