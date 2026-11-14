"""Guard the meta-refresh redirects that docs/conf.py writes for removed pages."""

import ast
import posixpath
from pathlib import Path
from types import SimpleNamespace

DOCS = Path(__file__).resolve().parents[2] / "docs"


def _load_redirect_hook():
    """Pull the redirect table and hook out of conf.py without running Sphinx setup."""
    tree = ast.parse((DOCS / "conf.py").read_text(encoding="utf-8"))
    table, hook = None, None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", "") == "REMOVED_PAGE_REDIRECTS" for t in node.targets
        ):
            table = ast.literal_eval(node.value)
        if isinstance(node, ast.FunctionDef) and node.name == (
            "_write_removed_page_redirects"
        ):
            hook = node
    assert table and hook, "REMOVED_PAGE_REDIRECTS or its hook is missing from conf.py"
    namespace = {
        "posixpath": posixpath,
        "Path": Path,
        "REMOVED_PAGE_REDIRECTS": table,
    }
    exec(compile(ast.Module([hook], []), "conf.py", "exec"), namespace)
    return table, namespace["_write_removed_page_redirects"]


def test_redirect_targets_exist_and_old_pages_are_gone():
    table, _ = _load_redirect_hook()
    for old, new in table.items():
        assert (DOCS / f"{new}.rst").is_file(), f"{old} redirects to missing {new}"
        assert not (DOCS / f"{old}.rst").exists(), f"{old} exists; drop its redirect"


def test_hook_writes_relative_meta_refresh_pages(tmp_path):
    table, hook = _load_redirect_hook()
    hook(SimpleNamespace(outdir=tmp_path, builder=SimpleNamespace(format="html")), None)
    page = (tmp_path / "tutorials/01_first_fit.html").read_text(encoding="utf-8")
    assert "url=routine/getting_started/first_fit.html" in page
    page = (tmp_path / "gui/user_guide.html").read_text(encoding="utf-8")
    assert "url=../tutorials/routine/gui_desktop/index.html" in page
    assert len(list(tmp_path.rglob("*.html"))) == len(table)


def test_hook_writes_nothing_on_error_or_non_html_builder(tmp_path):
    _, hook = _load_redirect_hook()
    html = SimpleNamespace(format="html")
    hook(SimpleNamespace(outdir=tmp_path, builder=html), RuntimeError("build failed"))
    hook(
        SimpleNamespace(outdir=tmp_path, builder=SimpleNamespace(format="latex")), None
    )
    assert not list(tmp_path.rglob("*.html"))
