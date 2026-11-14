"""Regression tests for the notebook <-> script tooling under scripts/.

Covers scripts/notebooks/{convert_examples,regenerate_notebooks,configure_notebooks,
run_scripts,run_notebooks}.py and scripts/quick_sitecustomize (review 2026-11).
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
NB_TOOLS = SCRIPTS_DIR / "notebooks"
EXAMPLE_SCRIPTS = sorted((REPO_ROOT / "examples" / "scripts").rglob("*.py"))

sys.path.insert(0, str(NB_TOOLS))
sys.path.insert(0, str(SCRIPTS_DIR))

import convert_examples as ce
import regenerate_notebooks as rn


def _code(nb: dict) -> str:
    return "\n".join(
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    )


def _is_docstring(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


# ----------------------------------------------------------------------------
# convert_examples.script_to_notebook
# ----------------------------------------------------------------------------


class TestScriptToNotebook:
    @pytest.mark.parametrize(
        "script", EXAMPLE_SCRIPTS, ids=lambda p: str(p.relative_to(REPO_ROOT))
    )
    def test_every_example_script_converts_losslessly(self, script, tmp_path):
        out = ce.script_to_notebook(script, tmp_path / "out.ipynb")
        nb = json.loads(out.read_text(encoding="utf-8"))
        assert nb["cells"], f"{script.name} produced an empty notebook"

        body = ast.parse(script.read_text(encoding="utf-8")).body
        if body and _is_docstring(body[0]):
            body = body[1:]
        expected = [ast.unparse(node) for node in body]
        got = [ast.unparse(node) for node in ast.parse(_code(nb)).body]
        assert got == expected

    def test_cells_carry_unique_ids(self, tmp_path):
        script = tmp_path / "s.py"
        script.write_text(
            "import os\n# " + "=" * 70 + "\n# prose\n# " + "=" * 70 + "\nx = 1\n",
            encoding="utf-8",
        )
        nb = json.loads(ce.script_to_notebook(script).read_text(encoding="utf-8"))
        ids = [c["id"] for c in nb["cells"]]
        assert len(ids) == 3 and len(set(ids)) == 3
        import nbformat

        nbformat.validate(nbformat.from_dict(nb))  # raw JSON: no id back-filling

    def test_inserted_magic_cells_get_ids_unique_in_the_notebook(self):
        from notebook_utils.cells import (
            create_ipython_display_import_cell,
            create_matplotlib_config_cell,
        )

        def code(cid):
            return {
                "id": cid,
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["import matplotlib.pyplot as plt\n"],
            }

        for _ in range(2):  # a taken id must be avoided, not reused
            nb = {"cells": [code("matplotlib-inline")]}
            modified, _msg = ce.configure_matplotlib_inline(nb)
            assert modified
            ids = [c["id"] for c in nb["cells"]]
            assert len(set(ids)) == len(ids) == 2

        first = create_matplotlib_config_cell()
        assert first["id"]
        again = create_matplotlib_config_cell([first])
        assert again["id"] != first["id"]
        assert create_ipython_display_import_cell([again])["id"]

    def test_closing_docstring_quotes_at_column_zero(self, tmp_path):
        script = tmp_path / "s.py"
        script.write_text(
            '"""\nTitle\n\nBody\n"""\nimport os\n\n\ndef f():\n    """Doc."""\n    return 1\n',
            encoding="utf-8",
        )
        nb = json.loads(ce.script_to_notebook(script).read_text(encoding="utf-8"))
        assert "import os" in _code(nb)
        assert "def f()" in _code(nb)

    def test_round_trip_keeps_cells(self, tmp_path):
        nb = {
            "cells": [
                {"cell_type": "markdown", "metadata": {}, "source": ["# Title\n"]},
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": ["x = 1\n", "print(x)\n"],
                },
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        src = tmp_path / "nb.ipynb"
        src.write_text(json.dumps(nb), encoding="utf-8")
        py = ce.notebook_to_script(src, tmp_path / "nb.py")
        back = json.loads(ce.script_to_notebook(py, tmp_path / "b.ipynb").read_text())
        assert back["cells"]
        assert "x = 1" in _code(back)

    def test_blank_lines_and_comments_survive_in_multiline_strings(self, tmp_path):
        script = tmp_path / "s.py"
        script.write_text(
            'T = """a\n\n# not a comment\nb"""\nprint(T)\n', encoding="utf-8"
        )
        nb = json.loads(ce.script_to_notebook(script).read_text(encoding="utf-8"))
        ns: dict = {}
        exec(_code(nb).split("print")[0], ns)
        assert ns["T"] == "a\n\n# not a comment\nb"

    def test_markdown_heading_level_kept(self, tmp_path):
        script = tmp_path / "s.py"
        banner = "# " + "=" * 70
        script.write_text(
            f"{banner}\n# ## Section\n# text\n{banner}\nx = 1\n", encoding="utf-8"
        )
        nb = json.loads(ce.script_to_notebook(script).read_text(encoding="utf-8"))
        md = "".join(nb["cells"][0]["source"])
        assert nb["cells"][0]["cell_type"] == "markdown"
        assert md.startswith("## Section")

    def test_indented_banner_inside_function_does_not_split_code(self, tmp_path):
        script = tmp_path / "s.py"
        banner = "# " + "=" * 70
        script.write_text(
            f"def f():\n    {banner}\n    # Step 1\n    {banner}\n    return 1\n",
            encoding="utf-8",
        )
        nb = json.loads(ce.script_to_notebook(script).read_text(encoding="utf-8"))
        ast.parse(_code(nb))
        assert len(nb["cells"]) == 1

    def test_syntax_error_writes_nothing(self, tmp_path):
        script = tmp_path / "bad.py"
        script.write_text("def (:\n", encoding="utf-8")
        with pytest.raises(SyntaxError):
            ce.script_to_notebook(script)
        assert not (tmp_path / "bad.ipynb").exists()

    def test_default_output_mirrors_scripts_to_notebooks(self, tmp_path):
        script = tmp_path / "examples" / "scripts" / "01_a" / "x.py"
        script.parent.mkdir(parents=True)
        script.write_text("x = 1\n", encoding="utf-8")
        out = ce.script_to_notebook(script)
        assert out == tmp_path / "examples" / "notebooks" / "01_a" / "x.ipynb"
        assert not (script.parent / "x.ipynb").exists()

    def test_swap_dir_part_uses_path_parts(self, tmp_path):
        assert ce.swap_dir_part(Path("a/scripts"), "scripts", "notebooks") == Path(
            "a/notebooks"
        )
        assert ce.swap_dir_part(Path("scripts/scripts/x"), "scripts", "n") == Path(
            "scripts/n/x"
        )
        assert ce.swap_dir_part(Path("a/b"), "scripts", "n") is None


# ----------------------------------------------------------------------------
# convert_examples: plt.show() / display(plt.gcf()) rewriting
# ----------------------------------------------------------------------------


class TestPltShowRewriting:
    def test_replacement_keeps_indent_on_every_line(self):
        src = "for i in range(2):\n    plt.plot([i])\n    plt.show()\nelse:\n    pass\n"
        new, modified = ce.replace_plt_show_in_cell(src)
        assert modified
        ast.parse(new)
        assert "    display(plt.gcf())\n    plt.close()\n" in new

    def test_existing_tight_layout_is_folded_not_duplicated(self):
        src = "if x:\n    plt.tight_layout()\n    plt.show()\n"
        new, _ = ce.replace_plt_show_in_cell(src)
        assert new.count("plt.tight_layout()") == 1
        ast.parse(new)

    def test_comment_mentioning_plt_show_is_untouched(self):
        src = "# call plt.show() later\nx = 1\n"
        new, modified = ce.replace_plt_show_in_cell(src)
        assert not modified and new == src

    def test_each_display_gets_its_own_figure_file(self):
        code = (
            "plt.plot([1])\ndisplay(plt.gcf())\nplt.close()\n"
            "plt.plot([2])\ndisplay(plt.gcf())\nplt.close()\n"
        )
        out = ce.post_process_script_code(code, "demo", {"count": 0, "imports": set()})
        assert "fig_01.png" in out and "fig_02.png" in out
        ast.parse(out)

    def test_display_inside_loop_stays_in_loop_body(self):
        code = "for i in range(3):\n    plt.plot([i])\n    display(plt.gcf())\n"
        out = ce.post_process_script_code(code, "demo", {"count": 0, "imports": set()})
        tree = ast.parse(out)
        loop = tree.body[0]
        assert isinstance(loop, ast.For)
        assert "savefig" in ast.unparse(loop)


# ----------------------------------------------------------------------------
# regenerate_notebooks
# ----------------------------------------------------------------------------


def _regen(tmp_path: Path, source: str) -> dict:
    script = tmp_path / "s.py"
    script.write_text(source, encoding="utf-8")
    out = rn.script_to_notebook_robust(script, tmp_path / "s.ipynb")
    return json.loads(out.read_text(encoding="utf-8"))


def _assert_cells_compile(nb: dict) -> None:
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            compile(
                "".join(cell["source"]).replace("%matplotlib inline", ""), "c", "exec"
            )


class TestRegenerateNotebooks:
    MAIN_SCRIPT = (
        '"""Title\n\nBody."""\n'
        "import os\n"
        "import jax\n"
        'jax.config.update("jax_enable_x64", True)\n'
        'os.environ["X"] = "1"\n'
        "\n"
        "@jax.jit\n"
        "def f(x):\n"
        "    return x\n"
        "\n"
        "class C:\n"
        "    pass\n"
        "\n"
        "LATE = f(3)\n"
        "\n"
        "def main():\n"
        "    print(f(1), C, LATE)\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    main()\n"
    )

    def test_main_path_keeps_toplevel_statements_decorators_and_classes(self, tmp_path):
        nb = _regen(tmp_path, self.MAIN_SCRIPT)
        _assert_cells_compile(nb)
        code = _code(nb)
        assert 'jax.config.update("jax_enable_x64", True)' in code
        assert 'os.environ["X"] = "1"' in code
        assert "@jax.jit\ndef f(x):" in code
        assert "class C:" in code
        assert "LATE = f(3)" in code
        # statements after the defs must not run before the defs
        assert code.index("def f(x)") < code.index("LATE = f(3)")

    def test_no_main_path_decorator_and_no_duplicate_constants(self, tmp_path):
        nb = _regen(
            tmp_path,
            "import jax\nQUICK = 1\n\n@jax.jit\ndef f(x):\n    return x\n\n"
            "print(f(QUICK))\n",
        )
        _assert_cells_compile(nb)
        code = _code(nb)
        assert "@jax.jit\ndef f(x):" in code
        assert code.count("QUICK = 1") == 1

    def test_nested_call_savefig_and_sys_exit(self, tmp_path):
        nb = _regen(
            tmp_path,
            "import os, sys\nimport matplotlib.pyplot as plt\n"
            'plt.savefig(os.path.join("d", "b.png"))\n'
            "sys.exit(len(sys.argv))\n",
        )
        _assert_cells_compile(nb)
        code = _code(nb)
        assert "plt.show()" in code and "savefig" not in code
        assert "sys.exit" not in code

    def test_replace_calls_leaves_unbalanced_fragment(self):
        assert rn._replace_calls("sys.exit(", r"\bsys\.exit", "pass") == "sys.exit("
        assert (
            rn._replace_calls('plt.savefig(")", x)', r"\bplt\.savefig", "plt.show()")
            == "plt.show()"
        )

    def test_syntax_error_is_an_error_not_an_empty_notebook(self, tmp_path):
        script = tmp_path / "bad.py"
        script.write_text("syntax error (\n", encoding="utf-8")
        with pytest.raises(SyntaxError):
            rn.script_to_notebook_robust(script, tmp_path / "bad.ipynb")
        assert not (tmp_path / "bad.ipynb").exists()

    def test_cli_exit_code_nonzero_on_failure(self, tmp_path, monkeypatch):
        scripts = tmp_path / "proj" / "scripts"
        scripts.mkdir(parents=True)
        (scripts / "ok.py").write_text("x = 1\n", encoding="utf-8")
        (scripts / "bad.py").write_text("syntax error (\n", encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["regenerate_notebooks.py", str(scripts)])
        with pytest.raises(SystemExit) as exc:
            rn.main()
        assert exc.value.code == 1
        assert (tmp_path / "proj" / "notebooks" / "ok.ipynb").exists()

    def test_output_dir_for_scripts_dir_without_trailing_component(
        self, tmp_path, monkeypatch
    ):
        scripts = tmp_path / "proj" / "scripts"
        scripts.mkdir(parents=True)
        (scripts / "a.py").write_text("x = 1\n", encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["regenerate_notebooks.py", str(scripts)])
        rn.main()
        assert (tmp_path / "proj" / "notebooks" / "a.ipynb").exists()
        assert not (scripts / "a.ipynb").exists()

    def test_directory_without_scripts_component_is_rejected(
        self, tmp_path, monkeypatch
    ):
        d = tmp_path / "plain"
        d.mkdir()
        (d / "a.py").write_text("x = 1\n", encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["regenerate_notebooks.py", str(d)])
        with pytest.raises(SystemExit) as exc:
            rn.main()
        assert exc.value.code == 1
        assert not (d / "a.ipynb").exists()

    @pytest.mark.parametrize(
        "script", EXAMPLE_SCRIPTS, ids=lambda p: str(p.relative_to(REPO_ROOT))
    )
    def test_every_example_script_regenerates_to_compilable_cells(
        self, script, tmp_path
    ):
        out = rn.script_to_notebook_robust(script, tmp_path / "o.ipynb")
        nb = json.loads(out.read_text(encoding="utf-8"))
        _assert_cells_compile(nb)

        # Every top-level class/def survives, decorators included.
        tree = ast.parse(script.read_text(encoding="utf-8"))
        code_tree = ast.parse(_code(nb).replace("%matplotlib inline", ""))
        want = {
            n.name: len(n.decorator_list)
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and n.name != "main"
        }
        have = {
            n.name: len(n.decorator_list)
            for n in code_tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        assert want.items() <= have.items()


# ----------------------------------------------------------------------------
# configure_notebooks CLI
# ----------------------------------------------------------------------------


class TestConfigureNotebooksCli:
    @pytest.fixture
    def cli(self):
        pytest.importorskip("click")
        from click.testing import CliRunner
        from configure_notebooks import main

        return CliRunner(), main

    def _nb(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "cells": [
                        {
                            "cell_type": "code",
                            "execution_count": None,
                            "metadata": {},
                            "outputs": [],
                            "source": ["import matplotlib.pyplot as plt\n"],
                        }
                    ],
                    "metadata": {},
                    "nbformat": 4,
                    "nbformat_minor": 5,
                }
            ),
            encoding="utf-8",
        )

    def test_relative_dir_is_relative_to_cwd(self, cli, tmp_path, monkeypatch):
        runner, main = cli
        self._nb(tmp_path / "nbs" / "a.ipynb")
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(main, ["--dir", "nbs", "--dry-run"])
        assert result.exit_code == 0, result.output
        assert "Directory not found" not in result.output

    def test_default_dir_is_repo_examples_notebooks(self, cli, tmp_path, monkeypatch):
        runner, main = cli
        monkeypatch.chdir(tmp_path)  # not the repo root
        result = runner.invoke(main, ["--dry-run"])
        assert result.exit_code == 0, result.output
        assert str(REPO_ROOT / "examples" / "notebooks") in result.output


# ----------------------------------------------------------------------------
# run_scripts / run_notebooks helpers
# ----------------------------------------------------------------------------


class TestRunnerHelpers:
    def test_script_ids_are_unique_for_same_named_examples(self):
        import run_scripts

        a = REPO_ROOT / "examples/scripts/04_gallery/physics/radioactive_decay.py"
        b = (
            REPO_ROOT
            / "examples/scripts/09_gallery_advanced/physics/radioactive_decay.py"
        )
        os_cwd = os.getcwd()
        try:
            os.chdir(REPO_ROOT)
            assert run_scripts._script_id(a) != run_scripts._script_id(b)
            assert run_scripts._script_id(a).startswith("04_gallery__")
        finally:
            os.chdir(os_cwd)

    def test_script_outside_examples_falls_back_to_name(self, tmp_path):
        import run_scripts

        assert run_scripts._script_id(tmp_path / "x.py") == "x"

    def test_notebook_ids_and_outside_path_do_not_raise(self, tmp_path, monkeypatch):
        pytest.importorskip("nbclient")
        import run_notebooks

        monkeypatch.chdir(REPO_ROOT)
        nb = tmp_path / "other.ipynb"
        assert run_notebooks._notebook_id(nb) == "other"
        monkeypatch.chdir(tmp_path)
        assert run_notebooks._relative_to_notebooks(nb) == Path("other.ipynb")

    def test_kernel_env_points_at_existing_sitecustomize_dir(self):
        pytest.importorskip("nbclient")
        import run_notebooks

        entries = run_notebooks._kernel_env()["PYTHONPATH"].split(os.pathsep)
        sc = [e for e in entries if e.endswith("notebook_sitecustomize")]
        assert sc and (Path(sc[0]) / "sitecustomize.py").is_file()
        assert (Path(entries[0]) / "io_patch.py").is_file()


# ----------------------------------------------------------------------------
# quick_sitecustomize: library callers get the size they asked for
# ----------------------------------------------------------------------------


_QUICK_CODE = """
import numpy as np
import jax.numpy as jnp

CAP = 10
space = ("linspace", "logspace", "geomspace")

# example code (__main__) is capped for every patched function
for mod in (np, jnp):
    for name in space:
        n = len(getattr(mod, name)(1, 2, 100))
        assert n == CAP, (mod.__name__, name, n)
rng = np.random
for call in (
    lambda: rng.normal(0, 1, 100),
    lambda: rng.normal(size=100),
    lambda: rng.uniform(0, 1, 100),
    lambda: rng.random(100),
    lambda: rng.random_sample(100),
    lambda: rng.rand(100),
    lambda: rng.randn(100),
):
    assert len(call()) == CAP

# library callers get exactly the size they asked for
for lib in ("numpy._x", "jax._x", "scipy._x", "matplotlib._x"):
    g = {"__name__": lib, "np": np, "jnp": jnp}
    exec("sizes = [len(f(1, 2, 100)) for f in (np.linspace, np.logspace,"
         " np.geomspace, jnp.linspace, jnp.logspace, jnp.geomspace)]", g)
    assert g["sizes"] == [100] * 6, (lib, g["sizes"])
# ... including through a real numpy-internal path
assert len(np.histogram(np.arange(100.), bins=50)[1]) == 51
"""


@pytest.mark.serial  # subprocess imports JAX (see CLAUDE.md on parallel JAX inits)
def test_quick_mode_caps_example_code_but_not_library_callers(tmp_path):
    env = {
        **os.environ,
        "NLSQ_EXAMPLES_QUICK": "1",
        "NLSQ_EXAMPLES_MAX_SAMPLES": "10",
        "JAX_PLATFORMS": "cpu",
        "MPLCONFIGDIR": str(tmp_path),
        "PYTHONPATH": str(SCRIPTS_DIR / "quick_sitecustomize"),
    }
    result = subprocess.run(
        [sys.executable, "-c", _QUICK_CODE],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _load_sitecustomize(monkeypatch):
    import importlib.util

    monkeypatch.delenv("NLSQ_EXAMPLES_QUICK", raising=False)  # no patches on import
    spec = importlib.util.spec_from_file_location(
        "_quick_sitecustomize_under_test",
        SCRIPTS_DIR / "quick_sitecustomize" / "sitecustomize.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, False),
        ("", False),
        ("0", False),
        ("False", False),
        (" no ", False),
        ("OFF", False),
        ("1", True),
        ("true", True),
    ],
)
def test_quick_flag_parsing(monkeypatch, value, expected):
    mod = _load_sitecustomize(monkeypatch)
    if value is not None:
        monkeypatch.setenv("NLSQ_EXAMPLES_QUICK", value)
    assert mod._quick_enabled() is expected


@pytest.mark.parametrize(
    "value, expected",
    [("10", 10), ("1", 1), ("0", 5000), ("-3", 5000), ("abc", 5000), (None, 5000)],
)
def test_max_samples_rejects_non_positive(monkeypatch, value, expected):
    mod = _load_sitecustomize(monkeypatch)
    if value is None:
        monkeypatch.delenv("NLSQ_EXAMPLES_MAX_SAMPLES", raising=False)
    else:
        monkeypatch.setenv("NLSQ_EXAMPLES_MAX_SAMPLES", value)
    assert mod._get_max_samples() == expected


# ----------------------------------------------------------------------------
# Text I/O in the tooling must name its encoding (cp1252 on Windows otherwise)
# ----------------------------------------------------------------------------


def _tooling_files() -> list[Path]:
    return sorted(
        p
        for d in ("notebooks", "notebook_utils")
        for p in (SCRIPTS_DIR / d).rglob("*.py")
    )


@pytest.mark.parametrize(
    "path", _tooling_files(), ids=lambda p: str(p.relative_to(SCRIPTS_DIR))
)
def test_text_io_specifies_encoding(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    missing = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        kwargs = {k.arg for k in node.keywords}
        if isinstance(func, ast.Name) and func.id == "open":
            mode = ""
            if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                mode = str(node.args[1].value)
            for k in node.keywords:
                if k.arg == "mode" and isinstance(k.value, ast.Constant):
                    mode = str(k.value.value)
            # skip binary mode and file-descriptor opens handled by name
            if "b" not in mode and "encoding" not in kwargs:
                missing.append(node.lineno)
        elif isinstance(func, ast.Attribute) and func.attr in (
            "read_text",
            "write_text",
        ):
            if "encoding" not in kwargs:
                missing.append(node.lineno)
    assert not missing, f"{path.name}: text I/O without encoding= at lines {missing}"


class TestBannerScriptToNotebook:
    SRC = (
        '"""Doc."""\n'
        "# " + "=" * 70 + "\n# # Title\n#\n# Prose here.\n# " + "=" * 70 + "\n"
        "import matplotlib.pyplot as plt\n"
        "fig_dir = Path(__file__).parent / 'figures'\n"
        "plt.plot([1])\n"
        "plt.savefig(fig_dir / 'f.png', dpi=300)\n"
        "plt.close()\n"
    )

    def test_prose_becomes_markdown_and_figures_show_inline(self, tmp_path):
        script = tmp_path / "s.py"
        script.write_text(self.SRC, encoding="utf-8")
        out = rn.banner_script_to_notebook(script, tmp_path / "s.ipynb")
        nb = json.loads(out.read_text(encoding="utf-8"))
        kinds = [c["cell_type"] for c in nb["cells"]]
        assert kinds == ["markdown", "code"]
        assert "".join(nb["cells"][0]["source"]).startswith("# Title")
        code = "".join(nb["cells"][1]["source"])
        assert "savefig" not in code and "fig_dir" not in code
        assert "plt.show()" in code
        assert [c["id"] for c in nb["cells"]] == ["cell-000", "cell-001"]

    def test_cli_flag_selects_banner_mode(self, tmp_path, monkeypatch):
        scripts = tmp_path / "proj" / "scripts"
        scripts.mkdir(parents=True)
        (scripts / "s.py").write_text(self.SRC, encoding="utf-8")
        monkeypatch.setattr(
            sys, "argv", ["regenerate_notebooks.py", str(scripts), "--banners"]
        )
        rn.main()
        nb = json.loads((tmp_path / "proj/notebooks/s.ipynb").read_text("utf-8"))
        assert nb["cells"][0]["cell_type"] == "markdown"

    def test_quick_exit_guard_skips_later_cells_instead_of_failing(
        self, tmp_path, monkeypatch
    ):
        src = (
            "import os, sys\n"
            'QUICK = os.environ.get("NLSQ_EXAMPLES_QUICK") == "1"\n'
            "if QUICK:\n"
            '    print("quick")\n'
            "    sys.exit(0)\n"
            "# " + "=" * 70 + "\n# more prose\n# " + "=" * 70 + "\n"
            "raise RuntimeError('must be skipped in quick mode')\n"
        )
        script = tmp_path / "s.py"
        script.write_text(src, encoding="utf-8")
        out = rn.banner_script_to_notebook(script, tmp_path / "s.ipynb")
        nb = json.loads(out.read_text(encoding="utf-8"))
        code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        assert "sys.exit" not in code[0]
        assert "input_transformers_cleanup" in code[0]

        def run_all(quick: str | None):
            from IPython.core.interactiveshell import InteractiveShell

            if quick is None:
                monkeypatch.delenv("NLSQ_EXAMPLES_QUICK", raising=False)
            else:
                monkeypatch.setenv("NLSQ_EXAMPLES_QUICK", quick)
            InteractiveShell.clear_instance()  # the hook mutates the shell
            shell = InteractiveShell.instance()
            try:
                return [shell.run_cell(c).error_in_exec for c in code]
            finally:
                InteractiveShell.clear_instance()

        pytest.importorskip("IPython")
        assert run_all("1") == [None, None]  # later cell skipped, nothing fails
        errors = run_all(None)  # guard is not vacuously always on
        assert errors[0] is None and isinstance(errors[1], RuntimeError)
