import os
import sys
from pathlib import Path

import matplotlib

# We don't import pyplot immediately to avoid backend locking if possible
# But we need to patch Figure and pyplot
import matplotlib.figure
import matplotlib.pyplot as plt


def patch_savefig():
    output_dir = os.environ.get("NLSQ_OUTPUT_DIR")
    if not output_dir:
        return

    def _safe_name(raw: str, fallback: str) -> str:
        # Path("..").name == ".." (not ""), so .name alone doesn't strip a
        # traversal token — reject it explicitly instead of trusting .name.
        name = Path(raw).name
        return name if name not in ("", "..", ".") else fallback

    # Sanitize to a bare filename: script_name feeds directly into the output
    # path below, and an env var containing ".." would escape output_dir.
    script_name = _safe_name(
        os.environ.get("NLSQ_CURRENT_SCRIPT", "unknown"), "unknown"
    )

    # We store the original methods
    _orig_fig_savefig = matplotlib.figure.Figure.savefig
    _orig_plt_savefig = plt.savefig

    def _resolve_target(fname):
        out_root = Path(output_dir)
        # Create a dedicated directory for artifacts of this script/notebook
        target_dir = out_root / "artifacts" / script_name
        resolved_target_dir = target_dir.resolve()

        # We try to preserve "figures/..." structure if present in fname
        # But fname could be absolute or relative
        p = Path(fname)

        # Resolve() collapses "..", "." and symlinks, so is_relative_to()
        # catches escapes a token-based filter can't (e.g. an fname made
        # entirely of ".." segments, or a symlink inside target_dir that
        # points outside it) — same idiom as
        # nlsq.cli.model_validation.validate_path().
        final_path = (target_dir / p).resolve()
        if not final_path.is_relative_to(resolved_target_dir):
            final_path = resolved_target_dir / _safe_name(fname, "figure.png")

        # Ensure directory exists
        final_path.parent.mkdir(parents=True, exist_ok=True)
        return final_path

    def _patched_fig_savefig(self, fname, *args, **kwargs):
        target_path = _resolve_target(fname)
        print(f"Redirecting figure {fname} to {target_path}")
        return _orig_fig_savefig(self, target_path, *args, **kwargs)

    def _patched_plt_savefig(fname, *args, **kwargs):
        target_path = _resolve_target(fname)
        print(f"Redirecting figure {fname} to {target_path}")
        return _orig_plt_savefig(target_path, *args, **kwargs)

    # Apply patches
    matplotlib.figure.Figure.savefig = _patched_fig_savefig
    plt.savefig = _patched_plt_savefig
    print(
        f"Patched matplotlib savefig to redirect to {output_dir}/artifacts/{script_name}"
    )
