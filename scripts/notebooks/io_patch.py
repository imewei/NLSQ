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

    script_name = os.environ.get("NLSQ_CURRENT_SCRIPT", "unknown")

    # We store the original methods
    _orig_fig_savefig = matplotlib.figure.Figure.savefig
    _orig_plt_savefig = plt.savefig

    def _resolve_target(fname):
        out_root = Path(output_dir)
        # Create a dedicated directory for artifacts of this script/notebook
        target_dir = out_root / "artifacts" / script_name

        # We try to preserve "figures/..." structure if present in fname
        # But fname could be absolute or relative
        p = Path(fname)

        # If it's absolute, we extract the name
        # If relative, we use it as is
        if p.is_absolute():
            rel_path = p.name
        else:
            rel_path = p

        # If users do "figures/fig1.png", we get "artifacts/script_name/figures/fig1.png"
        final_path = target_dir / rel_path

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
