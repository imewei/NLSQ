"""Notebook transformations using Strategy pattern.

This package provides individual transformation classes that can be composed
into processing pipelines. Each transformer implements a specific modification
to notebook cells.
"""

from .base import NotebookTransformer
from .imports import IPythonDisplayImportTransformer
from .matplotlib import MatplotlibInlineTransformer
from .plt_show import PltShowReplacementTransformer

__all__ = [
    "IPythonDisplayImportTransformer",
    "MatplotlibInlineTransformer",
    "NotebookTransformer",
    "PltShowReplacementTransformer",
]
