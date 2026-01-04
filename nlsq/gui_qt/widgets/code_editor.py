"""
NLSQ Qt GUI Code Editor Widget

This widget provides a Python code editor with syntax highlighting
for defining custom model functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QSyntaxHighlighter,
    QTextCharFormat,
)
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from PySide6.QtGui import QTextDocument

    from nlsq.gui_qt.theme import ThemeConfig

__all__ = ["CodeEditorWidget", "PythonHighlighter"]

# Python keywords for highlighting
PYTHON_KEYWORDS = {
    "and",
    "as",
    "assert",
    "async",
    "await",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "try",
    "while",
    "with",
    "yield",
    "True",
    "False",
    "None",
}

# Built-in functions
PYTHON_BUILTINS = {
    "abs",
    "all",
    "any",
    "bin",
    "bool",
    "bytes",
    "callable",
    "chr",
    "dict",
    "dir",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "getattr",
    "hasattr",
    "hash",
    "hex",
    "id",
    "input",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "object",
    "oct",
    "open",
    "ord",
    "pow",
    "print",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "setattr",
    "slice",
    "sorted",
    "str",
    "sum",
    "super",
    "tuple",
    "type",
    "vars",
    "zip",
}

# Scientific computing keywords
NUMPY_KEYWORDS = {
    "np",
    "jnp",
    "numpy",
    "jax",
    "exp",
    "log",
    "sin",
    "cos",
    "tan",
    "sqrt",
    "abs",
    "power",
    "array",
    "zeros",
    "ones",
    "linspace",
    "arange",
    "sum",
    "mean",
    "std",
    "where",
    "pi",
    "e",
}


class PythonHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for Python code."""

    def __init__(self, document: QTextDocument) -> None:
        """Initialize the syntax highlighter.

        Args:
            document: The QTextDocument to highlight
        """
        super().__init__(document)

        # Create text formats
        self._keyword_format = QTextCharFormat()
        self._keyword_format.setForeground(QColor("#569CD6"))  # Blue
        self._keyword_format.setFontWeight(QFont.Weight.Bold)

        self._builtin_format = QTextCharFormat()
        self._builtin_format.setForeground(QColor("#4EC9B0"))  # Cyan

        self._numpy_format = QTextCharFormat()
        self._numpy_format.setForeground(QColor("#DCDCAA"))  # Yellow

        self._string_format = QTextCharFormat()
        self._string_format.setForeground(QColor("#CE9178"))  # Orange

        self._comment_format = QTextCharFormat()
        self._comment_format.setForeground(QColor("#6A9955"))  # Green
        self._comment_format.setFontItalic(True)

        self._number_format = QTextCharFormat()
        self._number_format.setForeground(QColor("#B5CEA8"))  # Light green

        self._function_format = QTextCharFormat()
        self._function_format.setForeground(QColor("#DCDCAA"))  # Yellow

    def highlightBlock(self, text: str) -> None:
        """Apply highlighting to a block of text.

        Args:
            text: The text to highlight
        """
        # Highlight keywords
        self._highlight_words(text, PYTHON_KEYWORDS, self._keyword_format)

        # Highlight builtins
        self._highlight_words(text, PYTHON_BUILTINS, self._builtin_format)

        # Highlight numpy/jax keywords
        self._highlight_words(text, NUMPY_KEYWORDS, self._numpy_format)

        # Highlight function definitions
        self._highlight_function_defs(text)

        # Highlight numbers
        self._highlight_numbers(text)

        # Highlight strings (single and double quotes)
        self._highlight_strings(text)

        # Highlight comments (must be last to override other highlighting)
        self._highlight_comments(text)

    def _highlight_words(
        self, text: str, words: set[str], fmt: QTextCharFormat
    ) -> None:
        """Highlight specific words in the text."""
        import re

        for word in words:
            pattern = rf"\b{re.escape(word)}\b"
            for match in re.finditer(pattern, text):
                self.setFormat(match.start(), match.end() - match.start(), fmt)

    def _highlight_function_defs(self, text: str) -> None:
        """Highlight function definition names."""
        import re

        pattern = r"\bdef\s+(\w+)"
        for match in re.finditer(pattern, text):
            # Highlight the function name (group 1)
            start = match.start(1)
            length = len(match.group(1))
            self.setFormat(start, length, self._function_format)

    def _highlight_numbers(self, text: str) -> None:
        """Highlight numeric literals."""
        import re

        # Match integers, floats, and scientific notation
        pattern = r"\b\d+\.?\d*(?:[eE][+-]?\d+)?\b"
        for match in re.finditer(pattern, text):
            self.setFormat(
                match.start(), match.end() - match.start(), self._number_format
            )

    def _highlight_strings(self, text: str) -> None:
        """Highlight string literals."""
        import re

        # Single-quoted strings
        for match in re.finditer(r"'[^'\\]*(?:\\.[^'\\]*)*'", text):
            self.setFormat(
                match.start(), match.end() - match.start(), self._string_format
            )
        # Double-quoted strings
        for match in re.finditer(r'"[^"\\]*(?:\\.[^"\\]*)*"', text):
            self.setFormat(
                match.start(), match.end() - match.start(), self._string_format
            )

    def _highlight_comments(self, text: str) -> None:
        """Highlight comments."""
        import re

        for match in re.finditer(r"#.*$", text):
            self.setFormat(
                match.start(), match.end() - match.start(), self._comment_format
            )


class CodeEditorWidget(QWidget):
    """Python code editor with syntax highlighting.

    Provides:
    - Python syntax highlighting
    - Line numbers (simplified)
    - Syntax validation
    - Function name extraction
    """

    # Signals
    code_changed = Signal(str)
    validation_changed = Signal(bool, str)  # is_valid, error_message

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the code editor widget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Editor with monospace font
        editor_layout = QHBoxLayout()
        editor_layout.setSpacing(0)
        editor_layout.setContentsMargins(0, 0, 0, 0)

        self._editor = QPlainTextEdit()
        self._editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

        # Set monospace font
        font = QFont("Consolas, 'Courier New', monospace")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(10)
        self._editor.setFont(font)

        # Set placeholder text
        self._editor.setPlaceholderText(
            "import jax.numpy as jnp\n\n"
            "def model(x, a, b, c):\n"
            "    return a * jnp.exp(-b * x) + c"
        )

        # Apply syntax highlighter
        self._highlighter = PythonHighlighter(self._editor.document())

        editor_layout.addWidget(self._editor)
        layout.addLayout(editor_layout)

        # Validation status
        self._validation_label = QLabel("")
        self._validation_label.setWordWrap(True)
        layout.addWidget(self._validation_label)

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        self._editor.textChanged.connect(self._on_text_changed)

    def _on_text_changed(self) -> None:
        """Handle text changes."""
        code = self._editor.toPlainText()
        self.code_changed.emit(code)

        # Validate syntax
        is_valid, message = self.validate_syntax()
        self._update_validation_display(is_valid, message)
        self.validation_changed.emit(is_valid, message)

    def _update_validation_display(self, is_valid: bool, message: str) -> None:
        """Update the validation display.

        Args:
            is_valid: Whether the code is valid
            message: Validation message
        """
        if not self._editor.toPlainText().strip():
            self._validation_label.setText("")
            return

        if is_valid:
            funcs = self.get_function_names()
            if funcs:
                self._validation_label.setText(f"Functions found: {', '.join(funcs)}")
                self._validation_label.setStyleSheet("color: #4CAF50;")
            else:
                self._validation_label.setText("No functions defined")
                self._validation_label.setStyleSheet("color: #FF9800;")
        else:
            self._validation_label.setText(f"Error: {message}")
            self._validation_label.setStyleSheet("color: #f44336;")

    def get_code(self) -> str:
        """Get the current code.

        Returns:
            The code content
        """
        return self._editor.toPlainText()

    def set_code(self, code: str) -> None:
        """Set the code content.

        Args:
            code: The code to set
        """
        self._editor.setPlainText(code)

    def get_function_names(self) -> list[str]:
        """Get function names defined in the code.

        Returns:
            List of function names
        """
        from nlsq.gui_qt.adapters.model_adapter import list_functions_in_module

        code = self._editor.toPlainText()
        if not code.strip():
            return []

        return list_functions_in_module(code)

    def validate_syntax(self) -> tuple[bool, str]:
        """Validate the code syntax.

        Returns:
            Tuple of (is_valid, error_message)
        """
        import ast

        code = self._editor.toPlainText()
        if not code.strip():
            return True, ""

        try:
            ast.parse(code)
            return True, "Syntax valid"
        except SyntaxError as e:
            line = e.lineno if e.lineno else 1
            return False, f"Line {line}: {e.msg}"

    def set_theme(self, theme: ThemeConfig) -> None:
        """Apply theme to this widget.

        Args:
            theme: Theme configuration
        """
        # Update editor colors based on theme
        if theme.name == "dark":
            self._editor.setStyleSheet(
                f"background-color: {theme.surface}; "
                f"color: {theme.text_primary}; "
                "border: 1px solid #3d3d3d;"
            )
        else:
            self._editor.setStyleSheet(
                f"background-color: {theme.background}; "
                f"color: {theme.text_primary}; "
                "border: 1px solid #e0e0e0;"
            )

    def setReadOnly(self, read_only: bool) -> None:
        """Set read-only mode.

        Args:
            read_only: Whether to make the editor read-only
        """
        self._editor.setReadOnly(read_only)

    def setMaximumHeight(self, height: int) -> None:
        """Set maximum height.

        Args:
            height: Maximum height in pixels
        """
        self._editor.setMaximumHeight(height)
