# NLSQ Qt GUI Development Guide

This guide covers best practices, common pitfalls, and code review checklists for developing the NLSQ Qt desktop GUI.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Code Review Checklist](#code-review-checklist)
3. [State Management](#state-management)
4. [Qt Best Practices](#qt-best-practices)
5. [Testing Guidelines](#testing-guidelines)

---

(architecture-overview)=
## Architecture Overview

The Qt GUI follows a layered architecture:

```
Pages (QWidget-based UI)
    ↓
AppState (Qt signals for reactivity)
    ↓
Adapters (Data transformation)
    ↓
SessionState (Plain dataclass)
    ↓
Core NLSQ (minpack, fit, etc.)
```

### Directory Structure

```
nlsq/gui_qt/
├── __init__.py            # run_desktop() entry point
├── main_window.py         # MainWindow with sidebar navigation
├── app_state.py           # AppState (Qt signals wrapping SessionState)
├── session_state.py       # SessionState dataclass
├── presets.py             # Fitting presets
├── theme.py               # ThemeConfig, ThemeManager
├── autosave.py            # AutosaveManager for crash recovery
├── pages/                 # 5-page workflow (QWidget-based)
│   ├── __init__.py
│   ├── data_loading.py
│   ├── model_selection.py
│   ├── fitting_options.py
│   ├── results.py
│   └── export.py
├── widgets/               # Reusable Qt widgets
│   ├── code_editor.py
│   ├── param_config.py
│   └── ...
├── plots/                 # pyqtgraph-based scientific plots
│   ├── fit_plot.py
│   ├── residuals_plot.py
│   └── live_cost_plot.py
└── adapters/              # Bridge between GUI and core
    ├── __init__.py
    ├── fit_adapter.py
    ├── data_adapter.py
    ├── model_adapter.py
    └── export_adapter.py
```

---

(code-review-checklist)=
## Code Review Checklist

### State Management

- [ ] **Check for element-level None in lists**: Never assume a list is valid just because it's not `None`.

  ```python
  # BAD: Only checks if list exists
  if state.p0 is None:
      return False

  # GOOD: Also checks element values
  if state.p0 is None or all(v is None for v in state.p0):
      return False
  ```

- [ ] **Use AppState for UI updates**: Changes should go through `AppState` methods to emit proper signals.

- [ ] **Connect signals to slots**: Ensure UI components react to state changes via Qt signals.

### Error Handling

- [ ] **Show errors via QMessageBox**: Use Qt's native dialogs for user feedback.

  ```python
  from PySide6.QtWidgets import QMessageBox

  QMessageBox.warning(self, "Error", f"Operation failed: {error}")
  ```

- [ ] **Log errors with context**: Use structured logging with relevant context.

  ```python
  logger.warning("Auto p0 estimation failed | error=%s, model=%s", e, model_name)
  ```

### Bounds and Parameters

- [ ] **Convert None bounds to ±inf**: Bounds arrays with `None` values should be converted before numpy operations.

  ```python
  lower = [-float("inf") if v is None else float(v) for v in bounds[0]]
  upper = [float("inf") if v is None else float(v) for v in bounds[1]]
  ```

- [ ] **Validate p0 length matches model parameters**: Ensure p0 has the correct number of elements.

### UI Components

- [ ] **Apply theme consistently**: Use ThemeManager for consistent styling.

- [ ] **Use lazy imports in pages**: Import adapters inside methods to avoid loading Qt at module import time.

  ```python
  def _on_button_click(self) -> None:
      from nlsq.gui_qt.adapters.fit_adapter import run_fit

      # ...
  ```

---

(state-management)=
## State Management

### SessionState vs AppState

The GUI uses two state classes:

1. **SessionState** (`session_state.py`): Plain dataclass holding all workflow configuration
2. **AppState** (`app_state.py`): Qt-observable wrapper that emits signals for reactive updates

Key principles:

1. **AppState wraps SessionState**: Access underlying state via `app_state.state`
2. **Use AppState methods for changes**: Call methods like `set_data()`, `set_model()` to emit signals
3. **Connect to signals for updates**: Pages connect to `data_changed`, `model_changed`, etc.

```python
class FittingOptionsPage(QWidget):
    def __init__(self, app_state: AppState) -> None:
        self._app_state = app_state
        # Connect to state changes
        app_state.data_changed.connect(self._on_data_changed)
        app_state.model_changed.connect(self._on_model_changed)
```

### PageState for Navigation Guards

The `PageState` class derives navigation permissions from `SessionState`:

```python
page_state = PageState.from_session_state(app_state.state)
if page_state.can_access("fitting_options"):
    # Enable fitting options page
    pass
```

---

(qt-best-practices)=
## Qt Best Practices

### 1. Use Signals and Slots

**Problem**: Direct method calls between widgets create tight coupling.

**Solution**: Use Qt's signal/slot mechanism for loose coupling.

```python
class FitWorker(QObject):
    finished = Signal(object)  # Emit result
    error = Signal(str)  # Emit error message

    def run(self) -> None:
        try:
            result = perform_fit()
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
```

### 2. Run Long Operations in Threads

**Problem**: Long operations block the UI.

**Solution**: Use `QThread` with worker objects.

```python
class FitWorker(QObject):
    finished = Signal(object)
    progress = Signal(int, float)

    def run(self) -> None:
        # Perform fit in background thread
        result = execute_fit(...)
        self.finished.emit(result)


# Usage
self._thread = QThread()
self._worker = FitWorker(state)
self._worker.moveToThread(self._thread)
self._thread.started.connect(self._worker.run)
self._worker.finished.connect(self._on_fit_complete)
self._thread.start()
```

### 3. Lazy Import Qt Dependencies

**Problem**: Importing PySide6/PyQt at module level slows down CLI usage.

**Solution**: Import Qt dependencies inside functions when possible.

```python
def __getattr__(name: str):
    """Lazy import pages to avoid importing Qt at module load time."""
    if name == "DataLoadingPage":
        from nlsq.gui_qt.pages.data_loading import DataLoadingPage

        return DataLoadingPage
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

### 4. Use pyqtgraph for Scientific Plots

**Problem**: Matplotlib is slow for large datasets.

**Solution**: Use pyqtgraph with OpenGL for GPU-accelerated plotting.

```python
from pyqtgraph import PlotWidget


class FitPlotWidget(PlotWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setBackground("w")
        self._data_scatter = self.plot([], [], pen=None, symbol="o")
        self._fit_curve = self.plot([], [], pen="r")
```

---

(testing-guidelines)=
## Testing Guidelines

### pytest-qt Integration

Use `pytest-qt` for testing Qt widgets:

```python
import pytest


@pytest.fixture
def app_state() -> AppState:
    """Create a fresh AppState instance for testing."""
    from nlsq.gui_qt.app_state import AppState

    return AppState()


def test_set_data_emits_signal(qtbot, app_state):
    """Test that set_data emits data_changed signal."""
    import numpy as np

    with qtbot.waitSignal(app_state.data_changed, timeout=1000):
        app_state.set_data(xdata=np.array([1, 2, 3]), ydata=np.array([1, 4, 9]))
```

### Test Categories

1. **Unit tests**: Test individual functions and state logic
2. **Widget tests**: Test UI components with pytest-qt
3. **Integration tests**: Test full workflows with qtbot

### Common Patterns

```python
def test_page_state_from_session_state(app_state):
    """Test PageState derivation from SessionState."""
    import numpy as np
    from nlsq.gui_qt.pages import PageState

    # Initially, data not loaded
    page_state = PageState.from_session_state(app_state.state)
    assert page_state.data_loaded is False
    assert page_state.can_access("model_selection") is False

    # After setting data
    app_state.set_data(np.array([1, 2]), np.array([1, 4]))
    page_state = PageState.from_session_state(app_state.state)
    assert page_state.data_loaded is True
    assert page_state.can_access("model_selection") is True
```

---

## Common Patterns

### Safe State Access

```python
def _get_param_names(self) -> list[str]:
    """Get parameter names from the model."""
    from nlsq.gui_qt.adapters.model_adapter import get_model_info

    state = self._app_state.state
    if state.model_func is not None:
        info = get_model_info(state.model_func)
        return info.get("param_names", [])
    return []
```

### Conditional Readiness Check

```python
def _is_ready_to_fit(self) -> tuple[bool, str]:
    """Check if prerequisites are met.

    Returns (is_ready, message).
    """
    state = self._app_state.state

    if state.xdata is None or state.ydata is None:
        return False, "Data not loaded"

    if state.model_func is None:
        return False, "Model not selected"

    if state.p0 is None or all(v is None for v in state.p0):
        if not state.auto_p0:
            return False, "Initial parameters not set"

    return True, "Ready"
```

### Theme-Aware Widgets

```python
class MyWidget(QWidget):
    def __init__(self, theme_config: ThemeConfig) -> None:
        super().__init__()
        self._theme = theme_config
        self._apply_theme()

    def _apply_theme(self) -> None:
        """Apply current theme styling."""
        self.setStyleSheet(
            f"""
            QWidget {{
                background-color: {self._theme.background};
                color: {self._theme.text};
            }}
        """
        )
```
