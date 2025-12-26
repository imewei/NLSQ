# NLSQ GUI Development Guide

This guide covers best practices, common pitfalls, and code review checklists for developing the NLSQ Streamlit-based GUI.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Code Review Checklist](#code-review-checklist)
3. [Streamlit Gotchas](#streamlit-gotchas)
4. [State Management](#state-management)
5. [Error Handling](#error-handling)
6. [Testing Guidelines](#testing-guidelines)

---

(architecture-overview)=
## Architecture Overview

The GUI follows a layered architecture:

```
Pages (Streamlit UI)
    ↓
Adapters (Data transformation)
    ↓
Components (Reusable UI elements)
    ↓
State (SessionState management)
    ↓
Core NLSQ (minpack, fit, etc.)
```

### Directory Structure

```
nlsq/gui/
├── app.py                 # Main entry point
├── state.py               # SessionState dataclass
├── presets.py             # Fitting presets
├── pages/                 # Streamlit pages
│   ├── 1_Data_Loading.py
│   ├── 2_Model_Selection.py
│   ├── 3_Fitting_Options.py
│   ├── 4_Results.py
│   └── 5_Export.py
├── adapters/              # Bridge between GUI and core
│   ├── fit_adapter.py
│   ├── data_adapter.py
│   ├── model_adapter.py
│   └── export_adapter.py
├── components/            # Reusable UI components
│   ├── param_config.py
│   ├── live_cost_plot.py
│   └── plotly_*.py
└── utils/
    └── theme.py           # Theme management
```

---

(code-review-checklist)=
## Code Review Checklist

### State Management

- [ ] **Check for element-level None in lists**: Never assume a list is valid just because it's not `None`. Always check individual elements.

  ```python
  # BAD: Only checks if list exists
  if state.p0 is None:
      return False

  # GOOD: Also checks element values
  if state.p0 is None or all(v is None for v in state.p0):
      return False
  ```

- [ ] **Validate state before operations**: Check that required state fields are populated before using them.

- [ ] **Use type annotations**: All state fields should have proper type hints.

### Error Handling

- [ ] **Store errors in session state before `st.rerun()`**: Direct `st.error()` calls are lost on rerun.

  ```python
  # BAD: Error message lost after rerun
  try:
      result = run_operation()
  except Exception as e:
      st.error(f"Failed: {e}")  # Lost after st.rerun()
  st.rerun()

  # GOOD: Error persists in session state
  try:
      result = run_operation()
  except Exception as e:
      st.session_state.error = f"Failed: {e}"
  st.rerun()

  # Later in render:
  if st.session_state.error:
      st.error(st.session_state.error)
  ```

- [ ] **Return error messages from functions**: Functions should return errors rather than displaying them directly.

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

- [ ] **Apply dark theme CSS on all pages**: Call `apply_dark_theme_css()` at the start of each page's `main()` function.

- [ ] **Use unique keys for Streamlit widgets**: Avoid key conflicts across reruns.

- [ ] **Initialize session state variables**: Check and initialize state before use.

  ```python
  if "my_var" not in st.session_state:
      st.session_state.my_var = default_value
  ```

---

(streamlit-gotchas)=
## Streamlit Gotchas

### 1. `st.rerun()` Clears Displayed Messages

**Problem**: `st.error()`, `st.warning()`, and `st.success()` messages are cleared when `st.rerun()` is called.

**Solution**: Store messages in session state and display them on re-render.

```python
# Store
st.session_state.fit_error = "Fitting failed: convergence error"
st.rerun()

# Display (after rerun)
if st.session_state.fit_error:
    st.error(st.session_state.fit_error)
```

### 2. Multi-Page Apps Run Pages Independently

**Problem**: CSS, state, and configurations set in `app.py` don't automatically apply to other pages.

**Solution**: Each page must:
- Call shared initialization functions (e.g., `apply_dark_theme_css()`)
- Access shared state through `st.session_state`

### 3. Callbacks Execute Before Widget Rendering

**Problem**: Widget callbacks run during the rerun cycle before the UI is rendered.

**Solution**: Use session state flags to track operations, not return values from callbacks.

### 4. Widget Keys Must Be Unique

**Problem**: Duplicate keys cause Streamlit errors or unexpected behavior.

**Solution**: Include context in keys:
```python
st.text_input("Value", key=f"param_{param_name}_{index}")
```

### 5. File Uploaders Clear on Rerun

**Problem**: Uploaded files are cleared after any rerun.

**Solution**: Process and store file data immediately:
```python
if uploaded_file is not None:
    content = uploaded_file.read()
    st.session_state.file_content = content
```

---

(state-management)=
## State Management

### SessionState Design

The `SessionState` dataclass holds all workflow configuration. Key principles:

1. **Single source of truth**: All UI state flows through `st.session_state.nlsq_state`
2. **Type safety**: Use dataclass with type annotations
3. **Explicit defaults**: All fields have default values
4. **Copyable**: Implement `copy()` for state snapshots

### Auto p0 Handling

When `auto_p0=True`:
1. Check if model has `estimate_p0` method
2. Call `model.estimate_p0(xdata, ydata)` to get estimates
3. Fill in `None` values in `state.p0` with estimates
4. Preserve user-set values (non-None)
5. Log which values were filled

```python
if state.auto_p0 and has_auto_p0:
    estimated = model.estimate_p0(xdata, ydata)
    for i, v in enumerate(state.p0):
        if v is None:
            state.p0[i] = estimated[i]
```

---

(error-handling)=
## Error Handling

### Logging Configuration

Use structured logging with the `nlsq.gui.*` namespace:

```python
import logging

logger = logging.getLogger("nlsq.gui.fitting")

logger.info("Auto p0 applied | p0=%s", p0)
logger.warning("Estimation failed | error=%s", e)
```

### Error Return Pattern

Functions that can fail should return error messages:

```python
def run_fit(state: SessionState) -> str | None:
    """Execute fitting.

    Returns
    -------
    str | None
        Error message if failed, None if successful.
    """
    try:
        result = execute_fit(...)
        state.fit_result = result
        return None
    except Exception as e:
        return f"Fitting failed: {e}"
```

---

## Testing Guidelines

### Test Categories

1. **Unit tests**: Test individual functions in isolation
2. **Component tests**: Test UI components with mock data
3. **Integration tests**: Test full workflows end-to-end

### Regression Tests

Add regression tests for any bug fix:

```python
def test_p0_all_none_without_auto_p0_not_ready(self):
    """Regression: p0=[None, None, None] should NOT be ready without auto_p0.

    Bug: is_ready_to_fit only checked `if state.p0 is None` but didn't
    check if all elements in the list were None.
    """
    state.p0 = [None, None, None]
    state.auto_p0 = False

    all_none = all(v is None for v in state.p0)
    is_ready = not all_none or (state.auto_p0 and has_auto_p0)

    assert is_ready is False
```

### Test Naming Convention

Pattern: `test_<component>_<scenario>_<expected_behavior>`

Examples:
- `test_p0_all_none_without_auto_p0_not_ready`
- `test_auto_p0_fills_none_values`
- `test_bounds_conversion_handles_none`

---

## Common Patterns

### Safe State Access

```python
def get_session_state() -> SessionState:
    """Get or initialize session state."""
    if "nlsq_state" not in st.session_state:
        st.session_state.nlsq_state = initialize_state()
    return st.session_state.nlsq_state
```

### Conditional Readiness Check

```python
def is_ready_to_fit(state: SessionState) -> tuple[bool, str]:
    """Check if prerequisites are met.

    Returns (is_ready, message).
    """
    if state.xdata is None:
        return False, "Data not loaded"

    if state.p0 is None or all(v is None for v in state.p0):
        if not (state.auto_p0 and model_has_auto_p0):
            return False, "Initial parameters not set"

    return True, "Ready"
```

### Theme-Aware Pages

```python
def main() -> None:
    """Page entry point."""
    apply_dark_theme_css()  # Apply theme first
    init_page_state()  # Initialize page-specific state

    st.title("Page Title")
    # ... rest of page
```
