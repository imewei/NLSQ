Architecture Overview
=====================

Understanding NLSQ's architecture helps you use advanced features effectively
and extend the library for custom needs.

.. toctree::
   :maxdepth: 1

   overview
   jax_patterns

Chapter Overview
----------------

**Overview** (10 min)
   Package structure and module organization.

**JAX Patterns** (10 min)
   JIT compilation, autodiff, and GPU acceleration patterns.

Package Structure
-----------------

.. code-block:: text

   nlsq/
   ├── core/           # Core optimization algorithms
   │   ├── minpack.py         # fit(), curve_fit(), CurveFit
   │   ├── least_squares.py   # LeastSquares orchestrator
   │   └── trf.py             # Trust Region Reflective
   ├── streaming/      # Large dataset handling
   ├── caching/        # Performance optimization
   ├── stability/      # Numerical stability
   ├── facades/        # Lazy-loading wrappers
   └── gui_qt/         # Desktop application

Key Design Principles
---------------------

1. **Layered Architecture**: High-level API → Mid-level → Low-level
2. **Protocol-Based DI**: Loose coupling via interfaces
3. **Lazy Loading**: Minimize import time and memory
4. **Memory Awareness**: Automatic strategy selection
5. **JAX-First**: GPU acceleration by default
