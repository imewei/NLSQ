# Initialize JAX configuration through central config
from nlsq.config import JAXConfig

_jax_config = JAXConfig()

import jax.numpy as jnp
from jax import jit


class LossFunctionsJIT:
    """JIT-compiled robust loss functions for nonlinear least squares optimization.

    This class provides GPU/TPU-accelerated implementations of robust loss functions
    that reduce the influence of outliers in curve fitting. All loss functions are
    JIT-compiled for maximum performance and include analytical derivatives required
    for efficient optimization.

    Robust Loss Function Theory
    ----------------------------
    Standard least squares minimizes sum of squared residuals, making it sensitive
    to outliers. Robust loss functions ρ(z) replace z (squared residuals) with
    functions that grow more slowly for large residuals:

    Standard LS: min Σ f_i²
    Robust LS:   min Σ ρ(f_i²/σ²)

    where σ is the scaling parameter (f_scale) and z = (f/σ)².

    Available Loss Functions
    -------------------------
    1. **linear**: Standard least squares (ρ(z) = z)
       - No outlier protection
       - Fastest computation
       - Optimal for clean data without outliers

    2. **huber**: Huber loss function
       - ρ(z) = z if z ≤ 1, else 2√z - 1
       - Quadratic for small residuals, linear for large ones
       - Good balance between efficiency and robustness
       - Recommended for data with moderate outliers

    3. **soft_l1**: Soft L1 loss function
       - ρ(z) = 2(√(1+z) - 1)
       - Smooth approximation to L1 norm
       - More robust than Huber for severe outliers
       - Preserves differentiability everywhere

    4. **cauchy**: Cauchy (Lorentzian) loss function
       - ρ(z) = ln(1 + z)
       - Extremely robust to outliers
       - Can handle heavy-tailed error distributions
       - May converge slowly for well-behaved data

    5. **arctan**: Arctangent loss function
       - ρ(z) = arctan(z)
       - Bounded loss function
       - Very robust to extreme outliers
       - Useful for data with unknown error characteristics

    Mathematical Implementation
    ----------------------------
    Each loss function computes three quantities:
    - **ρ(z)**: Loss function value
    - **ρ'(z)**: First derivative for gradient computation
    - **ρ''(z)**: Second derivative for Hessian approximation

    The derivatives are used in the optimization algorithm:
    - Gradient: g = J^T (rho'(z) ⊙ f)
    - Hessian: H ≈ J^T diag(rho'(z)) J + J^T diag(rho''(z) ⊙ f²) J

    Performance Characteristics
    ----------------------------
    - **JIT Compilation**: All functions compiled for GPU/TPU acceleration
    - **Vectorized Operations**: Efficient batch processing of residuals
    - **Memory Optimization**: In-place operations where possible
    - **Numerical Stability**: Careful handling of edge cases and overflow

    Usage Example
    -------------
    ::

        from nlsq.loss_functions import LossFunctionsJIT

        # Initialize loss function handler
        loss_jit = LossFunctionsJIT()

        # Get robust loss function
        huber_loss = loss_jit.get_loss_function('huber')

        # Apply to residuals
        residuals = jnp.array([0.1, 5.0, 0.2, 10.0])  # Contains outliers
        f_scale = 1.0
        data_mask = jnp.ones_like(residuals, dtype=bool)

        # Compute loss with derivatives
        rho = huber_loss(residuals, f_scale, data_mask, cost_only=False)
        # rho[0] = loss values, rho[1] = first derivatives, rho[2] = second derivatives

        # Compute total cost only
        cost = huber_loss(residuals, f_scale, data_mask, cost_only=True)

    Loss Function Selection Guidelines
    -----------------------------------
    - **Clean Data**: Use 'linear' for maximum efficiency
    - **Few Outliers**: Use 'huber' for balanced robustness
    - **Many Outliers**: Use 'soft_l1' or 'cauchy'
    - **Unknown Data Quality**: Start with 'huber', upgrade if needed
    - **Extreme Outliers**: Use 'cauchy' or 'arctan'

    Scale Parameter (f_scale)
    --------------------------
    The scale parameter σ (f_scale) determines the transition point between
    quadratic and robust behavior:
    - **Too Small**: All residuals treated as outliers
    - **Too Large**: No outlier protection
    - **Optimal**: ~median absolute residual or robust MAD estimate
    - **Adaptive**: Can be estimated during optimization

    Technical Implementation Details
    --------------------------------
    - All functions handle both scalar and vector inputs
    - Derivatives computed analytically for accuracy
    - Special handling for z=0 to avoid numerical issues
    - Efficient masking for missing data points
    - Compatible with JAX transformations (grad, jit, vmap)
    """

    def __init__(self):
        self.stack_rhos = self.create_stack_rhos()

        self.create_huber_funcs()
        self.create_soft_l1_funcs()
        self.create_cauchy_funcs()
        self.create_arctan_funcs()

        self.IMPLEMENTED_LOSSES = {
            "linear": None,
            "huber": self.huber,
            "soft_l1": self.soft_l1,
            "cauchy": self.cauchy,
            "arctan": self.arctan,
        }
        self.loss_funcs = self.construct_all_loss_functions()

        self.create_zscale()
        self.create_calculate_cost()
        self.create_scale_rhos()

    def create_stack_rhos(self):
        @jit
        def stack_rhos(rho0, rho1, rho2):
            return jnp.stack([rho0, rho1, rho2])

        return stack_rhos

    def get_empty_rhos(self, z):
        dlength = len(z)
        rho1 = jnp.zeros([dlength])
        rho2 = jnp.zeros([dlength])
        return rho1, rho2

    def create_huber_funcs(self):
        @jit
        def huber1(z):
            mask = z <= 1

            return jnp.where(mask, z, 2 * z**0.5 - 1), mask

        @jit
        def huber2(z, mask):
            rho1 = jnp.where(mask, 1, z**-0.5)
            rho2 = jnp.where(mask, 0, -0.5 * z**-1.5)
            return rho1, rho2

        self.huber1 = huber1
        self.huber2 = huber2

    def huber(self, z, cost_only):
        rho0, mask = self.huber1(z)
        if cost_only:
            rho1, rho2 = self.get_empty_rhos(z)
        else:
            rho1, rho2 = self.huber2(z, mask)
        return self.stack_rhos(rho0, rho1, rho2)

    def create_soft_l1_funcs(self):
        @jit
        def soft_l1_1(z):
            t = 1 + z
            return 2 * (t**0.5 - 1), t

        @jit
        def soft_l1_2(t):
            rho1 = t**-0.5
            rho2 = -0.5 * t**-1.5
            return rho1, rho2

        self.soft_l1_1 = soft_l1_1
        self.soft_l1_2 = soft_l1_2

    def soft_l1(self, z, cost_only):
        rho0, t = self.soft_l1_1(z)
        if cost_only:
            rho1, rho2 = self.get_empty_rhos(z)
        else:
            rho1, rho2 = self.soft_l1_2(t)
        return self.stack_rhos(rho0, rho1, rho2)

    def create_cauchy_funcs(self):
        @jit
        def cauchy1(z):
            return jnp.log1p(z)

        @jit
        def cauchy2(z):
            t = 1 + z
            rho1 = 1 / t
            rho2 = -1 / t**2
            return rho1, rho2

        self.cauchy1 = cauchy1
        self.cauchy2 = cauchy2

    def cauchy(self, z, cost_only):
        rho0 = self.cauchy1(z)
        if cost_only:
            rho1, rho2 = self.get_empty_rhos(z)
        else:
            rho1, rho2 = self.cauchy2(z)
        return self.stack_rhos(rho0, rho1, rho2)

    def create_arctan_funcs(self):
        @jit
        def arctan1(z):
            return jnp.arctan(z)

        @jit
        def arctan2(z):
            t = 1 + z**2
            return 1 / t, -2 * z / t**2

        self.arctan1 = arctan1
        self.arctan2 = arctan2

    def arctan(self, z, cost_only):
        rho0 = self.arctan1(z)
        if cost_only:
            rho1, rho2 = self.get_empty_rhos(z)
        else:
            rho1, rho2 = self.arctan2(z)
        return self.stack_rhos(rho0, rho1, rho2)

    def create_zscale(self):
        @jit
        def zscale(f, f_scale):
            return (f / f_scale) ** 2

        self.zscale = zscale

    def create_calculate_cost(self):
        @jit
        def calculate_cost(f_scale, rho, data_mask):
            cost_array = jnp.where(data_mask, rho[0], 0)
            return 0.5 * f_scale**2 * jnp.sum(cost_array)

        self.calculate_cost = calculate_cost

    def create_scale_rhos(self):
        @jit
        def scale_rhos(rho, f_scale):
            rho0 = rho[0] * f_scale**2
            rho1 = rho[1]
            rho2 = rho[2] / f_scale**2
            return self.stack_rhos(rho0, rho1, rho2)

        self.scale_rhos = scale_rhos

    def construct_single_loss_function(self, loss):
        def loss_function(f, f_scale, data_mask=None, cost_only=False):
            z = self.zscale(f, f_scale)
            rho = loss(z, cost_only=cost_only)
            if cost_only:
                return self.calculate_cost(f_scale, rho, data_mask)
            rho = self.scale_rhos(rho, f_scale)
            return rho

        return loss_function

    def construct_all_loss_functions(self):
        loss_funcs = {}
        for key, loss in self.IMPLEMENTED_LOSSES.items():
            loss_funcs[key] = self.construct_single_loss_function(loss)

        return loss_funcs

    def get_loss_function(self, loss):
        if loss == "linear":
            return None

        if not callable(loss):
            return self.loss_funcs[loss]
        else:

            def loss_function(f, f_scale, data_mask=None, cost_only=False):
                z = self.zscale(f, f_scale)
                rho = loss(z)
                if cost_only:
                    return self.calculate_cost(f_scale, rho, data_mask)
                rho = self.scale_rhos(rho, f_scale)
                return rho

        return loss_function
