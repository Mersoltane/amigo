---
sidebar_position: 4
---

# Solve a Problem

Once you have defined your optimization problem by creating components and building a model, you can solve it using the `Optimizer` class.

## Creating an Optimizer

The optimizer is initialized with a model and, optionally, an initial guess and bounds:

```python
import amigo as am

# Simplest form: uses default values from add_input()
opt = am.Optimizer(model)

# With a custom initial guess
x = model.create_vector()
x["comp.x"] = 5.0
opt = am.Optimizer(model, x)

# With custom initial guess and bounds
lower = model.create_vector()
upper = model.create_vector()
lower["comp.x"] = -10.0
upper["comp.x"] = 10.0
opt = am.Optimizer(model, x, lower=lower, upper=upper)
```

### Choosing a Linear Solver

The optimizer uses a direct linear solver for the KKT system. If Intel MKL is available, the PARDISO solver is used automatically. Otherwise, it falls back to SciPy's sparse LU factorization.

```python
# Explicit solver selection (optional)
from amigo.optimizer import PardisoSolver, DirectScipySolver

problem = model.get_problem()
solver = PardisoSolver(problem)
opt = am.Optimizer(model, x, solver=solver)
```

Available solvers:

| Solver | Description | Requirements |
|--------|-------------|--------------|
| `PardisoSolver` | Intel MKL PARDISO (LDL^T with inertia detection) | `pypardiso` |
| `DirectScipySolver` | SciPy sparse LU factorization | None (built-in) |
| `DirectCudaSolver` | NVIDIA cuDSS on GPU | CUDA build |
| `DirectPetscSolver` | PETSc distributed solver | `petsc4py` |

## Running Optimization

```python
opt.optimize()
```

This runs the interior-point optimization loop and updates the model with the optimal solution.

### With Options

Pass a dictionary of options to customize the solver behavior:

```python
data = opt.optimize({
    "max_iterations": 500,
    "convergence_tolerance": 1e-8,
    "barrier_strategy": "heuristic",
})
```

The `optimize()` method returns a dictionary containing convergence information and iteration history:

```python
data = opt.optimize()

print(f"Converged: {data['converged']}")
print(f"Iterations: {len(data['iterations'])}")
print(f"Final residual: {data['iterations'][-1]['residual']:.6e}")
```

## Optimizer Options Reference

### Core Options

| Option | Default | Description |
|--------|---------|-------------|
| `max_iterations` | `100` | Maximum number of optimization iterations |
| `convergence_tolerance` | `1e-8` | KKT residual tolerance for convergence |
| `initial_barrier_param` | `1.0` | Initial value of the barrier parameter $\mu$ |
| `fraction_to_boundary` | `0.95` | Base fraction-to-boundary parameter $\tau$ |

### Barrier Strategy

| Option | Default | Description |
|--------|---------|-------------|
| `barrier_strategy` | `"heuristic"` | Strategy for updating $\mu$: `"heuristic"`, `"monotone"`, or `"quality_function"` |
| `monotone_barrier_fraction` | `0.1` | Factor for barrier reduction in monotone mode |
| `heuristic_barrier_gamma` | `0.1` | $\gamma$ parameter for heuristic (LOQO) barrier |
| `heuristic_barrier_r` | `0.95` | $r$ parameter for heuristic barrier |
| `progress_based_barrier` | `True` | Only reduce $\mu$ when making sufficient progress |
| `barrier_progress_tol` | `10.0` | Tolerance factor for barrier subproblem progress |
| `verbose_barrier` | `False` | Print barrier parameter updates |

**Heuristic (LOQO)** is the default and works well for most problems. It reduces the barrier parameter based on the current complementarity and a nonlinear formula.

**Monotone** uses a fixed $\mu$ until the subproblem is solved to sufficient accuracy, then reduces $\mu$ by the `monotone_barrier_fraction` factor. This is more conservative but can be more robust for difficult problems.

**Quality function** implements the Nocedal-Wachter-Waltz (2009) algorithm, which adaptively chooses $\mu$ by solving a one-dimensional optimization problem. It provides safeguarded nonmonotone barrier reduction and is recommended for challenging nonlinear problems.

### Quality Function Options

These options only apply when `barrier_strategy` is set to `"quality_function"`:

| Option | Default | Description |
|--------|---------|-------------|
| `quality_function_sigma_max` | `1000.0` | Maximum centering parameter |
| `quality_function_golden_iters` | `12` | Golden section search iterations |
| `quality_function_kappa_free` | `0.9999` | Nonmonotone acceptance threshold |
| `quality_function_l_max` | `5` | Window size for nonmonotone history |
| `quality_function_norm_scaling` | `True` | Scale KKT components by element count |

### Line Search

| Option | Default | Description |
|--------|---------|-------------|
| `use_armijo_line_search` | `True` | Use Armijo sufficient decrease condition |
| `armijo_constant` | `1e-4` | Armijo constant $c$ for sufficient decrease |
| `max_line_search_iterations` | `10` | Maximum backtracking steps |
| `backtracking_factor` | `0.5` | Step length reduction factor per backtrack |
| `second_order_correction` | `True` | Apply second-order correction on rejected steps |

### Step Control

| Option | Default | Description |
|--------|---------|-------------|
| `adaptive_tau` | `True` | Use adaptive fraction-to-boundary rule: $\tau = \max(\tau_\text{min}, 1 - \mu)$ |
| `tau_min` | `0.99` | Minimum fraction-to-boundary value |
| `equal_primal_dual_step` | `False` | Force equal step lengths for primal and dual |
| `check_update_step` | `False` | Verify step direction validity |

### Regularization

| Option | Default | Description |
|--------|---------|-------------|
| `regularization_eps_x` | `1e-8` | Primal regularization $\epsilon_x$ |
| `regularization_eps_z` | `1e-8` | Dual regularization $\epsilon_z$ |
| `adaptive_regularization` | `True` | Increase regularization on factorization failure |
| `max_regularization` | `1e-2` | Maximum regularization value |
| `regularization_increase_factor` | `10.0` | Factor to increase regularization |

### Convexification

| Option | Default | Description |
|--------|---------|-------------|
| `block_psd_convexification` | `False` | Use block PSD projection for Hessian convexification |
| `block_psd_hub_threshold` | `50` | Hub degree threshold for block detection |
| `curvature_probe_convexification` | `False` | Use three-layer inertia regularization |
| `max_inertia_corrections` | `8` | Maximum refactorizations for inertia correction |

**Block PSD convexification** performs eigenvalue decomposition on blocks of the Hessian matrix and clips negative eigenvalues. This is recommended for trajectory optimization problems where the Hessian has a block-diagonal structure.

**Curvature probe convexification** uses a three-layer approach: (1) per-variable regularization from the Hessian diagonal, (2) targeted inertia correction on nonconvex primal variables, and (3) global inertia correction as a fallback.

:::tip

For nonlinear trajectory optimization, `block_psd_convexification: True` with `barrier_strategy: "quality_function"` is a strong combination.

:::

### Multiplier Initialization

| Option | Default | Description |
|--------|---------|-------------|
| `init_least_squares_multipliers` | `True` | Initialize multipliers via least-squares solve |
| `init_affine_step_multipliers` | `False` | Initialize multipliers via affine scaling step |

### Acceptable Convergence

| Option | Default | Description |
|--------|---------|-------------|
| `acceptable_tol` | `None` | Acceptable residual threshold. Default: 100x `convergence_tolerance` |
| `acceptable_iter` | `10` | Number of stagnating iterations before accepting |

When the optimizer detects that the residual is no longer decreasing (stagnation), it can accept the current point if the residual is below `acceptable_tol`. This prevents the optimizer from iterating indefinitely on problems with numerical noise near the solution.

### Advanced Options

| Option | Default | Description |
|--------|---------|-------------|
| `gamma_penalty` | `1e3` | Penalty parameter for constraint violations |
| `record_components` | `[]` | List of component names to record at each iteration |
| `continuation_control` | `None` | Continuation component control parameter |
| `zero_hessian_variables` | `[]` | Variable names with zero Hessian (linear variables) |
| `regularization_eps_x_zero_hessian` | `1.0` | Strong regularization for zero-Hessian variables |
| `nonconvex_constraints` | `[]` | Constraint names for selective dual regularization |
| `max_consecutive_rejections` | `5` | Rejected steps before barrier increase |
| `barrier_increase_factor` | `5.0` | Factor to increase barrier when stuck |
| `convex_eps_z_coeff` | `1.0` | Coefficient for dual regularization: $\epsilon_z = C_z \cdot \mu$ |

## Accessing Results

After optimization, extract the solution from the model:

```python
# Get optimal values by component and variable name
x_opt = model.get_input("comp.x")
f_opt = model.get_objective("comp.f")
constraint = model.get_constraint("comp.g")

print(f"Optimal x: {x_opt}")
print(f"Objective: {f_opt}")
print(f"Constraint: {constraint}")
```

For vector variables across multiple instances, use the initial guess vector (which is updated in-place after optimization):

```python
import numpy as np

# The x vector is updated in-place after opt.optimize()
q_solution = np.array(x["dynamics.q[:, 0]"])
u_solution = np.array(x["dynamics.u[:, 0]"])
```

:::note

Wrap results in `np.array()` when you need to perform NumPy operations on them (fancy indexing, broadcasting, etc.).

:::

## Iteration Output

Set `print_level` or inspect the returned data to monitor convergence:

```python
data = opt.optimize({"max_iterations": 200})

# Iteration history
for it in data["iterations"]:
    print(f"Iter {it['iteration']:3d}  "
          f"residual={it['residual']:.4e}  "
          f"mu={it['barrier_param']:.4e}  "
          f"alpha_x={it['alpha_x']:.4f}")
```

The iteration data dictionary contains:
- `residual`: KKT residual norm
- `barrier_param`: Current barrier parameter $\mu$
- `alpha_x`: Primal step length
- `alpha_z`: Dual step length
- `objective`: Current objective value
- `complementarity`: Complementarity measure

## Complete Workflow

```python
import amigo as am
import numpy as np

# 1. Define component
class MyComponent(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("x", value=0.0, lower=-10.0, upper=10.0)
        self.add_objective("f")
        self.add_constraint("g", upper=0.0)

    def compute(self):
        x = self.inputs["x"]
        self.objective["f"] = x * x
        self.constraints["g"] = x - 5.0

# 2. Build model
model = am.Model("optimization")
model.add_component("comp", 1, MyComponent())
model.build_module()
model.initialize()

# 3. Set initial guess and bounds
x = model.create_vector()
x["comp.x"] = 3.0

# 4. Solve
opt = am.Optimizer(model, x)
data = opt.optimize({
    "convergence_tolerance": 1e-10,
    "max_iterations": 200,
})

# 5. Extract solution
x_opt = model.get_input("comp.x")
f_opt = model.get_objective("comp.f")

print(f"Converged: {data['converged']}")
print(f"Solution: x = {x_opt:.6f}, f(x) = {f_opt:.6f}")
```

See the [Optimizer API documentation](../api/optimizer.md) for the complete class reference.
