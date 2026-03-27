---
sidebar_label: Set an initial guess
sidebar_position: 3
---

# Set an Initial Guess

A well-chosen initial guess can be the difference between convergence and failure, especially for nonlinear and trajectory optimization problems. Amigo provides a flexible interface for setting initial values on any variable in the model.

## The ModelVector Interface

After calling `model.initialize()`, you can create a vector that maps to all design variables in the model. This vector supports string-based indexing using component and variable names.

```python
# Create a writable vector for the initial guess
x = model.create_vector()
```

### Setting Scalar Values

```python
x["comp.x"] = 5.0
```

### Setting Vector Values

```python
import numpy as np

# Set an entire vector variable
x["comp.forces"] = [1.0, 2.0, 3.0]

# Set with a NumPy array
x["comp.forces"] = np.array([1.0, 2.0, 3.0])
```

### Setting Values Across Instances

When a component has multiple instances, the first index in the brackets corresponds to the instance number:

```python
# Set the state vector at instance 5
x["dynamics.q[5, :]"] = [1.0, 0.0, 0.0, 0.0]

# Set all instances of a scalar variable
x["dynamics.u[:]"] = np.linspace(0.0, 1.0, num_steps)

# Set a specific element across all instances
x["dynamics.q[:, 0]"] = np.linspace(0.0, 10.0, num_steps)
```

### Reading Values Back

The same indexing syntax works for reading:

```python
current_values = x["comp.x"]
all_states = x["dynamics.q[:, :]"]
```

## Passing the Initial Guess to the Optimizer

The initial guess vector is passed directly to the `Optimizer` constructor:

```python
opt = am.Optimizer(model, x)
opt.optimize()
```

You can also pass custom bounds at the same time:

```python
lower = model.create_vector()
upper = model.create_vector()

lower["comp.x"] = -10.0
upper["comp.x"] = 10.0

opt = am.Optimizer(model, x, lower=lower, upper=upper)
```

:::note

If no initial guess is provided, Amigo uses the `value` specified in `add_input()` for each variable. If no bounds are provided, it uses the `lower` and `upper` specified in `add_input()`.

:::

## Strategies for Good Initial Guesses

### Linear Interpolation

For trajectory optimization, a linear interpolation between boundary conditions is often a good starting point:

```python
N = 101  # Number of time steps

# Interpolate between initial and final states
h_init = np.linspace(h0, hf, N)
v_init = np.linspace(v0, vf, N)

x["dynamics.q[:, 0]"] = h_init
x["dynamics.q[:, 1]"] = v_init
```

### Physical Reasoning

Use domain knowledge to construct physically meaningful guesses. For example, in a spacecraft re-entry problem, initial control angles might follow a reasonable profile:

```python
# Angle of attack ramps from 30 to 10 degrees
alpha_init = np.radians(np.linspace(30.0, 10.0, N))
x["shuttle.u[:, 0]"] = alpha_init
```

### Using Metadata Defaults

If `value` was set in `add_input()`, you can retrieve a vector populated with those defaults:

```python
x = model.get_values_from_meta("value")
```

This creates a `ModelVector` with every variable set to its declared default value.

### Warm-Starting from a Previous Solution

After solving a simpler version of the problem (e.g., fewer time steps or relaxed constraints), you can use that solution as an initial guess for the harder problem:

```python
# Solve a coarse problem first
opt_coarse = am.Optimizer(model_coarse, x_coarse)
opt_coarse.optimize()

# Interpolate the coarse solution onto the fine grid
x_fine["dynamics.q[:, 0]"] = np.interp(t_fine, t_coarse, x_coarse["dynamics.q[:, 0]"])
```

## Setting Custom Bounds

Variable bounds can be set per-component and per-instance, just like the initial guess:

```python
lower = model.create_vector()
upper = model.create_vector()

# Set bounds for all instances
lower["dynamics.q"] = -float("inf")
upper["dynamics.q"] = float("inf")

# Set bounds for a specific variable
lower["dynamics.u[:, 0]"] = np.radians(-90.0)
upper["dynamics.u[:, 0]"] = np.radians(90.0)

# Constrain a slack variable to [0, 1]
lower["heat.slack"] = 0.0
upper["heat.slack"] = 1.0
```

:::tip

For variables with no physical upper or lower limit, use `float("inf")` and `-float("inf")` respectively. Avoid using excessively large numerical values (e.g., `1e20`) as bounds, since these can affect the scaling of the problem.

:::

## Complete Example

A full initial guess setup for a trajectory optimization problem:

```python
import amigo as am
import numpy as np

# After model.build_module() and model.initialize()
N = 101

# Create vectors
x = model.create_vector()
lower = model.create_vector()
upper = model.create_vector()

# States: linear interpolation between boundary conditions
x["dynamics.q[:, 0]"] = np.linspace(260000.0, 80000.0, N)   # altitude
x["dynamics.q[:, 1]"] = np.zeros(N)                           # longitude
x["dynamics.q[:, 2]"] = np.zeros(N)                           # latitude
x["dynamics.q[:, 3]"] = np.linspace(25600.0, 2500.0, N)      # velocity

# Controls: physically motivated profile
x["dynamics.u[:, 0]"] = np.radians(np.linspace(30.0, 10.0, N))
x["dynamics.u[:, 1]"] = np.radians(-75.0)

# Bounds
lower["dynamics.u[:, 0]"] = np.radians(-90.0)
upper["dynamics.u[:, 0]"] = np.radians(90.0)
lower["dynamics.u[:, 1]"] = np.radians(-89.0)
upper["dynamics.u[:, 1]"] = np.radians(1.0)

# Solve
opt = am.Optimizer(model, x, lower=lower, upper=upper)
opt.optimize()
```

:::warning

A poor initial guess can cause the optimizer to converge to a local minimum or fail to converge entirely. When in doubt, start from a physically reasonable state and use conservative control inputs.

:::
