# Interior-Point Method: From KKT to Solution

A detailed algorithmic note on the interior-point optimizer implemented in Amigo,
following the IPOPT framework of Wachter & Biegler (2006) with extensions for
barrier parameter selection via the quality function of Nocedal, Wachter & Waltz (2009).

## 1. The Nonlinear Program

We consider the general NLP:

```
min   f(x)
s.t.  h(x) = 0               (m_e equalities)
      c_l <= c(x) <= c_u     (m_i inequalities)
      l <= x <= u             (variable bounds)
```

where `x` is the vector of `n` design variables, `f: R^n -> R` is the objective,
`h: R^n -> R^{m_e}` are equality constraints, and `c: R^n -> R^{m_i}` are
inequality constraints with finite or infinite bounds.

## 2. Slack Reformulation

### 2.1 The Problem with Inequalities

The original NLP has three kinds of constraints:

```
h(x) = 0                   equalities  (straightforward for Newton)
c_l <= c(x) <= c_u         inequalities (problematic)
l <= x <= u                 simple bounds (handled by barriers)
```

Equalities are natural for Newton's method: we linearize `h(x) = 0` and solve.
Simple bounds are natural for barrier methods: we add `- mu * ln(x - l)` terms.

But inequalities are neither. Consider a single constraint `c(x) <= 5`:

- If `c(x) = 3.0`, the constraint is *inactive* with margin 2.0. Newton doesn't
  need to do anything.
- If `c(x) = 4.99`, it is *nearly active*. Newton must be careful not to overshoot.
- If `c(x) = 5.0`, it is *active*. Now it behaves like an equality.
- We don't know in advance which constraints will be active at the solution.

An interior-point method needs a smooth way to handle all three cases. Direct
logarithmic barriers on `c(x)` (i.e., `- mu * ln(c_u - c(x))`) would work in
theory, but they mix the nonlinear function `c(x)` into the barrier term, which
complicates the Hessian structure -- the barrier contribution would involve second
derivatives of `c(x)`, coupling the barrier curvature with constraint nonlinearity.

### 2.2 Introducing Slack Variables

The solution is to decouple the constraint function from the bound enforcement by
introducing a new variable `s_j` for each inequality:

**Before (original NLP):**
```
min   f(x)
s.t.  h(x) = 0                          (m_e equalities)
      c_l <= c(x) <= c_u                 (m_i inequalities)
      l <= x <= u                        (n variable bounds)
```

**After (slack reformulation):**
```
min   f(x)
s.t.  h(x) = 0                          (m_e equalities, unchanged)
      c(x) - s = 0                       (m_i NEW equalities)
      c_l <= s <= c_u                    (m_i slack bounds, simple!)
      l <= x <= u                        (n variable bounds, unchanged)
```

The key transformation for each inequality is:

```
BEFORE:  c_l <= c(x) <= c_u

AFTER:   c(x) = s          (equality: the slack tracks the constraint value)
         c_l <= s <= c_u    (simple bound: the slack carries the bounds)
```

This splits each inequality into two pieces:
1. An **equality** `c(x) - s = 0` that Newton's method handles naturally via
   Jacobian linearization.
2. A **simple bound** `c_l <= s <= c_u` that the barrier method handles naturally
   via logarithmic terms `- mu * ln(s - c_l) - mu * ln(c_u - s)`.

### 2.3 Concrete Example

Consider a tire friction circle constraint `F_x^2 + F_y^2 <= (mu * F_z)^2`,
written as `c(x) <= 1` after normalization.

**Before:** The constraint `c(x) <= 1` is an inequality. We don't know whether the
tire is at the friction limit or well within it.

**After:**
```
c(x) - s = 0       Equality: s always equals the friction utilization c(x)
0 <= s <= 1         Bound: s must be between 0 and 1
```

The barrier adds `- mu * ln(s) - mu * ln(1 - s)` to the objective, creating a
smooth force that keeps `s` strictly between 0 and 1. As `mu -> 0`, the barriers
weaken and `s` can approach 0 (no friction used) or 1 (at the friction limit).

At the solution, if the tire IS at the limit, `s -> 1` and the barrier multiplier
`z_su = mu/(1 - s)` grows large -- this is the Lagrange multiplier for the
friction constraint. If the tire is NOT at the limit, `s` is interior and both
multipliers `z_sl, z_su` approach zero.

### 2.4 What Changes in the KKT System

The slack reformulation adds to the KKT system:

**New variables:** `s` (m_i slack values), `z_sl` (m_i lower bound multipliers),
`z_su` (m_i upper bound multipliers), `lam_c` (m_i equality multipliers for
`c(x) - s = 0`).

**New equations:**
```
c(x) - s = 0                            (primal feasibility of new equalities)
(s - c_l) * z_sl = mu                   (complementarity for slack lower bounds)
(c_u - s) * z_su = mu                   (complementarity for slack upper bounds)
-lam_c - z_sl + z_su = 0                (stationarity w.r.t. s)
```

The last equation is the slack stationarity condition: differentiating the
Lagrangian with respect to `s` gives `-lam_c` (from `c(x) - s = 0`) minus
`z_sl` (from the lower barrier) plus `z_su` (from the upper barrier) equals zero.
This relates the constraint multiplier `lam_c` to the bound multipliers.

**Crucially:** The slack equations are *diagonal* -- each slack `s_j` only
interacts with its own multipliers `z_sl_j, z_su_j` and constraint multiplier
`lam_c_j`. There is no coupling between different slacks. This is what makes the
condensation in Section 6 possible: we can eliminate all m_i slack variables and
their 2*m_i dual variables analytically, without any matrix operations.

### 2.5 Why Not Just Use Inequality Barriers Directly?

One might ask: why not skip slacks and put barriers directly on the inequalities?

```
phi_direct = f(x) - mu * sum_j ln(c(x)_j - c_l_j) - mu * sum_j ln(c_u_j - c(x)_j)
```

This works mathematically, but the Hessian of `phi_direct` contains terms like:

```
d^2/dx^2 [- mu * ln(c(x) - c_l)] = mu * J_c^T diag(1/(c-c_l)^2) J_c
                                    - mu * diag(1/(c-c_l)) * d^2c/dx^2
```

The first term is a dense rank-m_i outer product through the Jacobian. The second
involves second derivatives of the constraints. Both destroy the sparsity that
makes large-scale NLP solvers efficient.

With slacks, the barrier Hessian contribution is purely diagonal:

```
d^2/ds^2 [- mu * ln(s - c_l)] = mu / (s - c_l)^2
```

This adds only to the diagonal of the slack block, preserving sparsity perfectly.
The constraint nonlinearity stays in the equality `c(x) - s = 0`, where it belongs
(in the Jacobian and the Lagrangian Hessian `W`, which is already sparse).

## 3. Barrier Subproblem

The bound constraints are replaced by logarithmic barrier terms added to the
objective. For a given barrier parameter `mu > 0`, the barrier subproblem is:

```
min   phi_mu(x, s) = f(x) - mu * [ sum_i ln(x_i - l_i) + sum_i ln(u_i - x_i)
                                  + sum_j ln(s_j - c_l_j) + sum_j ln(c_u_j - s_j) ]
s.t.  h(x) = 0
      c(x) - s = 0
```

The logarithmic barriers enforce strict feasibility: as `x_i -> l_i`, the term
`-mu * ln(x_i - l_i) -> +inf`, creating an infinite penalty wall at the bound.

As `mu -> 0`, the barrier walls become infinitely steep and the solution of the
barrier subproblem converges to the solution of the original NLP. The algorithm
solves a sequence of barrier subproblems with decreasing `mu`.

## 4. First-Order KKT Conditions

The Lagrangian of the barrier subproblem is:

```
L = f(x) + lam_h^T h(x) + lam_c^T (c(x) - s)
    - mu * [sum ln(x-l) + sum ln(u-x) + sum ln(s-c_l) + sum ln(c_u-s)]
```

Differentiating and setting to zero gives the perturbed KKT conditions:

```
(1)  grad_f(x) + J_h^T lam_h + J_c^T lam_c - z_l + z_u = 0     (stationarity)
(2)  h(x) = 0                                                     (primal: equalities)
(3)  c(x) - s = 0                                                 (primal: slacks)
(4)  (X - L) z_l = mu * e                                         (complementarity: lower)
(5)  (U - X) z_u = mu * e                                         (complementarity: upper)
(6)  (S - C_l) z_sl = mu * e                                      (complementarity: slack lower)
(7)  (C_u - S) z_su = mu * e                                      (complementarity: slack upper)
(8)  -lam_c - z_sl + z_su = 0                                     (slack stationarity)
```

Here `z_l, z_u >= 0` are the bound multipliers for design variables, and
`z_sl, z_su >= 0` are the bound multipliers for slacks. The diagonal matrices
`(X - L)`, `(U - X)`, etc. contain the gaps to the bounds.

At `mu = 0`, conditions (4)-(7) become the standard complementary slackness
conditions `(x_i - l_i) * z_l_i = 0`, meaning either a variable is at its bound
or its multiplier is zero.

## 5. Newton System (Full Form)

Applying Newton's method to the KKT system (1)-(8) gives the full primal-dual
system. In block form:

```
[ W + Sigma_x    0      J_h^T    J_c^T ] [ dx    ]   [ r_x    ]
[   0        Sigma_s    0        -I    ] [ ds    ] = [ r_s    ]
[   J_h        0        0         0    ] [ dlam_h]   [ r_h    ]
[   J_c       -I        0         0    ] [ dlam_c]   [ r_c    ]
```

where:
- `W = grad^2_{xx} L` is the Hessian of the Lagrangian
- `Sigma_x = Z_l/(X-L) + Z_u/(U-X)` is the primal barrier diagonal
- `Sigma_s = Z_sl/(S-C_l) + Z_su/(C_u-S)` is the slack barrier diagonal
- `J_h, J_c` are the constraint Jacobians
- The right-hand side `r` contains the KKT residuals at the current point

The barrier diagonals `Sigma_x` and `Sigma_s` arise from linearizing the
complementarity conditions (4)-(7) and substituting the multiplier updates.

## 6. Condensed KKT System

The slack variables `s` and their multipliers `z_sl, z_su` can be eliminated
analytically because their equations are diagonal (no coupling between slack
components). From the slack block:

```
ds = (r_s + dlam_c) / Sigma_s
```

Substituting into the constraint block eliminates `ds`, giving the condensed
system:

```
[ W + Sigma_x     J^T    ] [ dx   ]   [ r_x'  ]
[    J          -D_c     ] [ dlam ] = [ r_c'  ]
```

where:
- `J = [J_h; J_c]` is the full constraint Jacobian (equalities + inequalities)
- `D_c` is a diagonal matrix:
  - `D_c[i] = 0` for equality rows
  - `D_c[i] = 1/Sigma_s[i]` for inequality rows (negative definite contribution)
- The modified RHS `r_c'` absorbs the condensed slack terms

This is the system actually factorized and solved. It has dimension
`(n + m_e + m_i)` instead of `(n + m_i + m_e + m_i)` -- the slacks and their
duals are gone.

**Structure**: The `(1,1)` block `W + Sigma_x` is `n x n` and expected to be
positive definite (at a local minimum with LICQ). The `(2,2)` block `-D_c` is
negative semidefinite. The full matrix is symmetric indefinite with target inertia
`(n, m, 0)`: `n` positive eigenvalues, `m = m_e + m_i` negative eigenvalues, and
zero null eigenvalues.

## 7. Back-Substitution

After solving the condensed system for `(dx, dlam)`, the remaining quantities
are recovered by back-substitution:

```
Slack step:
  ds[i] = (dlam_c[i] + d[i]) / Sigma_s[i]
  where d[i] is the condensed slack RHS contribution

Design variable dual steps:
  dz_l[i] = (mu - (x_i - l_i)*z_l[i] - z_l[i]*dx[i]) / (x_i - l_i)
  dz_u[i] = (mu - (u_i - x_i)*z_u[i] + z_u[i]*dx[i]) / (u_i - x_i)

Slack dual steps:
  dz_sl[j] = (mu - (s_j - c_l_j)*z_sl[j] - z_sl[j]*ds[j]) / (s_j - c_l_j)
  dz_su[j] = (mu - (c_u_j - s_j)*z_su[j] + z_su[j]*ds[j]) / (c_u_j - s_j)
```

These formulas come from linearizing the complementarity conditions. For example,
condition (4) says `(x-l)*z_l = mu`. Expanding at a trial point:
`(x + dx - l)*(z_l + dz_l) = mu`, which gives
`(x-l)*dz_l + z_l*dx = mu - (x-l)*z_l`, solving for `dz_l`.

## 8. Fraction-to-Boundary Rule

A full Newton step might violate the strict positivity requirements (`x > l`,
`z_l > 0`, etc.). The fraction-to-boundary rule computes maximum step sizes:

```
alpha_x_max = max{ alpha in (0,1] : x + alpha*dx >= l + (1-tau)*(x-l) }
            = min over i of { -tau*(x_i - l_i)/dx_i  if dx_i < 0, else 1 }

alpha_z_max = max{ alpha in (0,1] : z_l + alpha*dz_l >= (1-tau)*z_l }
            = min over i of { -tau*z_l_i/dz_l_i  if dz_l_i < 0, else 1 }
```

The parameter `tau in (0,1)` controls how close to the boundary the step can go.
IPOPT uses an adaptive rule:

```
tau = max(tau_min, 1 - mu)
```

where `tau_min = 0.99`. Early in the solve (large `mu`), `tau` is close to
`tau_min`, allowing aggressive steps. As `mu -> 0`, `tau -> 1`, preventing the
iterate from approaching bounds too closely.

Two separate step sizes are maintained:
- `alpha_x`: primal step (design variables and slacks)
- `alpha_z`: dual step (all multipliers)

This decoupling is essential: the primal iterates need to stay strictly feasible
with respect to bounds, while the duals need to stay positive, and these
constraints are generally different.

## 9. Inertia Correction (IPOPT Algorithm IC)

Reference: Wachter & Biegler, "On the Implementation of an Interior-Point Filter
Line-Search Algorithm for Large-Scale Nonlinear Programming" (2006), Section 3.1.

The condensed KKT matrix must have the correct inertia `(n, m, 0)` for the Newton
direction to be a descent direction. When the Hessian `W` is indefinite (nonconvex
problem) or the Jacobian `J` is rank-deficient, the inertia is wrong.

The inertia corrector adds diagonal regularization:

```
[ W + Sigma_x + delta_w*I     J^T         ] [ dx   ]   [ r_x  ]
[         J              -D_c - delta_c*I  ] [ dlam ] = [ r_c  ]
```

where `delta_w >= 0` regularizes the primal block (makes it more positive definite)
and `delta_c >= 0` regularizes the dual block (makes it more negative definite).

**Algorithm IC flow:**

```
IC-1: Try factorization with warm-started delta_w from previous iteration
      (scaled by kw_minus = 1/3). If correct inertia -> done.

IC-2: If zero eigenvalues detected, set delta_c = dc_bar * mu^kc
      (dc_bar = 1e-8, kc = 0.25). This handles rank-deficient Jacobians.

IC-3: Choose initial delta_w:
      - If history exists: delta_w = kw_minus * last_successful_delta
      - Otherwise: delta_w = dw_0 = 1e-4

IC-4: Factorize with (delta_w, delta_c) and check inertia.

IC-5: If wrong inertia, grow:
      - First try: delta_w *= kw_plus_bar = 100 (aggressive)
      - Subsequent: delta_w *= kw_plus = 8

IC-6: If delta_w > dw_max = 1e40, abort (singular system).
```

The solver obtains the inertia from the LDL^T factorization (Pardiso: exact
inertia from D diagonal; Scipy LU: approximate inertia via pivots).

## 10. Line Search

After computing the search direction, a line search determines the step length
`alpha in (0, 1]` such that `x_{k+1} = x_k + alpha * alpha_x * dx`.

### 10.1 Merit Function Line Search (Armijo)

The `l1` exact penalty merit function is:

```
phi_1(x; nu) = f(x) + nu * ||c(x)||_1
```

where `nu` is the penalty parameter. The Armijo condition requires:

```
phi_1(x + alpha*dx) <= phi_1(x) + eta * alpha * D(phi_1; dx)
```

with `eta = 1e-4`. The directional derivative `D(phi_1; dx)` is computed from
the gradient of `f` and the linearized constraint violation.

Backtracking: if the condition fails, `alpha *= beta` (typically `beta = 0.5`)
and retry.

### 10.2 Filter Line Search (IPOPT Algorithm A)

Reference: Wachter & Biegler (2006), Section 2.

The filter line search treats optimization as a bi-objective problem: minimize
both the constraint violation `theta` and the barrier objective `phi_mu`:

```
theta(x) = ||h(x)||_2 + ||c(x) - s||_2      (constraint violation)
phi_mu(x) = f(x) - mu * sum ln(s - c_l) - mu * sum ln(c_u - s)  (barrier objective)
```

A **filter** is a set of pairs `{(theta_j, phi_j)}` representing previously
rejected points. A trial point `(theta_t, phi_t)` is **filter-acceptable** if it
is not dominated by any filter entry:

```
theta_t < theta_j - delta   OR   phi_t < phi_j - delta   for all j in filter
```

**Two acceptance modes:**

**Case 1 -- Switching condition (Armijo on phi):**
When the current point has small constraint violation (`theta_k < theta_min`) and
the search direction provides sufficient predicted decrease in the barrier objective
(`-dphi^{s_phi} > delta * theta_k^{s_theta}`), we accept via Armijo:

```
phi_mu(x + alpha*dx) <= phi_mu(x) + eta_phi * alpha * dphi
```

This mode focuses purely on optimality improvement when feasibility is already good.
Accepted points do NOT augment the filter.

**Case 2 -- Filter acceptance:**
When the switching condition is not active, the trial point must show sufficient
decrease in either theta or phi:

```
theta_t <= (1 - gamma_theta) * theta_k    OR    phi_t <= phi_k - gamma_phi * theta_k
```

AND be acceptable to the filter. Accepted points augment the filter with
`(theta_k, phi_k)` (the CURRENT point, not the trial -- this is the "envelope"
property).

**Backtracking**: Only the primal step is backtracked (`alpha *= beta`). The dual
step size `alpha_z` remains at its full fraction-to-boundary value throughout.

**Second-order correction (SOC):** When the full step (`alpha = 1`) is rejected
due to increased constraint violation (`theta_trial > theta_k`), a corrector step
is computed. The constraint rows of the Newton RHS are replaced with the trial
point's nonlinear residual, and the system is re-solved (no new factorization).
This corrects the Maratos effect -- the phenomenon where a Newton step is rejected
near a solution because the nonlinear constraint curvature temporarily increases
the violation even though the step is locally optimal.

### 10.3 Feasibility Restoration

When the filter line search fails (all backtracking steps rejected), the algorithm
enters a feasibility restoration phase. This has two modes:

**Mode 1 (large theta):** The constraint violation is genuinely large. The
algorithm temporarily increases the barrier parameter and takes Newton steps
aimed purely at reducing `theta`:

```
for restore_mu in [max(saved_mu, 0.01), 0.1, 1.0]:
    barrier_param = restore_mu
    for k iterations:
        compute Newton step at restore_mu
        backtrack until theta decreases
    if theta < 0.9 * theta_start: break
restore barrier_param to saved value
```

Higher barrier parameters relax the bound constraints, giving the algorithm more
room to reduce infeasibility.

**Mode 2 (small theta):** The constraint violation is small but the filter is
blocking optimality progress. The algorithm tries KKT-descent steps (accept any
step that reduces the overall residual), and if that fails, clears the filter
entries to unblock progress.

## 11. Barrier Parameter Update Strategies

The barrier parameter `mu` controls the trade-off between optimality and
centrality. Reducing `mu` too quickly causes ill-conditioning (the barrier
diagonals `Sigma` become extremely large near bounds). Reducing too slowly wastes
iterations on subproblems that don't need high accuracy.

### 11.1 Monotone Strategy

The simplest approach: reduce `mu` by a fixed factor when the current barrier
subproblem is solved to sufficient accuracy:

```
if res_norm <= kappa_epsilon * mu:
    mu_new = kappa_mu * mu       (kappa_mu = 0.1, i.e., reduce by 10x)
```

This gives predictable, stable convergence but is conservative -- it may spend
too many iterations solving each subproblem to high accuracy before reducing `mu`.

### 11.2 Heuristic Strategy (LOQO)

A more aggressive approach that sets `mu` based on the current complementarity gap:

```
comp = (x - l)^T z_l / n           (average complementarity)
xi = min_i{(x_i - l_i) * z_l_i} / comp   (centrality measure)

heuristic_factor = min((1-r)*(1-xi)/xi, 2)^3
mu_new = gamma * heuristic_factor * comp
```

When the iterate is well-centered (`xi` close to 1), the factor is small and `mu`
drops aggressively. When poorly centered (`xi` close to 0), the factor is large and
`mu` stays high to re-center. This adapts to the geometry of the central path.

### 11.3 IPOPT Algorithm A-3 (Filter Line Search)

When using filter line search, the barrier update follows IPOPT's specific rule.
The barrier subproblem is considered solved when:

```
E_mu = max(||dual_err||, ||primal_err||, comp) <= kappa_eps * mu
```

Note that ALL THREE components (dual, primal, complementarity) must be small. A
common implementation error is checking only the overall residual norm, which can
be dominated by one component while others are still large -- leading to premature
`mu` reduction and instability.

When the condition is met:

```
mu_new = max(tol/10, min(kappa_mu * mu, mu^theta_mu), mu/10)
```

where `kappa_mu = 0.2`, `theta_mu = 1.5`. The `mu^1.5` term gives superlinear
convergence: as `mu` gets small, the reductions become proportionally larger (e.g.,
`(1e-4)^1.5 = 1e-6`). The `mu/10` floor prevents reductions larger than 10x,
which stabilizes the transition.

After reducing `mu`, the inner filter is cleared (it was built for the old barrier
subproblem) and `theta_0` is reset for the new subproblem.

### 11.4 Quality Function Strategy (Nocedal-Wachter-Waltz 2009)

Reference: "Adaptive Barrier Update Strategies for Nonlinear Interior Methods",
SIAM J. Optim., 2009.

Instead of solving each barrier subproblem to convergence before reducing `mu`, the
quality function approach selects the OPTIMAL `mu` at every iteration by minimizing
a measure of the predicted KKT error after one Newton step.

**The quality function `q_L(sigma)`:**

Given the current complementarity `comp = x^T z / n`, any `mu = sigma * comp` can
be achieved without re-factorizing the KKT matrix. This is because the Newton RHS
is affinely linear in `mu`:

```
r(mu) = r(0) + (mu / comp) * [r(comp) - r(0)]
```

So the Newton direction at any `mu` is:

```
p(sigma) = p(0) + sigma * [p(1) - p(0)]     (by linearity of K^{-1} r)
```

where `p(0)` is the affine-scaling direction (mu=0) and `p(1)` is the centering
direction (mu=comp). Only TWO back-solves are needed, regardless of how many
`sigma` values are evaluated.

The quality function measures the predicted KKT error:

```
q_L(sigma) = (1 - alpha_z)^2 * ||dual||^2 / n_d
           + (1 - alpha_x)^2 * ||primal||^2 / n_p
           + ||comp_trial||^2 / n_c
```

where `alpha_x(sigma)` and `alpha_z(sigma)` are the fraction-to-boundary step
sizes for the direction `p(sigma)`, and `comp_trial` is the complementarity at the
trial point. The normalization factors `1/n_d, 1/n_p, 1/n_c` prevent any single
component from dominating.

**The problem with golden-section search:**

The original NWW (2009) algorithm finds the optimal `sigma` via golden-section
search over `[sigma_min, sigma_max]`. Each evaluation of `q_L` requires:
1. Forming the direction `p(sigma)` (cheap: vector addition)
2. Back-substituting for all dual/slack updates (moderate cost)
3. Computing `alpha_x, alpha_z` via fraction-to-boundary (moderate cost)
4. Applying the trial step and computing trial complementarity (moderate cost)

With 12 golden-section iterations, this means 12 full back-substitution + step
computations per optimizer iteration. For large problems with thousands of
variables, this overhead becomes significant.

Worse, the golden-section search can be unreliable: `q_L(sigma)` is not
necessarily unimodal (it depends on step size clipping by the fraction-to-boundary
rule), so the golden-section minimum may be a local minimum, not global.

**The solution: Mehrotra predictor-corrector**

Instead of searching for the optimal `sigma`, we use the Mehrotra predictor-
corrector heuristic (Nocedal & Wright, Algorithm 14.3), which computes `sigma`
analytically from the affine step:

```
Step 1: Compute affine step p(0) and its max step alpha_aff.
Step 2: Compute trial complementarity comp_aff at the affine trial point.
Step 3: sigma_aff = min(1, (comp_aff / comp)^3)
```

The intuition: if the affine step (pure Newton, no centering) achieves a large
reduction in complementarity (`comp_aff << comp`), then little centering is
needed (`sigma` small, aggressive `mu` reduction). If the affine step barely
reduces complementarity, more centering is needed (`sigma` large, conservative
`mu`).

**NLP safeguards** (beyond classical Mehrotra):

For NLPs (vs. LPs), the affine step may be limited by constraint nonlinearity,
not just bound constraints. Additional sigma components handle this:

```
sigma_step  = (1 - alpha_aff_x)^2     (primal step limitation)
sigma_decay = kappa_decay * mu_old / comp   (IPOPT-style mu decay limit)
sigma_floor = mu_floor / comp          (absolute floor)
sigma_pc    = min(1, max(sigma_aff, sigma_step, sigma_decay, sigma_floor))
mu_pc       = sigma_pc * comp
```

- `sigma_step`: When the primal step is severely clipped (`alpha_aff_x` small),
  the affine direction hits constraint boundaries. More centering prevents this.
- `sigma_decay`: Prevents `mu` from crashing by more than `kappa_decay` per
  iteration (default 0.2, matching IPOPT's `kappa_mu`).
- `sigma_floor`: Ensures `mu >= mu_floor` (e.g., `tol * 0.01`).

This approach requires exactly 2 back-solves (affine + centering) instead of 12+
for golden-section. It is also more robust because the sigma selection is based
on actual step quality rather than a potentially non-unimodal objective.

### 11.5 Quality Function Mode Switching (Free/Monotone)

The quality function operates in two modes:

**Free mode (default):** The barrier parameter is chosen adaptively each
iteration by the predictor-corrector. A nonmonotonicity mechanism allows
temporary KKT error increases:

```
Accept if: phi_new <= kappa * max(recent KKT errors)     (kappa = 0.9999)
```

This permits exploring directions that temporarily worsen the KKT error but
lead to better solutions (e.g., moving along a constraint boundary).

**Monotone mode (fallback):** If the nonmonotonicity check fails or the line
search rejects the step, the quality function switches to monotone mode with
a fixed barrier:

```
mu_monotone = 0.8 * current_complementarity
```

In monotone mode, `mu` is only reduced when the residual satisfies
`res <= progress_tol * mu`, using the standard factor of 0.1.

The algorithm returns to free mode when the KKT error drops below the level
at which monotone mode was entered. This two-mode design gives the adaptive
strategy a safe fallback when the free-mode heuristic produces poor steps.

## 12. Stagnation Detection

When the optimizer makes insufficient progress, two mechanisms activate:

**Acceptable convergence:** If the residual has not improved significantly for
`acceptable_iter` (default 10) consecutive iterations AND the residual is below
`acceptable_tol` (default 100x the convergence tolerance), the algorithm
terminates with "acceptable convergence". The solution is near-optimal but did
not reach the tight tolerance.

**Forced barrier reduction:** If stagnation persists for `2 * acceptable_iter`
iterations and the barrier parameter is above the convergence tolerance, the
barrier is forcibly reduced by the monotone factor (0.1x). This can cause a
temporary spike in the KKT residual (because the complementarity target suddenly
drops) but may break the stagnation by forcing progress toward the central path.

**The instability risk:** When the barrier is forcibly reduced from `mu` to
`mu/10`, the complementarity conditions `(x-l)*z_l = mu` are suddenly violated by
a factor of 10. The next Newton step must simultaneously correct this mismatch
across ALL bound-active variables. If many variables are near bounds, this
correction can be large and destabilizing, causing residual spikes of 100-1000x.
The optimizer typically recovers within 10-30 iterations, but the time is wasted.

## 13. The Complete Algorithm Flow

```
INITIALIZE:
  Set mu = initial_barrier_param
  Initialize slacks: s = c(x), project into strict interior
  Initialize multipliers: z_l = z_u = mu, z_sl = mu/gap, z_su = mu/gap
  (Optional) Compute least-squares multiplier estimates

MAIN LOOP (for i = 0, 1, 2, ...):

  Step A: Evaluate KKT residual
    Compute gradient g = grad_f + J^T lam
    Form condensed Newton RHS r(mu)
    res_norm = ||r||

  Step B: Log iteration data

  Step C: Convergence and termination checks
    If res_norm < tol: CONVERGED
    If stagnation_count >= acceptable_iter and res_norm < acceptable_tol:
      ACCEPTABLE CONVERGENCE
    If stagnation_count >= 2*acceptable_iter: force barrier reduction

  Step D: Barrier parameter update
    [Filter LS] IPOPT A-3: if E_mu <= kappa_eps * mu, reduce mu
    [Quality function] Compute optimal mu via predictor-corrector
    [Heuristic] LOQO-style mu from complementarity
    [Monotone] mu *= kappa_mu if subproblem solved

  Step E: Compute search direction
    Assemble KKT matrix: W + Sigma_x + delta_w*I, J, -D_c - delta_c*I
    Inertia correction (Algorithm IC) if needed
    Factorize (LDL^T or LU)
    Solve for (dx, dlam)
    Back-substitute for (ds, dz_l, dz_u, dz_sl, dz_su)
    Compute alpha_x, alpha_z via fraction-to-boundary

  Step F: Line search and step acceptance
    [Filter LS] IPOPT Algorithm A: bi-objective filter
    [Merit LS] Armijo backtracking on l1 penalty
    If line search fails: feasibility restoration
    If restoration fails: reject step, increase barrier

  Step G: Post-step update
    x += alpha * alpha_x * dx
    s += alpha * alpha_x * ds
    z_l += alpha_z * dz_l   (dual uses full step in filter LS)
    lam += alpha_z * dlam
    Update gradient at new point

  REPEAT
```

## 14. Residual Components and Convergence Measure

The overall KKT residual `res_norm` is the norm of the condensed Newton RHS,
which combines all KKT violations into a single vector. For diagnostics, it is
decomposed into three components:

```
dual_err    = ||grad_f + J^T lam - z_l + z_u||     (stationarity violation)
primal_err  = ||h(x)|| + ||c(x) - s||              (constraint violation)
comp_err    = ||diag(X-L)*z_l - mu*e|| + ...        (complementarity violation)
```

The algorithm has converged when `max(dual_err, primal_err, comp_err) < tol`.

The complementarity error naturally approaches zero as `mu -> 0` and the Newton
steps drive `(x-l)*z_l -> mu`. If `comp_err` remains large relative to `mu`, it
indicates that some bound multiplier products `z_l * (x-l)` are stuck away from
the barrier target -- typically because the fraction-to-boundary rule prevents the
dual variables from moving fast enough.
