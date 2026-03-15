# BISC: Block Inertia-free Structural Convexification
## Minimum-Perturbation Hessian Convexification for OC-Structured Interior Point Methods

**Target:** SIAM Journal on Optimization

---

## 1. Motivation

Interior point methods (IPMs) for nonlinear optimal control need to solve a Newton system at each iteration. When we use **exact Hessians** (from automatic differentiation), the Hessian of the Lagrangian is **indefinite** — it has negative eigenvalues from the constraint curvature terms. This means the Newton step may not be a descent direction.

**Current practice (IPOPT):** Try a Cholesky factorization. If it fails (inertia check), add a scalar delta to the **entire** Hessian diagonal: H + delta*I. Retry. Increase delta. Retry again. This is:
- Wasteful: modifies ALL stages, even healthy ones
- Slow: multiple trial factorizations
- Over-regularizing: adds the same delta everywhere, destroying curvature info

**BISC idea:** Exploit the block structure of optimal control. Use a backward Riccati recursion, and at each stage, apply the **minimum-norm PSD projection** only to the small control Hessian block that actually needs it.

**Result:** One pass, no trial-and-error, minimum perturbation, preserves exact Newton near the solution.

---

## 2. Problem Formulation

### 2.1 Discrete-Time Optimal Control NLP

We consider the discrete-time optimal control problem with N stages:

**Objective:**

    min        Sigma_{k=0}^{N-1}  l_k(x_k, u_k)  +  l_N(x_N)
  {x_k, u_k}

**Subject to:**

    f_k(x_k, u_k) - x_{k+1} = 0,      k = 0, ..., N-1      (dynamics)
    g_k(x_k, u_k) <= 0,                 k = 0, ..., N        (path inequalities)
    x_0 = x_bar                                               (initial condition)

**Variables:**
- x_k in R^{n_x} : state at stage k
- u_k in R^{n_u} : control at stage k (absent at terminal stage N)
- f_k : R^{n_x} x R^{n_u} -> R^{n_x} : dynamics map
- l_k : stage cost
- g_k : inequality constraints (p_k of them at stage k)

**Stage variable:** z_k = (x_k, u_k) in R^{n_z}, where n_z = n_x + n_u for k < N, and z_N = x_N at terminal.


### 2.2 Interior Point Barrier Reformulation

Introduce slacks s_k >= 0 and form the barrier subproblem:

    min_{z,s}  Sigma_k l_k(z_k)  -  mu * Sigma_k  1^T log(s_k)

    s.t.  f_k(x_k, u_k) - x_{k+1} = 0,      k = 0, ..., N-1
          g_k(z_k) + s_k = 0,                  k = 0, ..., N

where **mu > 0** is the barrier parameter, driven to zero as the IPM progresses.

**Lagrangian** with multipliers lambda_k (dynamics) and nu_k (inequalities):

    L = Sigma_k l_k(z_k)  -  mu * Sigma_k 1^T log(s_k)
      + Sigma_k lambda_k^T * (f_k(x_k, u_k) - x_{k+1})
      + Sigma_k nu_k^T * (g_k(z_k) + s_k)

**Sign convention:** The dynamics residual is e_k = f_k(x_k, u_k) - x_{k+1} = 0.

The Jacobians of the dynamics are:
- de_k/dx_k = F_k  (state Jacobian, n_x x n_x)
- de_k/du_k = G_k  (control Jacobian, n_x x n_u)
- de_k/dx_{k+1} = -I_{n_x}

This sign convention gives **+G_k^T lambda_k** in the control stationarity (important for getting M_k right).


### 2.3 IPM Slack Elimination

The complementarity condition from the KKT system is:

    S_k * nu_k = mu * 1

where S_k = diag(s_k). This lets us eliminate slacks and inequality multipliers:

    nu_k = mu * S_k^{-1} * 1
    Delta_nu_k = S_k^{-1} * (mu*1 - nu_k * Delta_s_k - S_k * nu_k)

After elimination, the inequality constraints contribute a **barrier Hessian term**:

    Sigma_k = diag(nu_{k,i} / s_{k,i}) = diag(mu / s_{k,i}^2)

This matrix is always PSD (positive semi-definite) and appears in the augmented Hessian below.


### 2.4 The Reduced KKT System

After eliminating slacks and inequality multipliers, we get:

    | H~   J^T |   | Dz |       | r_d |
    |          | * |    | = -   |     |
    | J    0   |   | Dl |       | r_p |

where:
- Dz = (Dz_0, ..., Dz_N) : primal step (all stage variables)
- Dl = (Dl_0, ..., Dl_{N-1}) : dual step (dynamics multipliers)
- r_d : dual residual (gradient of Lagrangian)
- r_p : primal residual (dynamics violation)


### 2.5 The Augmented Hessian H~ (Block-Diagonal!)

**Key structural property:** H~ is **block-diagonal**:

    H~ = block-diag(H~_0, H~_1, ..., H~_N)

This is because f_k(x_k, u_k) depends only on z_k, not z_{k+1}. There are **no cross-stage second-derivative terms**. All cross-stage coupling comes from the constraint Jacobian J.

Each stage block decomposes as:

    H~_k = W_k + J_{g,k}^T * Sigma_k * J_{g,k}

**Part 1 — Lagrangian Hessian W_k (from AD, possibly indefinite):**

    W_k = nabla^2_{z_k} l_k  +  lambda_k^T * nabla^2_{z_k} f_k  +  nu_k^T * nabla^2_{z_k} g_k
          |_________________|    |_____________________________|    |__________________________|
           objective curvature     dynamics curvature (CAN BE         inequality curvature
           (usually PSD)           NEGATIVE -- source of              (can be negative)
                                   indefiniteness!)

The dynamics curvature term **lambda_k^T * nabla^2 f_k** is what makes W_k indefinite. This term carries genuine second-order information. Dropping it (= Gauss-Newton approximation) loses superlinear convergence.

**Part 2 — Barrier contribution (always PSD, decays with mu):**

    J_{g,k}^T * Sigma_k * J_{g,k},      Sigma_k = diag(mu / s_{k,i}^2)

This is PSD by construction. It regularizes directions of active inequality constraints. As mu -> 0, it vanishes for inactive constraints but stays large for active ones (s_{k,i} -> 0).

**Sub-block decomposition of H~_k:**

    H~_k = | Q~_k    N~_k  |      Q~_k in R^{n_x x n_x}  (state-state Hessian)
           | N~_k^T  R~_k  |      R~_k in R^{n_u x n_u}  (control-control Hessian)
                                   N~_k in R^{n_x x n_u}  (state-control cross term)

where:
- Q~_k = d^2L/dx_k^2 + (barrier terms on x_k)
- R~_k = d^2L/du_k^2 + (barrier terms on u_k)
- N~_k = d^2L/(dx_k du_k) + (barrier cross terms)


### 2.6 The Constraint Jacobian J (Block-Bidiagonal)

    J = | F_0  G_0  -I                                    |
        |              F_1  G_1  -I                        |
        |                          ...                     |
        |                          F_{N-1}  G_{N-1}  -I   |

Each row k: [F_k, G_k] * Dz_k - I * Dx_{k+1} = -r_{p,k}

i.e., **Dx_{k+1} = F_k * Dx_k + G_k * Du_k - r_{p,k}**  (linearized dynamics propagation)


### 2.7 Stationarity Equations (Stage-by-Stage)

**State stationarity at stage k:**

    Q~_k * Dx_k + N~_k * Du_k + F_k^T * Dlambda_k - Dlambda_{k-1} = -r_{x,k}

(The -Dlambda_{k-1} comes from de_{k-1}/dx_k = -I, so J^T contribution is -Dlambda_{k-1}.)

**Control stationarity at stage k:**

    N~_k^T * Dx_k + R~_k * Du_k + G_k^T * Dlambda_k = -r_{u,k}

(The +G_k^T comes from our sign convention e_k = f_k - x_{k+1}.)

**Dynamics at stage k:**

    F_k * Dx_k + G_k * Du_k - Dx_{k+1} = -r_{p,k}


---

## 3. Backward Riccati Recursion

### 3.1 Why Backward (Not Forward)?

**The cascade problem with forward elimination:**

In a forward Schur complement approach, we would eliminate x_0, then x_1, etc. At each stage, the Schur complement is:

    S_k = H_k - C_k^T * S_{k-1}^{-1} * C_k

If S_{k-1} has a small eigenvalue (barely positive), then S_{k-1}^{-1} has a huge eigenvalue, and the subtracted term C^T S^{-1} C can be enormous. This means S_k can become **arbitrarily negative** — one barely-PD stage causes a catastrophic cascade through all subsequent stages.

**Backward Riccati avoids this entirely** through the floor property (Section 3.4).


### 3.2 The Riccati Ansatz

We maintain the affine relation (backward from terminal):

    Dlambda_k = P_{k+1} * Dx_{k+1} + p_{k+1}

where:
- P_{k+1} in R^{n_x x n_x} : symmetric cost-to-go Hessian
- p_{k+1} in R^{n_x} : cost-to-go gradient

**Terminal condition:** At stage N, stationarity is Q~_N * Dx_N - Dlambda_{N-1} = -r_{x,N}, so:

    P_N = Q~_N       (terminal cost-to-go Hessian)
    p_N = r_{x,N}    (terminal cost-to-go gradient)


### 3.3 Deriving the Backward Recursion

Given P_{k+1} and p_{k+1}, we derive the update for stage k.

**Step 1 — Substitute dynamics into the ansatz:**

From the linearized dynamics: Dx_{k+1} = F_k * Dx_k + G_k * Du_k - r_{p,k}

Substitute into Dlambda_k = P_{k+1} * Dx_{k+1} + p_{k+1}:

    Dlambda_k = P_{k+1} * (F_k * Dx_k + G_k * Du_k - r_{p,k}) + p_{k+1}
              = P_{k+1} * F_k * Dx_k  +  P_{k+1} * G_k * Du_k  +  (p_{k+1} - P_{k+1} * r_{p,k})

**Step 2 — Substitute into control stationarity:**

    N~_k^T * Dx_k + R~_k * Du_k + G_k^T * [P_{k+1} * F_k * Dx_k + P_{k+1} * G_k * Du_k + ...] = -r_{u,k}

Collecting Du_k terms and Dx_k terms:

    [R~_k + G_k^T * P_{k+1} * G_k] * Du_k  +  [N~_k^T + G_k^T * P_{k+1} * F_k] * Dx_k  =  -r_{M,k}
     \_____________________________/             \___________________________________/
                  M_k                                           L_k^T

where:

    r_{M,k} = r_{u,k} + G_k^T * (p_{k+1} - P_{k+1} * r_{p,k})

**THE CONTROL HESSIAN:**

    M_k  =  R~_k  +  G_k^T * P_{k+1} * G_k        (n_u x n_u matrix)
             ^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^
         stage Hessian    cost-to-go propagated
         (can be indef)    through controls (PSD)

**The cross term:**

    L_k  =  N~_k + F_k^T * P_{k+1} * G_k           (n_x x n_u matrix)

**Step 3 — Solve for the control step** (requires M_k to be PD!):

    Du_k = -M_k^{-1} * (L_k^T * Dx_k + r_{M,k})

Define feedback gain and feedforward:
- K_k = M_k^{-1} * L_k^T     (n_u x n_x)
- k_k = M_k^{-1} * r_{M,k}   (n_u x 1)

So: **Du_k = -K_k * Dx_k - k_k**

**Step 4 — Riccati update for P_k:**

Substitute Du_k back into the state stationarity:

    P_k = Q~_k + F_k^T * P_{k+1} * F_k  -  L_k * M_k^{-1} * L_k^T

    p_k = (corresponding affine term from residuals)


### 3.4 The Anti-Cascade Floor Property

**Proposition 1.** If M_k is PD and N~_k = 0 (no state-control cross term), then:

    P_k >= Q~_k      (in the PSD ordering, meaning P_k - Q~_k is PSD)

**Proof.**

With N~_k = 0, we have L_k = F_k^T * P_{k+1} * G_k, and:

    P_k = Q~_k + F_k^T * P_{k+1} * F_k - F_k^T * P_{k+1} * G_k * M_k^{-1} * G_k^T * P_{k+1} * F_k
        = Q~_k + F_k^T * [P_{k+1} - P_{k+1} * G_k * (G_k^T * P_{k+1} * G_k + R~_k)^{-1} * G_k^T * P_{k+1}] * F_k

The bracketed term is the Schur complement of M_k in:

    Omega = | P_{k+1}          P_{k+1} * G_k              |
            | G_k^T * P_{k+1}  G_k^T * P_{k+1} * G_k + R~_k |

Since Omega = [I; G_k^T]^T * P_{k+1} * [I, G_k] + block-diag(0, R~_k), and P_{k+1} >= 0 (PSD, by induction), R~_k >= 0 (after BISC), we get Omega >= 0. The Schur complement of a PSD block in a PSD matrix is PSD.

Therefore: **P_k = Q~_k + F_k^T * (PSD) * F_k >= Q~_k**.  QED.

**What this means:** The backward Riccati **cannot amplify indefiniteness**. The cost-to-go Hessian P_k is always at least as positive as the stage Hessian Q~_k. If Q~_k is PD (which barrier terms help ensure), P_k is automatically PD. No cascade.

**For general N~_k != 0:** P_k >= Q~_k - N~_k * M_k^{-1} * N~_k^T (the Schur complement of M_k in H~_k). PD if and only if H~_k is PD given M_k PD.


### 3.5 Forward Sweep (Step Recovery)

After the backward pass, recover the Newton step forward:

    Dx_0 = (from initial condition handling)

    For k = 0, ..., N-1:
        Du_k      = -K_k * Dx_k - k_k                           (control step)
        Dx_{k+1}  = F_k * Dx_k + G_k * Du_k - r_{p,k}          (state propagation)
        Dlambda_k = P_{k+1} * Dx_{k+1} + p_{k+1}               (dual recovery)

**Total cost:** O(N * (n_x + n_u)^3) — **linear in horizon N**.


---

## 4. BISC: The Convexification Algorithm

### 4.1 Where Indefiniteness Enters

The backward Riccati requires **inverting M_k** at each stage. From the definition:

    M_k = R~_k + G_k^T * P_{k+1} * G_k

- G_k^T * P_{k+1} * G_k is **PSD** (since P_{k+1} >= 0 by induction)
- R~_k can be **indefinite** because it contains the dynamics curvature:

    R~_k = d^2l_k/du_k^2 + lambda_k^T * d^2f_k/du_k^2 + (barrier terms)
                            |___________________________|
                            CAN BE NEGATIVE (from AD)

When this negative curvature is large enough, M_k becomes indefinite. The Riccati breaks down.

**This is exactly where BISC intervenes.**


### 4.2 PD Cone Projection (Minimum Frobenius-Norm)

**Definition.** For a symmetric matrix A in R^{m x m} with eigendecomposition A = Q * Lambda * Q^T, and parameter delta > 0:

    proj_delta(A)  =  Q * max(Lambda, delta * I) * Q^T

where max is applied element-wise to the eigenvalue diagonal:

    max(Lambda, delta * I) = diag( max(lambda_1, delta), max(lambda_2, delta), ..., max(lambda_m, delta) )

**Lemma 1 (Optimality).** The modification E = proj_delta(A) - A satisfies:

    E = Q * max(delta*I - Lambda, 0) * Q^T

and E is the **unique minimizer** of:

    min_{E >= 0}  ||E||_F    subject to    lambda_min(A + E) >= delta

**Proof.** In the eigenbasis of A, the Frobenius norm decomposes over eigenvalues:

    ||E||_F^2 = Sigma_i e_i^2

and the constraint lambda_i(A) + e_i >= delta (where e_i >= 0 is the i-th eigenvalue of E in A's eigenbasis). The minimum is achieved by e_i = max(delta - lambda_i, 0) independently for each i. This is the standard PSD cone projection result.

**Key properties:**
1. Only "bad" eigenvalues (< delta) are modified; healthy ones are untouched
2. If A >= delta * I already, then E = 0 (no modification at all)
3. The modification is in A's own eigenbasis — minimal distortion
4. For n_u x n_u blocks (typically n_u = 1 to 10 in OC), eigendecomposition costs O(n_u^3) — negligible


### 4.3 The Cheng-Higham Modified Cholesky (1998)

For implementation, instead of computing a full eigendecomposition, we can use the **Cheng-Higham modified Cholesky algorithm**. This is based on symmetric indefinite factorization and is numerically more robust for larger blocks.

**Background:** Given a symmetric matrix A in R^{m x m} (not necessarily PD), the Bunch-Kaufman algorithm computes:

    P * A * P^T = L * B * L^T

where:
- P is a permutation matrix (pivoting for stability)
- L is unit lower triangular
- B is block diagonal with 1x1 and 2x2 blocks on the diagonal

The 2x2 blocks appear where A has negative or near-zero eigenvalues (Bunch-Kaufman pairs them for stability).

**Cheng-Higham algorithm step by step:**

    INPUT:  Symmetric A in R^{m x m}, threshold delta > 0
    OUTPUT: Modified A~ = A + E such that A~ >= delta * I, with ||E||_F minimized

    Step 1: Compute symmetric indefinite factorization
            P * A * P^T = L * B * L^T
            (using Bunch-Kaufman pivoting)

    Step 2: For each diagonal block B_j of B:

            Case (a) — B_j is 1x1, i.e. B_j = [b_jj]:
                If b_jj >= delta:  keep it (healthy)
                If b_jj < delta:   set b_jj~ = delta  (clip to delta)

            Case (b) — B_j is 2x2, i.e. B_j = |b_jj    b_j,j+1|:
                                                |b_j+1,j  b_j+1,j+1|
                Compute eigenvalues of B_j:
                    tau = (b_jj + b_j+1,j+1) / 2          (average)
                    rho = sqrt( ((b_jj - b_j+1,j+1)/2)^2 + b_j,j+1^2 )
                    lambda_1 = tau + rho
                    lambda_2 = tau - rho

                If both lambda_1, lambda_2 >= delta: keep B_j (healthy)
                Otherwise: replace eigenvalues
                    lambda_i~ = max(lambda_i, delta)
                    Reconstruct: B_j~ = V * diag(lambda_1~, lambda_2~) * V^T
                    where V is the 2x2 eigenvector matrix of B_j

    Step 3: Reconstruct the modified matrix:
            A~ = P^T * L * B~ * L^T * P

**Why this is equivalent to eigenvalue clipping for small blocks:**

For the n_u x n_u control Hessian M_k (typical size 1-10):
- The Bunch-Kaufman factorization with 1x1 and 2x2 blocks captures all eigenvalues
- Clipping the eigenvalues of each B_j block is equivalent to clipping the eigenvalues of M_k itself (the L and P transformations are invertible and the eigenvalues of B equal the eigenvalues of A up to the block structure)
- For very small blocks (n_u <= 3), direct eigendecomposition is simpler and equally fast
- For larger blocks (n_u > 5), Cheng-Higham is more numerically stable (avoids forming Q explicitly)

**In practice for BISC:** Since the control Hessian M_k is typically small (n_u x n_u with n_u = 1 to 10), either approach works. Use direct eigendecomposition for n_u <= 4, Cheng-Higham for n_u > 4.


### 4.4 Choice of delta (No Free Parameters)

    delta = kappa * mu,       kappa > 0 fixed (e.g., kappa = 0.1)

**Why this works:**
- **Early iterations** (mu large, far from solution): delta is large, providing robust regularization even when Hessians are very indefinite.
- **Near solution** (mu small): delta is small, modification is minimal, preserving Newton step quality.
- **At convergence** (mu -> 0): delta -> 0, and under SOSC the modification **vanishes completely** (Theorem 4).

**No trial-and-error.** No multiple factorizations. The parameter is determined by the IPM's own barrier parameter.

Compare with IPOPT: tries delta_0, fails, tries 4*delta_0, fails, tries 16*delta_0, ... up to 5-10 factorizations per iteration.


### 4.5 The Complete BISC Algorithm

    ==================================================================
    ALGORITHM: BISC (Block Inertia-free Structural Convexification)
    ==================================================================

    INPUT:  Stage Hessians {H~_k} = {Q~_k, R~_k, N~_k}
            Dynamics Jacobians {F_k, G_k}
            Residuals {r_{x,k}, r_{u,k}, r_{p,k}}
            Barrier parameter mu
            Convexification threshold delta = kappa * mu

    OUTPUT: Newton step {Dx_k, Du_k, Dlambda_k} for all k

    ---------- BACKWARD PASS (k = N down to 0) ----------

    1.  TERMINAL STAGE:
        P~_N = proj_delta(Q~_N)            // project terminal Hessian to PD
        E_N  = P~_N - Q~_N                 // terminal modification (often 0)
        p_N  = r_{x,N}

    2.  FOR k = N-1, N-2, ..., 0:

        (a) Form the control Hessian:
            M_k = R~_k + G_k^T * P~_{k+1} * G_k        // n_u x n_u

        (b) CONVEXIFY (the core BISC step):
            M~_k = proj_delta(M_k)                       // PD cone projection
            E_k^M = M~_k - M_k                           // PSD perturbation (min Frobenius norm)
            // If M_k already has all eigenvalues >= delta: E_k^M = 0 (no change!)

        (c) Compute cross term:
            L_k = N~_k + F_k^T * P~_{k+1} * G_k         // n_x x n_u

        (d) Compute feedback gain and feedforward:
            K_k = M~_k^{-1} * L_k^T                      // n_u x n_x  (solve via Cholesky of M~_k)
            k_k = M~_k^{-1} * r_{M,k}                    // n_u x 1
            where r_{M,k} = r_{u,k} + G_k^T * (p_{k+1} - P~_{k+1} * r_{p,k})

        (e) Riccati update:
            P_k = Q~_k + F_k^T * P~_{k+1} * F_k - L_k * K_k

        (f) Safety projection of cost-to-go:
            P~_k = proj_delta(P_k)                        // project if needed
            E_k^P = P~_k - P_k                            // often 0 by Prop. 1
            p_k = (affine term from residuals)

    ---------- FORWARD SWEEP (k = 0 up to N-1) ----------

    3.  Compute Dx_0 from initial condition.

    4.  FOR k = 0, 1, ..., N-1:
        Du_k      = -K_k * Dx_k - k_k                    // control step
        Dx_{k+1}  = F_k * Dx_k + G_k * Du_k - r_{p,k}   // state propagation
        Dlambda_k = P~_{k+1} * Dx_{k+1} + p_{k+1}       // dual recovery

    5.  Recover slack and inequality multiplier steps from eliminated equations.

    ==================================================================

**Remark on step 2f:** By Proposition 1 (floor property), when N~_k = 0 and Q~_k is PD, P_k is automatically PD, so E_k^P = 0. The barrier contribution to Q~_k (from state bounds and path constraints) typically ensures this. Step 2f is a safety net for the general case.


### 4.6 Computational Cost

    Per stage (backward pass):
      Matrix multiply G_k^T * P~_{k+1} * G_k:    O(n_x^2 * n_u + n_x * n_u^2)
      Eigendecomposition of M_k (n_u x n_u):      O(n_u^3)
      Solve M~_k^{-1} * (...) via Cholesky:       O(n_u^2 * n_x)
      Riccati update for P_k (n_x x n_x):         O(n_x^2 * n_u + n_x^3)

    Total backward pass:   O(N * (n_x + n_u)^3)
    Forward sweep:         O(N * (n_x + n_u)^2)
    Overall:               O(N * (n_x + n_u)^3)   -- LINEAR in horizon N

    Compare IPOPT on the same problem:
      With dense factorization:  O(N^3 * (n_x + n_u)^3)   (cubic in N!)
      With MA57 sparse:          O(N * (n_x + n_u)^3) * (2 to 5 trial factorizations)

    BISC: ONE backward + ONE forward pass. No retries.


---

## 5. Theoretical Results

### 5.1 Theorem 1 — Minimum Perturbation

**Theorem 1 (Stagewise Minimum Frobenius-Norm Perturbation).**
At each stage k, the BISC modification E_k^M to the control Hessian M_k satisfies:

    ||E_k^M||_F  =  min { ||E||_F  :  E >= 0,  lambda_min(M_k + E) >= delta }

That is, E_k^M is the **unique minimum-norm PSD perturbation** making M_k sufficiently positive definite.

**Proof.** Direct consequence of Lemma 1 (PD cone projection optimality) applied to M_k. QED.

**Corollary (Structured sparsity).** The total BISC modification is:

    ||E_total||_F^2 = Sigma_k ||E_k^M||_F^2 + Sigma_k ||E_k^P||_F^2

Each E_k^M lives in the n_u x n_u control block. Each E_k^P lives in the n_x x n_x state block. **Off-diagonal blocks and the constraint Jacobian J are never modified.**


### 5.2 Theorem 2 — Descent Direction

**Theorem 2 (Descent Guarantee).**
If P~_k > 0 for all k = 0, ..., N (ensured by BISC steps 1 and 2f), then the computed step (Dz, Dl) satisfies:

    Dz^T * nabla phi_mu(z) < 0

where phi_mu(z) = Sigma_k l_k(z_k) - mu * Sigma_k 1^T * log(-g_k(z_k)) is the log-barrier merit function.

**Proof sketch.**

The BISC step solves the modified KKT system:

    | H~ + E    J^T |   | Dz |       | nabla phi_mu |
    |               | * |    | = -   |              |
    | J         0   |   | Dl |       | c            |

where E = block-diag(E_0, ..., E_N) is the total BISC modification.

The **reduced Hessian** of the modified system is Z^T * (H~ + E) * Z, where Z spans the null space of J (i.e., J*Z = 0).

The backward Riccati with BISC computes an implicit LDL^T factorization of this reduced Hessian, where all diagonal blocks (M~_k and P~_k) are **PD by construction**. Therefore:

    Z^T * (H~ + E) * Z > 0     (positive definite)

From standard IPM theory (Nocedal & Wright, Ch. 19): when the reduced Hessian is PD, the KKT step is a descent direction:

    Dz^T * nabla phi_mu = -Dz^T * (H~ + E) * Dz < 0

QED.


### 5.3 Theorem 3 — BISC Perturbation is Smaller than IPOPT

**Theorem 3.**
Let delta_IP be IPOPT's inertia correction (scalar added to entire Hessian: H~ + delta_IP * I). Let E_BISC be the total BISC modification. Then:

    ||E_BISC||_F  <=  ||delta_IP * I||_F  =  delta_IP * sqrt(n_total)

with **strict inequality** whenever any stage has a healthy control Hessian (E_k^M = 0) or healthy cost-to-go (E_k^P = 0).

**Proof.**

IPOPT adds delta_IP to ALL n_total diagonal elements. Its perturbation norm:

    ||delta_IP * I||_F = delta_IP * sqrt(n_total)

BISC adds E_k^M **only to stages where M_k is indefinite**, and E_k^P **only where P_k is indefinite**. At each modified stage:

    ||E_k^M||_F^2 = Sigma_{i: lambda_i(M_k) < delta} (delta - lambda_i(M_k))^2

This sums over only the problematic eigenvalues, at only the problematic stages:

    ||E_BISC||_F^2 = Sigma_{k in S_bad} ||E_k||_F^2

where S_bad = {stages needing modification}. Since typically |S_bad| << N+1 (most stages are healthy), the inequality is strict.

**For any stage k where M_k has all eigenvalues >= delta:** BISC contributes 0, while IPOPT contributes delta_IP * sqrt(n_u) > 0. QED.


### 5.4 Theorem 4 — SOSC Recovery and Superlinear Convergence

**Definition (Second-Order Sufficient Condition).** SOSC holds at a KKT point (z*, lambda*, nu*) if the reduced Hessian of the Lagrangian is positive definite on the critical cone:

    d^T * nabla^2_{zz} L * d > 0    for all d != 0 with J*d = 0 and (active constraint conditions)

Let sigma > 0 be the minimum reduced Hessian eigenvalue at the solution.

**Theorem 4 (Exact Newton Recovery).**
Suppose SOSC holds with minimum eigenvalue sigma > 0. Then there exists a neighborhood of the solution such that for all iterates in this neighborhood:

    (i)   All control Hessians satisfy: lambda_min(M_k) >= sigma/c  for some constant c > 0
    (ii)  For delta < sigma/c (which holds when mu is sufficiently small): E_k^M = 0 at ALL stages
    (iii) Similarly: E_k^P = 0 at ALL stages
    (iv)  BISC produces the EXACT Newton step (zero modification everywhere)

**Proof.**

(i) Under SOSC, the reduced Hessian eigenvalues are continuous functions of the iterate and bounded below by sigma at the solution. By continuity, in a neighborhood they are bounded below by sigma/2 > 0. The control Hessian M_k is a principal submatrix of the Riccati-projected Hessian, so its eigenvalues are bounded below by the reduced Hessian eigenvalues (eigenvalue interlacing).

(ii) With delta = kappa * mu and mu -> 0, eventually delta < sigma/c. At this point, lambda_min(M_k) >= sigma/c > delta, so proj_delta(M_k) = M_k and E_k^M = 0.

(iii) By Proposition 1 (floor property), P_k >= Q~_k. The barrier contribution to Q~_k ensures it is PD near the solution. Therefore P_k is PD and E_k^P = 0.

(iv) With all modifications zero, BISC computes the **exact** Newton step.

**Corollary (Quadratic Convergence with AD).**
Since BISC eventually gives exact Newton steps, and AD provides exact Hessians, the IPM achieves quadratic convergence near the solution:

    ||(z_{j+1}, lambda_{j+1}) - (z*, lambda*)|| = O(||(z_j, lambda_j) - (z*, lambda*)||^2)

**This is the key result.** Other convexification methods (IPOPT's scalar delta_IP, Vanroye's dual regularization rho_d) always have nonzero perturbation and therefore cap convergence at superlinear at best.


### 5.5 Comparison with Existing Methods

**vs. Verschueren et al. (2017) — Sparsity-Preserving Convexification:**

| Aspect | Verschueren | BISC |
|--------|-------------|------|
| Context | SQP (QP subproblems) | IPM (barrier subproblem) |
| Projection | PROJECT or MIRROR heuristic | min Frobenius-norm projection |
| Parameter | epsilon (free, user-chosen) | delta = kappa * mu (tied to barrier) |
| Barrier benefit | N/A (no barrier) | Barrier terms reduce modification needed |
| Theory | Convergence of SQP with convexified QP | SOSC exact recovery, quadratic convergence |

Verschueren's PROJECT strategy (clip eigenvalues to epsilon) is the same **mechanism** as BISC's proj_delta, but BISC adds: (1) IPM context with barrier, (2) automatic delta from mu, (3) SOSC theory.

**vs. Vanroye et al. (2025) — Dual-Regularized Riccati for IPM:**

| Aspect | Vanroye | BISC |
|--------|---------|------|
| Modification type | Dual (add -rho_d * I to constraint block) | Primal (modify control Hessian M_k) |
| Selectivity | Global (ALL stages get rho_d) | Stage-selective (only indefinite stages) |
| Constraint preservation | Modified (changes J^T structure) | Preserved (J^T untouched) |
| Norm optimality | Scalar rho_d (over-regularizes most directions) | Frobenius-optimal per block |
| Near solution | Always rho_d > 0 | E = 0 under SOSC (exact Newton) |
| Convergence | Superlinear (at best) | Quadratic (exact Newton + exact Hessians) |


---

## 6. The Role of Automatic Differentiation

### 6.1 Why AD + BISC is the Right Combination

The Lagrangian Hessian at each stage:

    W_k = nabla^2 l_k(z_k)         (I: objective curvature, usually PSD)
        + lambda_k^T * nabla^2 f_k(z_k)    (II: dynamics curvature, CAN BE NEGATIVE)
        + nu_k^T * nabla^2 g_k(z_k)        (III: inequality curvature, can be negative)

**Without AD (Gauss-Newton):** Use only term I. Result is PSD. No convexification needed. But convergence is only **linear** because terms II and III are missing.

**With AD (exact Hessians):** All three terms are computed exactly. Convergence is **quadratic** (exact Newton). But W_k is indefinite. You NEED a convexification method.

**The convergence hierarchy:**

    Method                    | Hessian info   | Convergence  | Convexification
    --------------------------|----------------|--------------|------------------
    Gauss-Newton              | Term I only    | Linear       | Not needed
    L-BFGS                    | Approximate    | Superlinear  | Not needed (PD by construction)
    Exact Hessian + IPOPT     | All terms (AD) | Superlinear* | Global delta*I (wasteful)
    Exact Hessian + BISC      | All terms (AD) | Quadratic    | Min-norm per block (optimal)

    * IPOPT's scalar regularization always perturbs, capping at superlinear even near the solution.

**BISC unlocks the full potential of AD:** it preserves the curvature information as much as possible while guaranteeing descent at every iteration. Near the solution, the modification vanishes and you get pure quadratic Newton convergence.


### 6.2 What AD Provides to BISC

For the backward Riccati, BISC needs at each stage:

**First derivatives (Jacobians):**
- F_k = df_k/dx_k (n_x x n_x state Jacobian)
- G_k = df_k/du_k (n_x x n_u control Jacobian)
- J_{g,k} = dg_k/dz_k (p_k x n_z inequality Jacobian)

**Second derivatives (Hessian blocks):**
- Q~_k = d^2L/dx_k^2 + barrier_xx (n_x x n_x)
- R~_k = d^2L/du_k^2 + barrier_uu (n_u x n_u)
- N~_k = d^2L/(dx_k * du_k) + barrier_xu (n_x x n_u)

All computed by second-order AD (forward-over-reverse or forward-over-forward mode).

**Automatic structure detection:** The AD computation graph reveals that the Hessian is block-diagonal (f_k depends only on z_k, not z_{k+1}). The solver can automatically detect this OC structure and switch to Riccati-based BISC, without the user declaring "this is an optimal control problem."

**Cost of AD second derivatives:** O(n_z) times the cost of first derivatives (for forward-over-reverse). Typically small compared to the savings from fewer IPM iterations (quadratic vs linear convergence).


---

## 7. Summary: What Makes BISC Novel

**The five-axis novelty compared to ALL existing methods:**

    1. IPM context          — unlike Verschueren (SQP only)
    2. Primal modification  — unlike Vanroye (dual regularization)
    3. Stage-selective       — unlike IPOPT (global scalar) and Vanroye (all stages)
    4. Min Frobenius-norm    — unlike IPOPT (scalar overshoot) and Vanroye (scalar rho_d)
    5. SOSC exact recovery   — unlike IPOPT (always delta > 0) and Vanroye (always rho_d > 0)

**One sentence:** BISC applies the minimum-norm PSD projection to only the small control Hessian blocks that need it during backward Riccati, tied to the barrier parameter, giving guaranteed descent with zero modification near the solution.

**The AD narrative:** AD gives exact Hessians (enabling quadratic convergence) but creates indefiniteness. BISC handles the indefiniteness optimally. Together, AD + BISC deliver quadratic convergence for OC-structured IPMs — something no existing combination achieves.


---

## 8. Numerical Experiments (To Be Done)

**Planned tests:**

1. **Space shuttle reentry** (n_x=6, n_u=2, N=100): Highly nonlinear dynamics with large constraint curvature. Compare BISC vs IPOPT inertia correction vs Gauss-Newton.

2. **Race car lap optimization** (n_x=6-8, n_u=2-4, N=200-500): Real OCP with path constraints (track boundaries, tire forces). Show stage-selectivity — most stages healthy, few need modification.

3. **Convergence rate comparison:**
   - Gauss-Newton: linear rate (count iterations)
   - IPOPT + exact Hessian: superlinear (count iterations + trial factorizations)
   - BISC + exact Hessian: quadratic (count iterations, no trial factorizations)

4. **Perturbation norm comparison:** Plot ||E_BISC||_F vs ||delta_IP * I||_F across IPM iterations. Show BISC is always smaller, and converges to 0 while IPOPT's stays positive.

5. **Timing comparison:** Wall-clock time showing BISC's single pass vs IPOPT's multiple trial factorizations.

**Metrics to report:**
- IPM iteration count to convergence (tol = 1e-8)
- Total factorization count (BISC: always N, IPOPT: 2-5 * N per iteration)
- ||E||_F at each iteration (perturbation size)
- Final convergence rate (quadratic vs linear)
