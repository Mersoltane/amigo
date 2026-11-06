from dataclasses import dataclass, fields
import numpy as np
import amigo as am
from amigo import ExpressionComponent as ExprComp
import argparse
from numpy.typing import ArrayLike, NDArray
from scipy.special import lambertw
import matplotlib.pylab as plt
from scipy.sparse import csr_matrix

Vector = NDArray[np.float64]


@dataclass(frozen=True)
class ParallelMDOData:
    b: Vector
    c: Vector
    a: Vector
    h: Vector
    r: Vector
    mu: Vector
    nu: Vector
    d: float
    omega: Vector
    gamma: Vector


@dataclass(frozen=True)
class ParallelMDOSolution:
    x: Vector
    z: Vector
    y: Vector
    q: Vector
    X: float
    Y: float
    Q: float
    objective: float
    active_set: str
    y_is_unique: bool


@dataclass(frozen=True)
class ParallelMDOPostOptimality:
    """A post-optimal output and its total derivative with respect to data."""

    value: float
    gradient: ParallelMDOData
    active_set: str


def generate_data(N: int = 5, M: int = 7) -> ParallelMDOData:
    """Generate coefficients for which Y=C and Q=d are both active."""

    if N <= 0 or M <= 0:
        raise ValueError("N and M must be positive")

    rng = np.random.default_rng()

    # First-group coefficients.
    c = rng.uniform(0.5, 2.0, size=N)
    gamma = rng.uniform(0.5, 2.0, size=N)

    mu = rng.uniform(0.5, 1.5, size=N)
    mu /= mu.sum()

    nu = rng.uniform(0.5, 1.5, size=N)
    nu /= nu.sum()

    # Second-group coefficients.
    a = rng.uniform(0.5, 1.5, size=M)
    h = rng.uniform(0.5, 1.5, size=M)
    r = rng.uniform(0.0, 0.5, size=M)

    omega = rng.uniform(0.5, 1.5, size=M)
    omega /= omega.sum()

    # Aggregate coefficients.
    C = float(mu @ c)
    A = float(omega @ a)
    H = float(omega @ h)
    R = float(omega @ r)

    L = float(np.sum(nu**2 / gamma))
    Dx = H**2 * L

    s_lower = np.sqrt(C)
    Q_min = R + A * s_lower

    # Both aggregate bounds remain active whenever
    #
    # 0 < d - Q_min < min(delta_q, delta_y).
    #
    delta_q = float(lambertw(Dx * np.exp(-Q_min)).real)
    delta_y = 2.0 * Dx * s_lower / A

    delta_max = min(delta_q, delta_y)

    if not np.isfinite(delta_max) or delta_max <= 0.0:
        raise ValueError("Unable to generate a strictly both-active problem")

    # Stay away from active-set transitions.
    theta = rng.uniform(0.25, 0.75)
    delta = theta * delta_max
    d = float(Q_min + delta)

    # Exact optimizer when both aggregate bounds are active.
    x = delta * nu / (H * L * gamma)

    # Choose b so every z_i remains real:
    # z_i^2 = c_i - x_i + b_i*d > 0.
    b_min = np.maximum(0.0, (x - c) / d)
    b = b_min + rng.uniform(0.1, 0.5, size=N)

    return ParallelMDOData(
        b=b,
        c=c,
        a=a,
        h=h,
        r=r,
        mu=mu,
        nu=nu,
        d=d,
        omega=omega,
        gamma=gamma,
    )


def make_reduction(weights):
    model = am.Model()

    reduction = ExprComp(
        name="Source",
        inputs=["input"],
        data=["weight"],
        constraints={"con": "-input * weight"},
    )
    target = ExprComp(
        name="Target",
        inputs={"output": {"value": 1.0, "lower": 0.0}},
        constraints={"con": "output"},
    )

    model.add_component("source", len(weights), reduction)
    model.add_component("target", 1, target)

    model.link("source.con", "target.con")
    model.set_data("source.weight", weights)

    return model


parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
parser.add_argument("--N", type=int, default=5)
parser.add_argument("--M", type=int, default=4)
args = parser.parse_args()

# Set up the problem data
N = args.N
M = args.M
problem_data = generate_data(N, M)

# Compute the exact solution
comp1 = ExprComp(
    name="Comp1",
    inputs={
        "y": {},
        "z": {"value": 1.0, "lower": 0.0},
        "x": {"value": 1.0, "lower": 0.0},
        "Q": {"value": 0.0, "upper": problem_data.d},
    },
    data=["b", "c"],
    constraints={
        "con1": "y - z**2 - x + b * Q",
        "con2": {"expr": "y - c", "lower": 0.0, "upper": am.inf},
    },
)
comp2 = ExprComp(
    name="Comp2",
    inputs=["q", "Y", "X"],
    data=["a", "r", "h"],
    constraints={"con": "q - a * sqrt(Y) - h * X - r"},
)

obj1 = ExprComp(name="Obj1", inputs=["Y", "Q"], objective={"obj": "Y + exp(-Q)"})
obj2 = ExprComp(
    name="Obj2",
    inputs=["x"],
    data=["gamma"],
    objective={"obj": "0.5 * gamma * x**2"},
    outputs={"output": "0.5 * gamma * x**2"},
)

# Create the model
model = am.Model("mdo_problem")
model.add_component("comp1", N, comp1)
model.add_model("ycomp", make_reduction(problem_data.mu))
model.add_model("xcomp", make_reduction(problem_data.nu))

model.add_component("comp2", M, comp2)
model.add_model("qcomp", make_reduction(problem_data.omega))

# Add the objective terms
model.add_component("obj1", 1, obj1)
model.add_component("obj2", N, obj2)

# Link the variables
model.link("comp1.y", "ycomp.source.input")
model.link("comp2.q", "qcomp.source.input")
model.link("comp2.Y", "ycomp.target.output")

model.link("ycomp.target.output", "obj1.Y")
model.link("qcomp.target.output", "obj1.Q")
model.link("comp1.x", "obj2.x")
model.link("comp1.x", "xcomp.source.input")
model.link("comp2.X", "xcomp.target.output")

# Link the output
model.link("obj2.output[0]", "obj2.output[1:]")

# Set the remaining data
data = model.get_meta_view("value", "data")
data["comp1.b"] = problem_data.b
data["comp1.c"] = problem_data.c
data["comp2.a"] = problem_data.a
data["comp2.r"] = problem_data.r
data["comp2.h"] = problem_data.h
data["obj2.gamma"] = problem_data.gamma

if args.build:
    model.build_module(data_derivatives=True)

model.initialize()

x = model.create_vector()
opt = am.Optimizer(model, x)
data = opt.optimize(
    {
        "initial_barrier_param": 0.1,
        "max_iterations": 500,
        "convergence_tolerance": 1e-12,
    }
)

# Create the post-optimality derivatives
of = ["obj2.output[0]"]
dfdx, of_map, wrt_map = opt.compute_post_opt_derivatives(of=of)

hessian = opt.solver.hessian
nrows, ncols, nnz, rowp, cols = hessian.get_nonzero_structure()
data = np.ones(nnz)
H = csr_matrix((data, cols, rowp), shape=(nrows, ncols))

fig, ax = plt.subplots(figsize=(6, 6))
ax.spy(H, markersize=5.0)

fig.tight_layout()  # Minimize padding/overlap
ax.set_axis_off()  # Hide axes, ticks, and frame
ax.set_aspect("equal")  # Equal scale in x and y

fig.savefig("hessian_pattern.png", bbox_inches="tight", pad_inches=0)
