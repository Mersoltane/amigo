"""
Parallel Sellar problem
-----------------------

    minimize    Y + exp(-Q) + sum_i gamma_i x_i^2 / 2
    subject to  y_i = z_i^2 + x_i - b_i Q,      y_i >= c_i,    i = 1..N
                q_j = a_j sqrt(Y) + h_j X + r_j,               j = 1..M
                Y = mu^T y,  X = nu^T x,  Q = omega^T q,  Q <= d

- Two groups of disciplines share only Y, X, and Q
"""

import argparse
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import lambertw

import amigo as am
from amigo import ExpressionComponent as ExprComp

Vector = NDArray[np.float64]


@dataclass(frozen=True)
class ParallelSellarData:
    """Coefficients of the two discipline groups and of their coupling"""

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


def generate_data(N: int = 5, M: int = 4) -> ParallelSellarData:
    """Coefficients for which the bounds Y >= mu^T c and Q <= d are both active"""
    if N <= 0 or M <= 0:
        raise ValueError("N and M must be positive")

    # First group coefficients are fixed and evenly spaced
    c = np.linspace(0.5, 2.0, N)
    gamma = np.linspace(2.0, 0.5, N)

    mu = np.linspace(0.5, 1.5, N)
    mu /= mu.sum()

    nu = np.linspace(1.5, 0.5, N)
    nu /= nu.sum()

    # Second group coefficients are fixed and evenly spaced
    a = np.linspace(0.5, 1.5, M)
    h = np.linspace(1.5, 0.5, M)
    r = np.linspace(0.0, 0.5, M)

    omega = np.linspace(0.5, 1.5, M)
    omega /= omega.sum()

    # Aggregate coefficients of the coupling
    C = float(mu @ c)
    A = float(omega @ a)
    H = float(omega @ h)
    R = float(omega @ r)

    L = float(np.sum(nu**2 / gamma))
    Dx = H**2 * L

    s_lower = np.sqrt(C)
    Q_min = R + A * s_lower

    # Both aggregate bounds stay active while 0 < d - Q_min < min(delta_q, delta_y)
    delta_q = float(lambertw(Dx * np.exp(-Q_min)).real)
    delta_y = 2.0 * Dx * s_lower / A
    delta_max = min(delta_q, delta_y)

    if not np.isfinite(delta_max) or delta_max <= 0.0:
        raise ValueError("Unable to generate a strictly both-active problem")

    # Sit halfway inside the window where both bounds stay active
    delta = 0.5 * delta_max
    d = float(Q_min + delta)

    # The optimal x when both aggregate bounds are active
    x = delta * nu / (H * L * gamma)

    # Every z_i stays real since z_i^2 = c_i - x_i + b_i d remains positive
    b_min = np.maximum(0.0, (x - c) / d)
    b = b_min + np.linspace(0.1, 0.5, N)

    return ParallelSellarData(
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
    """Submodel whose target output is the weighted sum of its source inputs"""
    model = am.Model()

    source = ExprComp(
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

    model.add_component("source", len(weights), source)
    model.add_component("target", 1, target)

    # The shared row sums the weighted inputs against the output
    model.link("source.con", "target.con")
    model.set_data("source.weight", weights)

    return model


def build_model(data: ParallelSellarData):
    """Parallel Sellar model with the reductions that couple the two groups"""
    N = len(data.c)
    M = len(data.a)

    # First group discipline with the bound on its own y
    comp1 = ExprComp(
        name="Comp1",
        inputs={
            "y": {},
            "z": {"value": 1.0, "lower": 0.0},
            "x": {"value": 1.0, "lower": 0.0},
            "Q": {"value": 0.0, "upper": data.d},
        },
        data=["b", "c"],
        constraints={
            "con1": "y - z**2 - x + b * Q",
            "con2": {"expr": "y - c", "lower": 0.0, "upper": am.inf},
        },
    )

    # Second group discipline driven by the aggregates Y and X
    comp2 = ExprComp(
        name="Comp2",
        inputs=["q", "Y", "X"],
        data=["a", "r", "h"],
        constraints={"con": "q - a * sqrt(Y) - h * X - r"},
    )

    # Objective terms on the aggregates and on every x_i
    obj1 = ExprComp(name="Obj1", inputs=["Y", "Q"], objective={"obj": "Y + exp(-Q)"})
    obj2 = ExprComp(
        name="Obj2",
        inputs=["x"],
        data=["gamma"],
        objective={"obj": "0.5 * gamma * x**2"},
        outputs={"output": "0.5 * gamma * x**2"},
    )

    model = am.Model("parallel_sellar")
    model.add_component("comp1", N, comp1)
    model.add_model("ycomp", make_reduction(data.mu))
    model.add_model("xcomp", make_reduction(data.nu))

    model.add_component("comp2", M, comp2)
    model.add_model("qcomp", make_reduction(data.omega))

    model.add_component("obj1", 1, obj1)
    model.add_component("obj2", N, obj2)

    # The aggregate Y joins the first group to the second group and the objective
    model.link("comp1.y", "ycomp.source.input")
    model.link("comp2.Y", "ycomp.target.output")
    model.link("ycomp.target.output", "obj1.Y")

    # The aggregate Q joins the second group back to the first group and the objective
    model.link("comp2.q", "qcomp.source.input")
    model.link("qcomp.target.output", "obj1.Q")
    model.link("qcomp.target.output", "comp1.Q")

    # The aggregate X carries the first group x to the second group
    model.link("comp1.x", "obj2.x")
    model.link("comp1.x", "xcomp.source.input")
    model.link("comp2.X", "xcomp.target.output")

    # The output sums the x terms of every discipline
    model.link("obj2.output[0]", "obj2.output[1:]")

    # The discipline coefficients enter as data
    values = model.get_meta_view("value", "data")
    values["comp1.b"] = data.b
    values["comp1.c"] = data.c
    values["comp2.a"] = data.a
    values["comp2.r"] = data.r
    values["comp2.h"] = data.h
    values["obj2.gamma"] = data.gamma

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build",
        dest="build",
        action="store_true",
        default=False,
        help="Enable building",
    )
    parser.add_argument(
        "--N", type=int, default=5, help="Disciplines in the first group"
    )
    parser.add_argument(
        "--M", type=int, default=4, help="Disciplines in the second group"
    )
    args = parser.parse_args()

    model = build_model(generate_data(args.N, args.M))

    if args.build:
        model.build_module()

    model.initialize()

    x = model.create_vector()
    opt = am.Optimizer(model, x)
    opt.optimize(
        {
            "initial_barrier_param": 0.1,
            "max_iterations": 500,
            "convergence_tolerance": 1e-12,
        }
    )
