"""Soft lunar landing of Meditch solved with the hp mesh refinement of Liu, Hager, and Rao.

The phs mesh refinement of G. V. Haman III and A. V. Rao, Adaptive mesh
refinement and error estimation method for optimal control using direct
collocation, arXiv 2410.07488, 2024, can replace it for comparison.

This is the example of section V of F. Liu, W. W. Hager, and A. V. Rao, An hp
mesh refinement method for optimal control using discontinuity detection and
mesh size reduction, 53rd IEEE Conference on Decision and Control, 2014.
Minimize the integral of the thrust u from 0 to tf subject to y1' = y2 and
y2' = u - c with c = 1.5, y1(0) = 10, y2(0) = -2, y1(tf) = y2(tf) = 0, and
0 <= u <= 3. The optimal thrust is zero up to ts = 1.4154 and 3 after it.
"""

import argparse
import json
import math
import os

import numpy as np
import amigo as am
from amigo.trajectory import (
    HPRefinement,
    MeshSolution,
    PHSOptions,
    PHSRefinement,
    RadauMesh,
    RefinementOptions,
    integration_matrix,
    mesh_time,
    radau_weights,
)

C = 1.5
U_MAX = 3.0
Y0 = np.array([10.0, -2.0])
YF = np.array([0.0, 0.0])

# The coast and burn arcs meet the boundary conditions only at this switch time
T_SWITCH = (-12.0 + math.sqrt(612.0)) / 9.0
T_FINAL = 2.0 * T_SWITCH + 4.0 / 3.0
COST = U_MAX * (T_FINAL - T_SWITCH)


class Point(am.Component):
    """Landing dynamics and the quadrature term of the cost at one collocation point."""

    def __init__(self):
        super().__init__()

        # Radau weight of the point times the half width of its interval
        self.add_data("w", value=1.0)

        self.add_input("tf")
        self.add_input("y", shape=2, label="altitude and velocity")
        self.add_input("ydot", shape=2, label="rates")
        self.add_input("u", lower=0.0, upper=U_MAX, label="thrust")

        self.add_constraint("res", shape=2, label="dynamics")
        self.add_objective("obj")

    def compute(self):
        y = self.inputs["y"]
        ydot = self.inputs["ydot"]
        u = self.inputs["u"]
        self.constraints["res"] = [ydot[0] - y[1], ydot[1] - (u - C)]
        self.objective["obj"] = 0.5 * self.inputs["tf"] * self.data["w"] * u


class Radau(am.Component):
    """Collocation conditions of Eq. 10 on one interval with n collocation points."""

    def __init__(self, n, flipped):
        # Every number of points generates its own class
        super().__init__(name=f"{'Flipped' if flipped else ''}Radau{n}")
        self.n = n
        self.I = integration_matrix(n, flipped)

        # Half width of the interval in mesh coordinates
        self.add_data("h", value=1.0)

        self.add_input("tf")
        self.add_input("Y", shape=(n + 1, 2))
        self.add_input("F", shape=(n, 2))
        self.add_constraint("res", shape=(n, 2))

    def compute(self):
        Y = self.inputs["Y"]
        F = self.inputs["F"]
        scale = 0.5 * self.inputs["tf"] * self.data["h"]
        res = np.empty((self.n, 2), dtype=object)
        for i in range(self.n):
            for s in range(2):
                total = float(self.I[i, 0]) * F[0, s]
                for j in range(1, self.n):
                    total = total + float(self.I[i, j]) * F[j, s]
                res[i, s] = Y[i + 1, s] - Y[0, s] - scale * total
        self.constraints["res"] = res


class Boundary(am.Component):
    """Final time and the boundary conditions."""

    def __init__(self):
        super().__init__()
        self.add_input("tf", lower=0.1, upper=100.0, label="final time")
        self.add_input("y0", shape=2)
        self.add_input("yf", shape=2)
        self.add_constraint("initial", shape=2)
        self.add_constraint("final", shape=2)

    def compute(self):
        y0 = self.inputs["y0"]
        yf = self.inputs["yf"]
        self.constraints["initial"] = [y0[0] - Y0[0], y0[1] - Y0[1]]
        self.constraints["final"] = [yf[0] - YF[0], yf[1] - YF[1]]


def rates(states, controls, t):
    """Rates of altitude and velocity at a set of points."""
    states = np.atleast_2d(states)
    u = np.asarray(controls, dtype=float).reshape(len(states))
    return np.column_stack([states[:, 1], u - C])


def build_model(mesh, module_name):
    """The amigo model of the Radau collocation on one mesh, with the intervals grouped by points."""
    model = am.Model(module_name)
    M = mesh.num_collocation
    model.add_component("pts", M, Point())
    model.add_component("ends", 1, Boundary())

    groups = {}
    for k, n in enumerate(mesh.degrees):
        groups.setdefault(int(n), []).append(k)
    for n, intervals in groups.items():
        model.add_component(f"radau{n}", len(intervals), Radau(n, mesh.flipped))

    model.link("ends.tf[0]", "pts.tf[:]")
    last = mesh.num_intervals - 1
    for n, intervals in groups.items():
        model.link("ends.tf[0]", f"radau{n}.tf[:]")
        for idx, k in enumerate(intervals):
            a = int(mesh.offsets[k])
            model.link(f"pts.ydot[{a}:{a + n}, :]", f"radau{n}.F[{idx}, :, :]")

            # The uncollocated support point is shared with the neighbor or held by the boundary
            if mesh.flipped:
                model.link(f"pts.y[{a}:{a + n}, :]", f"radau{n}.Y[{idx}, 1:{n + 1}, :]")
                start = "ends.y0[0, :]" if k == 0 else f"pts.y[{a - 1}, :]"
                model.link(start, f"radau{n}.Y[{idx}, 0, :]")
            else:
                model.link(f"pts.y[{a}:{a + n}, :]", f"radau{n}.Y[{idx}, 0:{n}, :]")
                end = "ends.yf[0, :]" if k == last else f"pts.y[{a + n}, :]"
                model.link(end, f"radau{n}.Y[{idx}, {n}, :]")

    if mesh.flipped:
        model.link("ends.yf[0, :]", f"pts.y[{M - 1}, :]")
    else:
        model.link("ends.y0[0, :]", "pts.y[0, :]")
    return model, groups


def catalog_mesh(min_points, max_points, flipped):
    """A mesh with one interval for every number of points, so one build covers every refined mesh."""
    degrees = list(range(min_points, max_points + 1))
    return RadauMesh(np.linspace(-1.0, 1.0, len(degrees) + 1), degrees, flipped)


class LandingSolver:
    """Solve the landing on a mesh, starting from the previous solution when there is one."""

    def __init__(self, module_name, tf_guess, solver_options):
        self.module_name = module_name
        self.tf_guess = tf_guess
        self.solver_options = solver_options
        self.reports = []

    def guess(self, mesh, previous):
        """States on the support points, controls on the collocation points, and the final time."""
        if previous is None:
            tau = mesh.state_points()[:, None]
            states = Y0 + 0.5 * (tau + 1.0) * (YF - Y0)
            controls = np.full((mesh.num_collocation, 1), C)
            return states, controls, self.tf_guess
        states = previous.mesh.interpolate_states(previous.states, mesh.state_points())
        controls = previous.mesh.interpolate_controls(previous.controls, mesh.collocation_points())
        return states, controls, previous.tf

    def __call__(self, mesh, previous):
        model, groups = build_model(mesh, self.module_name)
        model.initialize()

        data = model.get_data_vector()
        h = mesh.half_widths
        data["pts.w"] = np.concatenate(
            [h[k] * radau_weights(n, mesh.flipped) for k, n in enumerate(mesh.degrees)]
        )
        for n, intervals in groups.items():
            data[f"radau{n}.h"] = h[intervals]
        data.get_vector().copy_host_to_device()

        states, controls, tf = self.guess(mesh, previous)
        coll = mesh.collocated_states()
        t = mesh_time(mesh.collocation_points(), 0.0, tf)
        x = model.create_vector()
        x["ends.tf"] = tf
        x["pts.y"] = states[coll]
        x["pts.u"] = np.clip(controls[:, 0], 0.0, U_MAX)
        x["pts.ydot"] = rates(states[coll], controls, t)
        if mesh.flipped:
            x["ends.y0"] = states[0]
        else:
            x["ends.yf"] = states[-1]

        opt = am.Optimizer(model, x)
        result = opt.optimize(self.solver_options)
        x = opt.get_optimized_point()

        states = np.empty((mesh.num_states, 2))
        states[coll] = np.asarray(x["pts.y"])
        if mesh.flipped:
            states[0] = np.asarray(x["ends.y0"]).ravel()
        else:
            states[-1] = np.asarray(x["ends.yf"]).ravel()
        controls = np.asarray(x["pts.u"]).reshape(-1, 1)
        solution = MeshSolution(mesh, states, controls, 0.0, float(x["ends.tf"][0]))
        self.reports.append(
            {"converged": bool(result["converged"]), "iterations": len(result["iterations"])}
        )
        return solution


def cost(solution):
    """The quadrature of the cost on the mesh of a solution."""
    mesh = solution.mesh
    w = np.concatenate(
        [h * radau_weights(n, mesh.flipped) for h, n in zip(mesh.half_widths, mesh.degrees)]
    )
    return 0.5 * (solution.tf - solution.t0) * float(w @ solution.controls[:, 0])


def switch_time(solution):
    """Time where the thrust polynomial first crosses half of its limit."""
    tau = np.linspace(-1.0, 1.0, 20001)
    u = solution.mesh.interpolate_controls(solution.controls, tau)[:, 0]
    above = np.nonzero(u > 0.5 * U_MAX)[0]
    if len(above) == 0 or above[0] == 0:
        return float("nan")
    i = above[0]
    frac = (0.5 * U_MAX - u[i - 1]) / (u[i] - u[i - 1])
    return float(mesh_time(tau[i - 1] + frac * (tau[i] - tau[i - 1]), solution.t0, solution.tf))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", default=False, help="Enable building")
    parser.add_argument("--flipped", action="store_true", default=False, help="Flipped Radau points")
    parser.add_argument(
        "--refinement", choices=["hp", "phs"], default="hp",
        help="Mesh refinement of Liu, Hager, and Rao or of Haman and Rao",
    )
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Mesh accuracy tolerance of the paper")
    parser.add_argument("--rmax", type=int, default=10, help="Most subintervals of the paper")
    parser.add_argument("--nmax", type=int, default=10, help="Most collocation points of the paper")
    parser.add_argument("--intervals", type=int, default=10, help="Initial intervals of the paper")
    parser.add_argument("--points", type=int, default=3, help="Initial collocation points of the paper")
    parser.add_argument("--Q", type=float, default=3.0, help="Smoothness threshold, not given in the paper")
    parser.add_argument(
        "--nmin", type=int, default=3,
        help="Fewest points after a reduction, not given in the paper, and Ref. 8 starts its meshes at this value",
    )
    parser.add_argument(
        "--max-meshes", type=int, default=None,
        help="Most meshes, 10 for the hp method where the paper gives none and the 40 of Haman and Rao for phs",
    )
    parser.add_argument("--ode-tol", type=float, default=1e-10, help="Flight tolerance of the phs error estimate")
    parser.add_argument("--tf-guess", type=float, default=5.0, help="Final time guess, not given in the paper")
    parser.add_argument("--nlp-tol", type=float, default=1e-8, help="NLP tolerance, the IPOPT default that Ref. 8 used")
    parser.add_argument("--out", default="results", help="Directory for the JSON results")
    args = parser.parse_args()
    if args.max_meshes is None:
        args.max_meshes = 10 if args.refinement == "hp" else 40

    module_name = "lunar_landing_hp_flipped" if args.flipped else "lunar_landing_hp"
    if args.build:
        catalog, _ = build_model(catalog_mesh(args.nmin, args.nmax, args.flipped), module_name)
        catalog.build_module()

    if args.refinement == "hp":
        options = RefinementOptions(
            smoothness_threshold=args.Q,
            tolerance=args.tolerance,
            max_subintervals=args.rmax,
            max_points=args.nmax,
            min_points=args.nmin,
            max_iterations=args.max_meshes,
        )
        refinement = HPRefinement(options, rates)
    else:
        options = PHSOptions(
            tolerance=args.tolerance,
            min_points=args.nmin,
            max_points=args.nmax,
            max_iterations=args.max_meshes,
            ode_tolerance=args.ode_tol,
        )
        refinement = PHSRefinement(options, rates)
    solve = LandingSolver(
        module_name,
        args.tf_guess,
        {"convergence_tolerance": args.nlp_tol, "max_iterations": 500},
    )
    history = refinement.run(RadauMesh.uniform(args.intervals, args.points, args.flipped), solve)

    print(f"\nExact solution: ts = {T_SWITCH:.6f}, tf = {T_FINAL:.6f}, J = {COST:.6f}")
    print("mesh  intervals  points  max error   tf          J           ts          NLP")
    records = []
    for i, (step, report) in enumerate(zip(history, solve.reports)):
        sol = step.solution
        mesh = sol.mesh
        print(
            f"{i:4d}  {mesh.num_intervals:9d}  {mesh.num_collocation:6d}  {step.errors.max():.3e}  "
            f"{sol.tf:.8f}  {cost(sol):.8f}  {switch_time(sol):.8f}  "
            f"{'converged' if report['converged'] else 'failed'} in {report['iterations']}"
        )
        records.append(
            {
                "mesh_points": mesh.points.tolist(),
                "degrees": mesh.degrees.tolist(),
                "errors": step.errors.tolist(),
                "smoothness": None if step.smoothness is None else step.smoothness.tolist(),
                "tf": sol.tf,
                "cost": cost(sol),
                "switch_time": switch_time(sol),
                "t_collocation": mesh_time(mesh.collocation_points(), sol.t0, sol.tf).tolist(),
                "t_states": mesh_time(mesh.state_points(), sol.t0, sol.tf).tolist(),
                "states": sol.states.tolist(),
                "controls": sol.controls.tolist(),
                "nlp": report,
            }
        )

    os.makedirs(args.out, exist_ok=True)
    name = f"lunar_landing{'_flipped' if args.flipped else ''}{'_phs' if args.refinement == 'phs' else ''}.json"
    with open(os.path.join(args.out, name), "w") as fp:
        json.dump({"exact": {"ts": T_SWITCH, "tf": T_FINAL, "cost": COST}, "meshes": records}, fp)
    print(f"Saved {os.path.join(args.out, name)}")


if __name__ == "__main__":
    main()
