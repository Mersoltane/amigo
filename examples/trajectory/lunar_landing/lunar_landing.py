"""
Soft lunar landing
------------------

    minimize    integral of u from 0 to tf
    subject to  y1' = y2,  y2' = u - c
                y1(0) = 10,  y2(0) = -2,  y1(tf) = y2(tf) = 0
                0 <= u <= 3
"""

import argparse
import json
import math
import os
import amigo as am
import numpy as np
from amigo.trajectory import (
    HPRefinement,
    MeshSolution,
    PHSOptions,
    PHSRefinement,
    RadauMesh,
    RefinementOptions,
    integration_matrix,
    mesh_time,
)

# Collocation on the standard or the flipped Radau points
flipped = False

# The hp refinement of Liu, Hager, and Rao or the phs refinement of Haman and Rao
refinement = "hp"

# Fewest and most collocation points in one interval
min_points = 3
max_points = 10

# Lunar gravity, thrust limit, and the initial and final states
C = 1.5
U_MAX = 3.0
Y0 = np.array([10.0, -2.0])
YF = np.array([0.0, 0.0])

# The coast and burn arcs meet the boundary conditions only at this switch time
T_SWITCH = (-12.0 + math.sqrt(612.0)) / 9.0
T_FINAL = 2.0 * T_SWITCH + 4.0 / 3.0
COST = U_MAX * (T_FINAL - T_SWITCH)


# Landing dynamics and the quadrature term of the cost at one collocation point
class Point(am.Component):
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

        return

    def compute(self):
        w = self.data["w"]

        tf = self.inputs["tf"]
        y = self.inputs["y"]
        ydot = self.inputs["ydot"]
        u = self.inputs["u"]

        self.constraints["res"] = [ydot[0] - y[1], ydot[1] - (u - C)]
        self.objective["obj"] = 0.5 * tf * w * u

        return


# Integral form of the Radau collocation on one interval with n points
class Radau(am.Component):
    def __init__(self, n, flipped):
        # Every number of points generates its own class
        if flipped:
            name = f"FlippedRadau{n}"
        else:
            name = f"Radau{n}"

        super().__init__(name=name)

        self.n = n
        self.I = integration_matrix(n, flipped)

        # Half width of the interval in mesh coordinates
        self.add_data("h", value=1.0)

        self.add_input("tf")
        self.add_input("Y", shape=(n + 1, 2))
        self.add_input("F", shape=(n, 2))

        self.add_constraint("res", shape=(n, 2))

        return

    def compute(self):
        h = self.data["h"]

        tf = self.inputs["tf"]
        Y = self.inputs["Y"]
        F = self.inputs["F"]

        scale = 0.5 * tf * h

        # Integrate the rates from the interval start to every later support point
        res = np.empty((self.n, 2), dtype=object)
        for i in range(self.n):
            for s in range(2):
                total = float(self.I[i, 0]) * F[0, s]
                for j in range(1, self.n):
                    total = total + float(self.I[i, j]) * F[j, s]
                res[i, s] = Y[i + 1, s] - Y[0, s] - scale * total

        self.constraints["res"] = res

        return


# Final time and the boundary conditions
class Boundary(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("tf", lower=0.1, upper=100.0, label="final time")
        self.add_input("y0", shape=2)
        self.add_input("yf", shape=2)

        self.add_constraint("initial", shape=2)
        self.add_constraint("final", shape=2)

        return

    def compute(self):
        y0 = self.inputs["y0"]
        yf = self.inputs["yf"]

        self.constraints["initial"] = [y0[0] - Y0[0], y0[1] - Y0[1]]
        self.constraints["final"] = [yf[0] - YF[0], yf[1] - YF[1]]

        return


def group_intervals(mesh):
    """Intervals of the mesh grouped by their number of points"""
    groups = {}

    for k in range(mesh.num_intervals):
        n = int(mesh.degrees[k])
        if n not in groups:
            groups[n] = []
        groups[n].append(k)

    return groups


def build_model(mesh, module_name):
    """Radau collocation of the landing on one mesh with intervals grouped by points"""
    groups = group_intervals(mesh)
    num_collocation = mesh.num_collocation
    last = mesh.num_intervals - 1

    model = am.Model(module_name)

    model.add_component("pts", num_collocation, Point())
    model.add_component("ends", 1, Boundary())

    for n in groups:
        model.add_component(f"radau{n}", len(groups[n]), Radau(n, mesh.flipped))

    # Link the final time to every point
    model.link("ends.tf[0]", "pts.tf[:]")

    for n in groups:
        intervals = groups[n]

        # Link the final time to every interval with n points
        model.link("ends.tf[0]", f"radau{n}.tf[:]")

        for idx in range(len(intervals)):
            k = intervals[idx]
            a = int(mesh.offsets[k])

            # Link the rates at the collocation points of the interval
            model.link(f"pts.ydot[{a}:{a + n}, :]", f"radau{n}.F[{idx}, :, :]")

            # Link the states and the support point from the neighbor or the boundary
            if mesh.flipped:
                model.link(f"pts.y[{a}:{a + n}, :]", f"radau{n}.Y[{idx}, 1:{n + 1}, :]")
                if k == 0:
                    model.link("ends.y0[0, :]", f"radau{n}.Y[{idx}, 0, :]")
                else:
                    model.link(f"pts.y[{a - 1}, :]", f"radau{n}.Y[{idx}, 0, :]")
            else:
                model.link(f"pts.y[{a}:{a + n}, :]", f"radau{n}.Y[{idx}, 0:{n}, :]")
                if k == last:
                    model.link("ends.yf[0, :]", f"radau{n}.Y[{idx}, {n}, :]")
                else:
                    model.link(f"pts.y[{a + n}, :]", f"radau{n}.Y[{idx}, {n}, :]")

    # Link the boundary state that sits on a collocation point
    if mesh.flipped:
        model.link("ends.yf[0, :]", f"pts.y[{num_collocation - 1}, :]")
    else:
        model.link("ends.y0[0, :]", "pts.y[0, :]")

    return model, groups


def catalog_mesh(min_points, max_points, flipped):
    """One interval per number of points so one build covers every refined mesh"""
    degrees = list(range(min_points, max_points + 1))
    points = np.linspace(-1.0, 1.0, len(degrees) + 1)

    return RadauMesh(points, degrees, flipped)


def set_mesh_data(model, mesh):
    """Quadrature weights of the points and half widths of the intervals"""
    groups = group_intervals(mesh)
    h = mesh.half_widths

    # Set the data by using a view of the data values
    data = model.get_meta_view("value", vtype="data")

    data["pts.w"] = mesh.quadrature_weights()

    for n in groups:
        data[f"radau{n}.h"] = h[groups[n]]

    return


def rates(states, controls, t):
    """Rates of altitude and velocity at a set of points"""
    states = np.atleast_2d(states)
    u = np.asarray(controls, dtype=float).reshape(len(states))

    return np.column_stack([states[:, 1], u - C])


def set_initial_guess(model, mesh, previous, tf_guess):
    """Initial point on a straight line or from the solution on the previous mesh"""
    if previous is None:
        # Draw the states on a straight line and set the thrust that cancels gravity
        tau = mesh.state_points()[:, None]
        states = Y0 + 0.5 * (tau + 1.0) * (YF - Y0)
        controls = np.full((mesh.num_collocation, 1), C)
        tf = tf_guess
    else:
        # Interpolate the states and the thrust of the previous mesh
        states = previous.mesh.interpolate_states(previous.states, mesh.state_points())
        controls = previous.mesh.interpolate_controls(
            previous.controls, mesh.collocation_points()
        )
        tf = previous.tf

    # The rates at the collocation points
    coll = mesh.collocated_states()
    t = mesh_time(mesh.collocation_points(), 0.0, tf)
    ydot = rates(states[coll], controls, t)

    # Set the initial point by using a view of the initial values
    x = model.get_meta_view("value")

    x["ends.tf"] = tf
    x["pts.y"] = states[coll]
    x["pts.u"] = np.clip(controls[:, 0], 0.0, U_MAX)
    x["pts.ydot"] = ydot

    # The uncollocated support point sits on the boundary
    if mesh.flipped:
        x["ends.y0"] = states[0]
    else:
        x["ends.yf"] = states[-1]

    return


# Solve the landing on one mesh from the solution on the previous mesh when there is one
class LandingSolver:
    def __init__(self, module_name, tf_guess, solver_options):
        self.module_name = module_name
        self.tf_guess = tf_guess
        self.solver_options = solver_options
        self.reports = []

        return

    def __call__(self, mesh, previous):
        """Solution of the landing on one mesh"""
        model = build_model(mesh, self.module_name)[0]
        set_mesh_data(model, mesh)
        set_initial_guess(model, mesh, previous, self.tf_guess)

        model.initialize()

        # Get the design variables
        x = model.create_vector()

        # Set up the optimizer and optimize
        opt = am.Optimizer(model, x=x)
        opt_data = opt.optimize(self.solver_options)

        # Copy the solution from the device to the host
        x.copy_device_to_host()

        # The states on the support points with the uncollocated one from the boundary
        states = np.empty((mesh.num_states, 2))
        states[mesh.collocated_states()] = np.asarray(x["pts.y"])

        if mesh.flipped:
            states[0] = np.asarray(x["ends.y0"]).ravel()
        else:
            states[-1] = np.asarray(x["ends.yf"]).ravel()

        controls = np.asarray(x["pts.u"]).reshape(-1, 1)
        tf = float(x["ends.tf"][0])

        solution = MeshSolution(
            mesh, states, controls, 0.0, tf, converged=bool(opt_data["converged"])
        )

        # Report whether the solve converged and how many iterations it took
        report = {
            "converged": bool(opt_data["converged"]),
            "iterations": len(opt_data["iterations"]),
        }
        self.reports.append(report)

        return solution


def cost(solution):
    """Quadrature of the cost on the mesh of a solution"""
    w = solution.mesh.quadrature_weights()

    return 0.5 * (solution.tf - solution.t0) * float(w @ solution.controls[:, 0])


def switch_time(solution):
    """Time where the thrust polynomial first crosses half of its limit"""
    tau = np.linspace(-1.0, 1.0, 20001)
    u = solution.mesh.interpolate_controls(solution.controls, tau)[:, 0]
    above = np.nonzero(u > 0.5 * U_MAX)[0]

    # A thrust that never rises past half or starts above it has no switch
    if len(above) == 0 or above[0] == 0:
        return float("nan")

    # Interpolate the crossing between the samples on either side of it
    i = above[0]
    frac = (0.5 * U_MAX - u[i - 1]) / (u[i] - u[i - 1])
    tau_switch = tau[i - 1] + frac * (tau[i] - tau[i - 1])

    return float(mesh_time(tau_switch, solution.t0, solution.tf))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build", dest="build", action="store_true", default=False, help="Enable building"
    )
    args = parser.parse_args()

    # Name the module after the collocation points
    if flipped:
        module_name = "lunar_landing_hp_flipped"
    else:
        module_name = "lunar_landing_hp"

    # Build one module with a Radau class for every number of points
    if args.build:
        catalog = catalog_mesh(min_points, max_points, flipped)
        model = build_model(catalog, module_name)[0]
        model.build_module()

    # Set up the mesh refinement
    if refinement == "hp":
        refinement_options = RefinementOptions(
            smoothness_threshold=3.0,
            tolerance=1e-6,
            max_subintervals=10,
            max_points=max_points,
            min_points=min_points,
            max_iterations=10,
        )
        mesh_refinement = HPRefinement(refinement_options, rates)
    else:
        refinement_options = PHSOptions(
            tolerance=1e-6,
            min_points=min_points,
            max_points=max_points,
            max_iterations=40,
            ode_tolerance=1e-10,
        )
        mesh_refinement = PHSRefinement(refinement_options, rates)

    opt_options = {
        "convergence_tolerance": 1e-8,
        "max_iterations": 500,
    }

    # Refine from ten equal intervals of three points
    solve = LandingSolver(module_name, tf_guess=5.0, solver_options=opt_options)
    initial_mesh = RadauMesh.uniform(10, 3, flipped)
    history = mesh_refinement.run(initial_mesh, solve)

    # Print the exact solution and the result on every mesh
    print(f"\nExact solution: ts = {T_SWITCH:.6f}, tf = {T_FINAL:.6f}, J = {COST:.6f}")
    print("mesh  intervals  points  max error   tf          J           ts          NLP")

    records = []

    for i in range(len(history)):
        step = history[i]
        report = solve.reports[i]
        sol = step.solution
        mesh = sol.mesh

        J = cost(sol)
        ts = switch_time(sol)

        if report["converged"]:
            status = "converged"
        else:
            status = "failed"

        print(
            f"{i:4d}  {mesh.num_intervals:9d}  {mesh.num_collocation:6d}  "
            f"{step.errors.max():.3e}  {sol.tf:.8f}  {J:.8f}  {ts:.8f}  "
            f"{status} in {report['iterations']}"
        )

        # Keep the mesh and the solution for the plot script
        if step.smoothness is None:
            smoothness = None
        else:
            smoothness = step.smoothness.tolist()

        t_collocation = mesh_time(mesh.collocation_points(), sol.t0, sol.tf)
        t_states = mesh_time(mesh.state_points(), sol.t0, sol.tf)

        records.append(
            {
                "mesh_points": mesh.points.tolist(),
                "degrees": mesh.degrees.tolist(),
                "errors": step.errors.tolist(),
                "smoothness": smoothness,
                "tf": sol.tf,
                "cost": J,
                "switch_time": ts,
                "t_collocation": t_collocation.tolist(),
                "t_states": t_states.tolist(),
                "states": sol.states.tolist(),
                "controls": sol.controls.tolist(),
                "nlp": report,
            }
        )

    # Save the meshes and the solutions for the plot script
    results_name = "lunar_landing"
    if flipped:
        results_name = results_name + "_flipped"
    if refinement == "phs":
        results_name = results_name + "_phs"

    path = os.path.join("results", results_name + ".json")
    exact = {"ts": T_SWITCH, "tf": T_FINAL, "cost": COST}

    os.makedirs("results", exist_ok=True)
    with open(path, "w") as fp:
        json.dump({"exact": exact, "meshes": records}, fp)

    print(f"Saved {path}")
