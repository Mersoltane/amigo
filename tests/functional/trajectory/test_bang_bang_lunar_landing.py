"""BangBangSolver on the soft lunar landing against its exact switch time, final time, and cost.

The landing of Meditch burns at full thrust after a coast, so the solver
must read the coast and burn arcs from the switching function of the single
phase and solve them with a free switch time. The arcs of the exact solution
are polynomials, so the solved switch time, final time, and cost must match
the closed form to collocation accuracy. The arc model written here is the
measure of what a new bang-bang problem still has to supply.
"""

import os
import sys

import numpy as np
import pytest

import amigo as am
from amigo.trajectory import (
    BangBangOptions,
    BangBangProblem,
    BangBangSolver,
    MeshSolution,
    PHSOptions,
    PHSRefinement,
    RadauMesh,
    integration_matrix,
    mesh_time,
    quadrature_costate,
    radau_weights,
    switching_arcs,
    uniform_arc_meshes,
)

EXAMPLE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples", "trajectory", "lunar_landing")
)
if EXAMPLE not in sys.path:
    sys.path.insert(0, EXAMPLE)
import lunar_landing as ll  # noqa: E402

SINGLE_MODULE = "landing_bang_single_test"
ARC_MODULE = "landing_bang_arcs_test"

# Smallest duration of an arc
FLOOR = 1e-3


class ArcPoint(am.Component):
    """Landing dynamics at one collocation point of an arc, at full thrust or at none."""

    def __init__(self, thrust):
        super().__init__(name="BurnPoint" if thrust else "CoastPoint")
        self.thrust = thrust
        self.add_input("y", shape=2, label="altitude and velocity")
        self.add_input("ydot", shape=2, label="rates")
        self.add_constraint("res", shape=2, label="dynamics")

    def compute(self):
        y = self.inputs["y"]
        ydot = self.inputs["ydot"]
        self.constraints["res"] = [ydot[0] - y[1], ydot[1] - (self.thrust - ll.C)]


class ArcRadau(am.Component):
    """Collocation conditions on one interval of an arc between its free start and end times."""

    def __init__(self, n):
        super().__init__(name=f"ArcRadau{n}")
        self.n = n
        self.I = integration_matrix(n, flipped=True)

        # Half width of the interval in the mesh coordinates of its arc
        self.add_data("h", value=1.0)

        self.add_input("a")
        self.add_input("b")
        self.add_input("Y", shape=(n + 1, 2))
        self.add_input("F", shape=(n, 2))
        self.add_constraint("res", shape=(n, 2))

    def compute(self):
        Y = self.inputs["Y"]
        F = self.inputs["F"]
        scale = 0.5 * (self.inputs["b"] - self.inputs["a"]) * self.data["h"]
        res = np.empty((self.n, 2), dtype=object)
        for i in range(self.n):
            for s in range(2):
                total = float(self.I[i, 0]) * F[0, s]
                for j in range(1, self.n):
                    total = total + float(self.I[i, j]) * F[j, s]
                res[i, s] = Y[i + 1, s] - Y[0, s] - scale * total
        self.constraints["res"] = res


class ArcEnds(am.Component):
    """Boundary conditions of the landing on the first and last states of the arcs."""

    def __init__(self):
        super().__init__()
        self.add_input("y0", shape=2)
        self.add_input("yf", shape=2)
        self.add_constraint("initial", shape=2)
        self.add_constraint("final", shape=2)

    def compute(self):
        y0 = self.inputs["y0"]
        yf = self.inputs["yf"]
        self.constraints["initial"] = [y0[0] - ll.Y0[0], y0[1] - ll.Y0[1]]
        self.constraints["final"] = [yf[0] - ll.YF[0], yf[1] - ll.YF[1]]


class ArcSpan(am.Component):
    """Start and end times of one arc with its duration held above the floor."""

    def __init__(self):
        super().__init__()
        self.add_input("start", lower=0.0, upper=100.0)
        self.add_input("end", lower=0.0, upper=100.0)
        self.add_constraint("duration", lower=FLOOR, upper=am.inf)

    def compute(self):
        self.constraints["duration"] = self.inputs["end"] - self.inputs["start"]


class BurnCost(am.Component):
    """Fuel of one burn arc at full thrust."""

    def __init__(self):
        super().__init__()
        self.add_input("start")
        self.add_input("end")
        self.add_objective("obj")

    def compute(self):
        self.objective["obj"] = ll.U_MAX * (self.inputs["end"] - self.inputs["start"])


def build_arc_model(kinds, meshes, module):
    """The amigo model with one phase per arc, each on its own mesh between its start and end times."""
    model = am.Model(module)
    counts = {"burn": 0, "coast": 0}
    slots = []
    for kind, mesh in zip(kinds, meshes):
        slots.append((kind, counts[kind], mesh.num_collocation))
        counts[kind] += mesh.num_collocation
    if counts["burn"]:
        model.add_component("burn", counts["burn"], ArcPoint(ll.U_MAX))
    if counts["coast"]:
        model.add_component("coast", counts["coast"], ArcPoint(0.0))
    model.add_component("ends", 1, ArcEnds())
    model.add_component("arcs", len(kinds), ArcSpan())
    burns = [p for p, kind in enumerate(kinds) if kind == "burn"]
    if burns:
        model.add_component("fuel", len(burns), BurnCost())
    for c, p in enumerate(burns):
        model.link(f"fuel.start[{c}]", f"arcs.start[{p}]")
        model.link(f"fuel.end[{c}]", f"arcs.end[{p}]")

    # Consecutive arcs share their switch time
    for p in range(len(kinds) - 1):
        model.link(f"arcs.end[{p}]", f"arcs.start[{p + 1}]")

    # The intervals of all arcs are grouped by their points, each as its arc, kind, points, offset, and half width
    intervals = []
    for p, (kind, mesh) in enumerate(zip(kinds, meshes)):
        base = slots[p][1]
        for k, n in enumerate(mesh.degrees):
            intervals.append((p, kind, int(n), base + int(mesh.offsets[k]), float(mesh.half_widths[k])))
    groups = {}
    for i, interval in enumerate(intervals):
        groups.setdefault(interval[2], []).append(i)
    for n, members in groups.items():
        model.add_component(f"radau{n}", len(members), ArcRadau(n))

    # Each interval starts at the last collocation point before it and spans its own arc
    position = {i: idx for members in groups.values() for idx, i in enumerate(members)}
    start = "ends.y0[0, :]"
    for i, (p, kind, n, a, _) in enumerate(intervals):
        idx = position[i]
        model.link(f"arcs.start[{p}]", f"radau{n}.a[{idx}]")
        model.link(f"arcs.end[{p}]", f"radau{n}.b[{idx}]")
        model.link(f"{kind}.y[{a}:{a + n}, :]", f"radau{n}.Y[{idx}, 1:{n + 1}, :]")
        model.link(f"{kind}.ydot[{a}:{a + n}, :]", f"radau{n}.F[{idx}, :, :]")
        model.link(start, f"radau{n}.Y[{idx}, 0, :]")
        start = f"{kind}.y[{a + n - 1}, :]"
    model.link("ends.yf[0, :]", start)
    return model, groups, intervals, slots


class SingleSolver:
    """Solve the single phase on a mesh and keep the multipliers of the last converged solve."""

    def __init__(self, nlp_options, counter):
        self.nlp_options = nlp_options
        self.counter = counter
        self.last = None

    def __call__(self, mesh, previous):
        model, groups = ll.build_model(mesh, SINGLE_MODULE)
        model.initialize()
        data = model.get_data_vector()
        h = mesh.half_widths
        data["pts.w"] = np.concatenate([h[k] * radau_weights(n, True) for k, n in enumerate(mesh.degrees)])
        for n, members in groups.items():
            data[f"radau{n}.h"] = h[members]
        data.get_vector().copy_host_to_device()

        if previous is None:
            tau = mesh.state_points()[:, None]
            states = ll.Y0 + 0.5 * (tau + 1.0) * (ll.YF - ll.Y0)
            controls, tf = np.full((mesh.num_collocation, 1), ll.C), 5.0
        else:
            states = previous.mesh.interpolate_states(previous.states, mesh.state_points())
            controls, tf = previous.mesh.interpolate_controls(previous.controls, mesh.collocation_points()), previous.tf
        x = model.create_vector()
        x["ends.tf"] = tf
        x["pts.y"] = states[mesh.collocated_states()]
        x["pts.u"] = np.clip(controls[:, 0], 0.0, ll.U_MAX)
        x["pts.ydot"] = ll.rates(states[mesh.collocated_states()], np.clip(controls, 0.0, ll.U_MAX), None)
        x["ends.y0"] = states[0]

        opt = am.Optimizer(model, x)
        result = opt.optimize(self.nlp_options)
        self.counter.iterations += len(result["iterations"])
        x = opt.get_optimized_point()
        states = np.empty((mesh.num_states, 2))
        states[mesh.collocated_states()] = np.asarray(x["pts.y"])
        states[0] = np.asarray(x["ends.y0"]).ravel()
        solution = MeshSolution(mesh, states, np.asarray(x["pts.u"]).reshape(-1, 1), 0.0, float(x["ends.tf"][0]),
                                converged=bool(result["converged"]))
        if result["converged"]:
            self.last = {"mesh": mesh, "point": opt.get_warm_start().point, "tf": solution.tf}
        return solution


class ArcSolve:
    """Solve the arcs on their meshes, with the records that give the costate of every solve."""

    def __init__(self, kinds, start, brackets, nlp_options, counter):
        self.kinds = kinds
        self.start = start
        self.brackets = brackets
        self.nlp_options = nlp_options
        self.counter = counter
        self.records = []

    def costate(self, record):
        """Costate at the collocation points of every arc of a converged solve, in the order of the combined arcs."""
        rows = []
        for p, (kind, a, m) in enumerate(record["slots"]):
            mu = np.asarray(record["point"][f"{kind}.res"])[a : a + m]
            rows.append(quadrature_costate(mu, record["meshes"][p], record["durations"][p]))
        return np.vstack(rows)

    def __call__(self, meshes, previous):
        model, groups, intervals, slots = build_arc_model(self.kinds, meshes, ARC_MODULE)
        model.initialize()
        data = model.get_data_vector()
        for n, members in groups.items():
            data[f"radau{n}.h"] = np.array([intervals[i][4] for i in members])
        data.get_vector().copy_host_to_device()

        guesses = []
        for g, mesh in zip(previous or self.start, meshes):
            if g.mesh is mesh:
                guesses.append(g)
            else:
                states = g.mesh.interpolate_states(g.states, mesh.state_points())
                controls = g.mesh.interpolate_controls(g.controls, mesh.collocation_points())
                guesses.append(MeshSolution(mesh, states, controls, g.t0, g.tf, dict(g.parameters)))
        x = model.create_vector()
        for (kind, a, m), g in zip(slots, guesses):
            coll = g.states[1:]
            x[f"{kind}.y[{a}:{a + m}, :]"] = coll
            thrust = ll.U_MAX if kind == "burn" else 0.0
            x[f"{kind}.ydot[{a}:{a + m}, :]"] = np.column_stack([coll[:, 1], np.full(m, thrust - ll.C)])
        x["ends.y0"] = guesses[0].states[0]
        x["arcs.start"] = [g.t0 for g in guesses]
        x["arcs.end"] = [g.tf for g in guesses]

        lower, upper = model.create_vector(), model.create_vector()
        lower["arcs.start[0]"] = 0.0
        upper["arcs.start[0]"] = 0.0
        for p, (low, high) in enumerate(self.brackets):
            lower[f"arcs.end[{p}]"] = low
            upper[f"arcs.end[{p}]"] = high

        opt = am.Optimizer(model, x, lower=lower, upper=upper)
        result = opt.optimize(self.nlp_options)
        self.counter.iterations += len(result["iterations"])
        x = opt.get_optimized_point()

        start = np.asarray(x["ends.y0"]).ravel()
        starts, ends = np.asarray(x["arcs.start"]).ravel(), np.asarray(x["arcs.end"]).ravel()
        solutions = []
        for p, ((kind, a, m), mesh) in enumerate(zip(slots, meshes)):
            coll = np.asarray(x[f"{kind}.y[{a}:{a + m}, :]"]).reshape(m, 2)
            thrust = ll.U_MAX if kind == "burn" else 0.0
            solutions.append(MeshSolution(mesh, np.vstack([start, coll]), np.full((m, 1), thrust),
                                          float(starts[p]), float(ends[p]), converged=bool(result["converged"])))
            start = coll[-1]
        record = None
        if result["converged"]:
            record = {"slots": slots, "intervals": intervals, "point": opt.get_warm_start().point,
                      "meshes": meshes, "durations": ends - starts}
        self.records.append(record)
        return solutions


class Landing(BangBangProblem):
    """The landing as a bang-bang problem for the solver, with the single phase and the arcs above."""

    def __init__(self):
        super().__init__()
        self.nlp_options = {"convergence_tolerance": 1e-10, "max_iterations": 500}
        self.solve = SingleSolver(self.nlp_options, self)

    def set_value(self, value):
        pass

    def single(self, mesh, guess, blend, warm="same"):
        return self.solve(mesh, guess)

    def single_costate(self):
        last = self.solve.last
        return quadrature_costate(np.asarray(last["point"]["pts.res"]), last["mesh"], last["tf"])

    def single_error(self, solution):
        if not solution.converged:
            return np.nan
        return float(self.refinement(1e-8).estimate(solution).errors.max())

    def switching(self, solution, costate):
        return 1.0 + np.asarray(costate)[:, 1]

    def read(self, source, sigma, noise):
        eta = source.controls[:, 0] / ll.U_MAX
        edges, signs = switching_arcs(source, sigma, eta, noise)
        return [("burn" if s < 0.0 else "coast", float(a), float(b))
                for s, a, b in zip(signs, edges[:-1], edges[1:])], 0.0, {}

    def arc_meshes(self, arcs):
        return uniform_arc_meshes([arcs[0][1]] + [end for _, _, end in arcs], intervals=8, degree=3, min_intervals=2)

    def arc_start(self, source, arcs, meshes, shift, parameters):
        guesses = []
        for (kind, a, b), mesh in zip(arcs, meshes):
            tau = 2.0 * (mesh_time(mesh.state_points(), a, b) - source.t0) / (source.tf - source.t0) - 1.0
            states = source.mesh.interpolate_states(source.states, np.clip(tau, -1.0, 1.0))
            thrust = ll.U_MAX if kind == "burn" else 0.0
            guesses.append(MeshSolution(mesh, states, np.full((mesh.num_collocation, 1), thrust), a, b))
        return guesses

    def arc_solver(self, kinds, start, brackets, record, budget):
        options = dict(self.nlp_options, max_iterations=budget) if budget else self.nlp_options
        return ArcSolve(kinds, start, brackets, options, self)

    def refinement(self, tolerance, max_iterations=None):
        kwargs = {} if max_iterations is None else {"max_iterations": max_iterations}
        return PHSRefinement(PHSOptions(tolerance=tolerance, min_points=3, max_points=8, **kwargs), ll.rates)

    def objective(self, solution):
        return ll.cost(solution)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    # Build both modules in a temporary directory so the repository stays untouched
    tmp_path = tmp_path_factory.mktemp("landing_bang")
    orig_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        ll.build_model(ll.catalog_mesh(3, 8, True), SINGLE_MODULE)[0].build_module()
        catalog = ll.catalog_mesh(3, 8, True)
        build_arc_model(["burn", "coast"], [catalog, catalog], ARC_MODULE)[0].build_module()
    finally:
        os.chdir(orig_dir)
    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))


def test_the_solver_reads_the_coast_and_burn_from_the_single_phase_and_lands_on_the_exact_switch(built):
    problem = Landing()
    solver = BangBangSolver(problem, BangBangOptions(tolerance=1e-8, homotopy=(), arc_floor=FLOOR))
    mesh = RadauMesh.uniform(16, 3, flipped=True)
    tau = mesh.state_points()[:, None]
    guess = MeshSolution(mesh, ll.Y0 + 0.5 * (tau + 1.0) * (ll.YF - ll.Y0),
                         np.full((mesh.num_collocation, 1), ll.C), 0.0, 5.0)

    step = solver.step(ll.U_MAX, start=guess)
    assert step.converged
    assert [stage.name for stage in step.stages] == ["jump"]
    assert step.stages[0].outcome == "serves"
    assert step.arcs is not None and step.arcs.error <= 1e-8

    # The arcs, their switch, the final time, and the cost match the closed form
    assert [kind for kind, _, _ in step.arcs.arcs] == ["coast", "burn"]
    assert step.arcs.arcs[0][2] == pytest.approx(ll.T_SWITCH, abs=1e-7)
    assert step.solution.tf == pytest.approx(ll.T_FINAL, abs=1e-7)
    assert problem.objective(step.solution) == pytest.approx(ll.COST, abs=1e-6)

    # The switching function of the arcs agrees with them everywhere
    assert step.arcs.check["stands"]
    assert step.iterations == problem.iterations
