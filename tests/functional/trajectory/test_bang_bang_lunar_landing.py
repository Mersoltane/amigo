"""
Bang-bang solve of the lunar landing
------------------------------------

- The landing of Meditch coasts and then burns at full thrust
- The solver reads the two arcs from the switching function of the single phase and solves them with a free switch time
- The exact arcs are polynomials, so the switch time, final time, and cost match the closed form to collocation accuracy
- The arc model written here is what a new bang-bang problem still has to supply
"""

import contextlib
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

# The lunar landing example supplies the single phase and the exact solution
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
    """Landing dynamics at one collocation point of an arc, at full thrust or at none"""

    def __init__(self, thrust):
        if thrust:
            name = "BurnPoint"
        else:
            name = "CoastPoint"

        super().__init__(name=name)

        self.thrust = thrust

        self.add_input("y", shape=2, label="altitude and velocity")
        self.add_input("ydot", shape=2, label="rates")

        self.add_constraint("res", shape=2, label="dynamics")

        return

    def compute(self):
        y = self.inputs["y"]
        ydot = self.inputs["ydot"]

        self.constraints["res"] = [ydot[0] - y[1], ydot[1] - (self.thrust - ll.C)]

        return


class ArcRadau(am.Component):
    """Collocation conditions on one interval of an arc between its free start and end times"""

    def __init__(self, n):
        super().__init__(name=f"ArcRadau{n}")

        self.n = n
        self.I = integration_matrix(n, flipped=True)

        # Half width of the interval in the mesh coordinates of its arc
        self.add_data("h", value=1.0)

        # Start and end times of the arc
        self.add_input("a")
        self.add_input("b")

        # States at the support points and rates at the collocation points
        self.add_input("Y", shape=(n + 1, 2))
        self.add_input("F", shape=(n, 2))

        self.add_constraint("res", shape=(n, 2))

        return

    def compute(self):
        h = self.data["h"]

        a = self.inputs["a"]
        b = self.inputs["b"]
        Y = self.inputs["Y"]
        F = self.inputs["F"]

        scale = 0.5 * (b - a) * h

        # The state rise to each support point is the integration row times the rates
        res = np.empty((self.n, 2), dtype=object)

        for i in range(self.n):
            for s in range(2):
                total = float(self.I[i, 0]) * F[0, s]

                for j in range(1, self.n):
                    total = total + float(self.I[i, j]) * F[j, s]

                res[i, s] = Y[i + 1, s] - Y[0, s] - scale * total

        self.constraints["res"] = res

        return


class ArcEnds(am.Component):
    """Boundary conditions of the landing on the first and last states of the arcs"""

    def __init__(self):
        super().__init__()

        self.add_input("y0", shape=2)
        self.add_input("yf", shape=2)

        self.add_constraint("initial", shape=2)
        self.add_constraint("final", shape=2)

        return

    def compute(self):
        y0 = self.inputs["y0"]
        yf = self.inputs["yf"]

        self.constraints["initial"] = [y0[0] - ll.Y0[0], y0[1] - ll.Y0[1]]
        self.constraints["final"] = [yf[0] - ll.YF[0], yf[1] - ll.YF[1]]

        return


class ArcSpan(am.Component):
    """Start and end times of one arc with its duration held above the floor"""

    def __init__(self):
        super().__init__()

        self.add_input("start", lower=0.0, upper=100.0)
        self.add_input("end", lower=0.0, upper=100.0)

        self.add_constraint("duration", lower=FLOOR, upper=am.inf)

        return

    def compute(self):
        start = self.inputs["start"]
        end = self.inputs["end"]

        self.constraints["duration"] = end - start

        return


class BurnCost(am.Component):
    """Fuel of one burn arc at full thrust"""

    def __init__(self):
        super().__init__()

        self.add_input("start")
        self.add_input("end")

        self.add_objective("obj")

        return

    def compute(self):
        start = self.inputs["start"]
        end = self.inputs["end"]

        self.objective["obj"] = ll.U_MAX * (end - start)

        return


def build_arc_model(kinds, meshes, module):
    """Model with one phase per arc, each on its own mesh between its start and end times"""
    model = am.Model(module)

    # The kind, first point, and number of points of every arc
    counts = {"burn": 0, "coast": 0}
    slots = []

    for p in range(len(kinds)):
        kind = kinds[p]
        mesh = meshes[p]

        slots.append((kind, counts[kind], mesh.num_collocation))
        counts[kind] += mesh.num_collocation

    # The collocation points of the burns and of the coasts
    if counts["burn"]:
        model.add_component("burn", counts["burn"], ArcPoint(ll.U_MAX))

    if counts["coast"]:
        model.add_component("coast", counts["coast"], ArcPoint(0.0))

    model.add_component("ends", 1, ArcEnds())
    model.add_component("arcs", len(kinds), ArcSpan())

    # The fuel of every burn
    burns = []

    for p in range(len(kinds)):
        if kinds[p] == "burn":
            burns.append(p)

    if burns:
        model.add_component("fuel", len(burns), BurnCost())

    for c in range(len(burns)):
        p = burns[c]

        model.link(f"fuel.start[{c}]", f"arcs.start[{p}]")
        model.link(f"fuel.end[{c}]", f"arcs.end[{p}]")

    # Consecutive arcs share their switch time
    for p in range(len(kinds) - 1):
        model.link(f"arcs.end[{p}]", f"arcs.start[{p + 1}]")

    # Every interval of every arc as its arc, kind, points, offset, and half width
    intervals = []

    for p in range(len(kinds)):
        kind = kinds[p]
        mesh = meshes[p]
        base = slots[p][1]

        for k in range(mesh.num_intervals):
            n = int(mesh.degrees[k])
            offset = base + int(mesh.offsets[k])
            half_width = float(mesh.half_widths[k])

            intervals.append((p, kind, n, offset, half_width))

    # The intervals are grouped by their number of points
    groups = {}

    for i in range(len(intervals)):
        n = intervals[i][2]

        if n not in groups:
            groups[n] = []

        groups[n].append(i)

    for n in groups:
        model.add_component(f"radau{n}", len(groups[n]), ArcRadau(n))

    # The position of every interval within its group
    position = {}

    for n in groups:
        members = groups[n]

        for idx in range(len(members)):
            position[members[idx]] = idx

    # Each interval starts at the last collocation point before it and spans its own arc
    start = "ends.y0[0, :]"

    for i in range(len(intervals)):
        p = intervals[i][0]
        kind = intervals[i][1]
        n = intervals[i][2]
        a = intervals[i][3]

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
    """Solve the single phase on a mesh and keep the multipliers of the last converged solve"""

    def __init__(self, nlp_options, counter):
        self.nlp_options = nlp_options
        self.counter = counter
        self.last = None

    def __call__(self, mesh, previous):
        built = ll.build_model(mesh, SINGLE_MODULE)
        model = built[0]
        groups = built[1]

        model.initialize()

        # The quadrature weights and the half widths of the intervals
        data = model.get_data_vector()
        h = mesh.half_widths

        weights = []

        for k in range(mesh.num_intervals):
            n = mesh.degrees[k]
            weights.append(h[k] * radau_weights(n, True))

        data["pts.w"] = np.concatenate(weights)

        for n in groups:
            data[f"radau{n}.h"] = h[groups[n]]

        data.get_vector().copy_host_to_device()

        # Start from a straight line or from the previous solution
        if previous is None:
            tau = mesh.state_points()[:, None]
            states = ll.Y0 + 0.5 * (tau + 1.0) * (ll.YF - ll.Y0)
            controls = np.full((mesh.num_collocation, 1), ll.C)
            tf = 5.0
        else:
            states = previous.mesh.interpolate_states(previous.states, mesh.state_points())
            controls = previous.mesh.interpolate_controls(previous.controls, mesh.collocation_points())
            tf = previous.tf

        x = model.create_vector()

        x["ends.tf"] = tf
        x["pts.y"] = states[mesh.collocated_states()]
        x["pts.u"] = np.clip(controls[:, 0], 0.0, ll.U_MAX)
        x["pts.ydot"] = ll.rates(states[mesh.collocated_states()], np.clip(controls, 0.0, ll.U_MAX), None)
        x["ends.y0"] = states[0]

        # Solve and count the iterations
        opt = am.Optimizer(model, x)
        result = opt.optimize(self.nlp_options)
        self.counter.iterations += len(result["iterations"])

        x = opt.get_optimized_point()

        # The solution with the state at the start of the phase
        states = np.empty((mesh.num_states, 2))
        states[mesh.collocated_states()] = np.asarray(x["pts.y"])
        states[0] = np.asarray(x["ends.y0"]).ravel()

        controls = np.asarray(x["pts.u"]).reshape(-1, 1)
        tf = float(x["ends.tf"][0])

        solution = MeshSolution(mesh, states, controls, 0.0, tf, converged=bool(result["converged"]))

        # Keep the multipliers of the last converged solve for its costate
        if result["converged"]:
            self.last = {"mesh": mesh, "point": opt.get_warm_start().point, "tf": solution.tf}

        return solution


class ArcSolve:
    """Solve the arcs on their meshes, with the records that give the costate of every solve"""

    def __init__(self, kinds, start, brackets, nlp_options, counter):
        self.kinds = kinds
        self.start = start
        self.brackets = brackets
        self.nlp_options = nlp_options
        self.counter = counter
        self.records = []

    def costate(self, record):
        """Costate at the collocation points of every arc of a converged solve, in the order of the arcs"""
        rows = []

        for p in range(len(record["slots"])):
            kind = record["slots"][p][0]
            a = record["slots"][p][1]
            m = record["slots"][p][2]

            mu = np.asarray(record["point"][f"{kind}.res"])[a : a + m]
            rows.append(quadrature_costate(mu, record["meshes"][p], record["durations"][p]))

        return np.vstack(rows)

    def __call__(self, meshes, previous):
        built = build_arc_model(self.kinds, meshes, ARC_MODULE)
        model = built[0]
        groups = built[1]
        intervals = built[2]
        slots = built[3]

        model.initialize()

        # The half width of every interval in its group
        data = model.get_data_vector()

        for n in groups:
            half_widths = []

            for i in groups[n]:
                half_widths.append(intervals[i][4])

            data[f"radau{n}.h"] = np.array(half_widths)

        data.get_vector().copy_host_to_device()

        # The guess of every arc, moved onto its mesh when the mesh changed
        sources = previous or self.start
        guesses = []

        for p in range(len(meshes)):
            g = sources[p]
            mesh = meshes[p]

            if g.mesh is mesh:
                guesses.append(g)
            else:
                states = g.mesh.interpolate_states(g.states, mesh.state_points())
                controls = g.mesh.interpolate_controls(g.controls, mesh.collocation_points())
                guesses.append(MeshSolution(mesh, states, controls, g.t0, g.tf, dict(g.parameters)))

        # The states and the rates of every arc at its collocation points
        x = model.create_vector()

        for p in range(len(slots)):
            kind = slots[p][0]
            a = slots[p][1]
            m = slots[p][2]

            thrust = 0.0
            if kind == "burn":
                thrust = ll.U_MAX

            coll = guesses[p].states[1:]

            x[f"{kind}.y[{a}:{a + m}, :]"] = coll
            x[f"{kind}.ydot[{a}:{a + m}, :]"] = np.column_stack([coll[:, 1], np.full(m, thrust - ll.C)])

        # The first state and the start and end of every arc
        guess_starts = []
        guess_ends = []

        for g in guesses:
            guess_starts.append(g.t0)
            guess_ends.append(g.tf)

        x["ends.y0"] = guesses[0].states[0]
        x["arcs.start"] = guess_starts
        x["arcs.end"] = guess_ends

        # The first arc starts at zero and every switch stays in its bracket
        lower = model.create_vector()
        upper = model.create_vector()

        lower["arcs.start[0]"] = 0.0
        upper["arcs.start[0]"] = 0.0

        for p in range(len(self.brackets)):
            lower[f"arcs.end[{p}]"] = self.brackets[p][0]
            upper[f"arcs.end[{p}]"] = self.brackets[p][1]

        # Solve and count the iterations
        opt = am.Optimizer(model, x, lower=lower, upper=upper)
        result = opt.optimize(self.nlp_options)
        self.counter.iterations += len(result["iterations"])

        x = opt.get_optimized_point()

        # The solution of every arc starts from the last state of the arc before it
        start = np.asarray(x["ends.y0"]).ravel()
        starts = np.asarray(x["arcs.start"]).ravel()
        ends = np.asarray(x["arcs.end"]).ravel()

        solutions = []

        for p in range(len(slots)):
            kind = slots[p][0]
            a = slots[p][1]
            m = slots[p][2]

            thrust = 0.0
            if kind == "burn":
                thrust = ll.U_MAX

            coll = np.asarray(x[f"{kind}.y[{a}:{a + m}, :]"]).reshape(m, 2)

            states = np.vstack([start, coll])
            controls = np.full((m, 1), thrust)

            solution = MeshSolution(
                meshes[p],
                states,
                controls,
                float(starts[p]),
                float(ends[p]),
                converged=bool(result["converged"]),
            )

            solutions.append(solution)
            start = coll[-1]

        # The record that gives the costate of a converged solve
        record = None

        if result["converged"]:
            record = {
                "slots": slots,
                "intervals": intervals,
                "point": opt.get_warm_start().point,
                "meshes": meshes,
                "durations": ends - starts,
            }

        self.records.append(record)

        return solutions


class Landing(BangBangProblem):
    """The landing as a bang-bang problem, with the single phase and the arcs above"""

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
        multipliers = np.asarray(last["point"]["pts.res"])

        return quadrature_costate(multipliers, last["mesh"], last["tf"])

    def single_error(self, solution):
        if not solution.converged:
            return np.nan

        estimate = self.refinement(1e-8).estimate(solution)

        return float(estimate.errors.max())

    def switching(self, solution, costate):
        return 1.0 + np.asarray(costate)[:, 1]

    def read(self, source, sigma, noise):
        # The throttle shows where the control jumps
        eta = source.controls[:, 0] / ll.U_MAX

        reading = switching_arcs(source, sigma, eta, noise)
        edges = reading[0]
        signs = reading[1]

        # A negative sign is a burn and a positive one a coast
        arcs = []

        for p in range(len(signs)):
            kind = "coast"
            if signs[p] < 0.0:
                kind = "burn"

            arcs.append((kind, float(edges[p]), float(edges[p + 1])))

        return arcs, 0.0, {}

    def arc_meshes(self, arcs):
        # The edges are the start of the first arc and the end of every arc
        edges = [arcs[0][1]]

        for arc in arcs:
            edges.append(arc[2])

        return uniform_arc_meshes(edges, intervals=8, degree=3, min_intervals=2)

    def arc_start(self, source, arcs, meshes, shift, parameters):
        guesses = []

        for p in range(len(arcs)):
            kind = arcs[p][0]
            a = arcs[p][1]
            b = arcs[p][2]
            mesh = meshes[p]

            # The source states at the support points of the arc
            tau = 2.0 * (mesh_time(mesh.state_points(), a, b) - source.t0) / (source.tf - source.t0) - 1.0
            states = source.mesh.interpolate_states(source.states, np.clip(tau, -1.0, 1.0))

            thrust = 0.0
            if kind == "burn":
                thrust = ll.U_MAX

            controls = np.full((mesh.num_collocation, 1), thrust)

            guesses.append(MeshSolution(mesh, states, controls, a, b))

        return guesses

    def arc_solver(self, kinds, start, brackets, record, budget):
        # A budget caps the iterations of every solve
        options = self.nlp_options
        if budget:
            options = dict(self.nlp_options, max_iterations=budget)

        return ArcSolve(kinds, start, brackets, options, self)

    def refinement(self, tolerance, max_iterations=None):
        if max_iterations is None:
            options = PHSOptions(tolerance=tolerance, min_points=3, max_points=8)
        else:
            options = PHSOptions(tolerance=tolerance, min_points=3, max_points=8, max_iterations=max_iterations)

        return PHSRefinement(options, ll.rates)

    def objective(self, solution):
        return ll.cost(solution)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Build both modules in a temporary directory so the repository stays untouched"""
    tmp_path = tmp_path_factory.mktemp("landing_bang")
    catalog = ll.catalog_mesh(3, 8, True)

    with contextlib.chdir(tmp_path):
        single = ll.build_model(catalog, SINGLE_MODULE)[0]
        single.build_module()

        arcs = build_arc_model(["burn", "coast"], [catalog, catalog], ARC_MODULE)[0]
        arcs.build_module()

    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))


def test_the_solver_reads_the_coast_and_burn_from_the_single_phase_and_lands_on_the_exact_switch(built):
    problem = Landing()
    options = BangBangOptions(tolerance=1e-8, homotopy=(), arc_floor=FLOOR)
    solver = BangBangSolver(problem, options)

    # Start from a straight line on a uniform mesh
    mesh = RadauMesh.uniform(16, 3, flipped=True)
    tau = mesh.state_points()[:, None]

    states = ll.Y0 + 0.5 * (tau + 1.0) * (ll.YF - ll.Y0)
    controls = np.full((mesh.num_collocation, 1), ll.C)
    guess = MeshSolution(mesh, states, controls, 0.0, 5.0)

    step = solver.step(ll.U_MAX, start=guess)

    # The jump alone serves, with arcs within the mesh tolerance
    stage_names = []

    for stage in step.stages:
        stage_names.append(stage.name)

    assert step.converged
    assert stage_names == ["jump"]
    assert step.stages[0].outcome == "serves"
    assert step.arcs is not None and step.arcs.error <= 1e-8

    # The arcs, their switch, the final time, and the cost match the closed form
    arc_kinds = []

    for arc in step.arcs.arcs:
        arc_kinds.append(arc[0])

    assert arc_kinds == ["coast", "burn"]
    assert step.arcs.arcs[0][2] == pytest.approx(ll.T_SWITCH, abs=1e-7)
    assert step.solution.tf == pytest.approx(ll.T_FINAL, abs=1e-7)
    assert problem.objective(step.solution) == pytest.approx(ll.COST, abs=1e-6)

    # The switching function of the arcs agrees with them everywhere
    assert step.arcs.check["stands"]
    assert step.iterations == problem.iterations
