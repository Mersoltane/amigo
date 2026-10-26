"""Tests of the decisions of the bang-bang solver, with scripted solves of the arcs and of the homotopy."""

from types import SimpleNamespace

import numpy as np
import pytest

from amigo.trajectory import (
    Arcs,
    BangBangOptions,
    BangBangProblem,
    BangBangSolver,
    MeshIteration,
    MeshSolution,
    RadauMesh,
    RefinementHistory,
    Stage,
    Step,
    mesh_time,
)


class Solution:
    """A solution that only knows its objective and whether it converged."""

    def __init__(self, objective, converged=True):
        self.objective = objective
        self.converged = converged
        self.mesh = SimpleNamespace(num_intervals=1, num_collocation=2)
        self.states = np.zeros((2, 7))


class Problem(BangBangProblem):
    """A problem whose single phase always converges with an objective of 100 and misses the mesh tolerance."""

    def __init__(self):
        super().__init__()
        self.singles = 0
        self.resets = 0

    def set_value(self, value):
        self.value = value

    def single(self, mesh, guess, blend, warm="same"):
        self.singles += 1
        self.iterations += 10
        return Solution(100.0)

    def single_costate(self):
        return np.zeros((2, 7))

    def single_error(self, solution):
        return 1.0

    def reset_single(self):
        self.resets += 1

    def switching(self, solution, costate):
        return np.ones(2)

    def objective(self, solution):
        return solution.objective


def arcs(objective, stands=True, error=1e-9):
    """Solved arcs with an objective, which their switching function either lets stand or asks to change."""
    check = {"worst": 0.0 if stands else 0.1, "stands": stands}
    return Arcs(Solution(objective), [], check, error, {"kinds": [], "solutions": [], "record": None})


class Scripted(BangBangSolver):
    """A solver whose solves of arcs return a script in order and whose homotopy and refinement always converge."""

    def __init__(self, script, options=None):
        super().__init__(Problem(), options)
        self.script = list(script)
        self.calls = []

    def solve_arcs(self, source, sigma, given=None, carry=None, noise=0.0, budget=None, effort=None, search=False):
        self.calls.append(("carry" if carry is not None else "read", budget))
        self.problem.iterations += 5
        return self.script.pop(0), []

    def smoothed(self, step, start):
        self.calls.append(("homotopy", None))
        return Solution(50.0), self.read_and_solve(step, Solution(50.0), 0.0, None)

    def refined(self, solution, blend, tolerance, meshes=None):
        self.calls.append(("refine", meshes))
        return Solution(60.0), 0.0, [{}]


def previous_step(value, carried=False):
    """A converged step at a value with arcs, which ran its single phase unless it carried them."""
    step = Step(value, Solution(30.0), arcs(30.0), None if carried else Solution(100.0))
    step.start = Solution(100.0)
    return step


def test_carried_arcs_that_serve_end_the_step_without_a_single_phase():
    solver = Scripted([arcs(20.0)])
    step = solver.step(0.99, previous_step(1.0))
    assert [s.name for s in step.stages] == ["carry"]
    assert step.solution.objective == 20.0
    assert solver.problem.singles == 0
    assert solver.calls == [("carry", 150)]


def test_a_step_beyond_the_carry_step_jumps_from_the_last_single_phase():
    solver = Scripted([arcs(21.0)])
    step = solver.step(0.8, previous_step(1.0))
    assert [s.name for s in step.stages] == ["jump"]
    assert solver.calls == [("read", 150)]
    assert step.solution.objective == 21.0
    assert step.start.objective == 100.0


def test_arcs_the_switching_function_asks_to_change_lead_to_the_homotopy_and_the_lighter_stands():
    solver = Scripted([arcs(22.0, stands=False), arcs(21.0)])
    step = solver.step(0.8, previous_step(1.0))
    assert [s.name for s in step.stages] == ["jump", "homotopy"]
    assert solver.calls == [("read", 150), ("homotopy", None), ("read", None)]
    assert step.solution.objective == 21.0
    assert step.start.objective == 50.0


def test_the_arcs_of_the_jump_stand_when_the_homotopy_finds_nothing_lighter():
    solver = Scripted([arcs(21.0, stands=False), arcs(22.0)])
    step = solver.step(0.8, previous_step(1.0))
    assert step.solution.objective == 21.0
    assert step.start.objective == 100.0


def test_without_the_homotopy_the_capped_refined_single_phase_comes_last_with_the_budget():
    solver = Scripted([None, arcs(23.0)], BangBangOptions(homotopy=()))
    step = solver.step(0.8, previous_step(1.0))
    assert [s.name for s in step.stages] == ["jump", "refined single phase"]
    assert solver.calls == [("read", 150), ("refine", 3), ("read", 150)]
    assert step.solution.objective == 23.0


def test_the_stage_order_does_not_change_after_a_level_the_homotopy_served():
    solver = Scripted([arcs(21.0)])
    previous = previous_step(1.0)
    previous.stages = [Stage("jump", "does not serve", 100), Stage("homotopy", "serves", 200)]
    step = solver.step(0.8, previous)
    assert [s.name for s in step.stages] == ["jump"]
    assert step.solution.objective == 21.0


def test_a_step_after_carried_arcs_forgets_the_warm_start_of_the_single_phase():
    solver = Scripted([arcs(21.0), arcs(21.0)])
    solver.step(0.8, previous_step(1.0, carried=True))
    assert solver.problem.resets == 1
    solver.step(0.7, previous_step(0.8))
    assert solver.problem.resets == 1


def test_a_step_whose_best_arcs_miss_the_mesh_tolerance_fails_even_when_their_solve_converged():
    solver = Scripted([arcs(21.0, stands=False, error=1e-3), arcs(22.0, error=1e-3)])
    step = solver.step(0.8, previous_step(1.0))

    # The lighter arcs stand as the result of the step, but the family must not continue from them
    assert step.solution.objective == 21.0
    assert step.met is False and not step.converged


def test_arcs_that_stand_but_miss_the_mesh_tolerance_do_not_end_the_step_early():
    solver = Scripted([arcs(21.0, error=1e-3), arcs(20.0)])
    step = solver.step(0.8, previous_step(1.0))
    assert [s.name for s in step.stages] == ["jump", "homotopy"]
    assert step.solution.objective == 20.0 and step.converged


def test_every_stage_records_the_iterations_it_took():
    solver = Scripted([arcs(22.0, stands=False), arcs(21.0)])
    step = solver.step(0.8, previous_step(1.0))

    # The jump solves the single phase once and its arcs once, the homotopy reads its arcs once
    assert [s.iterations for s in step.stages] == [15, 5]
    assert step.iterations == 20


def test_arcs_that_meet_the_tolerance_beat_lighter_ones_that_miss_it_and_otherwise_the_lower_objective_stands():
    solver = BangBangSolver(Problem(), BangBangOptions(tolerance=1e-8))

    def found(objective, error):
        return Arcs(Solution(objective), [], {"worst": 0.0, "stands": True}, error, {})

    assert solver.better(found(0.022, 1e-9), found(0.021, 1e-7))
    assert solver.better(found(0.021, 1e-9), found(0.022, 1e-9))

    # A gain within the tolerance of the objective is no gain
    assert not solver.better(found(0.022 - 1e-9, 1e-9), found(0.022, 1e-9))
    assert solver.best([None, found(0.022, 1e-9), found(0.021, 1e-9)]).solution.objective == 0.021


def test_a_family_visits_values_at_most_the_max_step_apart_and_every_value_asked_for():
    solver = BangBangSolver(Problem(), BangBangOptions(max_step=0.05))
    values = solver.visits([1.0, 0.99, 0.8])

    # The step from 0.99 to 0.8 is split into equal ratios of at most five percent
    assert values[:2] == [1.0, 0.99] and values[-1] == 0.8
    ratios = np.array(values[2:]) / np.array(values[1:-1])
    assert np.allclose(ratios, ratios[0]) and 1.0 / ratios[0] <= 1.05
    assert BangBangSolver(Problem()).visits([1.0, 0.5]) == [1.0, 0.5]


class Lazy(BangBangSolver):
    """A solver whose readings of arcs come from a script, for the lazy refinement of the smoothed problem."""

    def __init__(self, reads):
        super().__init__(Problem())
        self.reads = list(reads)
        self.budgets = []
        self.refines = 0

    def read_and_solve(self, step, single, noise, budget, effort=None, search=False):
        self.budgets.append(budget)
        return self.reads.pop(0)

    def refined(self, solution, blend, tolerance, meshes=None):
        self.refines += 1
        return Solution(45.0), 0.0, [{}]


def test_arcs_read_on_the_mesh_of_the_homotopy_that_stand_skip_the_smoothed_refinement():
    solver = Lazy([arcs(20.0)])
    smooth, found = solver.smoothed(Step(0.5), Solution(100.0))
    assert found.solution.objective == 20.0
    assert solver.refines == 0 and solver.budgets == [150]


def test_arcs_that_do_not_stand_lead_to_a_second_reading_on_the_refined_smoothed_problem():
    solver = Lazy([arcs(22.0, stands=False), arcs(21.0)])
    smooth, found = solver.smoothed(Step(0.5), Solution(100.0))
    assert solver.refines == 1 and solver.budgets == [150, 150]
    assert found.solution.objective == 21.0
    assert smooth.objective == 45.0


class OneMesh:
    """A refinement that solves the first meshes once and finds them within the tolerance."""

    def __init__(self, tolerance):
        self.tolerance = tolerance

    def run(self, meshes, solve, first=None, stop=None):
        solutions = list(first) if first is not None else solve(meshes, None)
        entry = [MeshIteration(s, np.full(s.mesh.num_intervals, 1e-9)) for s in solutions]
        if stop is not None:
            stop(solutions)
        return RefinementHistory(self.tolerance, [entry])


class Layouts(BangBangProblem):
    """A problem whose arcs solve to the times they start from, except that a burn starting after shrink shrinks to the floor.

    The control is one on a burn and zero on a coast, the objective is its
    integral, and the switching function is a fixed function of time.
    """

    def __init__(self, read, sigma, shrink=np.inf):
        super().__init__()
        self.arcs, self.sigma, self.shrink = read, sigma, shrink
        self.layouts = []

    def read(self, source, sigma, noise):
        return self.arcs, 0.0, {}

    def arc_meshes(self, arcs):
        return [RadauMesh.uniform(8, 3, True) for _ in arcs]

    def arc_start(self, source, arcs, meshes, shift, parameters):
        return [MeshSolution(m, np.zeros((m.num_states, 1)), np.zeros((m.num_collocation, 1)), a, b)
                for (_, a, b), m in zip(arcs, meshes)]

    def arc_solver(self, kinds, start, brackets, record, budget):
        problem = self

        class Solver:
            records = []

            def __call__(self, meshes, previous):
                problem.layouts.append(list(kinds))
                self.records.append({})
                edges = [start[0].t0] + [s.tf for s in start]
                for p in range(len(kinds) - 1):
                    if kinds[p] == "burn" and edges[p] >= problem.shrink:
                        edges[p + 1] = edges[p] + 1e-3
                return [MeshSolution(m, np.zeros((m.num_states, 1)), np.full((m.num_collocation, 1), float(k == "burn")),
                                     a, b) for k, m, a, b in zip(kinds, meshes, edges[:-1], edges[1:])]

            def costate(self, record):
                return None

        return Solver()

    def refinement(self, tolerance, max_iterations=None):
        return OneMesh(tolerance)

    def switching(self, solution, costate):
        return self.sigma(mesh_time(solution.mesh.collocation_points(), solution.t0, solution.tf))

    def objective(self, solution):
        return 0.5 * (solution.tf - solution.t0) * float(solution.mesh.quadrature_weights() @ solution.controls[:, 0])


def test_a_region_where_the_switching_function_has_the_wrong_sign_gets_an_arc_of_the_other_kind():
    def sigma(t):
        return np.where(t <= 2.0, np.exp(-((t - 1.0) / 0.3) ** 2) - 0.5, 1.0)

    problem = Layouts([("burn", 0.0, 2.0), ("coast", 2.0, 4.0)], sigma)
    found, _ = BangBangSolver(problem, BangBangOptions(arc_floor=1e-3)).solve_arcs(None, None)

    # The coast inside the burn saves its own length of burning, and the switching function then agrees
    assert problem.layouts == [["burn", "coast"], ["burn", "coast", "burn", "coast"]]
    assert found.check["stands"]
    assert [kind for kind, _, _ in found.arcs] == ["burn", "coast", "burn", "coast"]
    assert problem.objective(found.solution) == pytest.approx(2.0 - 2.0 * 0.3 * np.sqrt(np.log(2.0)), abs=0.02)


class Restless(Layouts):
    """A problem whose switching function asks for a coast somewhere new on every check, so no layout ever stands.

    Every solve of its arcs counts a fixed number of iterations, so the
    effort of a stage buys a known number of layouts.
    """

    def __init__(self):
        super().__init__([("burn", 0.0, 4.0)], None)
        self.checks = 0
        self.cost = 120

    def arc_solver(self, kinds, start, brackets, record, budget):
        solver = super().arc_solver(kinds, start, brackets, record, budget)
        problem = self

        class Counted:
            def __call__(self, meshes, previous):
                problem.iterations += problem.cost
                return solver(meshes, previous)

            def costate(self, record):
                return solver.costate(record)

            @property
            def records(self):
                return solver.records

        return Counted()

    def switching(self, solution, costate):
        self.checks += 1
        t = mesh_time(solution.mesh.collocation_points(), solution.t0, solution.tf)
        center = 0.4 + 0.4 * self.checks
        return np.where(np.abs(t - center) < 0.15, 1.0, -1.0)


def test_a_stage_that_spends_its_effort_returns_what_it_holds_while_the_full_dance_runs_longer():
    problem = Restless()
    found, _ = BangBangSolver(problem, BangBangOptions(arc_floor=1e-3)).solve_arcs(None, None, effort=300)
    assert len(problem.layouts) == 3
    assert found is not None and not found.check["stands"]

    problem = Restless()
    BangBangSolver(problem, BangBangOptions(arc_floor=1e-3)).solve_arcs(None, None)
    assert len(problem.layouts) == 8


def test_an_arc_that_shrinks_to_the_floor_goes_to_its_neighbors_without_refining_further():
    def sigma(t):
        return np.where(t <= 0.5, -1.0, 1.0)

    read = [("burn", 0.0, 0.5), ("coast", 0.5, 1.5), ("burn", 1.5, 2.5), ("coast", 2.5, 4.0)]
    problem = Layouts(read, sigma, shrink=1.0)
    found, _ = BangBangSolver(problem, BangBangOptions(arc_floor=1e-3)).solve_arcs(None, None)

    # The two coasts around the shrunk burn join into one
    assert problem.layouts == [["burn", "coast", "burn", "coast"], ["burn", "coast"]]
    assert found.check["stands"]
    assert found.arcs == [("burn", 0.0, 0.5), ("coast", 0.5, 4.0)]


class Failing(Scripted):
    """A solver whose steps fail below a value of the parameter."""

    def __init__(self, floor):
        super().__init__([])
        self.floor = floor
        self.values = []

    def step(self, value, previous=None, start=None):
        self.values.append(value)
        step = Step(value, Solution(value, converged=value >= self.floor))
        step.start = step.solution
        return step


def test_a_family_puts_a_value_halfway_before_a_value_that_fails():
    solver = Failing(0.5)
    seen = []
    path = solver.family([1.0, 0.8, 0.4], Solution(0.0), on_step=lambda step: seen.append(step.value))

    # The halfway values close in on the lowest value that converges until four halvings in a row fail
    assert solver.values[:4] == pytest.approx([1.0, 0.8, 0.4, (0.8 * 0.4) ** 0.5])
    assert seen == solver.values
    assert 0.5 <= path.last.value < 0.8
    assert not path.reached(0.4)
