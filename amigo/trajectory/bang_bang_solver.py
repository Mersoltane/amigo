"""The solve of a bang-bang problem at each value of a parameter, cheapest way first.

A step tries carried arcs and then the jump of Agamawi, Hager, and Rao.
The homotopy of Bertrand and Epenoy and a refined single phase follow.
Arcs refine with the phs method and relay where the check of Maurer, Buskens, Kim, and Kaya asks.
"""

from dataclasses import dataclass, field

import numpy as np

from ..algorithm.homotopy import Homotopy, HomotopyOptions
from .arcs import combine_arcs, insert_arcs, merge_short_arcs, sign_agreement, sub_mesh, switch_brackets, uniform_arc_meshes


@dataclass
class BangBangOptions:
    """Options of the bang-bang solve of a step and of a family."""

    # Mesh tolerance of the arcs and of the single phase
    tolerance: float = 1e-8

    # Smoothing values of the homotopy from energy toward the objective, and none to skip it
    homotopy: tuple = (1.0, 0.3, 0.1, 0.03, 0.01)

    # Mesh tolerance of the smoothed single phase whose arcs are read
    homotopy_tolerance: float = 1e-4

    # Share of its largest value within which a sign change of a switching function is noise
    switch_noise: float = 1e-4

    # Noise of the first reading of the smoothed single phase
    read_noise: float = 1e-4

    # Whether a step tries the arcs of the last value first
    carry: bool = True

    # Largest change of the parameter, as a share of it, across which the arcs of the last value are carried
    carry_step: float = 0.05

    # Iterations each solve of arcs may take before one rerun from its failed point
    budget: int = 150

    # Most meshes the refined single phase of the last stage tries before the step fails
    last_meshes: int = 3

    # Mesh tolerance the layouts of a solve of arcs settle at before the one that stands refines to the tolerance
    layout_tolerance: float = 1e-4

    # Most arc layouts one solve of arcs tries, with its contact events and its changes
    max_layouts: int = 8

    # Iterations a probing stage may spend testing that its arcs are right, for the carry, the jump, and the first homotopy reading
    probe_effort: int = 1000

    # Iterations a reading that searches for the structure may spend, for the homotopy and the refined single phase
    search_effort: int = 4000

    # Arcs shorter than twice this have collapsed
    arc_floor: float = 1e-3

    # Most halvings in a row of one step of a family before it stops
    max_bisections: int = 4

    # Largest change between the values a family visits, as a share of the value, with values put between, and 0 for none
    max_step: float = 0.0


@dataclass
class Stage:
    """One stage of a step with its outcome, the iterations it took, and the objective it reached."""

    name: str
    outcome: str
    iterations: int
    objective: float = np.nan


@dataclass
class Arcs:
    """Solved arcs with their combined solution, times, check, largest mesh error, and what a near value carries."""

    solution: object
    arcs: list
    check: dict
    error: float
    carry: dict


@dataclass
class Step:
    """The solution at one value of the parameter, its arcs, the single phase they came from, and its stages."""

    value: float
    solution: object = None
    arcs: Arcs = None
    single: object = None
    stages: list = field(default_factory=list)
    meshes: list = field(default_factory=list)

    # The solution the single phase of the next step starts from
    start: object = None

    # Whether the solution meets the mesh tolerance
    met: bool = True

    @property
    def converged(self):
        """Whether the step found a converged solution within the mesh tolerance."""
        return self.solution is not None and bool(self.solution.converged) and self.met

    @property
    def iterations(self):
        """Iterations of every stage of the step."""
        return sum(stage.iterations for stage in self.stages)


def mesh_steps(history, kinds):
    """The combined size, largest error, and convergence of every mesh of a history of arcs, with the one that stands marked."""
    best = history.best_index
    return [
        {
            "intervals": sum(step.solution.mesh.num_intervals for step in entry),
            "points": sum(step.solution.mesh.num_collocation for step in entry),
            "max_error": max(float(step.errors.max()) for step in entry),
            "converged": all(step.solution.converged for step in entry),
            "arcs": len(kinds),
            "chosen": k == best,
        }
        for k, entry in enumerate(history)
    ]


class BangBangProblem:
    """What a bang-bang problem supplies to BangBangSolver, its parameter, its single phase, and its arcs.

    The solves add their NLP iterations to iterations, which the stages of a
    step record.
    """

    # The kind of arc at the upper limit of the control and the kind at its lower limit
    kinds = ("burn", "coast")

    def __init__(self):
        self.iterations = 0

    def set_value(self, value):
        """Set the parameter of the family to a value."""
        raise NotImplementedError

    def single(self, mesh, guess, blend, warm="same"):
        """The single phase solved on a mesh from a guess.

        blend is None for the energy start, a smoothing value of the homotopy,
        or 0 for the objective itself. warm is "same" to start warm only from a
        solve of the same objective, "always" along a homotopy, or "never".
        """
        raise NotImplementedError

    def single_costate(self):
        """Costate at the collocation points of the last converged single phase solve."""
        raise NotImplementedError

    def single_error(self, solution):
        """Largest mesh error of a single phase solution, or nan when it did not converge."""
        raise NotImplementedError

    def reset_single(self):
        """Forget the warm start of the single phase, so its next solve starts from its guess alone."""

    def switching(self, solution, costate):
        """Switching function at the collocation points of a solution, negative where the control sits at its upper limit."""
        raise NotImplementedError

    def sign(self, kind):
        """Sign of the switching function on an arc of a kind, negative on an arc at the upper limit."""
        return -1.0 if kind == self.kinds[0] else 1.0

    def other(self, kind):
        """The kind of arc at the other limit of the control."""
        return self.kinds[1] if kind == self.kinds[0] else self.kinds[0]

    def read(self, source, sigma, noise):
        """Arcs read from a switching function as kinds with their start and end times, with the shift and the parameters of their start."""
        raise NotImplementedError

    def arc_meshes(self, arcs):
        """Uniform meshes of five points per interval with about forty intervals over all the arcs."""
        return uniform_arc_meshes([arcs[0][1]] + [end for _, _, end in arcs], intervals=40, degree=5, min_intervals=2)

    def arc_start(self, source, arcs, meshes, shift, parameters):
        """Solutions of the arcs on their meshes from a source solution."""
        raise NotImplementedError

    def arc_solver(self, kinds, start, brackets, record, budget):
        """A solve of the arcs that takes their meshes and previous solutions, with costate(record) and records.

        The first meshes start from the multipliers of record, a converged
        solve of these arcs or of the arcs a new layout came from, or cold
        when it is None.
        """
        raise NotImplementedError

    def primer(self, solution, costate):
        """The solution with the control the minimum principle gives wherever it rests, for arcs found again from it."""
        return solution

    def touches(self, kinds, solutions):
        """The arcs split into contact events where they touch a path constraint, as kinds, arcs, meshes, and start, or None."""
        return None

    def refinement(self, tolerance, max_iterations=None):
        """The mesh refinement of the single phase or of the arcs to a tolerance."""
        raise NotImplementedError

    def carried(self, carry, value):
        """The arcs a step carries to another value of the parameter."""
        return carry

    def objective(self, solution):
        """The objective of a solution, lower is better."""
        raise NotImplementedError


class BangBangSolver:
    """The solve of a bang-bang problem at each value of its parameter, the cheapest way first."""

    def __init__(self, problem, options=None):
        self.problem = problem
        self.options = BangBangOptions() if options is None else options

    def record(self, step, name, mark, outcome, solution=None):
        """Note a stage of a step with the iterations since mark."""
        objective = np.nan if solution is None or not solution.converged else self.problem.objective(solution)
        step.stages.append(Stage(name, outcome, self.problem.iterations - mark, objective))
        print(f"\nStage {name}: {outcome}, {self.problem.iterations - mark} iterations")

    def check(self, kinds, solutions, sigma):
        """Share of the points where the switching function has the sign of its arc, its largest disagreement, its values, and the arcs it asks for.

        The arcs stand when the switching function asks for no arc of the
        other kind anywhere.
        """
        problem, options = self.problem, self.options
        signs = [problem.sign(kind) for kind in kinds]
        agreement, worst = sign_agreement(signs, solutions, sigma)
        asked, edges = insert_arcs(kinds, solutions, sigma, signs, problem.other, options.switch_noise, options.arc_floor)
        stands = asked == list(kinds) and edges == [solutions[0].t0] + [s.tf for s in solutions]
        return {"agreement": agreement, "worst": worst, "sigma": [float(s) for s in sigma], "stands": stands,
                "kinds": asked, "edges": [float(t) for t in edges]}

    def relaid(self, solutions, costate, kinds, edges):
        """Kinds, arcs, meshes, and start of new arcs over solved ones, each on the part of their combined mesh it spans."""
        problem = self.problem
        combined = combine_arcs(solutions)
        source = problem.primer(combined, costate)
        arcs = [(kind, a, b) for kind, a, b in zip(kinds, edges[:-1], edges[1:])]
        meshes = [sub_mesh(combined.mesh, combined.t0, combined.tf, a, b, self.options.arc_floor) for _, a, b in arcs]
        return list(kinds), arcs, meshes, problem.arc_start(source, arcs, meshes, 0.0, dict(combined.parameters))

    def better(self, found, kept):
        """Whether solved arcs beat the kept ones, by meeting the mesh tolerance they miss or by a lower objective beyond it."""
        tolerance = self.options.tolerance
        meets, kept_meets = found.error <= tolerance, kept.error <= tolerance
        if meets != kept_meets:
            return meets
        best = self.problem.objective(kept.solution)
        return self.problem.objective(found.solution) < best - tolerance * max(1.0, abs(best))

    def best(self, candidates):
        """The best of some solved arcs, or None."""
        kept = None
        for found in candidates:
            if found is not None and (kept is None or self.better(found, kept)):
                kept = found
        return kept

    def solve_arcs(self, source, sigma, given=None, carry=None, noise=0.0, budget=None, effort=None, search=False):
        """Solved arcs from a source or given or carried, with free switch times and every mesh size.

        A layout relays at contact events, at collapsed arcs, and where the switching function disagrees.
        New arcs stand only when they lower the objective, as Shamsi requires, and no layout is tried twice.
        A search settles each layout at the layout tolerance and refines the one that stands.
        """
        problem, options = self.problem, self.options
        rough_tolerance = max(options.layout_tolerance, options.tolerance) if search else options.tolerance
        rough = problem.refinement(rough_tolerance)
        refining = problem.refinement(options.tolerance)
        steps, layout, kept, seen = [], None, None, set()
        mark = problem.iterations

        def spent():
            return effort is not None and problem.iterations - mark >= effort

        for attempt in range(options.max_layouts):
            if spent():
                return kept, steps
            record, bracketed = None, False
            if layout is not None:
                # A new layout starts from the multipliers of the solve it came from
                kinds, arcs, meshes, start, record = layout
            elif carry is not None:
                # Carried arcs keep their meshes and start warm from the multipliers they bring
                kinds, start, record = list(carry["kinds"]), carry["solutions"], carry["record"]
                arcs = [(kind, s.t0, s.tf) for kind, s in zip(kinds, start)]
                meshes = [s.mesh for s in start]
            else:
                # Switches read on the single phase stay in brackets as Agamawi, Hager, and Rao keep them
                arcs, shift, parameters = given if given is not None else problem.read(source, sigma, noise)
                bracketed = given is None
                kinds = [kind for kind, _, _ in arcs]
                meshes = problem.arc_meshes(arcs)
                start = problem.arc_start(source, arcs, meshes, shift, parameters)
            layout = None

            # Switches of solved arcs move freely, with the floor on every arc duration keeping them in order
            brackets = switch_brackets([arcs[0][1]] + [end for _, _, end in arcs]) if bracketed else []
            events = {}

            def stop(solutions):
                events["touch"] = problem.touches(kinds, solutions)
                events["short"] = min(s.tf - s.t0 for s in solutions) <= 2.0 * options.arc_floor
                return events["touch"] is not None or events["short"]

            solver = problem.arc_solver(kinds, start, brackets, record, budget)
            history = rough.run(meshes, solver, stop=stop)
            steps += mesh_steps(history, kinds)
            if events.get("touch") is not None:
                layout = events["touch"] + (solver.records[-1],)
                continue
            if events.get("short"):
                print("\nAn arc shrank to its floor, so its neighbors take it over")
                last = [step.solution for step in history[-1]]
                edges = merge_short_arcs(kinds, [last[0].t0] + [s.tf for s in last], options.arc_floor)
                layout = self.relaid(last, solver.costate(solver.records[-1]), *edges) + (solver.records[-1],)
                continue
            best = history.best_index
            solutions = [step.solution for step in history[best]]
            if not all(s.converged for s in solutions):
                return kept, steps
            record = solver.records[best]
            costate = solver.costate(record)
            combined = combine_arcs(solutions)
            check = self.check(kinds, solutions, problem.switching(combined, costate))

            # A layout the settled arcs disagree with is relaid before any mesh refines further
            if not check["stands"] and not spent():
                asked = self.asked(check)
                if asked not in seen:
                    seen.add(asked)
                    print(f"\nThe switching function disagrees with the {len(kinds)} arcs by {check['worst']:.1e} of "
                          f"its largest value, so {len(check['kinds'])} arcs go where it asks for them")
                    layout = self.relaid(solutions, costate, check["kinds"], check["edges"]) + (record,)
                    continue

            # The layout that settled refines on to the mesh tolerance from where it stopped
            if rough_tolerance > options.tolerance:
                solver = problem.arc_solver(kinds, solutions, brackets, record, budget)
                fine = refining.run([s.mesh for s in solutions], solver, first=solutions, stop=stop)
                steps += mesh_steps(fine, kinds)[1:]
                last_record = solver.records[-1] if solver.records else record
                if events.get("touch") is not None:
                    layout = events["touch"] + (last_record,)
                    continue
                if events.get("short"):
                    print("\nAn arc shrank to its floor, so its neighbors take it over")
                    last = [step.solution for step in fine[-1]]
                    edges = merge_short_arcs(kinds, [last[0].t0] + [s.tf for s in last], options.arc_floor)
                    layout = self.relaid(last, solver.costate(last_record), *edges) + (last_record,)
                    continue
                records = [record] + solver.records
                best = fine.best_index
                solutions = [step.solution for step in fine[best]]
                if not all(s.converged for s in solutions):
                    return kept, steps
                record = records[best]
                costate = solver.costate(record)
                combined = combine_arcs(solutions)
                check = self.check(kinds, solutions, problem.switching(combined, costate))
                history = fine
            arcs = [(kind, s.t0, s.tf) for kind, s in zip(kinds, solutions)]
            error = max(float(step.errors.max()) for step in history[best])
            found = Arcs(combined, arcs, check, error, {"kinds": kinds, "solutions": solutions, "record": record})
            if kept is not None and not self.better(found, kept):
                print(f"\nThe new arcs reach {problem.objective(combined):.6g} against "
                      f"{problem.objective(kept.solution):.6g}, so the last arcs stand")
                return kept, steps
            kept = found
            asked = self.asked(check)
            if check["stands"] or asked in seen or spent():
                return kept, steps
            seen.add(asked)
            print(f"\nThe switching function disagrees with the {len(kinds)} arcs by {check['worst']:.1e} of its "
                  f"largest value, so {len(check['kinds'])} arcs go where it asks for them")
            layout = self.relaid(solutions, costate, check["kinds"], check["edges"]) + (record,)
        return kept, steps

    def asked(self, check):
        """The layout a check asks for, as the signature that marks it as already tried."""
        return tuple(check["kinds"]), tuple(round(float(t), 6) for t in check["edges"])

    def refined(self, solution, blend, tolerance, meshes=None):
        """The single phase of a blend refined to a tolerance from a solution of it, its largest error, and the size of every mesh.

        The costate comes from the last solve, so the mesh that stands is
        solved once more when it was not the last. meshes caps the meshes the
        refinement tries.
        """
        problem = self.problem
        history = problem.refinement(tolerance, max_iterations=meshes).run(
            solution.mesh, lambda m, previous: problem.single(m, previous, blend), first=solution
        )
        best = history.best_index
        solution = history[best].solution
        steps = [
            {"intervals": entry.solution.mesh.num_intervals, "points": entry.solution.mesh.num_collocation,
             "max_error": float(entry.errors.max()), "converged": entry.solution.converged, "chosen": k == best}
            for k, entry in enumerate(history)
        ]
        if solution.converged and best != len(history) - 1:
            solution = problem.single(solution.mesh, solution, blend)
        return solution, steps[best]["max_error"], steps

    def smoothed(self, step, start):
        """Arcs from the single phase carried from energy toward the objective, and the smoothed solution they came from.

        The arcs are read on the mesh of the homotopy first, with a budget on
        each solve, and the smoothed problem is refined for a second reading
        only when they do not stand.
        """
        problem, options = self.problem, self.options
        mesh = start.mesh
        path = Homotopy().run(options.homotopy, lambda eps, previous: problem.single(mesh, previous, eps, "always"), start)
        step.meshes += [{"intervals": mesh.num_intervals, "points": mesh.num_collocation, "max_error": np.nan,
                         "converged": s.converged, "homotopy": s.value} for s in path]
        if path.last is None:
            return path[-1].solution, None
        eps = path.last.value
        smooth = path.last.solution
        found = self.read_and_solve(step, smooth, options.read_noise, options.budget, options.probe_effort)
        if self.serves(found):
            return smooth, found
        refined, _, steps = self.refined(smooth, eps, options.homotopy_tolerance)
        step.meshes += [dict(s, homotopy=eps) for s in steps[1:]]
        second = None
        if refined.converged:
            second = self.read_and_solve(step, refined, options.read_noise, options.budget, options.search_effort,
                                         search=True)
            smooth = refined
        return smooth, self.best([found, second])

    def read_and_solve(self, step, single, noise, budget, effort=None, search=False):
        """The arcs read from the last single phase solve and solved, with their meshes noted on the step."""
        sigma = self.problem.switching(single, self.problem.single_costate())
        found, steps = self.solve_arcs(single, sigma, noise=noise, budget=budget, effort=effort, search=search)
        step.meshes += [dict(s, arcs=True) for s in steps]
        return found

    def homotopy_stage(self, step, start):
        """The homotopy stage of a step, recorded with its outcome."""
        mark = self.problem.iterations
        smooth, found = self.smoothed(step, start)
        self.record(step, "homotopy", mark, "serves" if found is not None else "does not serve",
                    None if found is None else found.solution)
        return smooth, found

    def step(self, value, previous=None, start=None):
        """The solution at one value of the parameter from the step before it or from a start, the cheapest way first."""
        problem, options = self.problem, self.options
        problem.set_value(value)
        step = Step(value)
        smoothing = bool(options.homotopy)
        candidates = []

        # Arcs carried from a near value come first, with a budget on each solve
        near = options.carry and previous is not None and previous.arcs is not None and (
            abs(value - previous.value) <= options.carry_step * abs(value)
        )
        if near:
            mark = problem.iterations
            carried, steps = self.solve_arcs(None, None, carry=problem.carried(previous.arcs.carry, value),
                                             budget=options.budget, effort=options.probe_effort)
            step.meshes += [dict(s, arcs=True, carried=True) for s in steps]
            serves = self.serves(carried)
            self.record(step, "carry", mark, "serves" if serves else "does not serve",
                        None if carried is None else carried.solution)
            if serves:
                return self.finish(step, carried, carried.solution)
            candidates.append(carried)

        # The jump solves the single phase straight to the objective from the start of this step
        if start is None:
            start = previous.start

        # A step that carried its arcs left the warm start of the single phase at an older value
        if previous is not None and previous.single is None:
            problem.reset_single()
        mark = problem.iterations
        single = problem.single(start.mesh, start, 0.0)
        error = problem.single_error(single)
        missed = error > options.tolerance
        step.single = single
        step.meshes.append({"intervals": single.mesh.num_intervals, "points": single.mesh.num_collocation,
                            "max_error": error, "converged": single.converged})
        if single.converged and not missed:
            self.record(step, "jump", mark, "the single phase meets the tolerance", single)
            step.solution, step.start = single, single
            return step
        jumped = None
        if single.converged:
            jumped = self.read_and_solve(step, single, 0.0, options.budget, options.probe_effort)
            serves = self.serves(jumped)
            self.record(step, "jump", mark, "serves" if serves else "does not serve",
                        None if jumped is None else jumped.solution)
            if serves:
                return self.finish(step, jumped, single)
            candidates.append(jumped)
        else:
            self.record(step, "jump", mark, "the single phase did not converge")

        # The homotopy reads the arcs afresh from the same start
        if smoothing:
            smooth, found = self.homotopy_stage(step, start)
            candidates.append(found)
            kept = self.best(candidates)
            if kept is not None:
                return self.finish(step, kept, smooth if found is not None and kept is found else single)

        # Agamawi, Hager, and Rao refine the single phase of the objective when no arcs serve
        if not single.converged:
            step.solution, step.start = single, start
            return step
        mark = problem.iterations
        refined, error, steps = self.refined(single, 0.0, options.tolerance, meshes=options.last_meshes)
        step.meshes += steps[1:]
        found = None
        if refined.converged:
            found = self.read_and_solve(step, refined, 0.0, options.budget, options.search_effort, search=True)
        self.record(step, "refined single phase", mark, "serves" if found is not None else "does not serve",
                    None if found is None else found.solution)
        candidates.append(found)
        kept = self.best(candidates)
        if kept is not None:
            return self.finish(step, kept, refined)
        step.solution, step.start, step.met = refined, refined, error <= options.tolerance
        return step

    def serves(self, found):
        """Whether solved arcs end their step, by standing against their switching function within the mesh tolerance."""
        return found is not None and found.check["stands"] and found.error <= self.options.tolerance

    def finish(self, step, found, start):
        """The step with its arcs, starting the next one from a solution, and failed when they miss the mesh tolerance."""
        step.solution, step.arcs, step.start = found.solution, found, start
        step.met = found.error <= self.options.tolerance
        return step

    def visits(self, values):
        """The values a family visits, with values put geometrically between two that lie more than max_step apart."""
        step = self.options.max_step
        out = [values[0]]
        for value in values[1:]:
            last = out[-1]
            if step > 0.0 and last * value > 0.0:
                n = max(1, int(np.ceil(abs(np.log(value / last)) / np.log1p(step))))
                out += [last * (value / last) ** (k / n) for k in range(1, n)]
            out.append(value)
        return out

    def family(self, values, start, on_step=None, previous=None):
        """Steps along the values of the parameter from a start, each from the last that converged.

        A value that fails gets a value halfway to it before the family stops.
        on_step(step) sees every step as it ends, also those at values put between.
        Given previous, the family resumes from that step instead of the start.
        """
        def solve(value, prior):
            step = self.step(value, prior, start if prior is None and previous is None else None)
            if on_step is not None:
                on_step(step)
            return step

        return Homotopy(HomotopyOptions(max_bisections=self.options.max_bisections)).run(self.visits(values), solve,
                                                                                         start=previous)
