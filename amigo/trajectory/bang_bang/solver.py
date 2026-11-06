"""
Bang-bang solver
----------------

- The solve of a bang-bang problem at each value of a parameter, the cheapest way first
- A step tries carried arcs and then the jump of Agamawi, Hager, and Rao
- The homotopy of Bertrand and Epenoy and a refined single phase follow
- Arcs refine with the phs method and relay where the check of Maurer, Buskens, Kim, and Kaya asks
"""

import numpy as np

from ...algorithm.homotopy import Homotopy, HomotopyOptions
from .arcs import (
    arc_edges,
    arc_times,
    combine_arcs,
    insert_arcs,
    merge_short_arcs,
    sign_agreement,
    solution_edges,
    sub_mesh,
    switch_brackets,
)
from .options import BangBangOptions
from .step import Arcs, Stage, Step


def mesh_steps(history, kinds):
    """Combined size, largest error, and convergence of every mesh of a history of arcs"""
    best = history.best_index
    steps = []

    for k in range(len(history)):
        entry = history[k]

        # The arcs of one mesh together
        intervals = 0
        points = 0
        errors = []
        converged = True

        for step in entry:
            intervals += step.solution.mesh.num_intervals
            points += step.solution.mesh.num_collocation
            errors.append(float(step.errors.max()))

            if not step.solution.converged:
                converged = False

        # The mesh that stands is marked as chosen
        steps.append(
            {
                "intervals": intervals,
                "points": points,
                "max_error": max(errors),
                "converged": converged,
                "arcs": len(kinds),
                "chosen": k == best,
            }
        )

    return steps


def entry_solutions(entry):
    """Solutions of the arcs in one entry of a refinement history"""
    solutions = []

    for step in entry:
        solutions.append(step.solution)

    return solutions


def all_converged(solutions):
    """Whether every solution converged"""
    for solution in solutions:
        if not solution.converged:
            return False

    return True


class ArcEvents:
    """Stop test of a solve of arcs that notes a contact event or an arc that shrank to its floor"""

    def __init__(self, problem, kinds, floor):
        self.problem = problem
        self.kinds = kinds
        self.floor = floor

        # The contact event and the collapse that the last solve showed
        self.touch = None
        self.short = False

    def __call__(self, solutions):
        self.touch = self.problem.touches(self.kinds, solutions)

        # An arc shorter than twice the floor has collapsed
        durations = []

        for solution in solutions:
            durations.append(solution.tf - solution.t0)

        self.short = min(durations) <= 2.0 * self.floor

        return self.touch is not None or self.short


class SingleSolve:
    """Solve of the single phase of one blend on each mesh of a refinement"""

    def __init__(self, problem, blend):
        self.problem = problem
        self.blend = blend

    def __call__(self, mesh, previous):
        return self.problem.single(mesh, previous, self.blend)


class SmoothingSolve:
    """Solve of the single phase on one mesh at each smoothing value of the homotopy"""

    def __init__(self, problem, mesh):
        self.problem = problem
        self.mesh = mesh

    def __call__(self, eps, previous):
        return self.problem.single(self.mesh, previous, eps, "always")


class FamilyStep:
    """Solve of one value of a family"""

    def __init__(self, solver, start, on_step, previous):
        self.solver = solver
        self.start = start
        self.on_step = on_step
        self.previous = previous

    def __call__(self, value, prior):
        # Only a family that does not resume starts from the start, until a step converges
        start = None
        if prior is None and self.previous is None:
            start = self.start

        step = self.solver.step(value, prior, start)

        # on_step sees every step as it ends, also those at values put between
        if self.on_step is not None:
            self.on_step(step)

        return step


class BangBangSolver:
    """The solve of a bang-bang problem at each value of its parameter, the cheapest way first"""

    def __init__(self, problem, options=None):
        self.problem = problem

        if options is None:
            options = BangBangOptions()

        self.options = options

    def record(self, step, name, mark, outcome, solution=None):
        """Note a stage of a step with the iterations since mark"""
        objective = np.nan
        if solution is not None and solution.converged:
            objective = self.problem.objective(solution)

        iterations = self.problem.iterations - mark

        step.stages.append(Stage(name, outcome, iterations, objective))
        print(f"\nStage {name}: {outcome}, {iterations} iterations")

    def record_arcs(self, step, name, mark, found, serves):
        """Note a stage of a step that solved arcs, with whether they serve"""
        outcome = "does not serve"
        if serves:
            outcome = "serves"

        solution = None
        if found is not None:
            solution = found.solution

        self.record(step, name, mark, outcome, solution)

    def check(self, kinds, solutions, sigma):
        """Agreement of the switching function with the arcs, its values, and the arcs it asks for"""
        problem = self.problem
        options = self.options

        # The sign each arc expects of the switching function
        signs = []

        for kind in kinds:
            signs.append(problem.sign(kind))

        agreement = sign_agreement(signs, solutions, sigma)

        # The arcs the switching function asks for
        inserted = insert_arcs(kinds, solutions, sigma, signs, problem.other, options.switch_noise, options.arc_floor)
        asked = inserted[0]
        edges = inserted[1]

        # The arcs stand when the switching function asks for no arc of the other kind anywhere
        stands = asked == list(kinds) and edges == solution_edges(solutions)

        sigma_values = []

        for value in sigma:
            sigma_values.append(float(value))

        edge_values = []

        for t in edges:
            edge_values.append(float(t))

        return {
            "agreement": agreement[0],
            "worst": agreement[1],
            "sigma": sigma_values,
            "stands": stands,
            "kinds": asked,
            "edges": edge_values,
        }

    def relaid(self, solutions, costate, kinds, edges):
        """Kinds, arcs, meshes, and start of new arcs over solved ones"""
        problem = self.problem

        combined = combine_arcs(solutions)
        source = problem.primer(combined, costate)

        # Each new arc goes on the part of the combined mesh it spans
        arcs = []
        meshes = []

        for i in range(len(kinds)):
            a = edges[i]
            b = edges[i + 1]

            arcs.append((kinds[i], a, b))
            meshes.append(sub_mesh(combined.mesh, combined.t0, combined.tf, a, b, self.options.arc_floor))

        start = problem.arc_start(source, arcs, meshes, 0.0, dict(combined.parameters))

        return list(kinds), arcs, meshes, start

    def better(self, found, kept):
        """Whether solved arcs beat the kept ones, by meeting the mesh tolerance they miss or by a lower objective"""
        tolerance = self.options.tolerance

        meets = found.error <= tolerance
        kept_meets = kept.error <= tolerance

        if meets != kept_meets:
            return meets

        # The objective must drop by more than the tolerance
        best = self.problem.objective(kept.solution)

        return self.problem.objective(found.solution) < best - tolerance * max(1.0, abs(best))

    def best(self, candidates):
        """The best of some solved arcs, or None"""
        kept = None

        for found in candidates:
            if found is None:
                continue

            if kept is None or self.better(found, kept):
                kept = found

        return kept

    def spent(self, mark, effort):
        """Whether a solve of arcs used up the iterations it may spend since mark"""
        return effort is not None and self.problem.iterations - mark >= effort

    def asked(self, check):
        """The layout a check asks for, as the signature that marks it as already tried"""
        edges = []

        for t in check["edges"]:
            edges.append(round(float(t), 6))

        return tuple(check["kinds"]), tuple(edges)

    def collapsed_layout(self, kinds, entry, solver, record):
        """Layout where the neighbors of an arc that shrank to its floor take it over"""
        print("\nAn arc shrank to its floor, so its neighbors take it over")

        last = entry_solutions(entry)
        merged = merge_short_arcs(kinds, solution_edges(last), self.options.arc_floor)
        relaid = self.relaid(last, solver.costate(record), merged[0], merged[1])

        return relaid + (record,)

    def disagreeing_layout(self, kinds, solutions, costate, check, record):
        """Layout with the arcs where the switching function asks for them"""
        print(
            f"\nThe switching function disagrees with the {len(kinds)} arcs by {check['worst']:.1e} of "
            f"its largest value, so {len(check['kinds'])} arcs go where it asks for them"
        )

        relaid = self.relaid(solutions, costate, check["kinds"], check["edges"])

        return relaid + (record,)

    def solve_arcs(self, source, sigma, given=None, carry=None, noise=0.0, budget=None, effort=None, search=False):
        """Solved arcs from a source or given or carried, with free switch times and every mesh size"""
        problem = self.problem
        options = self.options

        # A search settles each layout at the layout tolerance and refines the one that stands
        rough_tolerance = options.tolerance
        if search:
            rough_tolerance = max(options.layout_tolerance, options.tolerance)

        rough = problem.refinement(rough_tolerance)
        refining = problem.refinement(options.tolerance)

        # The layouts tried so far and the best arcs kept
        steps = []
        layout = None
        kept = None
        seen = set()

        mark = problem.iterations

        for attempt in range(options.max_layouts):
            if self.spent(mark, effort):
                return kept, steps

            record = None
            bracketed = False

            # A layout holds the kinds, arcs, meshes, and start, and the record its multipliers come from
            if layout is not None:
                kinds = layout[0]
                arcs = layout[1]
                meshes = layout[2]
                start = layout[3]
                record = layout[4]

            # Carried arcs keep their meshes and start warm from the multipliers they bring
            elif carry is not None:
                kinds = list(carry["kinds"])
                start = carry["solutions"]
                record = carry["record"]

                arcs = arc_times(kinds, start)
                meshes = []

                for solution in start:
                    meshes.append(solution.mesh)

            # Switches read on the single phase stay in brackets as Agamawi, Hager, and Rao keep them
            else:
                if given is not None:
                    reading = given
                else:
                    reading = problem.read(source, sigma, noise)

                arcs = reading[0]
                shift = reading[1]
                parameters = reading[2]

                bracketed = given is None

                kinds = []

                for arc in arcs:
                    kinds.append(arc[0])

                meshes = problem.arc_meshes(arcs)
                start = problem.arc_start(source, arcs, meshes, shift, parameters)

            layout = None

            # Switches of solved arcs move freely, with the floor on every arc duration keeping them in order
            brackets = []
            if bracketed:
                brackets = switch_brackets(arc_edges(arcs))

            events = ArcEvents(problem, kinds, options.arc_floor)

            solver = problem.arc_solver(kinds, start, brackets, record, budget)
            history = rough.run(meshes, solver, stop=events)
            steps.extend(mesh_steps(history, kinds))

            # A contact event splits the arcs where they touch a path constraint
            if events.touch is not None:
                layout = events.touch + (solver.records[-1],)
                continue

            if events.short:
                layout = self.collapsed_layout(kinds, history[-1], solver, solver.records[-1])
                continue

            # The arcs on the mesh that stands
            best = history.best_index
            solutions = entry_solutions(history[best])

            if not all_converged(solutions):
                return kept, steps

            record = solver.records[best]
            costate = solver.costate(record)

            combined = combine_arcs(solutions)
            check = self.check(kinds, solutions, problem.switching(combined, costate))

            # A layout the settled arcs disagree with is relaid before any mesh refines further
            if not check["stands"] and not self.spent(mark, effort):
                asked = self.asked(check)

                if asked not in seen:
                    seen.add(asked)
                    layout = self.disagreeing_layout(kinds, solutions, costate, check, record)
                    continue

            # The layout that settled refines on to the mesh tolerance from where it stopped
            if rough_tolerance > options.tolerance:
                solver = problem.arc_solver(kinds, solutions, brackets, record, budget)

                fine_meshes = []

                for solution in solutions:
                    fine_meshes.append(solution.mesh)

                fine = refining.run(fine_meshes, solver, first=solutions, stop=events)
                steps.extend(mesh_steps(fine, kinds)[1:])

                last_record = record
                if solver.records:
                    last_record = solver.records[-1]

                if events.touch is not None:
                    layout = events.touch + (last_record,)
                    continue

                if events.short:
                    layout = self.collapsed_layout(kinds, fine[-1], solver, last_record)
                    continue

                # The first entry of the fine history is the settled layout itself
                records = [record] + solver.records

                best = fine.best_index
                solutions = entry_solutions(fine[best])

                if not all_converged(solutions):
                    return kept, steps

                record = records[best]
                costate = solver.costate(record)

                combined = combine_arcs(solutions)
                check = self.check(kinds, solutions, problem.switching(combined, costate))

                history = fine

            # The arcs this layout found
            arcs = arc_times(kinds, solutions)

            errors = []

            for step in history[best]:
                errors.append(float(step.errors.max()))

            error = max(errors)

            carry_on = {"kinds": kinds, "solutions": solutions, "record": record}
            found = Arcs(combined, arcs, check, error, carry_on)

            # New arcs stand only when they lower the objective, as Shamsi requires
            if kept is not None and not self.better(found, kept):
                print(
                    f"\nThe new arcs reach {problem.objective(combined):.6g} against "
                    f"{problem.objective(kept.solution):.6g}, so the last arcs stand"
                )
                return kept, steps

            kept = found

            # Stop when the arcs stand, when their layout was tried, or when the effort is spent
            asked = self.asked(check)

            if check["stands"] or asked in seen or self.spent(mark, effort):
                return kept, steps

            seen.add(asked)
            layout = self.disagreeing_layout(kinds, solutions, costate, check, record)

        return kept, steps

    def refined(self, solution, blend, tolerance, meshes=None):
        """The single phase of a blend refined to a tolerance, its largest error, and the size of every mesh"""
        problem = self.problem

        # meshes caps the meshes the refinement tries
        refinement = problem.refinement(tolerance, max_iterations=meshes)
        history = refinement.run(solution.mesh, SingleSolve(problem, blend), first=solution)

        best = history.best_index
        solution = history[best].solution

        # The size, the error, and the convergence of every mesh
        steps = []

        for k in range(len(history)):
            entry = history[k]

            steps.append(
                {
                    "intervals": entry.solution.mesh.num_intervals,
                    "points": entry.solution.mesh.num_collocation,
                    "max_error": float(entry.errors.max()),
                    "converged": entry.solution.converged,
                    "chosen": k == best,
                }
            )

        # The costate comes from the last solve, so the mesh that stands is solved once more when it was not the last
        if solution.converged and best != len(history) - 1:
            solution = problem.single(solution.mesh, solution, blend)

        return solution, steps[best]["max_error"], steps

    def read_and_solve(self, step, single, noise, budget, effort=None, search=False):
        """The arcs read from the last single phase solve and solved, with their meshes noted on the step"""
        sigma = self.problem.switching(single, self.problem.single_costate())

        result = self.solve_arcs(single, sigma, noise=noise, budget=budget, effort=effort, search=search)
        found = result[0]
        steps = result[1]

        for s in steps:
            step.meshes.append(dict(s, arcs=True))

        return found

    def serves(self, found):
        """Whether solved arcs end their step, by standing against their switching function within the mesh tolerance"""
        if found is None:
            return False

        return found.check["stands"] and found.error <= self.options.tolerance

    def smoothed(self, step, start):
        """Arcs from the single phase carried from energy toward the objective, and the smoothed solution they came from"""
        problem = self.problem
        options = self.options
        mesh = start.mesh

        # Carry the single phase along the smoothing values on the mesh of the start
        path = Homotopy().run(options.homotopy, SmoothingSolve(problem, mesh), start)

        for s in path:
            step.meshes.append(
                {
                    "intervals": mesh.num_intervals,
                    "points": mesh.num_collocation,
                    "max_error": np.nan,
                    "converged": s.converged,
                    "homotopy": s.value,
                }
            )

        if path.last is None:
            return path[-1].solution, None

        eps = path.last.value
        smooth = path.last.solution

        # The arcs are read on the mesh of the homotopy first, with a budget on each solve
        found = self.read_and_solve(step, smooth, options.read_noise, options.budget, options.probe_effort)

        if self.serves(found):
            return smooth, found

        # The smoothed problem is refined for a second reading only when they do not stand
        result = self.refined(smooth, eps, options.homotopy_tolerance)
        refined = result[0]
        steps = result[2]

        for s in steps[1:]:
            step.meshes.append(dict(s, homotopy=eps))

        second = None

        if refined.converged:
            second = self.read_and_solve(
                step, refined, options.read_noise, options.budget, options.search_effort, search=True
            )
            smooth = refined

        return smooth, self.best([found, second])

    def homotopy_stage(self, step, start):
        """The homotopy stage of a step, recorded with its outcome"""
        mark = self.problem.iterations

        result = self.smoothed(step, start)
        smooth = result[0]
        found = result[1]

        self.record_arcs(step, "homotopy", mark, found, found is not None)

        return smooth, found

    def finish(self, step, found, start):
        """The step with its arcs, starting the next one from a solution, and failed when they miss the mesh tolerance"""
        step.solution = found.solution
        step.arcs = found
        step.start = start
        step.met = found.error <= self.options.tolerance

        return step

    def step(self, value, previous=None, start=None):
        """The solution at one value of the parameter from the step before it or from a start, the cheapest way first"""
        problem = self.problem
        options = self.options

        problem.set_value(value)

        step = Step(value)
        smoothing = bool(options.homotopy)
        candidates = []

        # Arcs carried from a near value come first, with a budget on each solve
        near = False
        if options.carry and previous is not None and previous.arcs is not None:
            near = abs(value - previous.value) <= options.carry_step * abs(value)

        if near:
            mark = problem.iterations

            carry = problem.carried(previous.arcs.carry, value)
            result = self.solve_arcs(None, None, carry=carry, budget=options.budget, effort=options.probe_effort)
            carried = result[0]
            steps = result[1]

            for s in steps:
                step.meshes.append(dict(s, arcs=True, carried=True))

            serves = self.serves(carried)
            self.record_arcs(step, "carry", mark, carried, serves)

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
        step.meshes.append(
            {
                "intervals": single.mesh.num_intervals,
                "points": single.mesh.num_collocation,
                "max_error": error,
                "converged": single.converged,
            }
        )

        if single.converged and not missed:
            self.record(step, "jump", mark, "the single phase meets the tolerance", single)

            step.solution = single
            step.start = single

            return step

        if single.converged:
            jumped = self.read_and_solve(step, single, 0.0, options.budget, options.probe_effort)

            serves = self.serves(jumped)
            self.record_arcs(step, "jump", mark, jumped, serves)

            if serves:
                return self.finish(step, jumped, single)

            candidates.append(jumped)
        else:
            self.record(step, "jump", mark, "the single phase did not converge")

        # The homotopy reads the arcs afresh from the same start
        if smoothing:
            result = self.homotopy_stage(step, start)
            smooth = result[0]
            found = result[1]

            candidates.append(found)
            kept = self.best(candidates)

            # The next step starts from the smoothed solution when the homotopy arcs are kept
            if kept is not None:
                next_start = single
                if found is not None and kept is found:
                    next_start = smooth

                return self.finish(step, kept, next_start)

        # Agamawi, Hager, and Rao refine the single phase of the objective when no arcs serve
        if not single.converged:
            step.solution = single
            step.start = start

            return step

        mark = problem.iterations

        result = self.refined(single, 0.0, options.tolerance, meshes=options.last_meshes)
        refined = result[0]
        error = result[1]
        steps = result[2]

        step.meshes.extend(steps[1:])

        found = None

        if refined.converged:
            found = self.read_and_solve(step, refined, 0.0, options.budget, options.search_effort, search=True)

        self.record_arcs(step, "refined single phase", mark, found, found is not None)

        candidates.append(found)
        kept = self.best(candidates)

        if kept is not None:
            return self.finish(step, kept, refined)

        step.solution = refined
        step.start = refined
        step.met = error <= options.tolerance

        return step

    def visits(self, values):
        """The values a family visits, with values put geometrically between two that lie more than max_step apart"""
        step = self.options.max_step
        out = [values[0]]

        for value in values[1:]:
            last = out[-1]

            if step > 0.0 and last * value > 0.0:
                n = max(1, int(np.ceil(abs(np.log(value / last)) / np.log1p(step))))

                for k in range(1, n):
                    out.append(last * (value / last) ** (k / n))

            out.append(value)

        return out

    def family(self, values, start, on_step=None, previous=None):
        """Steps along the values of the parameter from a start, each from the last that converged"""
        # Every step starts from the last that converged, or from previous when the family resumes
        solve = FamilyStep(self, start, on_step, previous)

        # A value that fails gets a value halfway to it before the family stops
        homotopy = Homotopy(HomotopyOptions(max_bisections=self.options.max_bisections))

        return homotopy.run(self.visits(values), solve, start=previous)
