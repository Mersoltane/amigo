"""
Trajectory benchmark
--------------------

- Solves every problem of the set on each of its meshes from each of its guess families
- Every run solves in a fresh process with the same solver settings
- A run is solved when it converges to the reference objective of its problem
- A run that converges to another objective has found another optimum
- A run of an infeasible variant is solved when the solver reports local infeasibility
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import amigo as am
from amigo.algorithm.globalization import feasibility_restoration
from amigo.algorithm.solvers import amigo_solver
from academic.alp_rider import alp_rider
from academic.bryson_denham import bryson_denham
from aerospace.delta3_ascent import delta3_ascent
from aerospace.goddard import goddard
from academic.hypersensitive import hypersensitive
from aerospace.landing_6dof import landing_6dof
from aerospace.moon_lander import moon_lander
from aerospace.powered_descent_3dof import powered_descent_3dof
from aerospace.reorientation import reorientation
from robotics.robot_arm import robot_arm

HERE = os.path.dirname(os.path.abspath(__file__))

# Domain folder of every problem
GROUPS = {
    "alp_rider": "academic",
    "bryson_denham": "academic",
    "hypersensitive": "academic",
    "goddard": "aerospace",
    "delta3_ascent": "aerospace",
    "moon_lander": "aerospace",
    "powered_descent_3dof": "aerospace",
    "landing_6dof": "aerospace",
    "reorientation": "aerospace",
    "robot_arm": "robotics",
}

# Seconds allowed per run
TIMEOUT = 900

# Relative gap to the reference that still counts as the same optimum
TOLERANCE = 1e-3

# Solver settings of every run, all others stay at the amigo defaults
OPTIONS = {"max_iterations": 1000, "max_wall_time": 600.0}

# A run is a fresh interpreter that imports this runner and solves the run on its input
CHILD = "import run_benchmark\nrun_benchmark.run_child()"

# Guess families of a problem with a free final time
FREE_TIME = ("literature", "linear", "constant", "tf_half", "tf_double")

# Name, variant, meshes, guess families, and the reference objective with its source
PROBLEMS = [
    ("goddard", None, (50, 200), FREE_TIME, -1.01283, "COPS 3.0"),
    ("bryson_denham", None, (30, 120), ("linear", "constant"), 4.0, "analytic"),
    ("hypersensitive", None, (100, 400), ("literature", "linear", "constant"), 1.330807, "analytic"),
    ("alp_rider", None, (100, 400), ("literature", "constant"), None, "none verified"),
    ("moon_lander", None, (50, 200), FREE_TIME, 1.420313, "shooting"),
    ("moon_lander", "infeasible", (50,), ("literature", "linear"), None, "infeasible"),
    ("robot_arm", None, (50, 200), FREE_TIME, 9.14093, "COPS 3.0"),
    ("reorientation", None, (50, 150), FREE_TIME, None, "none verified"),
    ("powered_descent_3dof", None, (40, 120), FREE_TIME[1:], None, "tf 75 s only"),
    ("powered_descent_3dof", "infeasible", (40,), ("linear", "constant"), None, "infeasible"),
    ("delta3_ascent", None, (10, 30), FREE_TIME, -7529.661, "PSOPT"),
    ("delta3_ascent", "canonical", (10, 30), FREE_TIME, -7529.661, "PSOPT"),
    ("landing_6dof", None, (20, 60), FREE_TIME, None, "none verified"),
]

# The solver methods that the counters of a run wrap
FACTOR = amigo_solver.AmigoSolver.factor
RESTORE = feasibility_restoration.FeasibilityRestoration.restore

# Factorizations and restorations of the run in this process
COUNTS = {"factorizations": 0, "resto_calls": 0, "resto_iterations": 0}


def counted_factor(self, *args, **kwargs):
    """Factorization that counts itself"""
    COUNTS["factorizations"] += 1
    return FACTOR(self, *args, **kwargs)


def counted_restore(self, *args, **kwargs):
    """Restoration that counts its calls and its iterations"""
    info = RESTORE(self, *args, **kwargs)
    COUNTS["resto_calls"] += 1
    COUNTS["resto_iterations"] += info.iterations
    return info


def create_run(name, variant, num_intervals, guess):
    """Model of one run with the initial guess of its family"""
    infeasible = variant == "infeasible"
    canonical = variant == "canonical"

    if name == "alp_rider":
        model = alp_rider.create_alp_rider_model(num_intervals=num_intervals)
        alp_rider.set_initial_guess(model, num_intervals=num_intervals, guess=guess)

    elif name == "bryson_denham":
        model = bryson_denham.create_bryson_denham_model(num_intervals=num_intervals)
        bryson_denham.set_initial_guess(model, num_intervals=num_intervals, guess=guess)

    elif name == "delta3_ascent":
        model = delta3_ascent.create_delta3_ascent_model(
            num_intervals=num_intervals, canonical=canonical
        )
        delta3_ascent.set_initial_guess(
            model, num_intervals=num_intervals, guess=guess, canonical=canonical
        )

    elif name == "goddard":
        model = goddard.create_goddard_model(num_intervals=num_intervals)
        goddard.set_initial_guess(model, num_intervals=num_intervals, guess=guess)

    elif name == "hypersensitive":
        model = hypersensitive.create_hypersensitive_model(num_intervals=num_intervals)
        hypersensitive.set_initial_guess(
            model, num_intervals=num_intervals, guess=guess
        )

    elif name == "landing_6dof":
        model = landing_6dof.create_landing_6dof_model(num_intervals=num_intervals)
        landing_6dof.set_initial_guess(model, num_intervals=num_intervals, guess=guess)

    elif name == "moon_lander":
        model = moon_lander.create_moon_lander_model(
            num_intervals=num_intervals, infeasible=infeasible
        )
        moon_lander.set_initial_guess(model, num_intervals=num_intervals, guess=guess)

    elif name == "powered_descent_3dof":
        model = powered_descent_3dof.create_powered_descent_3dof_model(
            num_intervals=num_intervals, infeasible=infeasible
        )
        powered_descent_3dof.set_initial_guess(
            model, num_intervals=num_intervals, guess=guess
        )

    elif name == "reorientation":
        model = reorientation.create_reorientation_model(num_intervals=num_intervals)
        reorientation.set_initial_guess(model, num_intervals=num_intervals, guess=guess)

    elif name == "robot_arm":
        model = robot_arm.create_robot_arm_model(num_intervals=num_intervals)
        robot_arm.set_initial_guess(model, num_intervals=num_intervals, guess=guess)

    return model


def run_objective(name, variant, x):
    """Objective of a solution in the convention of the reference"""
    canonical = variant == "canonical"

    if name == "alp_rider":
        objective = alp_rider.get_objective(x)
    elif name == "bryson_denham":
        objective = bryson_denham.get_objective(x)
    elif name == "delta3_ascent":
        objective = delta3_ascent.get_objective(x, canonical=canonical)
    elif name == "goddard":
        objective = goddard.get_objective(x)
    elif name == "hypersensitive":
        objective = hypersensitive.get_objective(x)
    elif name == "landing_6dof":
        objective = landing_6dof.get_objective(x)
    elif name == "moon_lander":
        objective = moon_lander.get_objective(x)
    elif name == "powered_descent_3dof":
        objective = powered_descent_3dof.get_objective(x)
    elif name == "reorientation":
        objective = reorientation.get_objective(x)
    elif name == "robot_arm":
        objective = robot_arm.get_objective(x)

    return float(objective)


def solve_run(name, variant, num_intervals, guess, build):
    """Solve one run in this process and return its result"""
    model = create_run(name, variant, num_intervals, guess)

    # The canonical units build in a tree of their own
    if build and variant == "canonical":
        model.build_module(build_dir=Path("_amigo_build_canonical"))
    elif build:
        model.build_module()

    model.initialize()

    # Get the design variables
    x = model.create_vector()

    # Set up the optimizer and time the optimization alone
    opt = am.Optimizer(model, x=x)

    t0 = time.perf_counter()
    opt_data = opt.optimize(OPTIONS)
    elapsed = time.perf_counter() - t0

    # Copy the solution from the device to the host
    x.copy_device_to_host()

    return {
        "status": opt_data.get("status_message", ""),
        "acceptable": bool(opt_data.get("acceptable", False)),
        "objective": run_objective(name, variant, x),
        "iterations": len(opt_data["iterations"]) - 1,
        "time": elapsed,
    }


def run_child():
    """Solve the run read from the standard input and print its result for the runner"""
    run = json.loads(sys.stdin.read())

    name = run["name"]
    variant = run["variant"]
    num_intervals = run["nodes"]
    guess = run["guess"]
    build = run["build"]

    # Work in the folder of the problem where its model compiles and loads
    folder = os.path.join(HERE, GROUPS[name], name)
    os.chdir(folder)
    sys.path.insert(0, folder)

    # Count the factorizations and the restorations of the solve
    amigo_solver.AmigoSolver.factor = counted_factor
    feasibility_restoration.FeasibilityRestoration.restore = counted_restore

    # The counters reach the runner even when the solve fails
    try:
        result = solve_run(name, variant, num_intervals, guess, build)
        print("[result] " + json.dumps(result), flush=True)
    finally:
        print("[counts] " + json.dumps(COUNTS), flush=True)

    return


def run_case(name, variant, nodes, guess, build):
    """Solve one problem, mesh, and start in a fresh process"""
    run = {
        "name": name,
        "variant": variant,
        "nodes": nodes,
        "guess": guess,
        "build": build,
    }

    try:
        proc = subprocess.run(
            [sys.executable, "-c", CHILD],
            input=json.dumps(run),
            cwd=HERE,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return {"status": f"timeout after {TIMEOUT} s"}

    # The result and the counters of the run
    result = {}
    for line in proc.stdout.splitlines():
        if line.startswith("[result] "):
            result.update(json.loads(line[len("[result] ") :]))
        elif line.startswith("[counts] "):
            result.update(json.loads(line[len("[counts] ") :]))

    # A run without a result crashed with the last line of its output
    if "status" not in result:
        lines = (proc.stderr or proc.stdout).strip().splitlines() or ["no output"]
        result["status"] = "crash " + lines[-1][:80]

    return result


def classify(result, reference, infeasible):
    """Outcome of one run against the reference of its problem"""
    status = result.get("status", "")

    # An infeasible variant is solved when the solver reports local infeasibility
    if infeasible:
        if status.startswith("Converged to a locally infeasible"):
            return "solved"
        return "FAILED"

    if not status.startswith("Converged to the requested"):
        if result.get("acceptable"):
            return "acceptable"
        return "FAILED"

    if reference is None:
        return "converged"

    # The same optimum lies within the relative tolerance of the reference
    gap = abs(result["objective"] - reference) / max(1.0, abs(reference))
    if gap <= TOLERANCE:
        return "solved"

    return "other optimum"


def print_row(label, nodes, guess, result):
    """Print the table row of one run"""
    objective = "-"
    if "objective" in result:
        objective = f"{result['objective']:.6g}"

    outcome = result["outcome"]
    iterations = result.get("iterations", 0)
    factorizations = result.get("factorizations", 0)
    resto_calls = result.get("resto_calls", 0)
    elapsed = result.get("time", 0.0)
    status = result["status"][:48]

    print(
        f"{label:30s} {nodes:5d} {guess:11s} {outcome:14s} {objective:>13s}"
        f" {iterations:6d} {factorizations:6d} {resto_calls:5d} {elapsed:7.2f}  {status}",
        flush=True,
    )

    return


def print_counts(label, results):
    """Print the runs, the solved runs, the other optima, and the failures of some runs"""
    good = 0
    other = 0

    for result in results:
        if result["outcome"] in ("solved", "converged"):
            good += 1
        elif result["outcome"] in ("other optimum", "acceptable"):
            other += 1

    runs = len(results)
    failed = runs - good - other

    print(f"{label:30s} {runs:5d} {good:7d} {other:6d} {failed:7d}")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build",
        dest="build",
        action="store_true",
        default=False,
        help="Enable building",
    )
    args = parser.parse_args()

    results = []
    built = set()

    print(
        f"{'problem':30s} {'nodes':>5s} {'guess':11s} {'outcome':14s} {'objective':>13s}"
        f" {'iters':>6s} {'fact':>6s} {'resto':>5s} {'time':>7s}  status"
    )

    for problem in PROBLEMS:
        name = problem[0]
        variant = problem[1]
        meshes = problem[2]
        guesses = problem[3]
        reference = problem[4]

        # A variant runs under the name of its problem followed by the variant
        label = name
        if variant is not None:
            label = name + " " + variant

        for nodes in meshes:
            for guess in guesses:
                # Compile every problem and variant before its first run
                build = args.build and label not in built
                built.add(label)

                result = run_case(name, variant, nodes, guess, build)
                result["label"] = label
                result["outcome"] = classify(result, reference, variant == "infeasible")
                results.append(result)

                print_row(label, nodes, guess, result)

    print()
    print(f"{'problem':30s} {'runs':>5s} {'solved':>7s} {'other':>6s} {'failed':>7s}")

    # The labels in the order of their first run
    labels = []
    for result in results:
        if result["label"] not in labels:
            labels.append(result["label"])

    for label in labels:
        runs = []
        for result in results:
            if result["label"] == label:
                runs.append(result)

        print_counts(label, runs)

    print_counts("total", results)
