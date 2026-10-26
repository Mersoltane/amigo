"""Functional tests that every Hock-Schittkowski example converges to its published optimum."""

import glob
import os
import re
import runpy
import sys

import numpy as np
import pytest

import amigo as am
from amigo.algorithm import ipm_driver
from amigo.algorithm.convergence_check import CONVERGED, CONVERGED_ACCEPTABLE

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples", "test_problems")
SCRIPTS = sorted(glob.glob(os.path.join(ROOT, "hs*", "hs*.py")))

# Examples whose converged objective disagrees with the optimum stated in their docstring
DISAGREE = {
    "hs108": "a local optimum -0.675 above the stated -0.866",
}


def published_optimum(script):
    """Optimal objective f* stated in the docstring and the absolute precision of its digits."""
    with open(script) as f:
        match = re.search(r"f\*\s*=\s*(-?[0-9.]+(?:[eE][-+]?[0-9]+)?)(?:\s*/\s*([0-9.]+))?", f.read())
    if match is None:
        return None, None
    text, divisor = match.groups()
    if divisor:
        return float(text) / float(divisor), 0.0
    # Some docstrings truncate rather than round, so one unit of the last digit is allowed
    decimals = len(text.split(".")[1]) if "." in text and "e" not in text.lower() else 0
    return float(text), 10.0**-decimals


def params():
    for script in SCRIPTS:
        name = os.path.basename(script)[:-3]
        marks = []
        if name in DISAGREE:
            marks.append(pytest.mark.xfail(reason=f"Converges to {DISAGREE[name]}", strict=True))
        yield pytest.param(script, id=name, marks=marks)


@pytest.mark.parametrize("script", list(params()))
def test_problem_reaches_published_optimum(script, monkeypatch):
    expected, precision = published_optimum(script)
    if expected is None:
        pytest.skip("the example states no optimum")

    # The last solve of the script is the one checked
    solves = []
    optimize = ipm_driver.Optimizer.optimize

    def recorded(self, *args, **kwargs):
        data = optimize(self, *args, **kwargs)
        solves.append(data)
        return data

    monkeypatch.setattr(ipm_driver.Optimizer, "optimize", recorded)

    # Build only when the compiled module is missing next to the script
    directory = os.path.dirname(script)
    name = os.path.basename(script)[:-3]
    built = glob.glob(os.path.join(directory, f"{name}.*.pyd")) + glob.glob(
        os.path.join(directory, f"{name}.*.so")
    )
    monkeypatch.setattr(sys, "argv", [script] if built else [script, "--build"])
    monkeypatch.syspath_prepend(directory)
    monkeypatch.chdir(directory)
    runpy.run_path(script, run_name="__main__")

    data = solves[-1]
    assert data["status"] in (CONVERGED, CONVERGED_ACCEPTABLE), data["status_message"]
    objective = data["iterations"][-1]["objective"]
    assert objective == pytest.approx(expected, rel=1e-6, abs=precision + 1e-6)


def test_a_warm_start_from_the_solution_and_its_bound_duals_needs_fewer_iterations(monkeypatch):
    script = os.path.join(ROOT, "hs71", "hs71.py")
    expected, precision = published_optimum(script)
    directory = os.path.dirname(script)
    built = glob.glob(os.path.join(directory, "hs71.*.pyd")) + glob.glob(os.path.join(directory, "hs71.*.so"))
    monkeypatch.setattr(sys, "argv", [script] if built else [script, "--build"])
    monkeypatch.syspath_prepend(directory)
    monkeypatch.chdir(directory)
    scope = runpy.run_path(script, run_name="__main__")
    model, solved = scope["model"], scope["opt"]

    # Only the lower bound of x1 is active at the optimum
    zl, zu = solved.get_bound_duals()
    assert zl["hs71.x1"][0] > 1e-2
    assert zl["hs71.x2"][0] < 1e-6 and zu["hs71.x2"][0] < 1e-6
    assert np.all(np.asarray(zl["hs71.c2"]) == 0.0)

    # A cold restart from the solution re-centers at the initial barrier and takes as long as the first solve
    options = {"convergence_tolerance": 1e-8, "max_iterations": 100}
    cold = am.Optimizer(model, solved.get_optimized_point()).optimize(options)
    warm = am.Optimizer(model, warm_start=solved.get_warm_start()).optimize(options)
    assert warm["status"] in (CONVERGED, CONVERGED_ACCEPTABLE), warm["status_message"]
    assert len(warm["iterations"]) < len(cold["iterations"])
    assert warm["iterations"][-1]["objective"] == pytest.approx(expected, rel=1e-6, abs=precision + 1e-6)

    # The same duals set by hand give the same warm solve
    restart = am.Optimizer(model, solved.get_optimized_point())
    restart.set_bound_duals(zl, zu)
    by_hand = restart.optimize({**options, "warm_start": True})
    assert len(by_hand["iterations"]) == len(warm["iterations"])


@pytest.mark.parametrize("factor", [1e-2, 1e2])
@pytest.mark.parametrize("name", ["hs71", "hs76"])
def test_objective_scaling_keeps_the_optimum_and_reports_it_unscaled(name, factor, monkeypatch):
    script = os.path.join(ROOT, name, f"{name}.py")
    expected, precision = published_optimum(script)

    # Every solve of the script runs with the objective scaled by the factor
    solves = []
    optimize = ipm_driver.Optimizer.optimize

    def scaled(self, options={}):
        data = optimize(self, {**options, "objective_scaling_factor": factor})
        solves.append(data)
        return data

    monkeypatch.setattr(ipm_driver.Optimizer, "optimize", scaled)
    directory = os.path.dirname(script)
    built = glob.glob(os.path.join(directory, f"{name}.*.pyd")) + glob.glob(
        os.path.join(directory, f"{name}.*.so")
    )
    monkeypatch.setattr(sys, "argv", [script] if built else [script, "--build"])
    monkeypatch.syspath_prepend(directory)
    monkeypatch.chdir(directory)
    runpy.run_path(script, run_name="__main__")

    data = solves[-1]
    assert data["status"] in (CONVERGED, CONVERGED_ACCEPTABLE), data["status_message"]
    objective = data["iterations"][-1]["objective"]
    assert objective == pytest.approx(expected, rel=1e-6, abs=precision + 1e-6)
