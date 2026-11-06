"""
Hock-Schittkowski problems
--------------------------

- Every Hock-Schittkowski example converges to its published optimum
- A warm start from the solution and its bound duals takes fewer iterations
- Objective scaling keeps the optimum and reports it unscaled
"""

import glob
import os
import runpy
import sys

import numpy as np
import pytest

import amigo as am
from amigo.algorithm.interior_point import driver
from amigo.algorithm.interior_point.convergence_check import (
    CONVERGED,
    CONVERGED_ACCEPTABLE,
)

ROOT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "examples",
    "test_suites",
    "hock_schittkowski",
)
SCRIPTS = sorted(glob.glob(os.path.join(ROOT, "hs*", "hs*.py")))

# Published optimal objectives, written with the digits of their source
PUBLISHED_OPTIMA = {
    "hs6": "0",
    "hs14": "1.3934651",
    "hs26": "0",
    "hs29": "-22.627417",
    "hs35": "1/9",
    "hs39": "-1",
    "hs40": "-0.25",
    "hs44": "-15",
    "hs53": "176/43",
    "hs60": "0.0325682",
    "hs65": "0.9535288567",
    "hs71": "17.0140173",
    "hs76": "-4.681818",
    "hs77": "0.2415051",
    "hs79": "0.0787768209",
    "hs87": "8827.5977",
    "hs93": "135.076",
    "hs99": "-831079892",
    "hs100": "680.6300573",
    "hs104": "3.9511634396",
    "hs106": "7049.2480",
    "hs108": "-0.8660254038",
    "hs109": "5362.0693",
    "hs111": "-47.7610",
    "hs113": "24.3062091",
    "hs114": "-1768.80696",
    "hs116": "97.5875",
    "hs117": "32.34867897",
    "hs118": "664.82045",
    "hs119": "244.8997",
}

# Examples whose converged objective disagrees with their published optimum
DISAGREE = {
    "hs108": "a local optimum -0.675 above the stated -0.866",
}


def published_optimum(script):
    """Published optimal objective of a problem and the absolute precision of its digits"""
    name = os.path.basename(script)[:-3]
    text = PUBLISHED_OPTIMA.get(name)

    if text is None:
        return None, None

    # A fraction is exact
    if "/" in text:
        numerator = text.split("/")[0]
        denominator = text.split("/")[1]

        return float(numerator) / float(denominator), 0.0

    # Some sources truncate rather than round, so one unit of the last digit is allowed
    decimals = 0
    if "." in text and "e" not in text.lower():
        decimals = len(text.split(".")[1])

    return float(text), 10.0**-decimals


def params():
    """One case per script, marked when its optimum disagrees with the published one"""
    cases = []

    for script in SCRIPTS:
        name = os.path.basename(script)[:-3]

        marks = []
        if name in DISAGREE:
            marks.append(
                pytest.mark.xfail(reason=f"Converges to {DISAGREE[name]}", strict=True)
            )

        cases.append(pytest.param(script, id=name, marks=marks))

    return cases


def run_script(script, monkeypatch):
    """Run an example script in its folder, building its module only when it is missing"""
    directory = os.path.dirname(script)
    name = os.path.basename(script)[:-3]

    pyd = glob.glob(os.path.join(directory, f"{name}.*.pyd"))
    so = glob.glob(os.path.join(directory, f"{name}.*.so"))

    argv = [script, "--build"]
    if pyd + so:
        argv = [script]

    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.syspath_prepend(directory)
    monkeypatch.chdir(directory)

    return runpy.run_path(script, run_name="__main__")


class SolveLog:
    """The data of every solve run through the wrapped optimize, with an optional objective scaling"""

    def __init__(self):
        self.solves = []
        self.factor = None


# Optimize before the tests wrap it
OPTIMIZE = driver.Optimizer.optimize

LOG = SolveLog()


def logged_optimize(self, options={}):
    """Optimize with the scaling of the log, recording the data of every solve"""
    if LOG.factor is not None:
        options = dict(options)
        options["objective_scaling_factor"] = LOG.factor

    data = OPTIMIZE(self, options)
    LOG.solves.append(data)

    return data


@pytest.fixture
def log(monkeypatch):
    """A fresh log with optimize wrapped for one test"""
    LOG.solves = []
    LOG.factor = None

    monkeypatch.setattr(driver.Optimizer, "optimize", logged_optimize)

    return LOG


@pytest.mark.parametrize("script", params())
def test_problem_reaches_published_optimum(script, monkeypatch, log):
    optimum = published_optimum(script)
    expected = optimum[0]
    precision = optimum[1]

    if expected is None:
        pytest.skip("the example states no optimum")

    run_script(script, monkeypatch)

    # The last solve of the script is the one checked
    data = log.solves[-1]

    assert data["status"] in (CONVERGED, CONVERGED_ACCEPTABLE), data["status_message"]

    objective = data["iterations"][-1]["objective"]

    assert objective == pytest.approx(expected, rel=1e-6, abs=precision + 1e-6)


def test_a_warm_start_from_the_solution_and_its_bound_duals_needs_fewer_iterations(
    monkeypatch,
):
    script = os.path.join(ROOT, "hs71", "hs71.py")

    optimum = published_optimum(script)
    expected = optimum[0]
    precision = optimum[1]

    scope = run_script(script, monkeypatch)
    model = scope["model"]
    solved = scope["opt"]

    # Only the lower bound of x1 is active at the optimum
    duals = solved.get_bound_duals()
    zl = duals[0]
    zu = duals[1]

    assert zl["hs71.x1"][0] > 1e-2
    assert zl["hs71.x2"][0] < 1e-6 and zu["hs71.x2"][0] < 1e-6
    assert np.all(np.asarray(zl["hs71.c2"]) == 0.0)

    # A cold restart from the solution recenters at the initial barrier and takes as long as the first solve
    options = {"convergence_tolerance": 1e-8, "max_iterations": 100}

    cold = am.Optimizer(model, solved.get_optimized_point()).optimize(options)

    warm_start = solved.get_warm_start()
    warm = am.Optimizer(model, warm_start=warm_start).optimize(options)

    assert warm["status"] in (CONVERGED, CONVERGED_ACCEPTABLE), warm["status_message"]
    assert len(warm["iterations"]) < len(cold["iterations"])
    assert warm["iterations"][-1]["objective"] == pytest.approx(
        expected, rel=1e-6, abs=precision + 1e-6
    )

    # The solution restarted at the barrier it ended at is already converged
    assert warm_start.mu < 1e-6
    assert len(warm["iterations"]) <= 2

    # The same duals and barrier set by hand give the same warm solve
    restart = am.Optimizer(model, solved.get_optimized_point())
    restart.set_bound_duals(zl, zu)

    by_hand_options = dict(options)
    by_hand_options["warm_start"] = True
    by_hand_options["warm_start_mu_init"] = warm_start.mu

    by_hand = restart.optimize(by_hand_options)

    assert len(by_hand["iterations"]) == len(warm["iterations"])


@pytest.mark.parametrize("factor", [1e-2, 1e2])
@pytest.mark.parametrize("name", ["hs71", "hs76"])
def test_objective_scaling_keeps_the_optimum_and_reports_it_unscaled(
    name, factor, monkeypatch, log
):
    script = os.path.join(ROOT, name, f"{name}.py")

    optimum = published_optimum(script)
    expected = optimum[0]
    precision = optimum[1]

    # Every solve of the script runs with the objective scaled by the factor
    log.factor = factor

    run_script(script, monkeypatch)

    data = log.solves[-1]

    assert data["status"] in (CONVERGED, CONVERGED_ACCEPTABLE), data["status_message"]

    objective = data["iterations"][-1]["objective"]

    assert objective == pytest.approx(expected, rel=1e-6, abs=precision + 1e-6)
