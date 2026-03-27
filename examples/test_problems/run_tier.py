"""
Run HS benchmark problems by tier.

Usage:
  python run_tier.py                  # Run Tier 1 (stress-test)
  python run_tier.py --tier 2         # Run Tier 2 (larger / structural)
  python run_tier.py --tier all       # Run all tiers
  python run_tier.py --tier basic     # Run basic problems
  python run_tier.py --problem hs116  # Run single problem
  python run_tier.py --build          # Build + run Tier 1
  python run_tier.py --build --tier all  # Build + run everything
"""

import subprocess
import sys
import os
import argparse
import time

PROBLEMS = {
    "basic": {
        "description": "Basic validation (small, well-conditioned)",
        "problems": {
            "hs6": "1 var, 1 eq : simplest constrained",
            "hs14": "2 vars, 2 mixed : small QP",
            "hs26": "3 vars, 1 eq : Powell singular",
            "hs35": "3 vars, 1 ineq : bound-constrained QP",
            "hs39": "4 vars, 2 eq : feasibility test",
            "hs40": "4 vars, 3 eq : equality-heavy",
            "hs44": "4 vars, 6 ineq : many inequalities",
            "hs53": "5 vars, 3 eq : linear constraints",
            "hs60": "3 vars, 1 eq : nonlinear equality",
            "hs65": "3 vars, 1 ineq : simple nonconvex",
            "hs71": "4 vars, 1 ineq + 1 eq : classic IPOPT test",
            "hs76": "4 vars, 3 ineq : linear objective",
            "hs77": "5 vars, 2 eq : Hessian accuracy test",
            "hs79": "5 vars, 3 eq : nonlinear equalities",
        },
    },
    "tier1": {
        "description": "Stress-test (inertia, degeneracy, ill-conditioning)",
        "problems": {
            "hs93": "6 vars, 2 ineq : badly scaled",
            "hs100": "7 vars, 4 ineq : medium nonconvex",
            "hs104": "8 vars, 5 ineq : geometric programming",
            "hs106": "8 vars, 6 ineq : near-degenerate at solution",
            "hs108": "9 vars, 13 ineq : many constraints, rank-deficient J",
            "hs111": "10 vars, 3 ineq : log-barrier objective",
            "hs113": "10 vars, 8 ineq : medium inequality-heavy",
            "hs114": "10 vars, 8 ineq + 3 eq : division, mixed types",
            "hs116": "13 vars, 15 ineq : most constraints, degenerate J",
        },
    },
    "tier2": {
        "description": "Larger scale / structural challenges",
        "problems": {
            "hs87": "6+6 vars, 7 eq : piecewise cost, multiple local minima",
            "hs99": "7 vars, 2 eq : ODE-like trajectory, cumulative states",
            "hs109": "9 vars, 4 ineq + 6 eq : trig, infeasible start",
            "hs117": "15 vars, 5 ineq : nonconvex QP, matrix structure",
            "hs118": "15 vars, 29 ineq : large inequality system",
            "hs119": "16 vars, 8 eq : largest HS, dense quartic Hessian",
        },
    },
}


def build_problem(name, script_dir):
    """Build a single HS problem. Returns (name, success, msg)."""
    script = os.path.join(script_dir, name, f"{name}.py")
    prob_dir = os.path.join(script_dir, name)
    if not os.path.exists(script):
        return name, False, "file not found"

    # Check if already built
    import glob
    import shutil

    if glob.glob(os.path.join(prob_dir, "*.pyd")) or glob.glob(
        os.path.join(prob_dir, "*.so")
    ):
        return name, True, "already built"

    # Clean stale build directory to avoid CMake cache issues
    build_dir = os.path.join(prob_dir, "_amigo_build")
    if os.path.isdir(build_dir):
        shutil.rmtree(build_dir, ignore_errors=True)

    try:
        result = subprocess.run(
            [sys.executable, script, "--build"],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=prob_dir,
        )
        # Check if .pyd/.so was created (build success even if solve fails)
        if glob.glob(os.path.join(prob_dir, "*.pyd")) or glob.glob(
            os.path.join(prob_dir, "*.so")
        ):
            return name, True, "built"
        else:
            # Check both stdout and stderr for error clues
            combined = (result.stdout + "\n" + result.stderr).strip()
            lines = [l for l in combined.split("\n") if l.strip()]
            err = lines[-1] if lines else "unknown"
            return name, False, err[:80]
    except subprocess.TimeoutExpired:
        return name, False, "build timeout (300s)"
    except Exception as ex:
        return name, False, str(ex)[:80]


def run_problem(name, script_dir):
    """Run a single HS problem, return (name, success, time, msg)."""
    script = os.path.join(script_dir, name, f"{name}.py")
    if not os.path.exists(script):
        return name, False, 0, "file not found"

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=os.path.join(script_dir, name),
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            # Check for convergence in output
            if "converged" in result.stdout.lower():
                return name, True, elapsed, "converged"
            else:
                return name, False, elapsed, "did not converge"
        else:
            # Extract last line of stderr for error summary
            err = result.stderr.strip().split("\n")[-1] if result.stderr else "unknown"
            return name, False, elapsed, err[:80]
    except subprocess.TimeoutExpired:
        return name, False, 120, "timeout (120s)"
    except Exception as ex:
        return name, False, 0, str(ex)[:80]


def main():
    parser = argparse.ArgumentParser(description="Run HS benchmark problems by tier")
    parser.add_argument(
        "--tier",
        default="tier1",
        choices=["basic", "tier1", "tier2", "all"],
        help="Which tier to run (default: tier1)",
    )
    parser.add_argument(
        "--problem", default=None, help="Run a single problem (e.g., hs116)"
    )
    parser.add_argument(
        "--build", action="store_true", help="Build problems before running"
    )
    parser.add_argument(
        "--list", action="store_true", help="List all problems without running"
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.list:
        for tier, info in PROBLEMS.items():
            print(f"\n{'='*60}")
            print(f"  {tier}: {info['description']}")
            print(f"{'='*60}")
            for name, desc in info["problems"].items():
                print(f"  {name:8s} : {desc}")
        return

    if args.problem:
        if args.build:
            _, ok, msg = build_problem(args.problem, script_dir)
            print(f"  {args.problem:8s}  BUILD  {'OK' if ok else 'FAIL'}  {msg}")
            if not ok:
                return
        name, ok, t, msg = run_problem(args.problem, script_dir)
        status = "PASS" if ok else "FAIL"
        print(f"  {name:8s}  {status}  {t:6.1f}s  {msg}")
        return

    tiers = list(PROBLEMS.keys()) if args.tier == "all" else [args.tier]

    # Build phase
    if args.build:
        for tier in tiers:
            info = PROBLEMS[tier]
            print(f"\n{'='*60}")
            print(f"  Building {tier}: {info['description']}")
            print(f"{'='*60}")
            for name in info["problems"]:
                _, ok, msg = build_problem(name, script_dir)
                status = "OK" if ok else "FAIL"
                print(f"  {name:8s}  {status:4s}  {msg}")
        print()

    # Run phase
    for tier in tiers:
        info = PROBLEMS[tier]
        print(f"\n{'='*60}")
        print(f"  {tier}: {info['description']}")
        print(f"{'='*60}")

        passed = 0
        total = 0
        for name, desc in info["problems"].items():
            total += 1
            _, ok, t, msg = run_problem(name, script_dir)
            status = "PASS" if ok else "FAIL"
            if ok:
                passed += 1
            print(f"  {name:8s}  {status}  {t:6.1f}s  {msg}")

        print(f"\n  Results: {passed}/{total} passed")


if __name__ == "__main__":
    main()
