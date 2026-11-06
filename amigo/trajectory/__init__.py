"""
Trajectory optimization
-----------------------

  collocation/        Radau collocation
    mesh.py           Radau points, weights, matrices, and the hp mesh
    costate.py        Costates from the multipliers of the collocated dynamics
    phase.py          Declarations of one phase of a trajectory problem

  refinement/         Mesh refinement
    base.py           The refinement loop and the records it keeps
    hp.py             hp refinement of Liu, Hager, and Rao
    phs.py            phs refinement of Haman and Rao

  bang_bang/          Bang-bang solve
    switches.py       Switch signs and estimates from a switching function
    arcs.py           Solutions held as consecutive arcs that meet at the switches
    options.py        Options of the bang-bang solve
    step.py           What the solve at one value of the parameter records
    problem.py        What a bang-bang problem supplies to the solver
    solver.py         The solve at each value of the parameter, the cheapest way first
"""

from .collocation import (
    RadauMesh,
    carry_densities,
    defect_costate,
    defect_weights,
    differentiation_matrix,
    integration_matrix,
    lagrange_matrix,
    mesh_time,
    quadrature_costate,
    radau_points,
    radau_weights,
    support_points,
)
from .refinement import (
    Assessment,
    FlightError,
    HPRefinement,
    MeshIteration,
    MeshRefinement,
    MeshSolution,
    PHSOptions,
    PHSRefinement,
    RefinementHistory,
    RefinementOptions,
    best_iteration,
    path_violations,
    relative_errors,
    simulated_errors,
    touch_points,
)
from .bang_bang import (
    Arcs,
    BangBangOptions,
    BangBangProblem,
    BangBangSolver,
    Stage,
    Step,
    combine_arcs,
    insert_arcs,
    merge_short_arcs,
    sign_agreement,
    split_arcs,
    split_at_switches,
    stretch_arcs,
    sub_mesh,
    switch_brackets,
    switch_estimates,
    switch_signs,
    switching_arcs,
    uniform_arc_meshes,
)

__all__ = [
    "RadauMesh",
    "carry_densities",
    "differentiation_matrix",
    "integration_matrix",
    "lagrange_matrix",
    "mesh_time",
    "radau_points",
    "radau_weights",
    "support_points",
    "Assessment",
    "MeshIteration",
    "MeshRefinement",
    "MeshSolution",
    "RefinementHistory",
    "best_iteration",
    "HPRefinement",
    "RefinementOptions",
    "relative_errors",
    "FlightError",
    "PHSOptions",
    "PHSRefinement",
    "path_violations",
    "simulated_errors",
    "touch_points",
    "split_at_switches",
    "switch_estimates",
    "switch_signs",
    "Arcs",
    "BangBangOptions",
    "BangBangProblem",
    "BangBangSolver",
    "Stage",
    "Step",
    "combine_arcs",
    "defect_costate",
    "defect_weights",
    "insert_arcs",
    "quadrature_costate",
    "merge_short_arcs",
    "sign_agreement",
    "split_arcs",
    "stretch_arcs",
    "sub_mesh",
    "switch_brackets",
    "switching_arcs",
    "uniform_arc_meshes",
]
