"""
Bang-bang solve
---------------

- Switches and arcs read from a switching function
- The options, the records of a step, and what a problem supplies
- The solve at each value of a parameter, the cheapest way first
"""

from .switches import split_at_switches, switch_estimates, switch_signs
from .arcs import (
    combine_arcs,
    insert_arcs,
    merge_short_arcs,
    sign_agreement,
    split_arcs,
    stretch_arcs,
    sub_mesh,
    switch_brackets,
    switching_arcs,
    uniform_arc_meshes,
)
from .options import BangBangOptions
from .step import Arcs, Stage, Step
from .problem import BangBangProblem
from .solver import BangBangSolver
