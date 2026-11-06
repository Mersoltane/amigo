"""
Bang-bang steps
---------------

What the solve at one value of the parameter records
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Stage:
    """One stage of a step with its outcome, the iterations it took, and the objective it reached"""

    name: str
    outcome: str
    iterations: int
    objective: float = np.nan


@dataclass
class Arcs:
    """Solved arcs with their combined solution, times, check, largest mesh error, and what a near value carries"""

    solution: object
    arcs: list
    check: dict
    error: float
    carry: dict


@dataclass
class Step:
    """The solution at one value of the parameter, its arcs, the single phase they came from, and its stages"""

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
        """Whether the step found a converged solution within the mesh tolerance"""
        if self.solution is None:
            return False

        return bool(self.solution.converged) and self.met

    @property
    def iterations(self):
        """Iterations of every stage of the step"""
        total = 0

        for stage in self.stages:
            total += stage.iterations

        return total
