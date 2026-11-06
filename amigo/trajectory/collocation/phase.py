"""
Trajectory phase
----------------

Declarations of one phase of a trajectory problem
"""

import math
from dataclasses import dataclass

INF = math.inf


@dataclass(frozen=True)
class State:
    """Bounds of one state over the phase and at its two ends"""

    name: str
    lower: float = -INF
    upper: float = INF
    initial: tuple = (-INF, INF)
    final: tuple = (-INF, INF)


@dataclass(frozen=True)
class Control:
    """Bounds of one control over the phase"""

    name: str
    lower: float = -INF
    upper: float = INF


@dataclass(frozen=True)
class Integral:
    """Bounds of one quantity integrated over the phase"""

    name: str
    lower: float = -INF
    upper: float = INF


@dataclass(frozen=True)
class Constraint:
    """Bounds of one path or boundary constraint"""

    name: str

    # Equal bounds make an equality
    lower: float = -INF
    upper: float = INF

    # Only path constraints read the endpoints flag
    endpoints: bool = True


@dataclass(frozen=True)
class Time:
    """Bounds of the start or end time of the phase"""

    lower: float
    upper: float


class Phase:
    """One stretch of trajectory with its declarations and hooks"""

    def __init__(self):
        self.names = set()

        self.states = {}
        self.controls = {}
        self.integrals = {}

        # Path constraints hold at every mesh point and boundary constraints once per phase
        self.path_constraints = {}
        self.boundary_constraints = {}

        self.initial_time = None
        self.final_time = None

    def _store(self, group, record):
        """Check the name and the bounds of a record and store it"""
        name = record.name

        if not name.isidentifier():
            raise ValueError(f"{name} is not a valid name")

        if name in self.names:
            raise ValueError(f"{name} is already used")

        if record.lower > record.upper:
            raise ValueError(
                f"{name}: lower bound {record.lower} is above upper bound {record.upper}"
            )

        self.names.add(name)
        group[name] = record

    def _to_pair(self, value):
        """Turn None, a number, or a pair into a lower and an upper bound"""
        if value is None:
            return (-INF, INF)

        if isinstance(value, (int, float)):
            return (value, value)

        if len(value) != 2:
            raise ValueError(f"a bound pair needs two values, not {value}")

        # A pair gives the lower and the upper bound
        lower = value[0]
        upper = value[1]

        if lower > upper:
            raise ValueError(f"lower bound {lower} is above upper bound {upper}")

        return (lower, upper)

    def add_state(self, name, lower=-INF, upper=INF, initial=None, final=None):
        """Declare one state of the phase"""
        initial = self._to_pair(initial)
        final = self._to_pair(final)

        self._store(self.states, State(name, lower, upper, initial, final))

    def add_control(self, name, lower=-INF, upper=INF):
        """Declare one control of the phase"""
        self._store(self.controls, Control(name, lower, upper))

    def add_integral(self, name, lower=-INF, upper=INF):
        """Declare one quantity integrated over the phase"""
        self._store(self.integrals, Integral(name, lower, upper))

    def add_path_constraint(self, name, lower=-INF, upper=INF, endpoints=True):
        """Declare one constraint enforced at every point of the phase"""
        self._store(self.path_constraints, Constraint(name, lower, upper, endpoints))

    def add_boundary_constraint(self, name, lower=-INF, upper=INF):
        """Declare one constraint on the start and end values of the phase"""
        self._store(self.boundary_constraints, Constraint(name, lower, upper))

    def set_initial_time(self, value):
        """Fix or bound the start time of the phase"""
        bounds = self._to_pair(value)

        self.initial_time = Time(bounds[0], bounds[1])

    def set_final_time(self, value):
        """Fix or bound the end time of the phase"""
        bounds = self._to_pair(value)

        self.final_time = Time(bounds[0], bounds[1])

    # Hooks that a problem overrides

    def dynamics(self, x, u, p, t):
        """Rate of every state"""
        raise NotImplementedError("the phase must define dynamics")

    def integrands(self, x, u, p, t):
        """Integrand of every integral"""
        return {}

    def path(self, x, u, p, t):
        """Value of every path constraint"""
        return {}

    def boundary(self, e, p):
        """Value of every boundary constraint"""
        return {}
