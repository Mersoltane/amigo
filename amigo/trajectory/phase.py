"""Declaration of one phase of trajectory problem."""

import math
from dataclasses import dataclass

INF = math.inf


@dataclass(frozen=True)
class State:
    """Bounds of one state over the phase and at its two ends."""
    name: str
    lower: float = -INF
    upper: float = INF
    initial: tuple = (-INF, INF)
    final: tuple = (-INF, INF)


@dataclass(frozen=True)
class Control:
    """Bounds of one control over the phase."""
    name: str
    lower: float = -INF
    upper: float = INF


@dataclass(frozen=True)
class Integral:
    """Bounds on one quantity integrated over the phase."""
    name: str
    lower: float = -INF
    upper: float = INF


@dataclass(frozen=True)
class Constraint:
    """Bounds of one path or boundary constraint."""
    name: str
    lower: float = -INF  # Equal bounds make an equality
    upper: float = INF
    endpoints: bool = True  # Used only by path constraints


@dataclass(frozen=True)
class Time:
    """Bounds of the start or end time of the phase."""
    lower: float
    upper: float 



class Phase:
    """One stretch of trajectory with its declarations and hooks."""

    def __init__(self):
        self.names = set()
        self.states = {}
        self.controls = {}
        self.integrals = {}
        self.path_constraints = {}  #enforced at every mesh point
        self.boundary_constraints = {} #once per phase

        self.initial_time = None
        self.final_time = None



    def _store(self, group, record):
        """Check the record's name and bounds, reserve the name, and store the record."""
        name = record.name

        if not name.isidentifier():
            raise ValueError(f"{name} is not a valid name")
        
        if name in self.names:
            raise ValueError(f"{name} is already used")

        if record.lower > record.upper:
            raise ValueError(f"{name}: lower bound {record.lower} is above upper bound {record.upper}")

        
        self.names.add(name)
        group[name] = record


    def _to_pair(self,value):
        """Turn None, a number, or a pair into a lower and upper bound."""
        if value is None:
            return (-INF,INF)
        
        if isinstance(value, (int, float)):
            return (value, value)
        
        lo, hi = value
        if lo > hi:
            raise ValueError(f"lower bound {lo} is above upper bound {hi}")
        return (lo, hi)
        


    def add_state(self, name, lower = -INF, upper = INF, initial =None, final = None):
        """Declare one state of the phase."""
        self._store(self.states, State(name,lower,upper, self._to_pair(initial), self._to_pair(final)))

    def add_control(self, name, lower = -INF, upper = INF):
        """Declare one control of the phase."""
        self._store(self.controls, Control(name, lower, upper))


    def add_integral(self, name, lower = -INF, upper = INF):
        """Declare one quanity integrated over the phase."""
        self._store(self.integrals, Integral(name, lower, upper))


    def add_path_constraint(self, name, lower = -INF, upper = INF, endpoints=True):
        """Declare one constraint enforced at every point of the phase."""
        self._store(self.path_constraints, Constraint(name, lower, upper, endpoints))

    def add_boundary_constraint(self,name, lower=-INF, upper =INF):
        """Declare one constraint on the start and end values of the phase."""
        self._store(self.boundary_constraints,Constraint(name, lower, upper) )

    def set_initial_time(self, value):
        """Fix or bound the start time of the phase."""
        self.initial_time = Time(*self._to_pair(value))

    def set_final_time(self, value):
        """Fix or bound the end time of the phase."""
        self.final_time = Time(*self._to_pair(value))


# TODO: 
    def dynamics(self, x, u, p, t):
        """Return the rate of every state."""
        raise NotImplementedError("the phase must define dynamics")

    def integrands(self, x, u, p, t):
        """Return the integrand of every integral."""
        return {}

    def path(self, x, u, p, t):
        """Return the value of every path constraint."""
        return {}

    def boundary(self, e, p):
        """Return the value of every boundary constraint."""
        return {}

    

    