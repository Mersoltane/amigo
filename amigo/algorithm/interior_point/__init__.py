"""
Interior point loop
-------------------

- The optimizer and its main loop
- The state of an iteration and the evaluations that fill it
- The Newton step, the convergence tests, and the step contract
"""

from .driver import Optimizer, WarmStart
