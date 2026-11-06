"""
Modeling layer
--------------

- Components, expressions, and their metadata
- The model that links components and generates their C++ code
- The checks a model can run before an optimization
"""

from .component import Component
from .expression_component import ExpressionComponent
from .model import Model, ModelVector
from .diagnostics import Diagnostics
