"""
Interfaces to other frameworks
------------------------------

- OpenMDAO components that wrap an amigo model
"""

from .openmdao import (
    ExternalOpenMDAOComponent,
    ExplicitOpenMDAOPostOptComponent,
    AmigoIndepVarComp,
)
