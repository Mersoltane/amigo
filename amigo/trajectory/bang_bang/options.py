"""
Bang-bang options
-----------------
"""

from dataclasses import dataclass


@dataclass
class BangBangOptions:
    """Options of the bang-bang solve of a step and of a family"""

    # Mesh tolerance of the arcs and of the single phase
    tolerance: float = 1e-8

    # Smoothing values of the homotopy from energy toward the objective, and none to skip it
    homotopy: tuple = (1.0, 0.3, 0.1, 0.03, 0.01)

    # Mesh tolerance of the smoothed single phase whose arcs are read
    homotopy_tolerance: float = 1e-4

    # Share of its largest value within which a sign change of a switching function is noise
    switch_noise: float = 1e-4

    # Noise of the first reading of the smoothed single phase
    read_noise: float = 1e-4

    # Whether a step tries the arcs of the last value first
    carry: bool = True

    # Largest change of the parameter, as a share of it, across which the arcs of the last value are carried
    carry_step: float = 0.05

    # Iterations each solve of arcs may take before one rerun from its failed point
    budget: int = 150

    # Most meshes the refined single phase of the last stage tries before the step fails
    last_meshes: int = 3

    # Mesh tolerance the layouts of a solve of arcs settle at before the one that stands refines to the tolerance
    layout_tolerance: float = 1e-4

    # Most arc layouts one solve of arcs tries, with its contact events and its changes
    max_layouts: int = 8

    # Iterations the carry, the jump, and the first homotopy reading may spend testing their arcs
    probe_effort: int = 1000

    # Iterations the searching readings of the homotopy and the refined single phase may spend
    search_effort: int = 4000

    # Arcs shorter than twice this have collapsed
    arc_floor: float = 1e-3

    # Most halvings in a row of one step of a family before it stops
    max_bisections: int = 4

    # Largest change between the values a family visits as a share of the value, and 0 for none
    max_step: float = 0.0
