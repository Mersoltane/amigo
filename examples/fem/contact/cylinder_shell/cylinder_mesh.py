# Author: Jack Turbush
# Description: Mesh generation script for the cylindrical shell contact example.

import gmsh
import math

# R = 0.5  # radius
# L = 2.0  # length
# lc = 0.05  # mesh size

# gmsh.initialize()
# gmsh.model.add("cylinder")

# gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, L, R)
# gmsh.model.occ.synchronize()

# gmsh.option.setNumber("Mesh.Algorithm", 8)
# gmsh.option.setNumber("Mesh.RecombineAll", 1)
# gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
# gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
# gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)

# gmsh.model.mesh.generate(2)
# gmsh.model.mesh.recombine()

# gmsh.write("cylinder.inp")
# gmsh.write("cylinder.msh")
# gmsh.finalize()
# print("Done")

gmsh.initialize()
gmsh.model.add("cylinder")

# Parameters
R = 1.0
H = 2.0
n_circ = 512  # number of quads around circumference
n_axial = 256  # number of quads along height
# n_circ = 256  # number of quads around circumference
# n_axial = 128  # number of quads along height

# Use the built-in geo kernel
geo = gmsh.model.geo

# Create a rectangular parametric domain:
# s in [0, 2*pi*R], z in [0, H]
p1 = geo.addPoint(0.0, 0.0, 0.0)
p2 = geo.addPoint(2.0 * math.pi * R, 0.0, 0.0)
p3 = geo.addPoint(2.0 * math.pi * R, H, 0.0)
p4 = geo.addPoint(0.0, H, 0.0)

l1 = geo.addLine(p1, p2)
l2 = geo.addLine(p2, p3)
l3 = geo.addLine(p3, p4)
l4 = geo.addLine(p4, p1)

loop = geo.addCurveLoop([l1, l2, l3, l4])
surf = geo.addPlaneSurface([loop])

# Structured mesh constraints
# Number of points = number of elements + 1
geo.mesh.setTransfiniteCurve(l1, n_circ + 1)
geo.mesh.setTransfiniteCurve(l3, n_circ + 1)
geo.mesh.setTransfiniteCurve(l2, n_axial + 1)
geo.mesh.setTransfiniteCurve(l4, n_axial + 1)

geo.mesh.setTransfiniteSurface(surf)
geo.mesh.setRecombine(2, surf)

geo.synchronize()

# Generate the flat quad mesh
gmsh.model.mesh.generate(2)

node_tags, coords, _ = gmsh.model.mesh.getNodes()

for i, tag in enumerate(node_tags):
    s = coords[3 * i + 0]
    z = coords[3 * i + 1]

    theta = s / R
    x = R * math.cos(theta)
    y = R * math.sin(theta)

    gmsh.model.mesh.setNode(int(tag), [x, y, z], [])

# Optional: remove duplicate nodes on the seam
# This merges the theta=0 and theta=2*pi node columns.
gmsh.model.mesh.removeDuplicateNodes()

# Save
gmsh.write("cylinder.inp")
gmsh.write("cylinder.msh")

# Launch GUI if desired
# gmsh.fltk.run()

gmsh.finalize()
