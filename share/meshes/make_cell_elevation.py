#!/usr/bin/env python3
"""Write a PETSc binary Vec of cell-centre bed elevations for a gmsh 2.2 mesh.

RDycore reads such a file through `grid: cell_elevation: file:`, in the mesh
file's own element order (the natural ordering of the DMPlex it reads). The
values override the cell-centre z the geometry would give, which is how a bed
comes from a DEM rather than from the mesh.

The only shape provided is a ramp, because that is what the AMR lake-at-rest
gate needs: a bed that rises along x, on a mesh that is itself flat. A flat
mesh with a sloping bed exercises the file path and nothing else -- the cell
areas and edge lengths stay exactly what they would be without a bed.

Usage:
  make_cell_elevation.py ramp <input.msh> <rise> <output-prefix>

writes <output-prefix>.int32.bin and <output-prefix>.int64.bin, which are
identical: PETSc writes a Vec's header with 32-bit integers either way, and the
suffix only tells RDycore's ${PETSC_ID_TYPE} substitution which name to ask for.
"""

import struct
import sys

VEC_FILE_CLASSID = 1211214


def read_gmsh22(path):
    lines = open(path).read().splitlines()
    i = lines.index("$Nodes")
    nnodes = int(lines[i + 1])
    xyz = {}
    for k in range(nnodes):
        f = lines[i + 2 + k].split()
        xyz[int(f[0])] = tuple(float(v) for v in f[1:4])

    j = lines.index("$Elements")
    nelem = int(lines[j + 1])
    triangles = []
    for k in range(nelem):
        f = lines[j + 2 + k].split()
        if f[1] != "2":  # element type 2 is a 3-node triangle
            continue
        triangles.append([int(v) for v in f[5:8]])
    return xyz, triangles


def ramp(xyz, triangles, rise):
    xs = [p[0] for p in xyz.values()]
    length = max(xs) - min(xs)
    z = []
    for tri in triangles:
        xc = sum(xyz[v][0] for v in tri) / len(tri)
        z.append(rise * (xc - min(xs)) / length)
    return z


def write_petsc_vec(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack(">i", VEC_FILE_CLASSID))
        f.write(struct.pack(">i", len(values)))
        f.write(struct.pack(">%dd" % len(values), *values))


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] != "ramp":
        raise SystemExit(f"usage: {sys.argv[0]} ramp <input.msh> <rise> <output-prefix>")
    xyz, triangles = read_gmsh22(sys.argv[2])
    z = ramp(xyz, triangles, float(sys.argv[3]))
    for id_type in ("int32", "int64"):
        write_petsc_vec(f"{sys.argv[4]}.{id_type}.bin", z)
    print(f"{sys.argv[4]}.{{int32,int64}}.bin: {len(z)} cells, z from {min(z):.6g} to {max(z):.6g}")
