#!/usr/bin/env python3
"""Generates a mesh and initial conditions for RDycore's flow-over-a-bump tests
(Goutal and Maurel, 1997).

The domain is the strip [0, 25] x [0, 0.25] (200 x 2 quads) with the bed
z(x) = 0.2 - 0.05 (x - 10)^2 for 8 < x < 12 (z = 0 elsewhere) stored as the
vertex z coordinate. Side set 1 is the inflow (x = 0) and side set 2 the
outflow (x = 25). Each case starts from still water, h = max(0, eta - z_c)
with z_c the mean of the cell's vertex z values, and eta = 2 (subcritical),
0.66 (transcritical) or 0.33 (shock). Initial conditions are written as PETSc
binary Vecs for both 32- and 64-bit PETSc indices.

Usage: generate_transcritical_bump.py <mesh-dir> <condition-dir>
"""

import os
import sys

from generate_lake_at_rest import write_exodus, write_petsc_vec

NX, NY = 200, 2
LX, LY = 25.0, 0.25
CASES = {"subcritical": 2.0, "transcritical": 0.66, "shock": 0.33}


def bed(x):
    return 0.2 - 0.05 * (x - 10.0) ** 2 if 8.0 < x < 12.0 else 0.0


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    mesh_dir, cond_dir = sys.argv[1], sys.argv[2]

    nodes = []
    for j in range(NY + 1):
        for i in range(NX + 1):
            x = LX * i / NX
            nodes.append((x, LY * j / NY, bed(x)))
    quads = [[j * (NX + 1) + i, j * (NX + 1) + i + 1, (j + 1) * (NX + 1) + i + 1, (j + 1) * (NX + 1) + i] for j in range(NY) for i in range(NX)]
    inflow = [(j * NX, 4) for j in range(NY)]
    outflow = [(j * NX + NX - 1, 2) for j in range(NY)]
    write_exodus(os.path.join(mesh_dir, "bump_channel.exo"), "bump_channel", nodes, [("QUAD4", quads)], [(1, inflow), (2, outflow)])

    for label, eta in CASES.items():
        ic = []
        for q in quads:
            zc = sum(nodes[v][2] for v in q) / 4.0
            ic += [max(0.0, eta - zc), 0.0, 0.0]
        for id_type, nbytes in (("int32", 4), ("int64", 8)):
            write_petsc_vec(os.path.join(cond_dir, f"bump_channel_{label}.ic.{id_type}.bin"), ic, nbytes)


if __name__ == "__main__":
    main()
