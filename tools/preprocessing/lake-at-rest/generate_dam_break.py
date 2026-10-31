#!/usr/bin/env python3
"""Generates a mesh and initial conditions for RDycore's 1D dam-break tests.

The domain is the strip [0, 100] x [0, 0.5] (400 x 2 quads, flat bed) with a
dam at x = 50 separating h = 1 (left) from h = 0 (dry bed, Ritter solution) or
h = 0.1 (wet bed, Stoker solution). Initial conditions are written as PETSc
binary Vecs for both 32- and 64-bit PETSc indices.

Usage: generate_dam_break.py <mesh-dir> <condition-dir>
"""

import os
import sys

from generate_lake_at_rest import write_exodus, write_petsc_vec

NX, NY = 400, 2
LX, LY = 100.0, 0.5
X_DAM = 50.0
H_LEFT = 1.0
CASES = {"dry": 0.0, "wet": 0.1}


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    mesh_dir, cond_dir = sys.argv[1], sys.argv[2]

    nodes = [(LX * i / NX, LY * j / NY, 0.0) for j in range(NY + 1) for i in range(NX + 1)]
    quads = [[j * (NX + 1) + i, j * (NX + 1) + i + 1, (j + 1) * (NX + 1) + i + 1, (j + 1) * (NX + 1) + i] for j in range(NY) for i in range(NX)]
    write_exodus(os.path.join(mesh_dir, "dam_break_strip.exo"), "dam_break_strip", nodes, [("QUAD4", quads)])

    for label, h_right in CASES.items():
        ic = []
        for q in quads:
            xc = sum(nodes[v][0] for v in q) / 4.0
            ic += [H_LEFT if xc < X_DAM else h_right, 0.0, 0.0]
        for id_type, nbytes in (("int32", 4), ("int64", 8)):
            write_petsc_vec(os.path.join(cond_dir, f"dam_break_strip_{label}.ic.{id_type}.bin"), ic, nbytes)


if __name__ == "__main__":
    main()
