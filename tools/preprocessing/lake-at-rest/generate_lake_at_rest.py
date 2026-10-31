#!/usr/bin/env python3
"""Generates meshes and initial conditions for RDycore's lake-at-rest tests.

The domain is [0, 2] x [0, 1] with the smooth bump
    z(x, y) = 0.8 exp(-5 (x - 1)^2 - 50 (y - 0.5)^2)
stored as the vertex z coordinate. The water surface is flat at eta, so the
initial depth of each cell is h = max(0, eta - z_c), where z_c is the mean of
the cell's vertex z values (the bed elevation RDycore uses for edge-based
well-balancing when no cell-elevation file is given). With eta = 1 the bump is
submerged; with eta = 0.5 it emerges as an island with a stationary shoreline.

Meshes are written as Exodus II files via ncgen (from netCDF), and initial
conditions as PETSc binary Vecs (h, hu, hv per cell, in the file's cell order)
for both 32- and 64-bit PETSc indices.

Usage: generate_lake_at_rest.py <mesh-dir> <condition-dir>
"""

import math
import os
import struct
import subprocess
import sys
import tempfile

NX, NY = 40, 20
LX, LY = 2.0, 1.0
ETAS = {"submerged": 1.0, "island": 0.5}


def bed(x, y):
    return 0.8 * math.exp(-5.0 * (x - 1.0) ** 2 - 50.0 * (y - 0.5) ** 2)


def structured_nodes():
    nodes = []
    for j in range(NY + 1):
        for i in range(NX + 1):
            x, y = LX * i / NX, LY * j / NY
            nodes.append((x, y, bed(x, y)))
    return nodes


def node_id(i, j):
    return j * (NX + 1) + i  # 0-based


def quads(i_range):
    return [[node_id(i, j), node_id(i + 1, j), node_id(i + 1, j + 1), node_id(i, j + 1)] for j in range(NY) for i in i_range]


def triangles(i_range):
    tris = []
    for j in range(NY):
        for i in i_range:
            a, b, c, d = node_id(i, j), node_id(i + 1, j), node_id(i + 1, j + 1), node_id(i, j + 1)
            if (i + j) % 2 == 0:  # alternate diagonals to avoid a preferred direction
                tris += [[a, b, c], [a, c, d]]
            else:
                tris += [[a, b, d], [b, c, d]]
    return tris


def meshes():
    nodes = structured_nodes()
    return {
        "lake_at_rest_quads": (nodes, [("QUAD4", quads(range(NX)))]),
        "lake_at_rest_tris": (nodes, [("TRI3", triangles(range(NX)))]),
        "lake_at_rest_mixed": (nodes, [("QUAD4", quads(range(NX // 2))), ("TRI3", triangles(range(NX // 2, NX)))]),
    }


def format_values(values, fmt):
    return ", ".join(fmt % v for v in values)


def write_exodus(path, name, nodes, blocks, side_sets=()):
    """Writes an Exodus II file. Each side set is (id, [(element, side), ...])
    with 0-based element indices (numbered across blocks) and 1-based Exodus
    local side numbers (for QUAD4, side k joins vertices k and k+1)."""
    num_elem = sum(len(cells) for _, cells in blocks)
    lines = [f"netcdf {name} {{", "dimensions:", "\tnum_dim = 3 ;", f"\tnum_nodes = {len(nodes)} ;", f"\tnum_elem = {num_elem} ;",
             f"\tnum_el_blk = {len(blocks)} ;"]
    if side_sets:
        lines.append(f"\tnum_side_sets = {len(side_sets)} ;")
    for b, (_, cells) in enumerate(blocks, start=1):
        lines += [f"\tnum_el_in_blk{b} = {len(cells)} ;", f"\tnum_nod_per_el{b} = {len(cells[0])} ;"]
    for s, (_, sides) in enumerate(side_sets, start=1):
        lines.append(f"\tnum_side_ss{s} = {len(sides)} ;")
    lines += ["\ttime_step = UNLIMITED ;", "variables:", "\tint eb_prop1(num_el_blk) ;", '\t\teb_prop1:name = "ID" ;',
              "\tdouble coordx(num_nodes) ;", "\tdouble coordy(num_nodes) ;", "\tdouble coordz(num_nodes) ;"]
    for b, (elem_type, _) in enumerate(blocks, start=1):
        lines += [f"\tint connect{b}(num_el_in_blk{b}, num_nod_per_el{b}) ;", f'\t\tconnect{b}:elem_type = "{elem_type}" ;']
    if side_sets:
        lines += ["\tint ss_prop1(num_side_sets) ;", '\t\tss_prop1:name = "ID" ;']
    for s, _ in enumerate(side_sets, start=1):
        lines += [f"\tint elem_ss{s}(num_side_ss{s}) ;", f"\tint side_ss{s}(num_side_ss{s}) ;"]
    lines += ["", "// global attributes:", "\t\t:api_version = 4.93 ;", "\t\t:version = 4.93 ;", "\t\t:floating_point_word_size = 8. ;",
              "\t\t:file_size = 1. ;", "data:", ""]
    lines.append(" eb_prop1 = " + format_values(range(1, len(blocks) + 1), "%d") + " ;")
    for d, coord in enumerate("xyz"):
        lines.append(f" coord{coord} = " + format_values([n[d] for n in nodes], "%.17g") + " ;")
    for b, (_, cells) in enumerate(blocks, start=1):
        lines.append(f" connect{b} = " + format_values([v + 1 for c in cells for v in c], "%d") + " ;")
    if side_sets:
        lines.append(" ss_prop1 = " + format_values([ss_id for ss_id, _ in side_sets], "%d") + " ;")
    for s, (_, sides) in enumerate(side_sets, start=1):
        lines.append(f" elem_ss{s} = " + format_values([e + 1 for e, _ in sides], "%d") + " ;")
        lines.append(f" side_ss{s} = " + format_values([k for _, k in sides], "%d") + " ;")
    lines.append("}")
    with tempfile.NamedTemporaryFile("w", suffix=".cdl", delete=False) as f:
        f.write("\n".join(lines) + "\n")
        cdl = f.name
    try:
        subprocess.run(["ncgen", "-k", "64-bit offset", "-o", path, cdl], check=True)
    finally:
        os.remove(cdl)


def write_petsc_vec(path, values, index_bytes):
    classid = 1211214
    int_fmt = ">i" if index_bytes == 4 else ">q"
    with open(path, "wb") as f:
        f.write(struct.pack(int_fmt, classid))
        f.write(struct.pack(int_fmt, len(values)))
        f.write(struct.pack(">%dd" % len(values), *values))


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    mesh_dir, cond_dir = sys.argv[1], sys.argv[2]
    for name, (nodes, blocks) in meshes().items():
        write_exodus(os.path.join(mesh_dir, name + ".exo"), name, nodes, blocks)
        zc = [sum(nodes[v][2] for v in c) / len(c) for _, cells in blocks for c in cells]
        for label, eta in ETAS.items():
            ic = []
            for z in zc:
                ic += [max(0.0, eta - z), 0.0, 0.0]
            for id_type, nbytes in (("int32", 4), ("int64", 8)):
                write_petsc_vec(os.path.join(cond_dir, f"{name}_{label}.ic.{id_type}.bin"), ic, nbytes)


if __name__ == "__main__":
    main()
