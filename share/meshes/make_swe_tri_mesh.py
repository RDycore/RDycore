#!/usr/bin/env python3
"""Generate the triangular meshes used by the AMR demonstration cases.

Triangles rather than quads because the AMR path refines by simplex bisection
(DMPlexTransform's refine_sbr), which needs simplices. Gmsh is not required:
this writes the 2.2 ASCII format directly.

Two shapes, both with a cell set on each side of the midline so a YAML config
can set a different initial depth in each (RDycore's flow conditions are
per-region constants, not spatial expressions):

  channel   a plain rectangle -- a 1D Riemann problem run on a 2D mesh.

                y=W  +---------------+---------------+
                     |   cell set 1  |   cell set 2  |
                y=0  +---------------+---------------+
                    x=0            x=L/2            x=L

  ramp      the channel again, but tilted: the bed rises along x, so a lake at
            rest has a depth that varies from cell to cell while its surface
            does not. Used by the well-balanced-transfer gate.

  graded    the channel again, but with the cells growing geometrically along
            x, so the largest base cell has GRADED_RATIO times the area of the
            smallest. Every other mesh here is uniform, which is what let a
            refinement cap written as one area threshold over the whole mesh go
            unnoticed; RDycore's real meshes are graded like this one.

  dam       the planar_dam_10x5.geo geometry: a basin with a dam across the
            middle, pierced by a central breach. The dam blocks are holes, so
            their outlines are boundary, like the basin walls.

                y=Ly +--------+=======+--------+
                     |        | (hole)|        |
                     |        +=======+        |
                     |  set 1    gap    set 2  |
                     |        +=======+        |
                     |        | (hole)|        |
                y=0  +--------+=======+--------+
                    x=0                      x=Lx

Face sets: 1 is the outer boundary, 2 and 3 the top and bottom dam blocks.

Usage:
  make_swe_tri_mesh.py channel [cells-along-x] [output.msh]
  make_swe_tri_mesh.py ramp    [cells-along-x] [output.msh]
  make_swe_tri_mesh.py graded  [cells-along-x] [output.msh]
  make_swe_tri_mesh.py dam     [cells-along-x] [output.msh]
"""

import sys

# channel: long and thin, so the flow stays one-dimensional away from the ends
CHANNEL_L, CHANNEL_W = 50.0, 2.0

# ramp: the channel bed rises by this much over its length
RAMP_RISE = 1.0

# graded: area of the largest base cell over the smallest. Above 2^max_levels,
# so that a cap taken over the whole mesh lets the large end refine past the
# limit while the small end is still held at it.
GRADED_RATIO = 8.0

# dam: the planar_dam_10x5.geo dimensions
DAM_LX, DAM_LY = 10.0, 5.0
DAM_WX = 2.0  # dam thickness in x
DAM_LTOP = 1.0  # dam length from the top wall
DAM_LBOT = 2.0  # dam length from the bottom wall

PHYSICAL_NAMES = {
    "channel": [(2, 1, "left"), (2, 2, "right"), (1, 1, "boundary")],
    "ramp": [(2, 1, "left"), (2, 2, "right"), (1, 1, "boundary")],
    "graded": [(2, 1, "left"), (2, 2, "right"), (1, 1, "boundary")],
    "dam": [(2, 1, "upstream"), (2, 2, "downstream"), (1, 1, "boundary"), (1, 2, "top_wall"), (1, 3, "bottom_wall")],
}


class Channel:
    lx, ly = CHANNEL_L, CHANNEL_W

    @classmethod
    def xs(cls, nx):
        """Node positions along x, nx + 1 of them. Uniform unless overridden."""
        return [i * cls.lx / nx for i in range(nx + 1)]

    @staticmethod
    def is_hole(xc, yc):
        return False

    @staticmethod
    def cell_set(xc, yc):
        return 1 if xc < 0.5 * CHANNEL_L else 2

    @staticmethod
    def face_set(x0, y0, x1, y1):
        return 1

    @staticmethod
    def bed(x, y):
        return 0.0

    @staticmethod
    def check(nx, ny, dx, dy):
        pass


class Ramp(Channel):
    @staticmethod
    def bed(x, y):
        return RAMP_RISE * x / CHANNEL_L


class Graded(Channel):
    @classmethod
    def xs(cls, nx):
        r = GRADED_RATIO ** (1.0 / (nx - 1))  # each cell this much wider than the last
        widths = [r**i for i in range(nx)]
        scale = cls.lx / sum(widths)
        nodes = [0.0]
        for w in widths:
            nodes.append(nodes[-1] + scale * w)
        nodes[-1] = cls.lx  # exactly, so the right wall lands where face_set expects it
        return nodes


class Dam:
    lx, ly = DAM_LX, DAM_LY

    @classmethod
    def xs(cls, nx):
        return [i * cls.lx / nx for i in range(nx + 1)]

    @staticmethod
    def is_hole(xc, yc):
        if not (0.5 * (DAM_LX - DAM_WX) < xc < 0.5 * (DAM_LX + DAM_WX)):
            return False
        return yc > DAM_LY - DAM_LTOP or yc < DAM_LBOT

    @staticmethod
    def cell_set(xc, yc):
        return 1 if xc < 0.5 * (DAM_LX - DAM_WX) else 2

    @staticmethod
    def face_set(x0, y0, x1, y1):
        on_outer = (
            (x0 == 0.0 and x1 == 0.0)
            or (x0 == DAM_LX and x1 == DAM_LX)
            or (y0 == 0.0 and y1 == 0.0)
            or (y0 == DAM_LY and y1 == DAM_LY)
        )
        if on_outer:
            return 1
        return 2 if 0.5 * (y0 + y1) > 0.5 * DAM_LY else 3

    @staticmethod
    def bed(x, y):
        return 0.0

    @staticmethod
    def check(nx, ny, dx, dy):
        for extent, step, what in (
            (0.5 * (DAM_LX - DAM_WX), dx, "dam face"),
            (DAM_LTOP, dy, "top dam block"),
            (DAM_LBOT, dy, "bottom dam block"),
        ):
            if abs(extent / step - round(extent / step)) > 1e-12:
                raise SystemExit(f"{nx} cells along x does not resolve the {what}; try a multiple of 10")


SHAPES = {"channel": Channel, "ramp": Ramp, "graded": Graded, "dam": Dam}


def generate(shape, nx):
    ny = max(1, round(nx * shape.ly / shape.lx))
    dx, dy = shape.lx / nx, shape.ly / ny
    shape.check(nx, ny, dx, dy)

    xs = shape.xs(nx)
    coords = {(i, j): (xs[i], j * dy) for j in range(ny + 1) for i in range(nx + 1)}

    node_id, triangles, cell_sets = {}, [], []
    for j in range(ny):
        for i in range(nx):
            xc, yc = 0.5 * (xs[i] + xs[i + 1]), (j + 0.5) * dy
            if shape.is_hole(xc, yc):
                continue
            sw, se, nw, ne = (i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)
            # alternate the diagonal so the mesh carries no net directional bias
            tris = ((sw, se, ne), (sw, ne, nw)) if (i + j) % 2 == 0 else ((sw, se, nw), (se, ne, nw))
            for tri in tris:
                for n in tri:
                    node_id.setdefault(n, len(node_id) + 1)
                triangles.append(tri)
                cell_sets.append(shape.cell_set(xc, yc))

    # an edge used by one triangle is on the boundary (outer wall or a hole)
    uses = {}
    for tri in triangles:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            uses[frozenset((a, b))] = uses.get(frozenset((a, b)), 0) + 1
    edges = []
    for edge, n in uses.items():
        if n != 1:
            continue
        a, b = tuple(edge)
        (x0, y0), (x1, y1) = coords[a], coords[b]
        edges.append((a, b, shape.face_set(x0, y0, x1, y1)))
    edges.sort(key=lambda e: (e[2], node_id[e[0]], node_id[e[1]]))

    return coords, node_id, triangles, cell_sets, edges


def write_msh(path, names, bed, coords, node_id, triangles, cell_sets, edges):
    with open(path, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write(f"$PhysicalNames\n{len(names)}\n")
        for dim, tag, name in names:
            f.write(f'{dim} {tag} "{name}"\n')
        f.write("$EndPhysicalNames\n")

        f.write(f"$Nodes\n{len(node_id)}\n")
        for node, nid in sorted(node_id.items(), key=lambda kv: kv[1]):
            x, y = coords[node]
            f.write(f"{nid} {x:.17g} {y:.17g} {bed(x, y):.17g}\n")
        f.write("$EndNodes\n")

        f.write(f"$Elements\n{len(edges) + len(triangles)}\n")
        eid = 0
        for a, b, group in edges:
            eid += 1
            f.write(f"{eid} 1 2 {group} {group} {node_id[a]} {node_id[b]}\n")
        for tri, cell_set in zip(triangles, cell_sets):
            eid += 1
            f.write(f"{eid} 2 2 {cell_set} {cell_set} " + " ".join(str(node_id[n]) for n in tri) + "\n")
        f.write("$EndElements\n")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in SHAPES:
        raise SystemExit(f"usage: {sys.argv[0]} <{'|'.join(SHAPES)}> [cells-along-x] [output.msh]")
    name = sys.argv[1]
    shape = SHAPES[name]
    nx = int(sys.argv[2]) if len(sys.argv) > 2 else {"channel": 100, "graded": 40}.get(name, 20)
    ny = max(1, round(nx * shape.ly / shape.lx))
    out = sys.argv[3] if len(sys.argv) > 3 else f"{name}_{nx}x{ny}_tri.msh"
    mesh = generate(shape, nx)
    write_msh(out, PHYSICAL_NAMES[name], shape.bed, *mesh)
    print(f"{out}: {len(mesh[3])} triangles, {len(mesh[1])} nodes, {len(mesh[4])} boundary edges")
