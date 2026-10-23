#!/usr/bin/env python3
"""Check and plot the AMR dam-break demonstration output.

The channel case (driver/tests/amr/channel_wave_amr.yaml) is a wet-bed Riemann
problem run on a 2D mesh, so it has an exact solution -- Stoker's -- against
which the computed depth can be measured. This script evaluates that solution
independently of RDycore (the root of the Rankine-Hugoniot / Riemann-invariant
matching condition, found here by bisection) and reports the area-weighted L1
error in depth. L1 rather than Linf because the solution has a shock, where a
first-order scheme's pointwise error does not converge.

  error  print the L1 depth error against Stoker at a given output step
  plot   write a figure: top view of the depth with the mesh drawn over it,
         and the channel's cells projected onto a depth-vs-x cross section
         with the exact solution
  panels write a figure with one top view per output step given, for cases
         with no exact solution -- the mesh following the flow over time
  movie  animate the top view over every output step given, as a GIF (or MP4
         where the file name says so and ffmpeg is installed)

Usage:
  amr_dam_break.py error <case-0250.xmf> [--hl 2 --hr 1 --x0 25]
  amr_dam_break.py plot  <case-0250.xmf> [-o figure.png] [--hl 2 --hr 1 --x0 25]
  amr_dam_break.py panels <case-0000.xmf> <case-0500.xmf> ... [-o figure.png]
  amr_dam_break.py movie  <case-*.xmf> [-o movie.gif] [--fps 8]

Reads the XDMF file only to find the HDF5 files it points at, so it follows
the mesh from one refinement level to the next without being told.
"""

import argparse
import os
import xml.etree.ElementTree as ET

import h5py
import numpy as np

GRAVITY = 9.806  # must match src/swe/swe_types_petsc.h


def stoker(h_l, h_r, xi, g=GRAVITY):
    """Depth of the wet-bed dam-break solution at similarity variable xi = (x - x0) / t.

    Left rarefaction, right shock. h_m is the depth of the star region, fixed by
    matching the rarefaction's Riemann invariant to the shock's jump condition:

        2 (sqrt(g h_l) - sqrt(g h_m))  =  (h_m - h_r) sqrt( g (h_m + h_r) / (2 h_m h_r) )
    """
    if h_l <= h_r:
        raise SystemExit("this routine assumes h_l > h_r (dam breaking to the right)")

    def residual(h_m):
        rarefaction = 2.0 * (np.sqrt(g * h_l) - np.sqrt(g * h_m))
        shock = (h_m - h_r) * np.sqrt(g * (h_m + h_r) / (2.0 * h_m * h_r))
        return rarefaction - shock

    lo, hi = h_r, h_l  # residual is positive at h_r and negative at h_l
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    h_m = 0.5 * (lo + hi)
    u_m = 2.0 * (np.sqrt(g * h_l) - np.sqrt(g * h_m))
    shock_speed = h_m * u_m / (h_m - h_r)

    head = -np.sqrt(g * h_l)  # rarefaction head, travelling left
    tail = u_m - np.sqrt(g * h_m)  # rarefaction tail

    xi = np.asarray(xi, dtype=float)
    h = np.empty_like(xi)
    h[:] = h_r
    h[xi <= head] = h_l
    fan = (xi > head) & (xi < tail)
    h[fan] = (2.0 * np.sqrt(g * h_l) - xi[fan]) ** 2 / (9.0 * g)
    h[(xi >= tail) & (xi <= shock_speed)] = h_m
    return h


def read_step(xmf_path):
    """Return (time, centroids, areas, depth, vertices, triangles) for one output step."""
    root = ET.parse(xmf_path).getroot()
    base = os.path.dirname(os.path.abspath(xmf_path))

    time = float(root.find(".//Time").get("Value"))

    def load(ref):
        fname, dataset = ref.strip().split(":")
        with h5py.File(os.path.join(base, fname), "r") as f:
            return f[dataset][...]

    topology = load(root.find(".//Topology/DataItem").text)
    vertices = load(root.find(".//Geometry/DataItem").text)

    fields = {}
    for attr in root.findall(".//Attribute"):
        fields[attr.get("Name")] = load(attr.find("DataItem").text)

    # XDMF "Mixed" topology: one element-type code then its node ids; the demo
    # meshes are triangles throughout (type 4), so the stride is fixed
    topology = topology.astype(np.int64).reshape(-1, 4)
    if not np.all(topology[:, 0] == 4):
        raise SystemExit("expected an all-triangle mesh")
    triangles = topology[:, 1:]

    centroids = np.column_stack((fields["XC"], fields["YC"]))
    return time, centroids, fields["Area"], fields["Height"], vertices[:, :2], triangles


def l1_error(xmf_path, h_l, h_r, x0):
    time, centroids, areas, depth, _, _ = read_step(xmf_path)
    exact = stoker(h_l, h_r, (centroids[:, 0] - x0) / time)
    return time, len(areas), float(np.sum(np.abs(depth - exact) * areas) / np.sum(areas))


def plot(xmf_path, h_l, h_r, x0, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    time, centroids, areas, depth, vertices, triangles = read_step(xmf_path)

    fig, (top, side) = plt.subplots(
        2, 1, figsize=(10, 6), height_ratios=[1, 2], constrained_layout=True
    )

    polys = PolyCollection(vertices[triangles], array=depth, cmap="viridis", edgecolors="white", linewidths=0.2)
    top.add_collection(polys)
    top.autoscale_view()
    top.set_aspect("equal")
    top.set_xlabel("x (m)")
    top.set_ylabel("y (m)")
    top.set_title(f"depth at t = {time:g} s, {len(areas)} cells")
    fig.colorbar(polys, ax=top, label="h (m)", fraction=0.025)

    order = np.argsort(centroids[:, 0])
    side.plot(centroids[order, 0], depth[order], ".", ms=3, color="#1f77b4", label="RDycore cells")
    x = np.linspace(vertices[:, 0].min(), vertices[:, 0].max(), 2000)
    side.plot(x, stoker(h_l, h_r, (x - x0) / time), "-", lw=1.2, color="#d62728", label="Stoker exact")
    side.set_xlabel("x (m)")
    side.set_ylabel("h (m)")
    side.legend(loc="upper right")
    side.grid(alpha=0.3)

    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def panels(xmf_paths, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    steps = [read_step(p) for p in xmf_paths]
    lo = min(float(np.min(s[3])) for s in steps)
    hi = max(float(np.max(s[3])) for s in steps)

    fig, axes = plt.subplots(len(steps), 1, figsize=(7, 2.1 * len(steps)), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (time, _, areas, depth, vertices, triangles) in zip(axes, steps):
        polys = PolyCollection(vertices[triangles], array=depth, cmap="viridis", edgecolors="white", linewidths=0.25)
        polys.set_clim(lo, hi)
        ax.add_collection(polys)
        ax.autoscale_view()
        ax.set_aspect("equal")
        ax.set_title(f"t = {time:g} s, {len(areas)} cells", fontsize=10)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    fig.colorbar(axes[0].collections[0], ax=list(axes), label="h (m)", fraction=0.02)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def movie(xmf_paths, out, fps):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    steps = sorted(xmf_paths)
    first = read_step(steps[0])
    # the colour scale is fixed over the whole movie, and taken from every
    # frame so that none of them clips: the mesh is what should be moving
    depths = [read_step(p)[3] for p in steps]
    lo = min(float(np.min(d)) for d in depths)
    hi = max(float(np.max(d)) for d in depths)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(first[4][:, 0].min(), first[4][:, 0].max())
    ax.set_ylim(first[4][:, 1].min(), first[4][:, 1].max())
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(lo, hi), cmap="viridis"), ax=ax, label="h (m)", fraction=0.03)

    def draw(path):
        time, _, areas, depth, vertices, triangles = read_step(path)
        for c in list(ax.collections):
            c.remove()
        polys = PolyCollection(vertices[triangles], array=depth, cmap="viridis", edgecolors="white", linewidths=0.25)
        polys.set_clim(lo, hi)
        ax.add_collection(polys)
        ax.set_title(f"t = {time:g} s, {len(areas)} cells", fontsize=11)
        return (polys,)

    anim = animation.FuncAnimation(fig, draw, frames=steps, interval=1000 / fps, blit=False)
    if out.endswith(".mp4"):
        anim.save(out, writer=animation.FFMpegWriter(fps=fps))
    else:
        anim.save(out, writer=animation.PillowWriter(fps=fps))
    print(f"wrote {out} ({len(steps)} frames)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=("error", "plot", "panels", "movie"))
    p.add_argument("xmf", nargs="+")
    p.add_argument("--hl", type=float, default=2.0, help="initial depth left of the dam (m)")
    p.add_argument("--hr", type=float, default=1.0, help="initial depth right of the dam (m)")
    p.add_argument("--x0", type=float, default=25.0, help="dam position (m)")
    p.add_argument("-o", "--out", default="amr_dam_break.png")
    p.add_argument("--fps", type=float, default=8.0, help="movie: frames per second")
    p.add_argument("--tol", type=float, help="error: fail if the L1 error exceeds this")
    args = p.parse_args()

    if args.action == "movie":
        movie(args.xmf, args.out, args.fps)
    elif args.action == "panels":
        panels(args.xmf, args.out)
    elif args.action == "error":
        time, ncells, err = l1_error(args.xmf[0], args.hl, args.hr, args.x0)
        print(f"t = {time:g} s, {ncells} cells, L1 depth error vs Stoker = {err:.6f} m")
        if args.tol is not None and err > args.tol:
            raise SystemExit(f"L1 error {err:.6f} exceeds tolerance {args.tol}")
    else:
        plot(args.xmf[0], args.hl, args.hr, args.x0, args.out)
