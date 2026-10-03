#!/usr/bin/env python3
"""Step A: would a transmissive perimeter leak inward?

Donghui asked whether to open all the boundary edges. RDycore's free-outflow
condition copies the interior state into the ghost cell with no inflow guard
and no elevation threshold, and the Turning perimeter follows a catchment
divide, which slopes INWARD. So an opened perimeter can admit water as well as
release it, and Step B's counterfactual would then be measuring two changes at
once.

This answers the question from existing o37 hourly checkpoints, with no
allocation. For every auto-generated (boundary id 0) perimeter edge, take the
would-be transmissive volume flux from the interior state,

    q = (hu, hv) . n_out * L     [m^3/s],   positive = out of the domain.

THE STAIRCASE. The 30 m perimeter is a zig-zag, and a flow parallel to a
zig-zag crosses alternating facets inward and outward in nearly equal measure.
A per-edge sign split therefore reports a large inward total that is an
artifact of the facets, not of the flow. So the edges are first walked into
connected boundary loops and aggregated into runs of a given length, and the
sign split is taken over the runs. The run length at which the split stops
moving is the scale at which the inward flux is real. The NET is invariant.

MASS BALANCE. The baseline perimeter is closed, so d(storage)/dt + outlet =
rain. Computing that residual over all 73 checkpoints validates the flux
formula, the normals and the cell areas independently of any of the above: a
wrong normal convention or a factor-of-two area shows up as a rain series that
is negative or implausible.

usage: stepA_perimeter_flux.py [--balance] [hour ...]      (default 40 60 72)
"""
import os, struct, sys
import numpy as np
from netCDF4 import Dataset
from scipy.spatial import cKDTree

RUNDIR = "/pscratch/sd/m/madams/gpu-implicit"
MESH   = RUNDIR + "/Turning_30m_with_z.updated.with_sidesets.exo"
CDIR   = RUNDIR + "/checkpoints_o37"
OBS    = RUNDIR + "/turning30m_hwm_obs.txt"
MARKS  = RUNDIR + "/o37_marks.txt"
LAST   = 259200          # final step of the 72-h run; a mark peaking here never crests
RUNS   = [0, 100, 300, 1000, 3000]      # m, aggregation scales for the sign split

do_balance = "--balance" in sys.argv
hours = [int(a) for a in sys.argv[1:] if not a.startswith("--")] or [40, 60, 72]

# ---------------------------------------------------------------- mesh
nc = Dataset(MESH)
x = np.asarray(nc.variables["coordx"][:], dtype=np.float64)
y = np.asarray(nc.variables["coordy"][:], dtype=np.float64)
conn = np.asarray(nc.variables["connect1"][:], dtype=np.int64) - 1   # natural cell order
ncell, nnode = conn.shape[0], x.size
area = 0.5 * np.abs((x[conn[:, 1]] - x[conn[:, 0]]) * (y[conn[:, 2]] - y[conn[:, 0]])
                    - (x[conn[:, 2]] - x[conn[:, 0]]) * (y[conn[:, 1]] - y[conn[:, 0]]))
print(f"mesh: {ncell} cells, {nnode} nodes, area {area.sum()/1e6:.1f} km^2, "
      f"cell area median {np.median(area):.1f} m^2")

# every triangle contributes three directed edges; row 3c+k is edge k of cell c
E = np.stack([conn[:, [0, 1]], conn[:, [1, 2]], conn[:, [2, 0]]], axis=1).reshape(-1, 2)
key = np.minimum(E[:, 0], E[:, 1]) * nnode + np.maximum(E[:, 0], E[:, 1])
order = np.argsort(key, kind="stable")
ks = key[order]
first = np.empty(ks.size, bool); first[0] = True;  first[1:] = ks[1:] != ks[:-1]
last  = np.empty(ks.size, bool); last[-1] = True;  last[:-1] = ks[:-1] != ks[1:]
bnd = order[first & last]            # flattened edge indices owned by exactly one cell

# the one named side set is the outlet; everything else on the boundary is id 0.
# side_ss1 uses shell numbering (3,4,5) for the three edges of a TRI3.
ss_elem = np.asarray(nc.variables["elem_ss1"][:], dtype=np.int64) - 1
ss_side = np.asarray(nc.variables["side_ss1"][:], dtype=np.int64) - 3
assert ss_side.min() >= 0 and ss_side.max() <= 2, f"unexpected side numbering {ss_side}"
outlet = 3 * ss_elem + ss_side
assert np.isin(outlet, bnd).all(), "outlet side set is not on the mesh boundary"
is_out = np.isin(bnd, outlet)
print(f"boundary edges {bnd.size}: outlet {is_out.sum()}, perimeter (id 0) {(~is_out).sum()}")


def geom(idx):
    """length, outward unit normal, midpoint and owning cell of boundary edges."""
    a, b = E[idx, 0], E[idx, 1]
    ex, ey = x[b] - x[a], y[b] - y[a]
    L = np.hypot(ex, ey)
    nx, ny = ey / L, -ex / L
    mx, my = 0.5 * (x[a] + x[b]), 0.5 * (y[a] + y[b])
    c = idx // 3
    cx, cy = x[conn[c]].mean(1), y[conn[c]].mean(1)
    flip = (nx * (mx - cx) + ny * (my - cy)) < 0     # point away from the owning cell
    return L, np.where(flip, -nx, nx), np.where(flip, -ny, ny), mx, my, c


bL, bnx, bny, bmx, bmy, bcell = geom(bnd)
print(f"boundary length {bL.sum()/1e3:.1f} km (outlet {bL[is_out].sum():.0f} m)")

# ------------------------------------------------- walk the boundary into loops
# a manifold boundary node carries exactly two boundary edges; walk each loop.
ends = np.c_[E[bnd, 0], E[bnd, 1]]
inc = {}
for i, (p, q_) in enumerate(ends):
    inc.setdefault(p, []).append(i)
    inc.setdefault(q_, []).append(i)
deg = np.array([len(v) for v in inc.values()])
print(f"boundary nodes {len(inc)}, degree 2 on {(deg == 2).sum()} "
      f"({(deg != 2).sum()} non-manifold)")

seen = np.zeros(bnd.size, bool)
loops = []
for s in range(bnd.size):
    if seen[s]:
        continue
    loop, e, node = [], s, ends[s, 1]
    while True:
        seen[e] = True
        loop.append(e)
        nxt = [k for k in inc[node] if not seen[k]]
        if not nxt:
            break
        e = nxt[0]
        node = ends[e, 1] if ends[e, 0] == node else ends[e, 0]
    loops.append(np.array(loop))
loops.sort(key=lambda L_: -bL[L_].sum())
print(f"loops: {len(loops)}; lengths km " +
      " ".join(f"{bL[L_].sum()/1e3:.1f}" for L_ in loops[:6]) +
      (" ..." if len(loops) > 6 else ""))


def segments(scale):
    """group each loop's edges into consecutive runs of at least `scale` metres."""
    segs = []
    for L_ in loops:
        if scale <= 0:
            segs += [np.array([e]) for e in L_]
            continue
        acc, run = 0.0, []
        for e in L_:
            run.append(e); acc += bL[e]
            if acc >= scale:
                segs.append(np.array(run)); acc, run = 0.0, []
        if run:
            segs.append(np.array(run))
    return segs


SEGS = {s: segments(s) for s in RUNS}

# ------------------------------------------------- the censored downstream reach
nlines = open(OBS).read().split("\n")
nmark = int(nlines[0])
mcell = np.array([int(l.split()[0]) for l in nlines[1:nmark + 1]])
censored = np.zeros(nmark, bool)
for l in open(MARKS):
    if l.startswith("#"):
        continue
    p = l.split()
    if int(p[4]) >= LAST - 300:
        censored[int(p[0])] = True
mkx, mky = x[conn[mcell]].mean(1), y[conn[mcell]].mean(1)
_, nearest = cKDTree(np.c_[mkx, mky]).query(np.c_[bmx, bmy])
in_voronoi = censored[nearest] & ~is_out          # nearest mark never crests
ox, oy = bmx[is_out].mean(), bmy[is_out].mean()
R = np.hypot(mkx[censored] - ox, mky[censored] - oy).max()
in_radius = (np.hypot(bmx - ox, bmy - oy) <= R) & ~is_out
print(f"marks {nmark} ({censored.sum()} censored); censored reach: "
      f"Voronoi {in_voronoi.sum()} edges, within {R/1e3:.1f} km of outlet "
      f"{in_radius.sum()} edges")


def vec_offset(fn):
    head = open(fn, "rb").read(65536)
    size = os.path.getsize(fn)
    for off in range(len(head) - 8):
        cid, n = struct.unpack_from(">ii", head, off)
        if cid == 1211214 and n > 0 and off + 8 + 8 * n == size:
            return off + 8, n
    raise RuntimeError(f"no Vec header in {fn}")


def load(step):
    fn = f"{CDIR}/o37.rdycore.r.{step:06d}.bin"
    off, n = vec_offset(fn)
    with open(fn, "rb") as f:
        f.seek(off)
        return np.fromfile(f, dtype=">f8", count=n).reshape(ncell, 3)


def flux(U):
    hu, hv = U[:, 1], U[:, 2]
    return (hu[bcell] * bnx + hv[bcell] * bny) * bL


def split(q, sel, scale):
    """aggregate q over runs of `scale` metres inside `sel`, then split by sign."""
    tot = np.array([q[s[sel[s]]].sum() for s in SEGS[scale] if sel[s].any()])
    out, inn = tot[tot > 0].sum(), tot[tot < 0].sum()
    return out, inn


steps = sorted(int(f.split(".r.")[1].split(".bin")[0])
               for f in os.listdir(CDIR) if f.endswith(".bin"))

# ------------------------------------------------- mass balance over all hours
if do_balance:
    print("\n=== mass balance: d(storage)/dt + outlet = implied rain ===")
    print(f"{'hour':>5} {'storage 1e6 m3':>15} {'dV/dt m3/s':>12} {'outlet m3/s':>12} "
          f"{'implied rain m3/s':>18} {'mm/hr':>8}")
    prev = None
    for st in steps:
        U = load(st)
        V = float((U[:, 0] * area).sum())
        qo = flux(U)[is_out].sum()
        if prev is not None:
            dV = (V - prev[1]) / (st - prev[0])
            rain = dV + 0.5 * (qo + prev[2])
            print(f"{st/3600:>5.0f} {V/1e6:>15.1f} {dV:>12.1f} {qo:>12.1f} "
                  f"{rain:>18.1f} {rain/area.sum()*3600*1e3:>8.3f}")
        prev = (st, V, qo)
    sys.exit(0)

# ------------------------------------------------- the answer, hour by hour
for hr in hours:
    U = load(hr * 3600)
    h = U[:, 0]
    q = flux(U)
    print(f"\n=== hour {hr} ===  storage {float((h*area).sum())/1e6:.1f} x10^6 m^3, "
          f"max h {h.max():.2f} m, wet (h>1cm) {(h > 0.01).mean()*100:.0f}% of cells")
    print(f"  net flux [m^3/s, + = out]: perimeter {q[~is_out].sum():10.1f}   "
          f"outlet {q[is_out].sum():8.1f}   "
          f"ratio {abs(q[~is_out].sum()/q[is_out].sum()):.1f}x")
    print(f"  {'run':>6} | {'perimeter: out':>14} {'in':>12} {'|in|/out':>9} | "
          f"{'censored reach: out':>20} {'in':>12}")
    for s in RUNS:
        po, pi = split(q, ~is_out, s)
        vo, vi = split(q, in_voronoi, s)
        tag = "edge" if s == 0 else f"{s} m"
        print(f"  {tag:>6} | {po:>14.1f} {pi:>12.1f} {abs(pi)/po*100:>8.1f}% | "
              f"{vo:>20.1f} {vi:>12.1f}")
    # where: net flux by distance band from the outlet, and the worst 3 km runs
    d = np.hypot(bmx - ox, bmy - oy) / 1e3
    print("  net perimeter flux by distance from the outlet [m^3/s, + = out]:")
    edges = [0, 10, R / 1e3, 50, 1e9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (~is_out) & (d >= lo) & (d < hi)
        if not sel.any():
            continue
        print(f"    {lo:5.1f} - {min(hi, d.max()):5.1f} km   {q[sel].sum():10.1f}   "
              f"over {bL[sel].sum()/1e3:5.1f} km of divide, {sel.sum():4d} edges")
    tot3 = [(q[s].sum(), bL[s].sum(), bmx[s].mean(), bmy[s].mean(), np.hypot(bmx[s].mean()-ox, bmy[s].mean()-oy)/1e3)
            for s in SEGS[3000] if not is_out[s].any()]
    tot3.sort()
    print("  five most inward 3 km runs (m^3/s, km of run, x, y, km from outlet):")
    for v, l, mx_, my_, dd in tot3[:5]:
        print(f"    {v:10.1f}  {l/1e3:5.2f} km  {mx_:10.0f} {my_:10.0f}  {dd:5.1f} km")
