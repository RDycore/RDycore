#!/usr/bin/env python3
"""Build a per-cell NLCD land-cover Manning map for an RDycore Exodus mesh.

Samples the NLCD 2021 land-cover raster over each mesh cell and writes

  <out>_manning.bin   PETSc binary Vec, per-cell Manning n (natural order)
  <out>_class.bin     PETSc binary Vec, per-cell dominant NLCD class code
  <out>_summary.csv   per-class cell counts, area fraction, and n

The class-to-roughness table is Donghui Xu's (OFMmesh
ex2/code/Step03_Process_Manning.m), which is what the RDycore continental
configuration uses.

NLCD is natively 30 m, so on the 30 m Harvey mesh this is essentially a 1:1
mapping and a single sample per cell suffices. On coarser meshes each cell
covers many pixels, so we sample a k x k grid across the cell (k chosen from
the cell size), average the Manning values -- matching how the 1 km continental
map was built -- and record the majority class.

The raster subset is fetched once from the MRLC WCS service; see fetch_nlcd.sh.

Usage:
  make_nlcd_manning.py <mesh.exo> <nlcd.tif> <albers_bbox> <out_prefix>
    albers_bbox = "xmin,ymin,xmax,ymax" in EPSG:5070, as requested from WCS

Requires: numpy, pyproj, tifffile, netCDF4.
"""

import sys
from pathlib import Path

import numpy as np
import tifffile
from netCDF4 import Dataset
from pyproj import Transformer

MESH_CRS = "EPSG:32610"  # see data/harvey_gauges/README.md
NLCD_CRS = "EPSG:5070"

# NLCD class code -> (name, Manning n)   [Donghui Xu, OFMmesh ex2]
NLCD = {
    11: ("Open Water", 0.038),
    12: ("Perennial Ice/Snow", 0.038),
    21: ("Developed, Open Space", 0.040),
    22: ("Developed, Low Intensity", 0.090),
    23: ("Developed, Medium Intensity", 0.120),
    24: ("Developed, High Intensity", 0.160),
    31: ("Barren Land", 0.027),
    41: ("Deciduous Forest", 0.150),
    42: ("Evergreen Forest", 0.120),
    43: ("Mixed Forest", 0.140),
    51: ("Dwarf Scrub", 0.038),
    52: ("Shrub/Scrub", 0.115),
    71: ("Grassland/Herbaceous", 0.038),
    72: ("Sedge/Herbaceous", 0.038),
    81: ("Pasture/Hay", 0.038),
    82: ("Cultivated Crops", 0.035),
    90: ("Woody Wetlands", 0.098),
    95: ("Emergent Herbaceous Wetlands", 0.068),
}
DEFAULT_N = 0.038  # for nodata pixels (open water value; rare inside the domain)


def write_petsc_vec(path, values):
    """PETSc binary Vec: int32 classid (1211214), int32 length, then float64, big-endian."""
    with open(path, "wb") as f:
        np.array([1211214, len(values)], dtype=">i4").tofile(f)
        np.asarray(values, dtype=">f8").tofile(f)


def main():
    mesh_path, tif_path, bbox_str, out_prefix = sys.argv[1:5]
    xmin, ymin, xmax, ymax = (float(v) for v in bbox_str.split(","))

    img = tifffile.imread(tif_path)
    if img.ndim == 3:
        img = img[..., 0]
    nrow, ncol = img.shape
    px = (xmax - xmin) / ncol
    py = (ymax - ymin) / nrow

    with Dataset(mesh_path) as nc:
        x = np.array(nc.variables["coordx"][:])
        y = np.array(nc.variables["coordy"][:])
        conn = np.array(nc.variables["connect1"][:]) - 1
    cx, cy = x[conn].mean(1), y[conn].mean(1)
    # triangle area, for the sampling footprint
    ax, ay = x[conn[:, 0]], y[conn[:, 0]]
    bx, by = x[conn[:, 1]], y[conn[:, 1]]
    dx, dy = x[conn[:, 2]], y[conn[:, 2]]
    area = 0.5 * np.abs((bx - ax) * (dy - ay) - (dx - ax) * (by - ay))
    radius = np.sqrt(area / np.pi)
    ncell = len(cx)

    # one sample per ~30 m of cell width, capped for cost
    k = int(np.clip(np.median(2 * radius) / 30.0, 1, 9))
    offs = np.zeros(1) if k == 1 else np.linspace(-0.7, 0.7, k)
    print(f"{ncell} cells, median cell width {np.median(2*radius):.0f} m -> {k}x{k} samples/cell")

    tr = Transformer.from_crs(MESH_CRS, NLCD_CRS, always_xy=True)
    codes = np.array(sorted(NLCD))
    code_to_n = np.array([NLCD[c][1] for c in codes])

    n_of_cell = np.full(ncell, DEFAULT_N)
    class_of_cell = np.zeros(ncell, dtype=int)
    votes = np.zeros((ncell, len(codes)), dtype=np.int32)
    nsum = np.zeros(ncell)
    nhit = np.zeros(ncell, dtype=int)

    for oi in offs:
        for oj in offs:
            sx = cx + oi * radius
            sy = cy + oj * radius
            ax5, ay5 = tr.transform(sx, sy)
            col = ((ax5 - xmin) / px).astype(int)
            row = ((ymax - ay5) / py).astype(int)  # raster row 0 is ymax
            ok = (col >= 0) & (col < ncol) & (row >= 0) & (row < nrow)
            v = np.zeros(ncell, dtype=np.uint8)
            v[ok] = img[row[ok], col[ok]]
            for ci, c in enumerate(codes):
                m = v == c
                votes[m, ci] += 1
                nsum[m] += NLCD[c][1]
                nhit[m] += 1

    good = nhit > 0
    n_of_cell[good] = nsum[good] / nhit[good]
    class_of_cell[good] = codes[np.argmax(votes[good], axis=1)]

    write_petsc_vec(f"{out_prefix}_manning.bin", n_of_cell)
    write_petsc_vec(f"{out_prefix}_class.bin", class_of_cell.astype(float))

    with open(f"{out_prefix}_summary.csv", "w") as f:
        f.write("nlcd_code,name,manning,num_cells,pct_cells,pct_area\n")
        for c in codes:
            m = class_of_cell == c
            if m.sum():
                f.write(f"{c},\"{NLCD[c][0]}\",{NLCD[c][1]},{m.sum()},"
                        f"{100*m.mean():.3f},{100*area[m].sum()/area.sum():.3f}\n")

    print(f"n: min {n_of_cell.min():.4f} mean {n_of_cell.mean():.4f} max {n_of_cell.max():.4f} "
          f"(area-weighted mean {np.average(n_of_cell, weights=area):.4f})")
    print(f"cells with no NLCD coverage: {(~good).sum()}")
    print(f"wrote {out_prefix}_manning.bin, {out_prefix}_class.bin, {out_prefix}_summary.csv")


if __name__ == "__main__":
    main()
