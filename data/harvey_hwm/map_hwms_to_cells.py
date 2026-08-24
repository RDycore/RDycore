#!/usr/bin/env python3
"""Map USGS STN HWMs to Turning 30m mesh cells and QC against bed elevation.

The Turning 30m mesh is in EPSG:32610 (WGS 84 / UTM zone 10N).
This script reads the raw HWM JSON, projects marks to the mesh CRS,
performs point-in-triangle tests, converts vertical datum to meters,
and outputs a detailed QC report.

The critical QC check: marks that fall BELOW their cell's bed elevation
are unusable (same issue as the gauges). This script reports:
  - Coverage: count and spatial spread of in-domain marks
  - Vertical datum: what the marks are in, what the mesh is in
  - Below-bed fraction: how many marks are invalid
  - Quality-code breakdown: by USGS quality flag and environment
  - Recommended filter: usable marks by quality threshold

Usage: python3 map_hwms_to_cells.py [mesh.exo] [hwm_raw.json] [outdir]
Requires: numpy, netCDF4, pyproj, json.
"""

import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from netCDF4 import Dataset
from pyproj import Transformer

HERE = Path(__file__).parent
MESH = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path(__file__).parent.parent.parent /
    "rescued-from-main-checkout" /
    "share" / "meshes" /
    "Turning_30m_with_z.updated.with_sidesets.exo"
)
HWM_JSON = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "hwm_raw.json"
OUTDIR = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE

MESH_CRS = "EPSG:32610"
FT_TO_M = 0.3048

# USGS quality codes (hwm_quality_id): 1=Excellent, 2=Good, 3=Fair, 4=Poor
QUALITY_MAP = {1: "excellent", 2: "good", 3: "fair", 4: "poor"}

# Vertical datum IDs (vdatum_id): 2=NAVD88, others need mapping
DATUM_MAP = {
    1: "NGVD29",
    2: "NAVD88",
    3: "Local",
    4: "Mean Sea Level",
}

def load_mesh(mesh_path):
    """Load mesh vertices, connectivity, and bed elevation."""
    with Dataset(mesh_path) as nc:
        x = np.array(nc.variables["coordx"][:])
        y = np.array(nc.variables["coordy"][:])
        z = np.array(nc.variables["coordz"][:])
        conn = np.array(nc.variables["connect1"][:]) - 1  # 1-based -> 0-based

    # Triangle vertices per cell
    xa, ya = x[conn[:, 0]], y[conn[:, 0]]
    xb, yb = x[conn[:, 1]], y[conn[:, 1]]
    xc, yc = x[conn[:, 2]], y[conn[:, 2]]
    zcell = z[conn].mean(axis=1)

    return (xa, ya, xb, yb, xc, yc, zcell, x, y, z, conn)

def point_in_triangle(px, py, xa, ya, xb, yb, xc, yc, tol=1.0):
    """Test if point (px, py) is inside triangle using barycentric signs.
    Returns True if point is inside (or within tolerance of edge).
    tol: tolerance in m^2-scale for on-edge points.
    """
    d1 = (xa - px) * (yb - py) - (xb - px) * (ya - py)
    d2 = (xb - px) * (yc - py) - (xc - px) * (yb - py)
    d3 = (xc - px) * (ya - py) - (xa - px) * (yc - py)
    return ((d1 >= -tol) & (d2 >= -tol) & (d3 >= -tol)) | \
           ((d1 <= tol) & (d2 <= tol) & (d3 <= tol))

def main():
    print("\n=== Loading mesh ===")
    if not MESH.exists():
        print(f"ERROR: mesh not found at {MESH}")
        return False

    xa, ya, xb, yb, xc, yc, zcell, x, y, z, conn = load_mesh(MESH)
    print(f"Mesh: {MESH.name}")
    print(f"  Cells: {len(zcell)}")
    print(f"  Vertices: {len(x)}")
    print(f"  Bed elevation range: {z.min():.2f} to {z.max():.2f} m")

    print(f"\n=== Loading HWMs ===")
    if not HWM_JSON.exists():
        print(f"ERROR: HWM JSON not found at {HWM_JSON}")
        return False

    with open(HWM_JSON) as f:
        hwm_list = json.load(f)
    print(f"Loaded {len(hwm_list)} HWMs")

    # Setup coordinate transformation
    t = Transformer.from_crs("EPSG:4326", MESH_CRS, always_xy=True)

    # Process HWMs
    print(f"\n=== Coverage check: projecting and testing point-in-triangle ===")
    rows_in = []
    rows_out = []

    for hwm in hwm_list:
        lon, lat = hwm.get("longitude_dd"), hwm.get("latitude_dd")
        if lon is None or lat is None:
            continue

        px, py = t.transform(float(lon), float(lat))

        # Point-in-triangle test
        inside = point_in_triangle(px, py, xa, ya, xb, yb, xc, yc)
        hits = np.flatnonzero(inside)

        rec = {
            "hwm_id": hwm.get("hwm_id"),
            "lon": lon,
            "lat": lat,
            "x_utm10": f"{px:.1f}",
            "y_utm10": f"{py:.1f}",
            "elev_ft": hwm.get("elev_ft"),
            "vdatum_id": hwm.get("vdatum_id"),
            "vdatum": DATUM_MAP.get(hwm.get("vdatum_id"), "unknown"),
            "hwm_quality_id": hwm.get("hwm_quality_id"),
            "hwm_quality": QUALITY_MAP.get(hwm.get("hwm_quality_id"), "unknown"),
            "hwm_environment": hwm.get("hwm_environment", "unknown"),
            "survey_date": hwm.get("survey_date"),
            "waterbody": hwm.get("waterbody", ""),
        }

        if hits.size:
            cell = int(hits[0])
            rec["cell"] = cell
            rec["cell_z_m"] = f"{zcell[cell]:.2f}"

            # Convert elevation to meters
            elev_ft = float(hwm.get("elev_ft", 0)) if hwm.get("elev_ft") else None
            if elev_ft is not None:
                elev_m = elev_ft * FT_TO_M
                rec["elev_m"] = f"{elev_m:.3f}"

                # QC: check if mark is above cell bed
                cell_z = float(rec["cell_z_m"])
                above_bed = elev_m - cell_z
                rec["above_bed_m"] = f"{above_bed:.3f}"
                rec["is_above_bed"] = "yes" if above_bed >= 0 else "no"

            rows_in.append(rec)
        else:
            rows_out.append(rec)

    print(f"In-domain HWMs: {len(rows_in)} of {len(hwm_list)}")
    print(f"Outside mesh: {len(rows_out)}")

    # Compute coverage statistics
    if rows_in:
        xs_in = [float(r["x_utm10"]) for r in rows_in]
        ys_in = [float(r["y_utm10"]) for r in rows_in]
        print(f"  X range: {min(xs_in):.0f} to {max(xs_in):.0f} m")
        print(f"  Y range: {min(ys_in):.0f} to {max(ys_in):.0f} m")
        print(f"  Mesh X: {x.min():.0f} to {x.max():.0f} m")
        print(f"  Mesh Y: {y.min():.0f} to {y.max():.0f} m")

    # QC Analysis
    print(f"\n=== QC Analysis ===")

    # Vertical datum check
    datum_counts = defaultdict(int)
    for r in rows_in:
        datum_counts[r["vdatum"]] += 1
    print("Vertical datum distribution (in-domain marks):")
    for datum, count in sorted(datum_counts.items()):
        print(f"  {datum}: {count}")

    # Check the gauge CSV to confirm mesh vertical datum
    gauge_csv = HERE / "turning30m_gauges_cells.csv"
    if gauge_csv.exists():
        with open(gauge_csv) as f:
            gauges = list(csv.DictReader(f))
        print(f"\nGauge reference (turning30m_gauges_cells.csv):")
        print(f"  Gauges: {len(gauges)}")
        if gauges:
            print(f"  Gauge datum: {gauges[0].get('datum')} (sample)")
            # Gauge WSE in meters is: stage_ft * FT_TO_M (stage is already in NAVD88)
            # Model h + z_b should be comparable
            print(f"  Gauge cell_z_m: {gauges[0].get('cell_z_m')} (sample)")

    # Below-bed analysis
    print(f"\nBelow-bed analysis:")
    above_bed_marks = [r for r in rows_in if r.get("is_above_bed") == "yes"]
    below_bed_marks = [r for r in rows_in if r.get("is_above_bed") == "no"]

    print(f"  Above cell bed: {len(above_bed_marks)} ({100*len(above_bed_marks)/len(rows_in):.1f}%)")
    print(f"  Below cell bed: {len(below_bed_marks)} ({100*len(below_bed_marks)/len(rows_in):.1f}%)")

    if above_bed_marks:
        above_bed_heights = [float(r["above_bed_m"]) for r in above_bed_marks]
        print(f"  Above-bed height range: {min(above_bed_heights):.2f} to {max(above_bed_heights):.2f} m")
        print(f"  Above-bed height mean: {np.mean(above_bed_heights):.2f} m")

    # Quality code breakdown
    print(f"\nQuality code breakdown (in-domain):")
    quality_counts = defaultdict(lambda: defaultdict(int))
    quality_above_bed = defaultdict(lambda: defaultdict(int))

    for r in rows_in:
        quality = r["hwm_quality"]
        quality_counts["all"][quality] += 1
        env = r["hwm_environment"]
        quality_counts[env][quality] += 1

        if r.get("is_above_bed") == "yes":
            quality_above_bed["all"][quality] += 1
            quality_above_bed[env][quality] += 1

    for env in ["all"] + sorted([k for k in quality_counts.keys() if k != "all"]):
        print(f"  Environment: {env}")
        for quality in ["excellent", "good", "fair", "poor"]:
            total = quality_counts[env].get(quality, 0)
            usable = quality_above_bed[env].get(quality, 0)
            pct = 100*usable/total if total > 0 else 0
            print(f"    {quality:10s}: {total:4d} total, {usable:4d} usable ({pct:5.1f}%)")

    # Recommendations
    print(f"\n=== Recommendations ===")
    excellent_usable = quality_above_bed["all"].get("excellent", 0)
    good_usable = quality_above_bed["all"].get("good", 0) + excellent_usable
    fair_usable = quality_above_bed["all"].get("fair", 0) + good_usable

    print(f"Usable marks (above cell bed) by quality threshold:")
    print(f"  Excellent+ (1): {excellent_usable} marks")
    print(f"  Good+ (1-2): {good_usable} marks")
    print(f"  Fair+ (1-3): {fair_usable} marks")

    # Save the detailed mapping CSV
    print(f"\n=== Writing outputs ===")
    out_csv = OUTDIR / "turning30m_hwm_cells.csv"
    fieldnames = [
        "hwm_id", "lon", "lat", "x_utm10", "y_utm10", "cell",
        "elev_ft", "elev_m", "cell_z_m", "above_bed_m", "is_above_bed",
        "vdatum_id", "vdatum", "hwm_quality_id", "hwm_quality",
        "hwm_environment", "survey_date", "waterbody"
    ]

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_in)

    print(f"Wrote {out_csv.name} ({len(rows_in)} marks)")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
