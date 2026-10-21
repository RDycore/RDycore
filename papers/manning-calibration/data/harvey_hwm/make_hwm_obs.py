#!/usr/bin/env python3
"""Write the rdycore_adjoint high-water-mark table from the QC'd mapping.

Reads turning30m_hwm_cells.csv (see QC_REPORT.md), applies the recommended
filter (above cell bed AND quality <= fair by default), and writes the
plain-text table -adjoint_hwm_file expects:

    n_marks
    cell_id wse_m
    ...

cell_id is the 0-based natural (mesh-file order) cell ID; wse_m is the
surveyed peak water-surface elevation in meters NAVD88 (the mesh datum).
Several marks may map to the same cell; each stays an independent row
(independent measurements of that cell's peak).

Usage: python3 make_hwm_obs.py [--max-quality 3] [--environment Riverine]
                               [csv_in] [table_out]
Defaults: turning30m_hwm_cells.csv -> turning30m_hwm_obs.txt, quality <= 3
(1=excellent, 2=good, 3=fair, 4=poor), both environments.
"""

import argparse
import csv
from pathlib import Path

HERE = Path(__file__).parent

ap = argparse.ArgumentParser()
ap.add_argument("csv_in", nargs="?", default=HERE / "turning30m_hwm_cells.csv")
ap.add_argument("table_out", nargs="?", default=HERE / "turning30m_hwm_obs.txt")
ap.add_argument("--max-quality", type=int, default=3)
ap.add_argument("--environment", default=None, help="Riverine or Coastal; default both")
args = ap.parse_args()

rows = []
n_total = n_below = n_quality = n_env = 0
with open(args.csv_in) as f:
    for r in csv.DictReader(f):
        n_total += 1
        if r["is_above_bed"] != "yes":
            n_below += 1
            continue
        if int(r["hwm_quality_id"]) > args.max_quality:
            n_quality += 1
            continue
        if args.environment and r["hwm_environment"] != args.environment:
            n_env += 1
            continue
        rows.append((int(r["cell"]), float(r["elev_m"])))

with open(args.table_out, "w") as f:
    f.write(f"{len(rows)}\n")
    for cell, wse in rows:
        f.write(f"{cell} {wse:.10g}\n")

print(
    f"{args.table_out}: {len(rows)} marks kept of {n_total} "
    f"(dropped {n_below} below bed, {n_quality} quality > {args.max_quality}, {n_env} wrong environment)"
)
