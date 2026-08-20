#!/usr/bin/env python3
"""Build a gauge WSE observation table for rdycore_adjoint -adjoint_calibrate_gauges.

Reads houston_gauges_cells.csv (from map_gauges_to_cells.py) and the USGS
stage series (height/USGS_gage_height_inst/<site>_gage_height.csv, feet above
gauge datum), converts to water-surface elevation in meters (NAVD88:
WSE = gage_datum_ft + stage_ft, then ft -> m), interpolates to the model's
observation times t0 + k*obs_dt (k = 1..K), and writes the plain-text table
the driver reads:

    line 1: ngauges K
    line 2: ngauges natural cell IDs
    lines 3..K+2: time_seconds  wse_1 ... wse_n   (nan = missing)

IMPORTANT: --t0 must be on the same clock as the CSV timestamps (the
HydroShare export carries no timezone). QC by comparing a documented crest
time (e.g. Buffalo Bayou at Piney Point, 63.94 ft) before trusting timing.

--subset rain-driven excludes the reservoir-affected gauges (Addicks/Barker
pools, outlets, and downstream Buffalo Bayou): on the 1-km mesh the dam
embankments are smoothed away and gates/releases are unmodeled, so those
hydrographs cannot be matched by any Manning field.
"""

import argparse
import csv
import math
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).parent
FT_TO_M = 0.3048

# reservoir-affected sites (see plans/RESULTS-manning-draft.md, 2026-08-19)
DAM_AFFECTED = {
    "08072500",  # Barker Res nr Addicks
    "08072600",  # Buffalo Bayou at SH6 (Barker outlet)
    "08073000",  # Addicks Res
    "08073100",  # Langham Ck at Addicks Res outflow
    "08073500",  # Buffalo Bayou nr Addicks (downstream)
    "08073600",  # Buffalo Bayou at W Belt Dr (downstream)
    "08073700",  # Buffalo Bayou at Piney Point (downstream)
    "08074710",  # Buffalo Bayou at Turning Basin
}

MAX_GAP = timedelta(hours=2)  # do not interpolate across larger data gaps


def read_series(path):
    times, vals = [], []
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) < 4 or row[1] == "site_no":
                continue
            try:
                t = datetime.strptime(row[2], "%Y-%m-%d %H:%M:%S")
                v = float(row[3])
            except ValueError:
                continue
            times.append(t)
            vals.append(v)
    return times, vals


def interp(times, vals, t):
    """Linear interpolation; nan outside the record or across a gap > MAX_GAP."""
    if not times or t < times[0] or t > times[-1]:
        return math.nan
    lo, hi = 0, len(times) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if times[mid] <= t:
            lo = mid
        else:
            hi = mid
    if times[hi] - times[lo] > MAX_GAP:
        return math.nan
    if times[hi] == times[lo]:
        return vals[lo]
    w = (t - times[lo]) / (times[hi] - times[lo])
    return vals[lo] * (1 - w) + vals[hi] * w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t0", required=True, help='model start on the CSV clock, e.g. "2017-08-26 00:00:00"')
    ap.add_argument("--obs-dt", type=float, required=True, help="seconds between observations")
    ap.add_argument("--num-obs", type=int, required=True, help="number of observation times K")
    ap.add_argument("--cells-csv", default=HERE / "houston_gauges_cells.csv")
    ap.add_argument("--height-dir", default=HERE / "height/USGS_gage_height_inst")
    ap.add_argument("--subset", choices=["all", "rain-driven"], default="rain-driven")
    ap.add_argument("--out", default=HERE / "obs_table.txt")
    args = ap.parse_args()

    t0 = datetime.strptime(args.t0, "%Y-%m-%d %H:%M:%S")

    gauges = []
    with open(args.cells_csv) as f:
        for g in csv.DictReader(f):
            if g["has_stage_ts"] != "True":
                continue
            if args.subset == "rain-driven" and g["site_no"] in DAM_AFFECTED:
                continue
            gauges.append(g)

    out = Path(args.out)
    with open(out, "w") as f:
        f.write(f"{len(gauges)} {args.num_obs}\n")
        f.write(" ".join(g["cell"] for g in gauges) + "\n")
        series = []
        for g in gauges:
            times, stage_ft = read_series(Path(args.height_dir) / f"{g['site_no']}_gage_height.csv")
            datum_ft = float(g["gage_datum_ft"])
            wse_m = [(datum_ft + s) * FT_TO_M for s in stage_ft]
            series.append((times, wse_m))
        n_missing = 0
        for k in range(1, args.num_obs + 1):
            t_model = k * args.obs_dt
            t_wall = t0 + timedelta(seconds=t_model)
            row = [f"{t_model:.10g}"]
            for times, wse in series:
                v = interp(times, wse, t_wall)
                if math.isnan(v):
                    n_missing += 1
                row.append(f"{v:.6g}")
            f.write(" ".join(row) + "\n")

    total = len(gauges) * args.num_obs
    print(f"wrote {out}: {len(gauges)} gauges ({args.subset}) x {args.num_obs} obs, "
          f"{total - n_missing}/{total} present, window {t0} + {args.num_obs * args.obs_dt / 3600:.1f} h")
    for g in gauges:
        print(f"  {g['site_no']} cell {g['cell']:>5s}  {g['station'][:50]}")


if __name__ == "__main__":
    main()
