#!/usr/bin/env python3
"""Write a crest-windowed HWM observation table.

The peak-WSE misfit should only see marks whose crest falls inside the
calibration window: a mark cresting outside contributes a window-edge
residual, not a peak. Crest times come from the per-mark dump of a
72-hour eval-only forward (h_model peak step per mark, uncalibrated
field -- the marks themselves record no time, so this is the only crest
estimate there is). Marks in the model's undrainable downstream reach
never crest (peak at the final step) and are excluded by construction.

Usage:
  filter_hwm_by_crest.py <full_obs_table> <marks_dump> <start_hr> <end_hr> <out_table>

Mark order in the dump is the obs-table line order.
"""
import sys

obs_path, dump_path, h0, h1, out_path = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]

lines = open(obs_path).read().split("\n")
n = int(lines[0])
rows = [l for l in lines[1 : n + 1]]

keep = []
for l in open(dump_path):
    if l.startswith("#"):
        continue
    p = l.split()
    m, step = int(p[0]), int(p[4])
    if h0 * 3600 <= step <= h1 * 3600:
        keep.append(m)

with open(out_path, "w") as f:
    f.write(f"{len(keep)}\n")
    for m in keep:
        f.write(rows[m] + "\n")
print(f"{len(keep)} of {n} marks crest in [h{h0:g}, h{h1:g}] -> {out_path}")
