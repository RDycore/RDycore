#!/usr/bin/env python3
"""Compares a 1D dam-break run with the Ritter (dry bed) or Stoker (wet bed)
analytic solution.

Usage: check_dam_break.py <h5dump> <grid.h5> <solution.h5> <group> <t> <h_left> <h_right> <x_dam> <max_rel_L1>

Reads the cell x centroids from the grid file and the depth from the solution
group, and fails (exit 1) if the relative L1 depth error exceeds max_rel_L1 or
if any depth is negative. Uses only the standard library.
"""

import math
import re
import subprocess
import sys

GRAVITY = 9.806


def read_dataset(h5dump, path, dataset):
    out = subprocess.run([h5dump, "-d", dataset, "-y", "-w", "0", "-m", "%.17g", path], capture_output=True, text=True, check=True).stdout
    body = out[out.index("DATA {") + len("DATA {"):]
    body = body[: body.index("}")]
    return [float(v) for v in re.split(r"[,\s]+", body.strip()) if v]


def stoker_middle_depth(h_left, h_right):
    """Depth between the rarefaction and the shock of a wet-bed dam break."""
    c_left = math.sqrt(GRAVITY * h_left)

    def residual(hm):
        shock_speed = math.sqrt(GRAVITY * hm * (hm + h_right) / (2.0 * h_right))
        return 2.0 * (c_left - math.sqrt(GRAVITY * hm)) - (hm - h_right) * shock_speed / hm

    lo, hi = h_right, h_left
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def analytic_depth(x, t, h_left, h_right, x_dam):
    xi = (x - x_dam) / t
    c_left = math.sqrt(GRAVITY * h_left)
    if xi <= -c_left:
        return h_left
    if h_right == 0.0:  # Ritter
        return (2.0 * c_left - xi) ** 2 / (9.0 * GRAVITY) if xi < 2.0 * c_left else 0.0
    hm = stoker_middle_depth(h_left, h_right)
    cm = math.sqrt(GRAVITY * hm)
    um = 2.0 * (c_left - cm)
    shock_speed = um * hm / (hm - h_right)
    if xi <= um - cm:
        return (2.0 * c_left - xi) ** 2 / (9.0 * GRAVITY)
    if xi <= shock_speed:
        return hm
    return h_right


def main():
    if len(sys.argv) != 10:
        sys.exit(__doc__)
    h5dump, grid, solution, group = sys.argv[1:5]
    t, h_left, h_right, x_dam, max_rel_l1 = (float(v) for v in sys.argv[5:10])

    xc = read_dataset(h5dump, grid, "/fields/XC")
    h = read_dataset(h5dump, solution, group + "/Height")
    exact = [analytic_depth(x, t, h_left, h_right, x_dam) for x in xc]

    rel_l1 = sum(abs(a - b) for a, b in zip(h, exact)) / sum(abs(b) for b in exact)
    min_h = min(h)
    print(f"relative L1 depth error = {rel_l1:.4e} (max {max_rel_l1:g}), min depth = {min_h:.4e}")
    if rel_l1 > max_rel_l1 or min_h < 0.0:
        sys.exit(1)


if __name__ == "__main__":
    main()
