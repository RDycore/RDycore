#!/usr/bin/env python3
"""Compares a steady flow over a bump with the analytic solution (Goutal and
Maurel, 1997).

Usage: check_transcritical_bump.py <h5dump> <grid.h5> <solution.h5> <group> <case> <dx> <max_rel_L1> <max_q_dev>
       check_transcritical_bump.py --boundary-states

The channel is [0, 25] with the frictionless bed z(x) = 0.2 - 0.05 (x - 10)^2
for 8 < x < 12 (z = 0 elsewhere). The cases are
    subcritical:   q = 4.42, h = 2 at the outflow
    transcritical: q = 1.53, critical flow at the crest, supercritical outflow
    shock:         q = 0.18, h = 0.33 at the outflow, with a hydraulic jump
The analytic depth is evaluated on the discrete bed seen by RDycore, the mean
of each cell's vertex elevations (the vertices are dx apart in x).

Fails (exit 1) if the relative L1 depth error exceeds max_rel_L1, if the
largest relative deviation of the cell discharge from q (away from the jump)
exceeds max_q_dev, or
if any depth is negative. With --boundary-states, prints the exact inflow and
outflow states used as Dirichlet boundary conditions. Uses only the standard
library.
"""

import math
import re
import subprocess
import sys

GRAVITY = 9.806
Z_MAX = 0.2
X_CREST = 10.0
CASES = {"subcritical": (4.42, 2.0), "transcritical": (1.53, None), "shock": (0.18, 0.33)}


def bed(x):
    return Z_MAX - 0.05 * (x - X_CREST) ** 2 if 8.0 < x < 12.0 else 0.0


def specific_energy(h, q):
    return h + q * q / (2.0 * GRAVITY * h * h)


def critical_depth(q):
    return (q * q / GRAVITY) ** (1.0 / 3.0)


def depth(energy, q, z, supercritical):
    """Root of h + q^2/(2 g h^2) = energy - z on the requested branch."""
    hc = critical_depth(q)
    e = energy - z
    if e <= 1.5 * hc:  # at (or, by round-off, above) the crest: critical flow
        return hc
    lo, hi = (1e-12, hc) if supercritical else (hc, e)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        # specific energy decreases with h on the supercritical branch
        if (specific_energy(mid, q) > e) != supercritical:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def shock_position(q, e_up, e_down):
    """Location in (X_CREST, 12) where the supercritical depth on e_up and the
    subcritical depth on e_down are conjugate depths of a hydraulic jump."""

    def residual(x):
        h1 = depth(e_up, q, bed(x), True)
        h2 = depth(e_down, q, bed(x), False)
        froude2 = q * q / (GRAVITY * h1**3)
        return 0.5 * h1 * (math.sqrt(1.0 + 8.0 * froude2) - 1.0) - h2

    lo, hi = X_CREST, 12.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def analytic_depth(case, x, z):
    q, h_out = CASES[case]
    if case == "subcritical":
        return depth(specific_energy(h_out, q), q, z, False)
    e_crest = 1.5 * critical_depth(q) + Z_MAX
    if x <= X_CREST:
        return depth(e_crest, q, z, False)
    if case == "transcritical":
        return depth(e_crest, q, z, True)
    e_down = specific_energy(h_out, q)
    return depth(e_crest, q, z, True) if x < shock_position(q, e_crest, e_down) else depth(e_down, q, z, False)


def read_dataset(h5dump, path, dataset):
    out = subprocess.run([h5dump, "-d", dataset, "-y", "-w", "0", "-m", "%.17g", path], capture_output=True, text=True, check=True).stdout
    body = out[out.index("DATA {") + len("DATA {"):]
    body = body[: body.index("}")]
    return [float(v) for v in re.split(r"[,\s]+", body.strip()) if v]


def main():
    if sys.argv[1:] == ["--boundary-states"]:
        for case, (q, _) in CASES.items():
            h_in, h_out = analytic_depth(case, 0.0, 0.0), analytic_depth(case, 25.0, 0.0)
            print(f"{case}: q = {q}, inflow h = {h_in:.12g}, outflow h = {h_out:.12g}")
        return
    if len(sys.argv) != 9:
        sys.exit(__doc__)
    h5dump, grid, solution, group, case = sys.argv[1:6]
    dx, max_rel_l1, max_q_dev = (float(v) for v in sys.argv[6:9])
    q = CASES[case][0]

    xc = read_dataset(h5dump, grid, "/fields/XC")
    h = read_dataset(h5dump, solution, group + "/Height")
    hu = read_dataset(h5dump, solution, group + "/MomentumX")
    exact = [analytic_depth(case, x, 0.5 * (bed(x - 0.5 * dx) + bed(x + 0.5 * dx))) for x in xc]

    rel_l1 = sum(abs(a - b) for a, b in zip(h, exact)) / sum(exact)
    # a captured hydraulic jump has a cell whose discharge is not q
    x_jump = shock_position(q, 1.5 * critical_depth(q) + Z_MAX, specific_energy(CASES[case][1], q)) if case == "shock" else None
    q_dev = max(abs(v - q) for x, v in zip(xc, hu) if x_jump is None or abs(x - x_jump) > 4.0 * dx) / q
    min_h = min(h)
    print(f"{case}: relative L1 depth error = {rel_l1:.4e} (max {max_rel_l1:g}), "
          f"max relative discharge deviation = {q_dev:.4e} (max {max_q_dev:g}), min depth = {min_h:.4e}")
    if rel_l1 > max_rel_l1 or q_dev > max_q_dev or min_h < 0.0:
        sys.exit(1)


if __name__ == "__main__":
    main()
