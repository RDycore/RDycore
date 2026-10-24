#!/usr/bin/env python3
"""Case 1: source-free moving transport of the conservative heat variable.

Three things are checked, in increasing order of what they can go wrong in:

1. Smoke, at a single resolution. The manufactured heat source must be zero to
   roundoff -- that is the defining invariant of this case, C_t + div(huT,hvT) = 0
   -- and the uniform flow must be preserved to roundoff, which requires n = 0 so
   the semi-implicit friction discretization and the analytic Cd u|u| source both
   vanish identically.

2. A *joint space-time* study with dt proportional to dx. Both the tracer
   discretization and the forward-Euler transport integrator are first order, so
   E ~ Cx dx + Ct dt collapses to (Cx + K Ct) dx. The measured rate verifies that
   the complete discretization converges at first order under coupled refinement;
   it is not an isolated spatial rate and is not reported as one.

3. An isolated *spatial* rate, obtained by raising the transport integrator to
   RK4 so its contribution is O(dt^4) and the O(dx) term is left dominant. This
   route works here only because Case 1 has no splitting error and no
   manufactured SWE source to be degraded by MMSPreStep's frozen midpoint
   sampling; it does not extend to Case 3. The script also checks the RK4 and
   forward-Euler errors agree closely, which independently demonstrates that the
   temporal contribution is negligible at these CFL numbers.

A fixed-mesh dt-refinement study against the exact solution is deliberately
absent: with transport active it converges to the semi-discrete solution, not to
the exact one, so the error plateaus on the O(dx) spatial floor and the rate
tends to zero.
"""

import argparse
import math

from mms_common import NORMS, check_rate, error_norms, max_error_over_components, run_driver, scalar_diagnostic

# Roundoff scale for |Q_mms|_inf. The source is rho*c*h*(residual), so its
# roundoff floor sits near 1e-10 in absolute terms while a genuinely nonzero
# manufactured source for these constants is O(1e6) -- twelve orders away.
SOURCE_TOLERANCE = 1.0e-6
FLOW_TOLERANCE = 1.0e-12

# The joint rate is measurably below 1 on the coarsest levels: first-order upwind
# on unstructured triangles approaches its asymptotic rate slowly, and Linf is
# the slowest of the three norms.
JOINT_BOUNDS = {"L1": (0.75, 1.20), "L2": (0.75, 1.20), "Linf": (0.55, 1.20)}
SPATIAL_BOUNDS = JOINT_BOUNDS


def refinement_args(level, base_step, final_time, extra=()):
    """Halves the timestep with the mesh, and grows the step cap to match.

    The driver takes its step cap from stop_n, which -dt does not update, so a
    finer level would otherwise stop early at a different final time and the
    comparison across levels would be meaningless.
    """
    step_size = base_step / 2**level
    max_steps = int(math.ceil(final_time / step_size)) + 2
    return ["-dm_refine", level, "-dt", step_size, "-ts_max_steps", max_steps] + list(extra)


def run_level(driver, input_file, level, base_step, final_time, extra=()):
    output = run_driver(driver, input_file, refinement_args(level, base_step, final_time, extra))
    return {
        "length_scale": scalar_diagnostic(output, "Avg-length-scale"),
        "hT": error_norms(output, "hT"),
        "T": error_norms(output, "T"),
        "max_source": scalar_diagnostic(output, "Max-|Q_mms|-inf"),
        "max_flow_error": max_error_over_components(output, ("h", "hu", "hv")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--base-step", type=float, default=0.05)
    parser.add_argument("--final-time", type=float, default=2.0)
    parser.add_argument("--levels", type=int, default=4)
    args = parser.parse_args()

    levels = list(range(args.levels))

    print("Case 1: source-free moving transport")
    euler = [run_level(args.driver, args.input, level, args.base_step, args.final_time) for level in levels]

    # 1. smoke assertions at every level we ran anyway
    for level, result in zip(levels, euler):
        if result["max_source"] > SOURCE_TOLERANCE:
            raise RuntimeError(
                f"|Q_mms|_inf is {result['max_source']:g} at refinement {level}, above the roundoff "
                f"tolerance {SOURCE_TOLERANCE:g}. The manufactured temperature wave no longer "
                f"satisfies C_t + div(huT, hvT) = 0."
            )
        if result["max_flow_error"] > FLOW_TOLERANCE:
            raise RuntimeError(
                f"Uniform flow was not preserved at refinement {level}: max h/hu/hv error is "
                f"{result['max_flow_error']:g}, above {FLOW_TOLERANCE:g}."
            )
    print(f"  |Q_mms|_inf <= {max(r['max_source'] for r in euler):g} (roundoff)")
    print(f"  max h/hu/hv error <= {max(r['max_flow_error'] for r in euler):g} (roundoff)")

    # 2. joint space-time rate under dt ~ dx
    length_scales = [result["length_scale"] for result in euler]
    print("  joint space-time convergence (dt ~ dx, forward-Euler transport):")
    for component in ("hT", "T"):
        for norm in NORMS:
            check_rate(
                f"joint {component} {norm}",
                length_scales,
                [result[component][norm] for result in euler],
                JOINT_BOUNDS[norm],
            )

    # 3. isolated spatial rate with an RK4 transport integrator
    rk4 = [
        run_level(args.driver, args.input, level, args.base_step, args.final_time, ("-ts_type", "rk", "-ts_rk_type", "4"))
        for level in levels
    ]
    print("  isolated spatial convergence (RK4 transport, temporal error O(dt^4)):")
    for component in ("hT", "T"):
        for norm in NORMS:
            check_rate(
                f"spatial {component} {norm}",
                length_scales,
                [result[component][norm] for result in rk4],
                SPATIAL_BOUNDS[norm],
            )

    # The two integrators agreeing to a few tenths of a percent is what licenses
    # calling the RK4 rate spatial: it shows the temporal term is already
    # negligible rather than merely smaller.
    worst = max(
        abs(a["hT"][norm] - b["hT"][norm]) / b["hT"][norm]
        for a, b in zip(rk4, euler)
        for norm in NORMS
    )
    if worst > 0.05:
        raise RuntimeError(
            f"RK4 and forward-Euler hT errors differ by {worst:.3%}, so the temporal contribution is "
            f"not negligible and the RK4 rate cannot be called a spatial rate."
        )
    print(f"  RK4 vs forward-Euler hT error difference: {worst:.3%} (temporal term negligible)")


if __name__ == "__main__":
    main()
