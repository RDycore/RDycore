#!/usr/bin/env python3
"""Case 0: method-dependent temporal convergence of the isolated MMS heat source.

The flow is stationary here (u = v = 0, h constant), so the transport operator
contributes nothing and the spatial error is identically zero. That is what makes
a fixed-mesh dt-refinement study against the *exact* solution valid for this case
and invalid once transport is active -- see check_heat_moving_source.py, which
uses a same-mesh reference instead for exactly that reason.

What this measures is the manufactured source *quadrature*, not the heat TS
itself. In the prescribed-source branch the heat residual is f = udot - S/(rho c)
with S a per-cell array that does not vary with the stage time, so every
consistent one-step method produces the same update and backward Euler and
Crank-Nicolson differ only in what MMSPostStep bakes into the source (right
endpoint versus trapezoidal average). The identical-error guard at the end checks
precisely that the branch is live.
"""

import argparse
import math

from mms_common import NORMS, check_rate, error_norms, max_error_over_components, run_driver

# Case 0 has h identically 1, so the conservative hT error and the derived T
# error coincide; both are checked to keep the two reporting paths honest.
COMPONENTS = ("hT", "T")
FLOW_TOLERANCE = 1.0e-10
EXPECTED = {"beuler": (0.85, 1.15), "cn": (1.8, 2.2)}


def run_case(driver, input_file, method, step_size, final_time):
    max_steps = math.ceil(final_time / step_size) + 1
    output = run_driver(driver, input_file, ["-dt", step_size, "-ts_max_steps", max_steps, "-heat_ts_type", method])

    max_flow_error = max_error_over_components(output, ("h", "hu", "hv"))
    if max_flow_error > FLOW_TOLERANCE:
        raise RuntimeError(
            f"Lake-at-rest flow error {max_flow_error:g} exceeds tolerance for {method}, dt={step_size:g}"
        )

    return {component: error_norms(output, component) for component in COMPONENTS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--final-time", type=float, default=20.0)
    args = parser.parse_args()

    step_sizes = [0.1, 0.05, 0.025, 0.0125]
    all_errors = {}
    for method, bounds in EXPECTED.items():
        case_errors = [run_case(args.driver, args.input, method, step_size, args.final_time) for step_size in step_sizes]

        print(f"{method}:")
        all_errors[method] = {}
        for component in COMPONENTS:
            for norm in NORMS:
                errors = [errors[component][norm] for errors in case_errors]
                all_errors[method][(component, norm)] = errors
                check_rate(f"{method} {component} {norm}", step_sizes, errors, bounds)

    for key in all_errors["beuler"]:
        if math.isclose(all_errors["beuler"][key][0], all_errors["cn"][key][0], rel_tol=1.0e-6):
            raise RuntimeError(f"Backward Euler and Crank-Nicolson produced identical coarse-step {key} errors")


if __name__ == "__main__":
    main()
