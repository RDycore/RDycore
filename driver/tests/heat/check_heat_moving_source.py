#!/usr/bin/env python3
"""Case 2: moving transport plus a nonzero prescribed heat source.

Measures the temporal-plus-splitting order of the complete Lie-split update by
*self-convergence against a same-mesh fine-timestep reference*, not against the
exact solution. That distinction is the whole point of this script. With
transport active, refining dt on a fixed mesh does not drive the error against
the exact solution to zero: the solution converges to the semi-discrete
(method-of-lines) solution, whose distance from the exact solution is the O(dx)
upwind transport error. A study against the exact solution would therefore
plateau on that spatial floor and report a rate near zero. Differencing two runs
that share a spatial discretization cancels the floor exactly and leaves the
temporal and splitting error alone.

Expected result: first order for *both* backward Euler and Crank-Nicolson. Lie
splitting is first order regardless of how accurately the second solve is
integrated, so Crank-Nicolson lowers the error constant without raising the
order. Case 1 has no splitting error at all (its source vector field is
identically zero), which is why the two cases measure different things.

The reference is only meaningful if it is itself converged, so the script
recomputes it at half the reference timestep and confirms the reported errors
barely move.
"""

import argparse
import math

from mms_common import NORMS, check_rate, error_norms, fit_rate, run_driver

METHODS = ("beuler", "cn")
RATE_BOUNDS = (0.85, 1.15)

# At first order the reference's own temporal error biases the finest measured
# point by roughly dt_ref/dt_min, so the reference must be far below the finest
# measured step -- dt_min/8 would already bias that point by ~12%.
REFERENCE_FACTOR = 32

# Halving dt_ref must not move the fitted rate by more than a small fraction of
# the rate tolerance, or the reference is not converged enough to support the
# order claim.
REFERENCE_RATE_TOLERANCE = 0.05
REFERENCE_ERROR_TOLERANCE = 0.05


def solve_args(level, step_size, final_time, method, extra=()):
    max_steps = int(math.ceil(final_time / step_size)) + 2
    return ["-dm_refine", level, "-dt", step_size, "-ts_max_steps", max_steps, "-heat_ts_type", method] + list(extra)


def make_reference(driver, input_file, level, step_size, final_time, method, path):
    run_driver(driver, input_file, solve_args(level, step_size, final_time, method, ("-mms_save_final_state", path)))
    return path


def reference_errors(driver, input_file, level, step_sizes, final_time, method, reference):
    """Difference norms of each dt against the shared fine-timestep reference."""
    errors = {norm: [] for norm in NORMS}
    for step_size in step_sizes:
        output = run_driver(
            driver, input_file, solve_args(level, step_size, final_time, method, ("-mms_reference_solution", reference))
        )
        norms = error_norms(output, "hT", prefix="ref ")
        for norm in NORMS:
            errors[norm].append(norms[norm])
    return errors


def exact_errors(driver, input_file, level, step_size, final_time, method):
    output = run_driver(driver, input_file, solve_args(level, step_size, final_time, method))
    return error_norms(output, "hT")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--level", type=int, default=1, help="fixed mesh refinement level for the study")
    parser.add_argument("--base-step", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--final-time", type=float, default=2.0)
    parser.add_argument(
        "--no-validate-reference",
        action="store_true",
        help="skip recomputing the reference at half its timestep (faster, but the order claim is then unbacked)",
    )
    args = parser.parse_args()

    step_sizes = [args.base_step / 2**i for i in range(args.steps)]
    reference_step = step_sizes[-1] / REFERENCE_FACTOR

    print("Case 2: moving transport plus a prescribed heat source")
    print(f"  fixed mesh at refinement {args.level}; reference dt = {reference_step:g} (dt_min/{REFERENCE_FACTOR})")

    rates = {}
    for method in METHODS:
        reference = make_reference(
            args.driver, args.input, args.level, reference_step, args.final_time, method, f"reference_{method}.bin"
        )
        errors = reference_errors(args.driver, args.input, args.level, step_sizes, args.final_time, method, reference)
        print(f"  {method} temporal + splitting convergence (vs same-mesh reference):")
        rates[method] = {
            norm: check_rate(f"{method} hT {norm}", step_sizes, errors[norm], RATE_BOUNDS) for norm in NORMS
        }

        if args.no_validate_reference:
            continue

        # Validate the reference by halving its timestep: if the reported errors
        # and the fitted rate barely move, the reference's own temporal error is
        # not contaminating the measurement.
        finer = make_reference(
            args.driver,
            args.input,
            args.level,
            reference_step / 2,
            args.final_time,
            method,
            f"reference_{method}_half.bin",
        )
        finer_errors = reference_errors(args.driver, args.input, args.level, step_sizes, args.final_time, method, finer)
        for norm in NORMS:
            shift = max(
                abs(a - b) / b for a, b in zip(finer_errors[norm], errors[norm])
            )
            rate_shift = abs(fit_rate(step_sizes, finer_errors[norm]) - rates[method][norm])
            if shift > REFERENCE_ERROR_TOLERANCE or rate_shift > REFERENCE_RATE_TOLERANCE:
                raise RuntimeError(
                    f"{method} reference at dt = {reference_step:g} is not converged: halving it moved the "
                    f"{norm} errors by {shift:.3%} and the rate by {rate_shift:.4f}."
                )
            print(f"    reference check ({norm}): errors moved {shift:.3%}, rate moved {rate_shift:.4f}")

    # Backward Euler samples the manufactured source at the right endpoint while
    # Crank-Nicolson uses the endpoint average. In the prescribed-source branch
    # the heat residual has no state dependence, so every consistent one-step
    # method gives the same update and that quadrature is the *only* thing the TS
    # type changes -- identical errors would mean the branch is dead code.
    coarse = {method: exact_errors(args.driver, args.input, args.level, step_sizes[0], args.final_time, method) for method in METHODS}
    for norm in NORMS:
        if math.isclose(coarse["beuler"][norm], coarse["cn"][norm], rel_tol=1.0e-6):
            raise RuntimeError(f"Backward Euler and Crank-Nicolson produced identical coarse-step {norm} errors")
    print(f"  backward Euler and Crank-Nicolson differ at dt = {step_sizes[0]:g} (source quadrature is live)")


if __name__ == "__main__":
    main()
