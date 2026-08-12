#!/usr/bin/env python3
"""Smoke assertions for the moving-water heat MMS cases (Cases 1 and 2).

Running the driver and checking only its exit status would pass no matter how
large the errors were, since a single-resolution MMS run prints norms rather than
judging them. This script turns the printed norms into assertions, across both
rank counts and whichever operator backend it is pointed at:

* the uniform flow must be preserved to roundoff (h, hu, hv), which is what makes
  Case 1 an isolated test of passive heat transport;
* the manufactured heat source must be zero to roundoff in the source-free case
  and genuinely nonzero in the source case, so that neither case silently
  degenerates into the other; and
* the two rank counts must agree, since the partitioning changes only the order
  of operations.
"""

import argparse
import shlex

from mms_common import NORMS, error_norms, max_error_over_components, run_driver, scalar_diagnostic

FLOW_TOLERANCE = 1.0e-12
ZERO_SOURCE_TOLERANCE = 1.0e-6
NONZERO_SOURCE_FLOOR = 1.0e3
# Repartitioning reorders the flux summation but changes nothing else, so the
# rank counts should agree far more tightly than this.
RANK_AGREEMENT_TOLERANCE = 1.0e-8


def run(driver, input_file, np, mpiexec, extra):
    output = run_driver(driver, input_file, extra, mpi_prefix=shlex.split(mpiexec) + ["-n", str(np)])
    return {
        "hT": error_norms(output, "hT"),
        "T": error_norms(output, "T"),
        "max_source": scalar_diagnostic(output, "Max-|Q_mms|-inf"),
        "max_flow_error": max_error_over_components(output, ("h", "hu", "hv")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--source", choices=("zero", "nonzero"), required=True,
                        help="whether the manufactured heat source is expected to vanish (Case 1) or not (Case 2)")
    parser.add_argument("--driver-args", default="", help="extra arguments passed through to the driver")
    args = parser.parse_args()

    extra = shlex.split(args.driver_args)
    results = {np: run(args.driver, args.input, np, args.mpiexec, extra) for np in args.ranks}

    for np, result in results.items():
        if result["max_flow_error"] > FLOW_TOLERANCE:
            raise RuntimeError(
                f"np={np}: uniform flow was not preserved -- max h/hu/hv error is "
                f"{result['max_flow_error']:g}, above {FLOW_TOLERANCE:g}."
            )
        if args.source == "zero" and result["max_source"] > ZERO_SOURCE_TOLERANCE:
            raise RuntimeError(
                f"np={np}: |Q_mms|_inf is {result['max_source']:g}, above the roundoff tolerance "
                f"{ZERO_SOURCE_TOLERANCE:g}. The manufactured temperature wave no longer satisfies "
                f"C_t + div(huT, hvT) = 0."
            )
        if args.source == "nonzero" and result["max_source"] < NONZERO_SOURCE_FLOOR:
            raise RuntimeError(
                f"np={np}: |Q_mms|_inf is only {result['max_source']:g}, below {NONZERO_SOURCE_FLOOR:g}. "
                f"This case is supposed to exercise a substantial prescribed heat source."
            )
        print(
            f"  np={np}: max h/hu/hv error {result['max_flow_error']:g}, "
            f"|Q_mms|_inf {result['max_source']:g}, hT L1 {result['hT']['L1']:g}, T L1 {result['T']['L1']:g}"
        )

    baseline = results[args.ranks[0]]
    for np in args.ranks[1:]:
        for component in ("hT", "T"):
            for norm in NORMS:
                a, b = results[np][component][norm], baseline[component][norm]
                if abs(a - b) > RANK_AGREEMENT_TOLERANCE * max(abs(b), 1.0e-30):
                    raise RuntimeError(
                        f"np={np} and np={args.ranks[0]} disagree on {component} {norm}: {a:g} vs {b:g}"
                    )
    if len(args.ranks) > 1:
        print(f"  rank counts {args.ranks} agree to within {RANK_AGREEMENT_TOLERANCE:g} relative")


if __name__ == "__main__":
    main()
