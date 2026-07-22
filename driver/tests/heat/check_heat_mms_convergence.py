#!/usr/bin/env python3
"""Check method-dependent temporal convergence of the isolated MMS heat source."""

import argparse
import math
import re
import subprocess


TEMPERATURE_ERROR = re.compile(
    r"^\s*temperature:\s+L1\s*=\s*([+\-0-9.eE]+).*?"
    r"L2\s*=\s*([+\-0-9.eE]+).*?Linf\s*=\s*([+\-0-9.eE]+)",
    re.MULTILINE,
)
FLOW_ERROR = re.compile(
    r"^\s*(h|hu|hv)\s*:\s+L1\s*=\s*([+\-0-9.eE]+).*?"
    r"L2\s*=\s*([+\-0-9.eE]+).*?Linf\s*=\s*([+\-0-9.eE]+)",
    re.MULTILINE,
)


def convergence_rate(step_sizes, errors):
    x = [math.log(value) for value in step_sizes]
    y = [math.log(value) for value in errors]
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    return sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / sum(
        (a - x_mean) ** 2 for a in x
    )


def run_case(driver, input_file, method, step_size, final_time):
    max_steps = math.ceil(final_time / step_size) + 1
    command = [
        driver,
        input_file,
        "-dt",
        str(step_size),
        "-ts_max_steps",
        str(max_steps),
        "-heat_ts_type",
        method,
    ]
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    output = result.stdout + result.stderr
    if result.returncode:
        raise RuntimeError(f"{' '.join(command)} failed:\n{output}")

    temperature_matches = TEMPERATURE_ERROR.findall(output)
    if not temperature_matches:
        raise RuntimeError(f"Could not find the temperature error in output:\n{output}")

    flow_matches = FLOW_ERROR.findall(output)
    if len(flow_matches) < 3:
        raise RuntimeError(f"Could not find all flow error norms in output:\n{output}")
    max_flow_error = max(float(value) for match in flow_matches[-3:] for value in match[1:])
    if max_flow_error > 1.0e-10:
        raise RuntimeError(
            f"Lake-at-rest flow error {max_flow_error:g} exceeds tolerance for {method}, dt={step_size:g}"
        )

    return {
        norm: float(value)
        for norm, value in zip(("L1", "L2", "Linf"), temperature_matches[-1])
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--final-time", type=float, default=20.0)
    args = parser.parse_args()

    step_sizes = [0.1, 0.05, 0.025, 0.0125]
    expected = {"beuler": (0.85, 1.15), "cn": (1.8, 2.2)}
    all_errors = {}
    for method, bounds in expected.items():
        case_errors = [
            run_case(args.driver, args.input, method, step_size, args.final_time)
            for step_size in step_sizes
        ]
        all_errors[method] = {
            norm: [errors[norm] for errors in case_errors]
            for norm in ("L1", "L2", "Linf")
        }

        print(f"{method}:")
        for norm, errors in all_errors[method].items():
            rate = convergence_rate(step_sizes, errors)
            if not bounds[0] <= rate <= bounds[1]:
                raise RuntimeError(
                    f"{method} temperature {norm} rate {rate:.6g} is outside "
                    f"[{bounds[0]}, {bounds[1]}]; errors={errors}"
                )
            print(f"  {norm}: rate={rate:.6f}, errors={errors}")

    for norm in ("L1", "L2", "Linf"):
        if math.isclose(
            all_errors["beuler"][norm][0],
            all_errors["cn"][norm][0],
            rel_tol=1.0e-6,
        ):
            raise RuntimeError(
                f"Backward Euler and Crank-Nicolson produced identical coarse-step {norm} errors"
            )


if __name__ == "__main__":
    main()
