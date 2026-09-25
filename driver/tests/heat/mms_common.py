"""Shared helpers for the heat MMS convergence checkers.

The three moving-water studies differ in what they refine, but they all run the
MMS driver, scrape componentwise error norms out of its output, and fit a rate.
The plateau check is the reason this lives in one place: refining one
discretization parameter while another holds the error at a floor produces a
sequence that still admits a least-squares slope, and reporting that slope as a
convergence rate is exactly the failure mode these studies exist to avoid.
"""

import math
import re
import subprocess


def _norm_regex(label, prefix=""):
    return re.compile(
        r"^\s*" + prefix + re.escape(label) + r"\s*:\s+L1\s*=\s*([+\-0-9.eE]+).*?"
        r"L2\s*=\s*([+\-0-9.eE]+).*?Linf\s*=\s*([+\-0-9.eE]+)",
        re.MULTILINE,
    )


NORMS = ("L1", "L2", "Linf")


def run_driver(driver, input_file, args, mpi_prefix=()):
    """Runs the MMS driver and returns its combined output."""
    command = list(mpi_prefix) + [driver, input_file] + [str(arg) for arg in args]
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    output = result.stdout + result.stderr
    if result.returncode:
        raise RuntimeError(f"{' '.join(command)} failed:\n{output}")
    return output


def error_norms(output, label, prefix=""):
    """Extracts the last reported {L1, L2, Linf} norms for one component.

    `prefix` selects the reference-difference rows ("ref ") rather than the
    error-against-exact rows.
    """
    matches = _norm_regex(label, prefix).findall(output)
    if not matches:
        raise RuntimeError(f"Could not find '{prefix}{label}' error norms in output:\n{output}")
    return dict(zip(NORMS, (float(value) for value in matches[-1])))


def max_error_over_components(output, labels):
    """Largest norm reported for any of the given components, used for smoke checks."""
    return max(value for label in labels for value in error_norms(output, label).values())


def scalar_diagnostic(output, label):
    match = re.search(r"^\s*" + re.escape(label) + r"\s*:\s*([+\-0-9.eE]+)", output, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not find the '{label}' diagnostic in output:\n{output}")
    return float(match.group(1))


def fit_rate(step_sizes, errors):
    """Least-squares slope of log(error) against log(step size)."""
    x = [math.log(value) for value in step_sizes]
    y = [math.log(value) for value in errors]
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    return sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / sum((a - x_mean) ** 2 for a in x)


def assert_no_plateau(description, step_sizes, errors, min_ratio=1.15):
    """Fails if the error stops responding to refinement.

    Each successive step size halves, so a converging first-order sequence
    roughly halves the error too. A consecutive ratio near 1 means the error has
    hit a floor set by some *other* discretization parameter, and a fitted slope
    over such a sequence is meaningless. Report that as a plateau rather than
    letting it surface as a low rate.

    The threshold separates "not converging" from "converging slowly", not
    "first order" from "less than first order": Linf on unstructured triangles
    legitimately shows ratios near 1.3 on the coarsest levels, while a genuine
    floor pins the ratio at essentially 1.
    """
    for coarse, fine, e_coarse, e_fine in zip(step_sizes, step_sizes[1:], errors, errors[1:]):
        ratio = e_coarse / e_fine if e_fine else math.inf
        if ratio < min_ratio:
            raise RuntimeError(
                f"PLATEAU: {description} error barely changed from {coarse:g} to {fine:g} "
                f"({e_coarse:g} -> {e_fine:g}, ratio {ratio:.3f} < {min_ratio}). The error has "
                f"reached a floor set by another discretization parameter; no convergence rate "
                f"can be measured from this sequence."
            )


def check_rate(description, step_sizes, errors, bounds, min_ratio=1.15):
    """Reports a rate, having first ruled out a plateau."""
    assert_no_plateau(description, step_sizes, errors, min_ratio)
    rate = fit_rate(step_sizes, errors)
    low, high = bounds
    if not low <= rate <= high:
        raise RuntimeError(f"{description} rate {rate:.6g} is outside [{low}, {high}]; errors={errors}")
    print(f"  {description}: rate={rate:.4f}, errors={[f'{e:.6g}' for e in errors]}")
    return rate
