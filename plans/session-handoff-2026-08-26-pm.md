# Session Handoff — August 26, 2026 (afternoon)

Branch `adams/gpu-implicit`, committed and pushed through **`4a4d792d`**.
Laptop tree clean, 21 adjoint/calibration ctests pass (19 previous plus 2
new). PM tree at the same commit, built as **`build-claude-gpu6`**.

**Read first:** `plans/PROJECT-STATE-2026-08-26.md` (standing top-level
document) and `plans/session-handoff-2026-08-26.md` (the morning's
operational layer). This file records the one thing that changed:
**Section 3 of that handoff — "get a calibration to run to completion" —
is now closed in code and running on the machine.**

---

## 1. What was wrong, and the evidence it is fixed

o43 did zero TAO iterations. The diagnosis was already on the record;
what was missing was a cheap way to test it and a fix. Both exist now.

**`o48_smoke.sh`** runs the *exact* production wiring — 2.93M cells, GPU
types, the 46 real cluster-A marks, the h29 checkpoint, the rain offset —
over a 600-step window instead of 43,200, so a control costs five minutes
instead of twelve hours. Both cases start from the NLCD prior and report
the same J to every digit, so they differ only in the optimization
problem:

| | first step | TAO its | J | MAE |
| --- | --- | --- | --- | --- |
| A: o43 setup (variable n, beta 1e2) | ‖g‖ = 667 | **0** | 702.20 -> 702.20 | 0.6630 -> 0.6630 |
| B: new (variable alpha, sigma_n 0.015) | ‖g‖/J0 = **0.10** | 1 | 702.20 -> **681.50** | 0.6630 -> **0.6127** |

The o43 failure was never about the long window or the real data. It was
a scaling defect, and the adjoint was never implicated — the FD gates
always passed.

## 2. The driver changes (48485e24, 4a4d792d)

- **`-adjoint_classes_relative`** — the optimizer's variable is
  alpha_k = n_k/n_prior_k and its objective is J/J(start). Both O(1), so
  BLMVM's first trial `x - g` is a ~10% roughness change instead of five
  orders too large. Bounds become alpha in [0.3, 3.0]
  (`-adjoint_classes_alpha_min/max`), a physical statement that also
  excludes the region where the implicit solve diverges — which a bound
  of n > 0.005 did not.
- **`-adjoint_sigma_n <s>`** — beta = 1/s^2 from a prior standard
  deviation on n. Report sigma_n, never a bare beta.
- **`-adjoint_classes_grad_dump <file>`** — (code, n, dJ/dn) at the start
  point, written *before* TAO runs. Both the fallback (an explicit line
  search along -g as eval-only forwards, 36 min instead of 2 hr per
  trial) and the diagnostic that aims `-adjoint_classes_active`.
- **Per-iteration dump.** `-adjoint_classes_dump` now writes after every
  TAO iteration via `TaoMonitorSet`, so a job stopped by the 12-hour wall
  still leaves the chain a start point. Kill-tested.
- **`-adjoint_jred_gate` now covers the classes branch.** o43 reported a
  "final" objective equal to its initial one and no test caught it.

Parameter files stay in physical Manning units in both modes, so
dump/init chains across them.

**Verification.** The alpha gradient passes the central-difference gate
at 2.3e-6 (new ctest, eps 1e-4; the default 1e-3 probe is truncation-
limited at 3e-5, and the error scales as eps^2 across 3e-5..3e-2, which
is what says truncation rather than a wrong chain rule). The dJ/dn
recovered from the alpha gradient equals the absolute mode's gradient at
the same point to 1e-8 — the dump's precision — class by class. New
ctests: `adjoint_classes_relative_fd_np_1`, `adjoint_classes_relative_np_1`.

## 3. Why sigma_n = 0.015

beta is dimensional; the number has to come from somewhere. Against the
measured J(alpha) curve (o44/o45) with sum_k n_prior_k^2 = 0.1399:

| sigma_n | beta | uniform-mode minimum | J_mis | reg | MAE |
| --- | --- | --- | --- | --- | --- |
| 0.010 | 10000 | alpha 0.88 | 790.0 | 9.7 | ~0.708 |
| **0.015** | **4444** | **alpha 0.70** | **757.3** | **28.6** | **~0.690** |
| 0.020 | 2500 | alpha 0.33 (bound) | 678.6 | 79.5 | ~0.643 |

sigma_n = 0.015 is about the loosest prior leaving an interior minimum in
the dominant identifiable direction. At 0.020 the problem runs to the
bound again — the same pathology beta = 1e2 had at a relative roughness
change of 13.6. **This is still an open question for the science team**
(PROJECT-STATE S6 Q3): how far can the NLCD lookup plausibly be wrong?
The answer sets this directly.

## 4. Running / queued on PM

| job | what |
| --- | --- |
| `57624518` | `o48_calibrate_rel.sh 8` — 15-class calibration, relative + sigma_n 0.015, 12-hr slot |
| `57624199` | `o49_alpha_bar.sh` — eval-only forwards at alpha 0.7 and 0.8 |

`o48` writes `o48_p_<jobid>.txt` (parameters, physical n, rewritten every
iteration) and `o48_g_<jobid>.txt` (start-point gradient, written in the
first ~2 hr). Continue with
`sbatch o48_calibrate_rel.sh <its> o48_p_<jobid>.txt`.

**Expected outcome, which is the point:** descent to roughly the
alpha-curve level, MAE 0.7188 -> ~0.69. A much larger improvement would
be surprising and worth distrusting. Classes reaching the alpha bounds is
the documented over-parameterization result (o28/o30/o47), not a bug —
and the cue to run the staged `o50_calibrate_active.sh <its> <codes>`,
aimed by the classes with the largest |dJ/dn| * n_prior in the gradient
dump (that product is the gradient in the variable being optimized).

## 5. What to watch for

- **Line-search overshoot.** In the smoke's single iteration BLMVM
  expanded well past `-g`: classes 23 and 24 hit the alpha 0.3 bound on
  the first accepted step while J fell only 3% and ‖g‖ rose. On a
  600-step window the peaks are barely resolved so this is not a
  production result, but if the same thing happens on the real window,
  `-tao_ls_type armijo` (which does not expand) or tighter alpha bounds
  are the levers.
- **Per-class balance.** sigma_n = 0.015 was chosen from the *uniform*
  direction. Individual weakly-constrained classes can still be
  unbounded — that is the over-parameterization story, not a defect. The
  gradient dump makes checking it arithmetic rather than another 12-hour
  experiment.

## 6. Cost basis update

- The Turning mesh **does not strong-scale past one node**: 40 TSSteps
  take 14.4 s on 4 ranks and 40.3 s on 64 (`b2i_dev_n4.log` vs
  `b2i_dev_n64.log`, same problem — per-rank flops drop exactly 16x).
  Run production on `-N 1 / srun -n 4 -G 4` and buy iterations by
  chaining slots, not by adding nodes. n8/n16 was never measured.
- Budget: ~35 node-hr used before today; today's smoke and builds are
  under 1, and the two queued jobs are ~12 and ~2.

## 7. Builds

PM `build-claude-gpu6` (current, has the per-iteration dump);
`build-claude-gpu5` is one commit behind and is what `o49` will launch
from — harmless, it uses only the unchanged eval-only path. Laptop
`build-claude`.
