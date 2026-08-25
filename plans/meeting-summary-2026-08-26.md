# Harvey Manning calibration — state for the Aug 26 meeting

One page per topic; every number is from a committed, reproducible run
(`plans/RESULTS-gpu-implicit.md` has the full entries). The paper's two
temporary sections carry the same content in reviewer form.

## 1. The baseline is two populations, and one of them is a model defect

72-hr rain-forced forward, NLCD prior, free outflow, beuler dt 1 s,
2.93M cells, **converged inner tolerances** (ksp 1e-4/snes 1e-5;
259,200 implicit solves, zero failures). Peak-WSE against the 108
QC-passed USGS marks:

| population | n | MAE | bias |
| --- | --- | --- | --- |
| crest inside the window | 71 | **1.51 m** | +1.20 m |
| never crest | 37 | 7.06 m | +7.06 m (all model-high) |
| all | 108 | 3.41 m | +3.21 m |

The 37 are not late-cresting — they are a **structural drainage
failure**: every one peaks at hour 72 with exactly zero recession,
still filling at +0.02–0.06 m/hr while the other 71 are already
draining; mean standing water 10.7 m and rising, arriving laterally
after the rain ends. The downstream Buffalo Bayou reach has no working
exit at 30 m. No Manning value repairs a missing exit.

**Decision asked of no one — position taken:** the calibration target
is the 71 admissible marks (1.51 m baseline); the drainage defect is
reported as a finding and deserves a look at outlet placement /
conveyance before any longer-window run. (Consistent with the existing
exclusion of the eight reservoir-influenced gauges.)

**For discussion:** whether both numbers (3.41 / 1.51) are reported
side-by-side in the paper. Current draft: yes, with the decomposition.

## 2. Window placement (from the measured crest times)

Crests are bimodal (h29–42 and h60–72 clusters). No 12-hr window
covers them: best 12-hr (h29–41) captures 61% of genuine crests; 24-hr
(h29–53) 68%; only 48-hr reaches 100%. Hourly restart checkpoints
exist at every candidate start (validated bit-exact on the production
GPU config), so window trials cost no new forwards. Marks whose crest
falls outside a chosen window contribute a window-edge residual, not a
peak — they need explicit zero-weighting or inclusion by choice.

## 3. Noise says: stronger regularization, longer window

Basin twin at HWM-grade noise (0.15 m, 1-hr window): misfit falls
9.4×, MAE 0.23→0.12 m — while 5 of 15 classes pin to bounds (pasture,
428k cells, to the 0.30 ceiling; three classes to the floor). At
signal ≈ noise the optimizer buys misfit with unphysical parameters;
the bound-pinning is the visible symptom. β = 1e-4 is too weak at this
window length. A lower MAE with pinned classes is a worse result, not
a better one — proposed report: MAE **plus** the roughness field's
physicality, with discrepancy-principle stopping as the follow-up.

## 4. The gradient question (the honesty item)

Status, measured today at basin scale, converged tolerances:

- The peak-misfit FD gate **fails** at basin scale (2.1e-1 on a
  domain-wide direction, ~2e-3 on single-class directions) and the
  failure does **not** shrink as inner solves tighten.
- The instrumented gate localizes it **below the observable**: zero
  argmax moves, zero wet/dry flips at the marks across every probe;
  evaluations reproduce to nine digits (noise excluded); error scales
  with direction support; |FD| < |adjoint| on the wide direction.
- Leading hypothesis: **branch switching in the wet/dry dynamics**
  (probe shifts which cells cross the depth cutoff mid-trajectory) —
  i.e., the objective is nonsmooth at a level the adjoint correctly
  one-sides but a central difference averages. Alternatives not yet
  excluded: genuine curvature (ε² truncation), support-correlated
  defect.
- Discriminators in flight: probe-step sweep (kink ~ε, curvature ~ε²,
  defect flat), all-15-class support sweep (error vs class area over
  4 decades), second-difference analysis (kink ~1/ε).
  **[Slot: verdicts land tonight — o39 / o41.]**

What this does and does not touch: the forward results (baseline,
drainage) involve no adjoint. The twin calibrations descended and
recovered area-dominant classes, consistent with a gradient correct up
to kink-scale detail. But the paper now says plainly: at basin scale
the gradient is validated by descent behavior, not by the FD gate,
until this closes.

## 5. Solver settings (settled; no discussion needed)

`ksp 1e-4 / snes 1e-5` reproduces the fully converged answer exactly
and is the production setting; the old loose setting changed the 72-hr
baseline by 1e-4 relative and moved zero of 108 peak steps (the
tolerance concern is closed). Tightening 1e-2→1e-3 is *faster* (fewer
Newton iterations); there is no speed argument for loose. The claim
that "1e-3 passes the FD gates" is withdrawn — see item 4.

## 6. Staged and ready

- `o35_calibrate_job.sh` on PM: parameterized calibration
  (`START_HOUR WINDOW_HOURS TAO_ITS [INIT_FILE]`), restarts from the
  converged-tolerance hourly checkpoints, chainable across queue slots
  (`-adjoint_classes_dump/init`, warm start exact in the parameters,
  gated by ctests). Needs: window choice (item 2), β (item 3), and the
  gradient verdict (item 4) before burning ~20–30 node-hr.
- Machinery all landed and gated: mid-event window IC (bit-exact),
  free-outflow BC (259,200 solves/0 fail), HWM misfit + eval-only +
  per-mark dump, failure-tolerant trials, kink-probe instrumentation.

Budget: ~22 of ≤100 GPU-node-hr spent across everything to date.
