# Manning paper: state of play for a writing session (2026-09-16)

*Read this first, then `plans/RESULTS-gpu-implicit.md` sections o62-o66 and
`plans/o63-gauge-weight-audit.md`. Everything below traces to a file in
`logs/` or a line in RESULTS. Paper: `papers/manning-calibration/manning-calibration.tex`,
32 pp, builds with latexmk. Branch `adams/gpu-implicit`, pushed, tree clean.*

## The one-paragraph version

Roughness can explain at most ~15% of a 0.72 m model-survey error on the 46
cluster-A high-water marks. The marks determine about three combinations of
the fifteen NLCD classes at a +/-30% prior, and what those three reach is
below the lookup's own range. The gauges cannot arbitrate: 99.5% of the gauge
misfit is a constant per-gauge offset, which roughness cannot produce. So the
error is in the water balance and the mesh, not the friction.

## What is SETTLED and may be written

1. **The ceiling.** Roughness accounts for 15% of the 0.72 m mark error; a
   single global factor at most 0.08 m, and only below any lookup entry.
2. **The mark spectrum, now validated.** lambda = 12.0, 2.98, 1.13, ... at
   sigma_alpha 0.30 -> three supported combinations. NEW 09-15: the
   sensitivity matrix reproduces the o62 adjoint gradient to within 4% on
   every class (22 -28.4 vs -29.5, 23 +89.7 vs +90.7, 90 +65.1 vs +66.2).
   Sec 6.4's construction has an independent check for the first time; say so.
3. **The three-class field is determined and indefensible.** Same answer for
   23 and 90 from three priors (n = 0.036, 0.029, below developed-open's
   0.040); developed-low is the prior's, moving 1.63x -> 3x while MAE moves
   0.002 m.
4. **The gauge misfit is an offset, not friction.** J 31313 = 31145 offset
   (99.5%) + 168 shape (0.54%). RMSE 3.243 m overall, **0.238 m in shape
   alone**. Buffalo Bayou at Houston +4.90 m, 82% of J. Per-gauge table in
   RESULTS o66. This is the strongest single number in the gauge story and it
   needs no spectrum.
5. **The two observables disagree in sign** at the lookup on developed-medium
   (gauges -53129, marks +756 in dJ/dn) and the gauge step worsens the marks
   at first order (+171 predicted, +87 measured). From the adjoint, so robust.
6. **The initial condition has more authority than roughness**: 0.075 m vs
   0.020 m at a 20% perturbation.

## What is RETRACTED -- do not write these

- **Every gauge Gauss-Newton number** (o65): eigenvalue counts 5/3/1/0, the
  75.6 deg eigenvector angle, "98% of the marks' subspace lies inside the
  gauges'", the demeaned GN step. The gauge sensitivity matrix is invalid at a
  5% class step: the +5% and -5% columns of one class differ by more than
  their own magnitude, and the implied gradient misses the adjoint by 2x to
  13x on three of four leading classes (o66). Central differences do not fix
  it. A valid gauge spectrum has not been computed.
- **"With demeaning the observables agree."** Direct evaluation of the
  demeaned objective agrees with the marks on developed-medium only; 22 and 90
  disagree, on differences of 1-5 J-units out of 168.
- **"A physically impossible field"** (Sec 6.1, ~line 1239). Donghui rules
  n = 0.36 for developed-medium "a little bit high, but reasonable".

## MEASURED vs LINEARIZED -- a labelling rule, from Emil (2026-09-16)

Emil asked whether the 0.33/0.34 roughness values came from a calibration or
from "the local sensitivity model". They were linearized, and that is why they
were withdrawn. His question generalizes into a check to run over the whole
paper:

**Every parameter value, error reduction and skill number must be identifiable
as one of two things: MEASURED (a forward or a calibration actually ran and
produced it) or LINEARIZED (it comes from the sensitivity model / the
Gauss-Newton Hessian).** Where the text does not make that plain, fix it.

- Measured: the 0.72 m prior MAE, the 0.6154 / 0.6290 / 0.7609 scored fields,
  the 15% ceiling, the 84% three-vs-fifteen share, the gauge J values, the
  99.5%/0.54% offset-shape split.
- Linearized: every eigenvalue, every "supported parameter" count, the
  error-reduction column of the spectrum table, the degrees of freedom for
  signal, and everything in tab:scaling.

**A caveat to state rather than let a reviewer find it:** the spectrum
describes the linearization at the prior, while the calibration traverses
alpha from 1 to 3. At the o63 step the measured objective change was 35% of
its linear prediction. So a spectrum, even a converged one, does not describe
the calibration's path -- it describes information content at the prior. The
paper currently blurs these.

**Emil's other point, still open:** he wants the finite-difference derivatives
shown to stabilize as the perturbation shrinks before any conclusion is drawn
from them. We have that only for developed-high. The test is 6 forwards at
+/-1% for classes 22, 23, 90, ~4.4 node-hours, blocked on the Perlmutter
maintenance. Until it runs, the paper must not assert anything about a gauge
spectrum in either direction -- not that one exists, and not that one cannot.

## Sec 6.1 rewrite, gated on decision 1 (do NOT restructure Sec 6)

Replacement text for individual claims is in `plans/team-decision-list.md`
and `plans/o63-gauge-weight-audit.md`. The three that must change:

1. **The mechanism paragraph (~1229-1231) is factually wrong.** It says the
   modelled stage at the gauges is too low. It is too HIGH at four of five
   gauges (+4.90, +1.94, +1.97, +1.20; Langham -0.61). It also has the
   conveyance sign backwards for the marks.
2. **"A factor of ten apart, each at an edge of the same prior"** overstates:
   3.0/0.3 are box edges set by the box, and o63 ran one projected BLMVM step
   stopped by the wall clock (TAO tolerances 1e-12, never converged). Say
   "one accepted iteration, stopped by the wall clock; two classes projected
   onto the upper bound".
3. **"Physically impossible field"** -> the gauge field's values are ones a
   reviewer would accept, which is what makes it the useful cautionary case.

**Untraced and must be re-sourced or cut:** Sec 6.1 line 1195, "modelled
water-surface elevation sits 1--11 m above observed stage at all 13 gauges",
from commit c0ab9162 (08-24). I could not trace it to any log, and the o65
per-gauge residual is the measurement that should replace it.

## Other open edits

- **Three passages predate o63** and contradict it: Sec 5 ~1470 and Sec 6.3
  ~1741 both say out-of-sample skill "has to be tested by cross-validation
  within the upstream band"; Conclusions ~2177, ~2186 never mention the
  validation. One clause each; text in the audit note.
- **Denominators.** 15% / 84% / 85% / 83% / 12.5% / 23% / 12% use three
  different denominators. Settle on metres of MAE at the marks: fraction of
  the prior's 0.7188 m for "how much of the error", the 0.107 m fifteen-class
  reduction quoted once (84%) for "three vs fifteen", and gauges in RMSE only
  (3.24 -> 2.84 m). Never the 23% of gauge J beside the 12.5%. Lines 87, 91,
  146, 212, 1244-45, 1753, 1942, 1949-50, 1981, 2065, 2177.
- **Notation in eq. (gn), Sec 6.4.** H is the observation operator in Methods
  (line 392) and the Gauss-Newton Hessian in 6.4 (1781), both on one line; k
  is the time index and the class index in the same expression; W is never
  defined. Three one-line fixes, no numbers change.

## Coauthor decisions still open

1. Lead with the measurement or with the three-parameter calibration. Donghui
   leans toward keeping the 15% prominent. **Mark's call.**
2. Emil's within-mark hold-out, ~24 node-hours. Nobody has answered.
3. (closed) 20% antecedent water: "possible, not quantified"; cite Kiang et
   al. 2018 for the scale of streamflow uncertainty, not for stored water.
4. (closed) GMD, Gautam's preference as PI; timing is not a decision.

Also outstanding: **is n = 0.36 defensible on the full developed ladder?**
Donghui compared 22 and 23 only. With 21 and 24 frozen at the lookup, the
gauge field leaves developed-high (0.16) smoother than developed-low (0.27).
By his own criterion no field we have is consistent.

## Machine state

Perlmutter went down for maintenance 09-16. Nothing of ours is queued; all
o65/o66 artifacts are fetched into `logs/o65/` and committed. No run is needed
for any of the writing above.

Two candidate runs, both blocked on the maintenance, neither blocking the
writing:
- **Emil's convergence test**: 22, 23, 90 at +/-1%, 6 forwards, ~4.4
  node-hours. Decides whether a gauge spectrum can be computed at all. Until
  it runs the paper asserts nothing either way.
- **Emil's within-mark hold-out** (coauthor decision 2, ~24 node-hours),
  unanswered by the group.

Do not chase a gauge spectrum for the paper's argument; the offset split
answers that question without one.
