# Manning paper: state of play (2026-09-17, mid-restructure)

*Read this first, then `plans/team-decision-list.md` (Emil's two 09-16
replies at the end), `plans/RESULTS-gpu-implicit.md` sections o62-o66 and
`plans/o63-gauge-weight-audit.md`. Paper:
`papers/manning-calibration/manning-calibration.tex`, 33 pp, builds clean
with latexmk. Branch `adams/gpu-implicit`. **Tree is NOT clean: the
09-16 review edits and the 09-17 restructure edits are uncommitted** (tex, bib, fig_spectrum.tex, pdf,
o58_gauss_newton.py, this file, the decision list, and six new files in
`logs/o61/`). Mark has not asked for a commit.*

## The one-paragraph version

Roughness can explain at most ~15% of a 0.72 m model-survey error on the 46
cluster-A high-water marks. The marks determine about three combinations of
the fifteen NLCD classes at a +/-30% prior (one at the absolute prior), and
what those three reach is below the lookup's own range. The gauges cannot
arbitrate: 99.5% of the gauge misfit is a constant per-gauge level error of
metres, and the one roughness change tested against it removed 0.4 m of
3.24 m at three times the lookup. So the error is in the water balance and
the mesh, not the friction.

## What the 09-16 review session did (uncommitted)

Full review pass, then claim-level edits with Mark's clearance to reorder.
Emil's feedback drove the order.

1. **Measured / linearized / held-out labelling** (Emil's point 1). Sec 6.4
   now opens with the caveat that the spectrum is a linearization at the
   lookup and not the calibration's path (o63 step: 35% of linear). New
   paragraph "Fit, uncertainty, and skill" at the end of Sec 6.5 states the
   three categories. tab:learned and tab:scaling captions say linearized;
   "would support" replaces "supports" in the more-marks paragraph.
2. **Offset/shape split written the way Emil allows** (point 2): a
   decomposition of the residual, not of roughness information. "An offset
   roughness cannot produce" appears nowhere. The measured counterpart is
   stated: o63 removed 0.4 m of the 3.24 m RMSE at 3x; the remaining 2.8 m
   is unmeasured. Abstract, contribution 4 and Sec 6.5 all use this form.
3. **Mark-sensitivity step-refinement caveat** (point 3): Sec 6.4
   "Assembling it" reports the 4% adjoint check (22 -28.4/-29.5, 23
   +89.7/+90.7, 90 +65.1/+66.2) AND says the 5% one-sided columns have not
   been shown to converge under step refinement, and that the gauge
   construction failed that test, so no gauge spectrum is in the paper.
4. **Sec 6.4 count fixed.** "The answer is one" -> "The count depends on
   the prior width": lambda = 2.78, 0.68, 0.61, ... (one) at sigma_n 0.015
   and 12.7, 2.99, 1.13, 0.85, 0.25, ... (three, dofs 3.4) at sigma_alpha
   0.30, ladder 1/1/2/3/4 at 10/15/20/30/50%. Both series in fig:spectrum.
   Sources: `logs/o61/o61_spectrum_sigma_n0.015.txt` (reproduces the paper
   exactly) and `o61_spectrum_sigma_alpha{0.10,0.15,0.20,0.30,0.50}.txt`,
   from `o58_gauss_newton.py --sigma-alpha` (new option, also prints
   per-class posterior widths). tab:learned now has both priors.
5. **Cross-observable test moved to a new Sec 6.5** "Validation against a
   second observable" (after the spectrum, before Sec 7), so the marks and
   the three classes exist before they are used. Rewritten claim by claim
   per the audit: five gauges/134 records, one iteration stopped by the wall
   clock, sign statement with the adjoint gradients, +171 predicted/+87
   measured, no "factor of ten", no "physically impossible", RMSE not 23%.
   New tab:gaugeresid (per-gauge offset/shape, from RESULTS o66 item 5,
   reproduced this session from `logs/o63/obs_turning_h29_41.txt` and
   `logs/o65/o65_gauge_base.txt(.zb)`: J 31313.3 = 31145.1 + 168.3, RMSE
   3.243 / 0.238, exact). tab:crossobs gained an RMSE column.
6. **Sec 6.1 line 1195 replaced.** "1-11 m above at all 13 gauges" traced
   to `plans/campaign-wednesday.md:156-168`, an 08-24 unforced early-window
   measurement; not a production-window number, and Langham is -0.61 m.
   Now a forward reference to Sec 6.5's per-gauge table. The mechanism
   paragraph is gone with the move. tab:gauges regenerated from the logged
   obs table (six endpoints moved 2-9 cm).
7. **Contributions**: five items; the Jacobian/gradient verification
   bullets folded into the capability bullet (Mark agreed they were
   preconditions, not results).
8. **Stale-since-o63 passages** fixed: Sec 6.1 bands paragraph, Sec 6.3
   in-sample caveat, conclusions ("15% in sample"; next steps lead with the
   failed validation; the "compute the spectrum" step removed as done).
9. **Sec 7 plausibility softened**: 20% is "an illustrative perturbation,
   not a measured error"; kiang2018streamflow added to the bib for the
   scale of streamflow uncertainty; the spin-up needed to quantify it named.
10. **Denominators**: 12.5%, 23%, 12%, 83%, 90%/94% gone; 84% is stated
    once as the share of the absolute-prior fifteen-class 0.107 m; the
    abstract says "about 0.10 m, 15%, under either prior width".
11. **Notation**: GN Hessian is G, H is the observation operator, W is
    defined, j indexes classes. "Four A100 nodes" -> four GPUs on one node.
12. **Cuts** (~180 lines, offset by the additions; still 32 pp): Algorithm
    3 (BLMVM step), the lambda/mu walkthrough shortened, the chained-restart
    anecdote, ARK-IMEX/libCEED/PETSc implementation facts (two sentences
    remain), the 300-vs-2000 semiconvergence prose, the pilot-lambda aside,
    the Methods pre-announcement of Sec 6 results. The three-vs-fifteen
    comparison is now tab:threefifteen with the prose halved.
13. **Glosses at first use** (Mark, 09-16: the audience may not know these
    methods; one clause each, no tutorial): 4D-Var, checkpointing,
    Tikhonov term, identifiable, semiconvergence, null space,
    representation error, argmax, Hessian / whitened / posterior
    precision, Gauss-Newton, secant, spectral gap, Rodgers' degrees of
    freedom. The abstract now says "the eigenvalues of the calibration
    problem" instead of naming the Hessian. Sec 6.5 defines fit,
    linearized uncertainty and held-out skill in one sentence each.
    Paper is 33 pp after these.

## RESTRUCTURE SESSION: start here (2026-09-17)

Mark cleared a full restructure ("rearrange the paper as you like") and
closed several questions. **The paper builds: 33 pp, no undefined refs.**
Everything is uncommitted.

**Mark's rulings, 09-17:**
- Rearrange freely. Rewrite the abstract (done). Ask about minor
  non-structural points as they come up rather than batching.
- **The "support" prohibition is lifted** -- use the word where it is
  clearest. (This retires the 08-27 review item.)
- **Zenodo DOI and archive location: deferred.** There will be a full
  read-through after the restructure, so the two placeholders in Code
  and Data Availability stay for now.
- **Emil's hold-out (decision 2) and the developed-ladder question to
  Donghui: no response, treat as not important.** Use judgement, clean
  up later if they answer. Mark will add both to the email that goes
  with the restructured draft. So nothing waits on them: the hold-out
  stays future work in the conclusions, and the ladder question was
  dissolved by making tab:ladder's last column factual (done).
- Boundary conditions are parked; the note to Donghui is drafted in
  `plans/team-decision-list.md`. Nothing in the paper depends on it.

**DONE in the 09-17 sessions (all verified by a clean build, 33 pp, no
undefined refs). Everything remains UNCOMMITTED; Mark has not asked for
a commit.**
1. **Abstract rewritten** into four paragraphs (capability; measurement
   plus the count in plain language; gauges; initial condition).
2. **Sec 6.3 reframed to option A** (what a calibration achieves against
   the bound; 15% stated as central AND in-sample; Donghui's "a fit is
   not thereby the true field" carried).
3. **tab:ladder's judgment column made factual**, caption states the
   lookup's range (0.027 to 0.160, lowest developed 0.040).
4. **Sec 6.1 lake paragraph rewritten (Fable, 09-17 pm).** The untraced
   pond statistics and the `% [PROVENANCE]` comment are gone. What
   remains traces: 37 never-cresting marks, +7 m (tab:baseline; RESULTS
   o37 drainage entry ~line 1355), zero recession, ~10 m arriving
   laterally after the rain, 35 of 6,198 boundary nodes, 13-edge outlet
   at the low point, boundary median 29.9 m (RESULTS mesh forensics
   ~line 2180). The overtopping mechanism is stated as "our reading of
   the event, not a measurement of this paper" -- Mark ruled it is
   project work, so it says "we" and does NOT name Donghui -- with a
   **red note asking the coauthors for a citation** (Harvey crossing the
   Buffalo Bayou divide, or the limits of closed watershed domains).
   Contribution 5, the tab:baseline/tab:bands captions ("reach the
   model cannot drain", not "no working exit") and the conclusions'
   third next step ("the closed-perimeter idealization ... a divide the
   event exceeded") match.
5. **Peak observable moved into Methods.** New paragraph "The peak
   observable" at the end of Sec 2.2.2 (sec:adjoint): the max-over-window
   misfit, the argmax injection, the FD-gate discussion and tab:fdwindow.
   WSE is now defined there (it was first used undefined in Sec 2.3).
   Sec 6.1 keeps the survey/QC motivation and points back with "The peak
   misfit and its adjoint are those of Section 2.2.2". Sec 6.4's
   branch-surface pointer now cites sec:adjoint.
6. **House-style pass on the intro and Secs 6-8.** Prose dashes: intro
   10 -> 0, Sec 6 51 -> 0, Sec 7 10 -> 0, conclusions 4 -> 0 (the only
   `---` left in those sections are empty table cells). Secs 2-5 were
   touched only by the moves and the audit fixes (fig:map's caption
   still carries five dashes). Also: "knob" -> "uniform scale" (metaphor
   rule), "physically defensible/indefensible" -> stated ranges,
   "defensible" defined in Sec 6.2 as "inside the stated prior",
   "reassuring/unreassuring half" -> "first/second part", "teach nothing
   about" -> "would narrow the prior by 2% or less", the question-form
   paragraph heads renamed. **Mark's ruling: keep "authority" as a
   defined term** (Sec 6.2 defines authority, ceiling and defensible);
   the abstract's pre-definition use was replaced by plain words.
7. **Labelling check** (Emil rule 1): bare "determine" for the
   three-class fit rewritten twice in Sec 6.4 ("so those values come from
   the marks and not from the start point"; "the calibration of those
   three reaches ..."); "narrow" -> "would narrow" where linearized; the
   intro and Sec 6 opener say "outweigh the prior on about three
   combinations at +/-30%".
8. **Full re-trace done** (five parallel audits, ~900 numbers; reports in
   `plans/trace-2026-09-17/trace_{A..E}.md`, one per block of the paper).
   Fixes applied, each verified against the source before editing:
   - tab:alpha caption: "one mark stays dry throughout" -> dry at every
     point below alpha 0.8 and none above (its own table said 0/46).
   - tab:ic caption: "No mark is dry at any point" deleted (dry counts
     exist only for a = 0.6, 0.7, 1.0).
   - Sec 7: "three more orders of magnitude of freedom" -> "almost six"
     (8.8M unknowns / 15 classes).
   - Abstract: "the same three classes move the opposite way" -> "two of
     the same three" (22 rises under both, RESULTS 2552-53); "level
     error of metres at each gauge" -> "of one to five metres at four of
     the five" (Langham is -0.61 m); "tracks the shape of every
     hydrograph to 0.24 m" -> "tracks hydrograph shape to 0.24 m"
     (Houston's shape rms is 0.38 m); "below any entry in the table" ->
     "with the whole table scaled to 0.3 ... smoothest classes below
     smooth concrete" (alpha 0.3 spans 0.008-0.048). Contribution 4 and
     the fig:authority caption changed to match.
   - Conclusions: "bitwise-identical trajectories" (that was a
     device-vs-device A/B) -> same binary and partition, identical to
     every printed digit; "tens of TAO evaluations instead of hundreds"
     -> nine or fewer iterations (tab:threefifteen) vs the hundreds of
     the per-cell twins.
   - Intro: red note that the 471M-cell validation has no citation.
   - Sec 5.1: "five orders of magnitude" -> four (0.1 m / 5 um); tab:snr
     caption "seven orders" -> four, "0.1-0.3 m" -> the 0.15 m used;
     "~5% of cells" -> "each under 3%" (RESULTS o41 table: others 12k-86k
     cells). **Red note on tab:snr: the noisy rows (0.20 -> 1.8,
     4e-7 -> 0.83) and the beta sweep to 1 trace only to a commit
     message; re-run and log, or drop.**
   - Sec 6.1: "reservoir gauges hold their water for the whole window"
     -> for every record they have (20 and 4 of 48 slots); "nine of the
     thirteen gauges" -> "all but the three Buffalo Bayou gauges"; the
     QC sentence now separates the 62% above-bed rejection (324 -> 122)
     from the quality filter (-> 108). The four tab:gauges bed
     elevations the audit could not find are in
     `logs/o65/o65_gauge_base.txt.zb` (gauges 2, 4, 7, 11).
   - Sec 6.4: argmax counts recomputed on the 156-of-690 set: 107 -> 112
     (72%), 1.6% -> 1.4%, "more than an hour" -> "more than half an
     hour" (bin 2100-6900 steps); tab:scaling last row 7e5 -> 1.2e5
     (lambda_15 = 3.867e-4 in the o61 sigma_n log; the 7e5 was the
     pilot's); "84%" -> "84% and 83%" for the +/-30% and +/-50% runs.
   - Sec 6.5: the 35% saturation Sec 6.4 cites is now printed here
     (-7354 measured vs -21283 predicted, o63 audit line 66); "each
     cross-score is worse than the lookup" -> the mark-calibrated field
     improves the gauges by 0.05 m of 3.24 (tab:crossobs says so).
   - Methods: the seven-class bound list in "Absolute or fractional
     prior" was the sigma_alpha 0.30 run's first step, cited as the
     absolute-prior run's. Replaced with the recorded absolute-prior
     first step (o52, RESULTS ~1790: 23, 24, 90 to the floor, 22 to
     2.62x). Stability limit lost its spurious factor 2 (the paper's own
     2.4 ms is 1/rate); beta "six orders" -> "nearly ten"; "resolutions
     two orders apart" -> "a factor of 30"; "Three configurations share
     it" -> four calibrations plus the 72-h forward; verification twin is
     "planar, two regions"; ||g|| 1.5e3 attributed to the 600-step
     window. tab:verification rows 1-6 were regenerated locally with
     ctest (`logs/verification/ctest_jacobian_2026-09-17.txt`): every
     printed value reproduces (1.561e-8, 1.463e-8, 1.917e-8, 7.589e-9,
     2.427e-10).
   - Kept after checking: the 19x (RESULTS 767 calls that pair the
     single-node like-for-like; the intervening fix removed device-side
     PCIe transfers, which does not touch the host timing). Not changed:
     "about 25% more wall time" (26% vs rung A), "about nine minutes"
     (9.4 measured).

**Three red notes are in the text** (grep `textcolor{red}`): the 471M
citation (intro), the tab:snr noisy rows (Sec 5.1), the overtopping
citation (Sec 6.1). Mark said red notes are fine while editing.

**NEXT STEPS:** Mark's full read-through of the restructured draft;
then the email to the coauthors (Emil's hold-out, the developed-ladder
question, the three red notes, the perimeter note already drafted in
the decision list). Nothing on the machine: every run is on hold for the
group's feedback. Also worth a line in RESULTS: its lines ~2016-17 still
carry the pilot tab:scaling values (71 -> 1, 108 -> 2); the paper and
the o61 log have the production values (2 and 3).

**Overleaf checked 09-17: nothing to merge.** Its head is my own 09-10
push (363971e), no coauthor commits since, no `[CLAUDE/` requests, no
red items. Three margin-comment threads from Gautam (08-31, 09-01)
remain unreadable from git and predate the whole rewrite; Mark would
have to relay them.

## The labelling rule, defined (Emil, 09-16)

Every parameter value, error reduction, count and "skill" number in the
paper must be identifiable as exactly one of three kinds. The kinds differ
in what produced the number, what it licenses a reader to conclude, and
what it does not.

**1. Achieved fit (measured, in-sample).**
*What it is:* the misfit of one specific roughness field on the
observations its calibration used, obtained by running the forward model
at that field. The field may come from an optimizer or from a scan; the
number is what the model did there.
*Examples:* the prior's 0.7188 m; 0.6116 / 0.6154 / 0.6274 / 0.6290 /
0.6295 m; every J and MAE in tab:alpha, tab:halves, tab:ic, tab:ladder,
tab:threefifteen; the 15% ceiling; the 84% three-vs-fifteen share; the
gauge J 31313 and its 99.5% / 0.5% offset-shape split (a decomposition of
a measured residual is still a measurement).
*Licenses:* "at this field, on these observations, the error is X."
Comparisons between fits on the same observations (three vs fifteen,
uniform vs class-by-class, IC vs roughness scans).
*Does not license:* that the field is right, that it would transfer, or
any error bar. A fit reported alone says nothing about generalization.
*Wording:* "reaches", "removes", "scored on", "measured"; always with the
observation set and the prior width.

**2. Linearized uncertainty (local, at the prior).**
*What it is:* anything derived from the sensitivity matrix S (finite-
difference secants of the observation at the lookup, 5% one-sided) or
from the adjoint gradient at the lookup, through the Gauss-Newton Hessian
G = S^T W S / sigma^2 whitened by the prior. It depends on the prior width
and on the secant step size.
*Examples:* every eigenvalue; every "supported" count (one / three); the
Rodgers degrees of freedom; the posterior widths in tab:learned; every row
of tab:scaling; the 2.9% / 85.6% / 11.5% split of the calibrated
displacement (measured displacement, linearized basis: label both); the
first-order prediction +171 of the gauge step's effect on the marks; any
Gauss-Newton step (the withdrawn 0.33/0.34).
*Licenses:* "under the linear model at the lookup, with this prior width,
the observations out-weigh the prior on N combinations", and which
combinations. It ranks classes by how much the data could narrow them.
*Does not license:* the values a calibration reaches, the error it
removes, or anything along the path alpha 1 -> 3 (the one step measured
against its linear prediction moved 35% of the predicted amount). A
converged local spectrum still would not describe the calibration.
*Preconditions before any of it is quoted:* the secant checked against
the adjoint gradient (S^T W r / sigma^2 vs dJ/dalpha; marks: within 4%
on every class), AND the secant shown to converge under step refinement
(a ladder, not one small-% point). The marks have the first and not yet
the second; the gauges failed the first, so no gauge number is quoted.
*Wording:* "would narrow", "would support", "in the linearization", "at
the lookup", always with the prior width. Never "learned", "determines",
"supports" bare.

**3. Held-out predictive skill.**
*What it is:* the score of a calibrated field on observations that did
NOT enter its objective: a second observable in the same basin and
window (cross-observable), or a hold-out within the same observable
(spatial folds). Each such score is itself a forward run, i.e. a
measurement; what makes it skill rather than fit is that the field never
saw those observations.
*Examples:* the gauge-calibrated field on the marks (0.7609 m, worse than
the prior's 0.7188) and the mark-calibrated field on the gauges (3.19 m
RMSE vs 3.24). These are the ONLY skill numbers in the paper and both are
negative. The within-mark hold-out (Emil's two folds, decision 2) does
not exist.
*Licenses:* "this field generalizes / does not generalize to observations
it was not fit to."
*Does not license:* nothing about the size of the in-sample improvement
(that is a fit) or about identifiability (that is the spectrum).
*Wording:* "scored on the other observable", "held-out", "validation";
never "skill" for an in-sample number.

**Mixed and paired numbers.** A measured quantity projected onto
linearized directions (the 9.2 sigma split) carries both labels. A
prediction must always be printed beside its measurement (+171 predicted,
+87 measured; 0.6895 predicted, 0.6894 measured). A fit and an
uncertainty for the same field may sit in one sentence only if each word
carries its label ("reaches 0.6290 m; the marks would narrow the prior on
three combinations").

**Where the paper says this:** Sec 6.4's opening (limits of the spectrum),
Sec 6.4 "Assembling it" (the two preconditions), Sec 6.5 "Fit,
uncertainty, and skill" (the three definitions), the captions of
tab:learned, tab:scaling, tab:threefifteen and tab:crossobs.

## Still untraced / open in the text (after the 09-17 re-trace)

- **tab:snr noisy rows** (0.20 -> 1.8 at 0.1 mm; 4e-7 -> 0.83 at 1 mm;
  the beta sweep to 1): commit-message provenance only. Red note in the
  caption. Re-run on the laptop (the verification twin runs in seconds)
  and log, or drop the two rows.
- **471M-cell validation**: no citation in the bib. Red note.
- **Overtopping mechanism** in Sec 6.1: stated as our reading, red note
  asking for a citation.
- The **lake pond statistics** (1 of 37 vs 6 of 71, 1.7%, 20.0 +/- 1.1 m,
  6.2-20.2 m, -0.93, quarter of the perimeter) are OUT of the paper. They
  survive only in the 09-07 memory note; the paper no longer needs them.
- **tab:alpha, tab:halves, tab:ic** rest entirely on RESULTS; no raw
  artefacts for o44/o45/o47/o49/o54 are in the repo (only the campaign
  scripts). tab:ladder, the 0.5818 IC point and the spectra do have logs.
- Secs 2-5 provenance is `plans/RESULTS-manning-draft.md`,
  `plans/campaign-wednesday.md`, `plans/PROJECT-STATE-2026-08-26.md` and
  the in-tree tests; the full row-by-row tables are in
  `plans/trace-2026-09-17/`.

## Coauthor decisions still open

1. **(closed 09-16, Donghui) Lead with the measurement, option A.** "I
   agree with CLAUDE that we should not aim to deliver a roughness table.
   There are uncertainties from many sources that will affect the
   calibration of the Manning coefficient. Even the calibrated one leads
   to improvement, it may not represent the truth." That is A, and it is
   consistent with his 09-14 lean toward keeping the 15% prominent. The
   abstract and contributions already read this way. **Remaining work:
   Sec 6.3 still frames the fifteen-class field as the headline
   calibration ("clears the bar", the supporting/limiting paragraphs) and
   must be brought into line -- reduce to the ladder table plus a short
   reading, with the three-class result carrying the emphasis.** His "may
   not represent the truth" is worth one sentence in Sec 6.3 or the
   conclusions: a calibrated field that improves the fit is not thereby
   the true field, which is the paper's own equifinality point.
2. Emil's within-mark hold-out, ~24 node-hours. Nobody has answered.
3. (closed) 20% antecedent water: possible, not quantified; Kiang cited.
4. (closed) GMD.

## The lake: attribution CLOSED by Donghui (2026-09-16)

"The domain extent was from actual watershed boundary. Due to the
watershed is flat, it is possible for the water to flow through the
watershed boundary for the very extreme Harvey event. We cannot simulate
this overtopping as we set the closed boundary in RDycore."

That is the authority the email asked for, from the coauthor who built
the domain, and it matches our own mesh forensics (35 of 6,198 boundary
nodes on the bounding box, so a delineated watershed; the side set at the
perimeter's low point). **The paper may now attribute the ponding to the
domain idealization** -- a closed catchment-divide perimeter under a storm
that exceeded the divide -- rather than list candidate causes. So the
proposed "list the causes and give up" edit is superseded: the lake
paragraph gets a rewrite that names the idealization and credits the
reason, not a cut. The untraced pond statistics can still go (or be
re-derived); the attribution no longer depends on them.

Note for Sec 7: this is consistent with Xu et al. 2025 finding the outlet
BC choice less important than precipitation and mesh resolution -- they
varied the OUTLET condition, not the perimeter walls.

### Donghui's follow-up question: open all the boundary edges?

**It is a yaml-only change, no code.** The untagged perimeter is already
collected into one auto-generated boundary (`InitBoundaries`,
`src/rdysetup.c:432-440`): unassigned edges get
`unassigned_edge_boundary_id`, which starts at 0 and only increments on a
collision with an id in the mesh file. The Turning mesh has one side set
(ss1, id 1), so the perimeter should be **grid_boundary_id 0**; verify
against the debug line "Adding boundary 0 for N unassigned boundary
edges" (expect N ~ 6,198 nodes' worth of edges minus the 13 outlet
edges). Declaring it in the yaml and binding `free-outflow` to it is then
a two-block edit, no rebuild:

```yaml
boundaries:
  - name: outlet
    grid_boundary_id: 1
  - name: perimeter
    grid_boundary_id: 0
boundary_conditions:
  - boundaries: [outlet, perimeter]
    flow: <the free-outflow condition>
```

**The hazard, and it is real.** `free-outflow` is transmissive: the ghost
state is a plain copy of the interior state (`ApplyFreeOutflowBC`,
`src/swe/swe_petsc.c:510-529`), with **no inflow guard and no elevation
threshold** -- unlike critical outflow, whose inflow branch returns an
identically zero flux. With identical left and right states the Roe
dissipation vanishes and the edge passes the interior advective flux in
whichever direction the interior momentum points. At a catchment divide
the terrain slopes inward, so the generic perimeter cell has inward
momentum, and a transmissive perimeter would **manufacture inflow along
the rim** while draining the ponded reach. The rim is the upstream band,
where all 46 calibration marks live.

**The physically right condition is an elevation-thresholded overflow**
(a weir at the divide): outflow only where the water surface exceeds the
local boundary elevation, zero otherwise. That is a small new BC (an
inflow guard plus an elevation comparison), plus its Jacobian block and
FD gate for the adjoint path.

**How it must be written, if the team takes this on.** Make it a flux
formula in the head over the divide, e.g. the weir form
`Q = C_w L (eta - z_div)^{3/2}` for `eta > z_div` and zero below, NOT a
switch between transmissive and reflecting. The distinction is this
paper's own Sec 4 material:
- The weir formula is continuous at the threshold AND its first
  derivative vanishes there (`d/d eta ~ (eta - z_div)^{1/2} -> 0`), so
  Newton sees a continuous residual with a continuous Jacobian. Only the
  curvature is singular, which is mild.
- A naive `if eta > z_div then open else wall` switch JUMPS, and it is
  the SAME BUG this project already hit, not an analogy. The
  critical-outflow branch "zeroes BOTH states when uperp < 0 (wall) and
  otherwise imposes the critical ghost -- at the uperp = 0 crossing the
  flux jumps by the full wet-onto-dry Roe flux, O(g h^2/2) ~ 1e-2 in
  residual norm" (RESULTS, "the SECOND discontinuity", 2026-08-22). That
  jump pinned Newton at step 4 of the class twin and at step 498 of the
  600-step control, at every h_anuga tried, and it was diagnosed by
  swapping the outlet to reflecting. o26 cured it by replacing the
  condition with the transmissive one, and the paper reports that as a
  cure rather than an improvement (tab:cure).
- Two things make a perimeter switch worse than the outlet one was. It
  would sit on ~6,000 perimeter edges instead of the outlet's 13, and
  cells would cross the threshold continuously as the flood rises. And
  the mitigation of the time, `-snes_rtol 1e-3`, was explicitly a
  "crutch" that "retires if/when the critical-outflow switch is
  smoothed"; o26 retired it, and production now runs at nonlinear 1e-5,
  which is what licenses the paper's claim that the verified gradients
  ARE the production gradients (Sec 2.2.2). A new switch would put that
  back in play.
So the weir formula is the safe choice and the threshold switch is the
trap. Worth saying to whoever implements it.

**Scope warning.** If an open perimeter changes the upstream water
balance, every production number in the paper (0.7188 baseline, the alpha
scan, the IC scan, the spectrum's 16 forwards, the calibrations) is
measured on a superseded configuration: ~60+ node-hours to redo. For THIS
paper the open-perimeter run is a diagnostic to report, not a
configuration change.

## The downstream-reach runs: PLANNED, NOT SCHEDULED

**Mark cancelled the overnight run on 09-16 evening: wait for the group's
feedback on Donghui's "open all the boundary edges" question before
spending machine time.** There is no cron; nothing fires on its own.
Restart this by re-reading the three steps below when the group replies.

**Run in this order; each step is independently reportable, so stop
wherever the budget ends.**

**Step A (zero node-hours, login node): does a transmissive perimeter
leak inward?** This is the guard-rail on Donghui's suggestion and needs
no run. From an existing o37 hourly checkpoint plus the mesh, compute for
every perimeter (auto-generated boundary) edge the would-be transmissive
flux h*u_n*L from the interior state, and split the sum by sign at two or
three times (say hours 40, 60, 72). Report inward total, outward total,
and the outward part restricted to the censored downstream reach. If the
inward part is negligible against the outward, Donghui's open perimeter
is safe as written; if it is comparable, the paper (and he) should hear
that an elevation-thresholded overflow is the right condition. Same kind
of script as the 08-27 mesh forensics and `depression_check.py`.

**Step B (~2 node-hours, no code change): the open-perimeter
counterfactual.** Copy the production yaml, add the perimeter boundary at
`grid_boundary_id: 0` bound to free-outflow (see the lake section above
for the block and for how to verify the id from the debug line), and run
one 72-hour forward from the same initial state as the baseline, same
rain, same NLCD prior, 1 node, m4267_g, submitted from a login shell
(~2 h by the baseline's own timing). Then score all 108 marks as the
baseline did and answer two questions: do the 37 never-cresting marks now
crest and drain, and do the 46 upstream marks move? The pair
(37 drain / 46 unchanged) would be a clean, quotable confirmation of
Donghui's mechanism that costs the paper nothing. If the 46 move, say so
plainly -- it means the baseline depends on the perimeter condition,
which is a finding and also a reason not to change the configuration
mid-paper (see the scope warning above).

**Step C (driver work, ~1-2 node-hours, LAST): the outlet-flux check.**
Still worth having but no longer the decisive test, since Donghui
supplied the attribution and step B tests it directly. Answers the
narrower question: is the outlet at capacity, or does the domain simply
hold water?
1. Driver: the adjoint driver has no boundary-flux output. Add an option
   (e.g. `-adjoint_outlet_flux_dump <file>`) that sums the volume flux
   through the free-outflow side set each step (or each observation
   interval) and writes time, Q_out. The free-outflow ghost map is the
   identity (RESULTS o26); the Kokkos boundary-flux kernels are where the
   edge fluxes exist. Keep it eval-only (no adjoint).
2. Build `build-claude-gpu12` from a shared-QOS build job following
   `cmake-claude-gpu11.sh` in `~/Codes/rdycore-manning` on Perlmutter.
   Never sbatch from inside a job (see memory). Echo the binary path,
   size and git describe into the log.
3. Run: one forward, 1 node, m4267_g, submitted from a login shell, from
   the o37 hourly checkpoint at hour 48 (`checkpoints_o37/`, hour 29 is
   `o37.rdycore.r.104400.bin`, so hour 48 is `...r.172800.bin`; verify),
   rain re-aligned as in `plans/campaigns/o65_gauge_spectrum.sh` (WIN
   line), through hour 72: 86,400 steps at dt 1 s, ~40 min on one node
   by the 72-h forward's two hours. A shorter window is fine if Q_out is
   steady.
4. Compare: Q_out(t) against (a) the domain's total stored water from the
   hourly o37 checkpoints (sum h*area; rain is 1.7% of the event in
   hours 60-72 and zero after 72), and (b) the censored reach's storage
   rate. Outlet at capacity: Q_out large and steady while the reach still
   rises. Domain holds water: Q_out small against the storage change.
5. Record in RESULTS-gpu-implicit.md as a new entry; mirror the logs to
   `logs/o67/`. Then the paper's candidate-cause list can become a
   finding (that edit waits for Mark).

## Machine state

Perlmutter down for maintenance 09-16. Nothing queued. Two candidate runs,
neither blocking the writing:
- **Emil's convergence test, redesigned per his 09-16 reply**: a step
  LADDER (e.g. 0.5, 1, 2, 5%, both sides) for 22, 23, 90 on the gauge
  observable, not a single +/-1% point. Decides whether a gauge spectrum
  can be computed at all. Until it runs the paper asserts nothing either
  way, and it now says so in Sec 6.4. The same ladder on the MARK columns
  would close the step-refinement caveat the paper now carries.
- **Emil's within-mark hold-out** (decision 2).

Do not chase a gauge spectrum for the paper's argument.
