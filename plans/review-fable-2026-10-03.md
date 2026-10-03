# Referee read of manning-calibration.tex (2026-10-03, Fable 5.1)

**Status, 10-03 evening: APPLIED to the .tex in the same session, on
Mark's instruction ("address this review"). Builds clean, 35 pp, no
undefined refs.** A1-A20 are all addressed; the hydrology attribution
(contribution 5, the divide) is unchanged and the new outlet-inflow
measurement is stated beside it (A2's text-only fix). Three further
inconsistencies found and fixed during the pass: Sec 6.3 opened with "The
scan bounds what any roughness field can do" (same overclaim as A1); Sec
6.2 called the uniform scan "close to its most identifiable" direction
(Sec 6.4's overlap of 0.218 says otherwise); Sec 2.2.2 said adjoint paths
"use the RK family" (only the explicit ones do). The Sec 6.4 displacement
paragraph (A4) now prints exact projections computed this session from
the o61 peak dumps and the o59 field table: absolute prior, 57% of the
squared displacement along v_1 alone (log bins 2.9 / 85.6 / 11.5
reproduce to within the field's three-digit rounding); +/-30% prior,
|w| = 5.69 sigma, 46% in the three data-determined directions, 39% along
v_1. Script inline in the session; worth adding a `--field` option to
`o58_gauss_newton.py` so it is reproducible from the repo. The questions
in B stand; B2(b) is now done in the text.

Scope: `papers/manning-calibration/manning-calibration.tex` at the tip of
`adams/gpu-implicit` (2402 lines, 34 pp), read against
`plans/PAPER-SESSION-HANDOFF.md`, `plans/RESULTS-gpu-implicit.md` (o48
through Step A), `plans/team-decision-list.md`,
`plans/o63-gauge-weight-audit.md`, the 09-17 trace reports and the logs
under `logs/`. Nothing in the .tex was edited. No run was made; the one
new number below (the integrated outlet inflow) is arithmetic on the
existing `logs/stepA/stepA_mass_balance.log`.

Line numbers are those of the .tex as read today. "RESULTS" is
`plans/RESULTS-gpu-implicit.md`; "HANDOFF" is the paper handoff.

---

## A. Findings, ranked by severity

### A1. WRONG. The ceiling was not measured from forward runs alone, and the forward runs did not bound it

**Where.** Abstract l.81-84: "roughness calibrated as well as we can
calibrate it removes about 15% ... and that ceiling is measurable from
forward runs alone." Contribution 2, l.211-213: "from forward runs alone:
about 15%". Sec 6.2 l.1540-1541: "the six forward runs of Table alpha
bound the authority of the whole roughness field"; l.1549-1551: "three
steps bound what calibrating that parameter can ever accomplish".
Conclusions l.2300-2302: "it bounded roughness at 4% ... and 11% ...; the
calibration then reached 15%".

**Why it fails.** The forward-only scan reaches 0.080 m (tab:alpha,
alpha 0.30, 0.6392 m) = 11%, and 0.029 m = 4% inside the prior. The 15%
(0.107 m, 0.6116 m) is the nine-iteration adjoint calibration, which the
paper's own Sec 6.3 l.1619-1622 says beats the scan by 0.078 m, "more than
the scan alone justifies". A quantity the calibration exceeded by a third
is not a bound, and the number the abstract attaches to "forward runs
alone" is the calibration's. By the paper's own definition (l.1393-1397)
the "ceiling" is the scan's 4%, which is neither 15% nor the calibration.

**Evidence.** tab:alpha l.1429/1434; tab:ladder l.1577-1578; RESULTS
l.2072-2085 (o59: calibration 0.6116 vs scan 0.6392).

**Fix.** Abstract: "...removes about 15% ...; a forward-only scan of a
single global scale puts the reachable amount at 11%, and at 4% inside the
prior, so the scale of the answer is known before any optimizer runs."
Contribution 2 the same way. Sec 6.2: "bound" -> "estimate the scale of",
and say the calibration exceeded the scan's figure by 0.027 m because the
scan's direction was not the informative one (Sec 6.4 says this already).
Conclusions: "bounded" -> "put".

### A2. UNSUPPORTED (alternative not ruled out). The production outlet admitted water for 38 of the 72 hours, and the paper does not say so

**Where.** Sec 4 l.945-956 presents the transmissive outlet as "not an
improvement but a cure". Sec 6.1 l.1294-1297 "the downstream Buffalo Bayou
reach, which the model cannot drain"; l.1299-1304 "The exit is not
missing"; l.1305-1317 and contribution 5 l.231-235 attribute the ponding to
the closed catchment-divide perimeter. Sec 7 l.2249-2250 "it holds too
much of [water] at hour 29", with the candidate list at l.2183-2184 and
l.2260-2262.

**What is measured.** `logs/stepA/stepA_mass_balance.log`, outlet column
(+ = out): -4,946 m3/s at hour 1, -8,522 at hour 3, -1,772 at hour 10,
-630 at hour 20, -95 at hour 29, -1.4 at hour 38, +11 at hour 39, +661 at
hour 72. Trapezoid integral, hours 1-38: **-193e6 m3 of inflow** (this
review's arithmetic on the log); hours 39-72: about +36e6 m3 out. Net over
the 72-hour baseline: about **157e6 m3 entered the domain through its only
exit**. Storage at hour 29 is 1,021e6 m3, so the inflow by hour 29 is
about **19% of the water the production initial condition holds**. RESULTS
l.2886-2891 records the sign and the 38 hours; the integral is new here.

**Why it bears on the text.** (a) Sec 4: critical outflow had an inflow
guard (l.353-355, "the code's inflow branch producing an identically zero
flux"); its replacement has none, and the production forward then admitted
inflow for 38 hours. "Cure" is true of Newton and silent about mass.
(b) Sec 6.1: in the 72-hour baseline the exit was, in net, a source. The
divide attribution is not thereby wrong, but it is no longer the only
measured mechanism acting on that reach, and "the exit is not missing"
invites the reading that the exit behaved as an exit. (c) Sec 7: the
hour-29 checkpoint contains about 19% of its water from a boundary that
should have passed none; the paper's illustrative perturbation is 20%. The
candidate list names rainfall and antecedent state and omits the boundary.
(d) The 46 upstream marks: no bearing is established. The inflow enters at
the low end; whether any of it reaches the upstream band by hour 29 is
unmeasured (question B1).

**Caveat on the measurement.** It rests on the orientation of the outlet
side set's normals in the Step A script. Two things say the sign is right:
the hour-72 flux is outward (+661 m3/s) after the rain ends, which is the
physical direction; and the flipped sign would require a domain-average
rainfall of 55-69 mm/hr for three consecutive hours at hours 2-4 (the
implied hyetograph with the stated sign peaks at 45.8 mm/hr at hour 28, on
the night the heaviest band crossed Houston). The mass balance alone does
not fix the sign, because the rain is the unknown it solves for.

**Fix (text only; the interpretation is the team's, B1).** One sentence in
Sec 6.1 after l.1304 stating the measurement with its caveat; "boundary
inflow at the outlet" added to the Sec 7 candidate list; Sec 4 l.953 "a
cure" qualified: a cure for Newton that also removed the inflow guard, with
the measured consequence named.

### A3. WRONG. The abstract says all three classes land in the same place under every prior

**Where.** Abstract l.86-89: "where those three land is the same under
every prior we ran, below the table's own range for developed ground."

**Why it fails.** Developed-low lands at 1.63x, 2.34x and 3.0x under the
three priors (tab:threefifteen l.1912-1915), and Sec 6.4 l.1972-1975 says
its value "is set by the prior". Two of the three land in the same place.

**Fix.** "two of those three land in the same place under every prior we
ran, below the table's own range for developed ground, and the third goes
where the prior lets it."

### A4. WRONG (internal contradiction). The calibrated displacement does not lie "in the one direction the data constrains"

**Where.** Sec 6.4 l.1850-1857: "So the displacement is not motion in
directions the data does not constrain. It lies in the one direction the
data constrains." Four sentences earlier, l.1826-1830: of the 9.2 sigma,
"only 2.9% lies in the single data-constrained direction; 85.6% lies in
directions where data and prior are comparable". Sec 6.3 l.1657-1660
quotes the 2.9% to argue the opposite point (equifinality).

**Why it fails.** At the absolute prior the leading eigenvector pairs
developed-medium and woody wetland with opposite signs,
v_0 = 23(-0.64) 90(+0.59) 22(-0.36) 81(-0.23); the calibration moved both
DOWN (23 to -5.60 sigma, 90 to -4.22 sigma, 22 to +3.76 sigma, RESULTS
l.2112-2128), which is nearly orthogonal to v_0 and lies along the second
eigenvector v_1 = 23(+0.64) 90(+0.61) 81(+0.30) 22(-0.22) with
lambda = 0.68, a direction the absolute prior rates slightly above the
data. Projecting the o59 displacement on the printed components gives
about half of its squared norm on v_1 and under 1% on v_0 (the log's 2.9%
includes the components not printed). Under the +/-30% prior the same
direction is v_1 = 90(+0.77) 23(+0.59) 22(-0.19) with lambda = 2.99, the
second of three data-determined directions.

**Evidence.** `logs/o61/o61_spectrum_sigma_n0.015.txt` lines 9-11;
`logs/o61/o61_spectrum_sigma_alpha0.30.txt` lines 9-11; RESULTS
l.2053-2058 (the 2.9 / 85.6 / 11.5 split).

**Fix.** Replace "It lies in the one direction the data constrains" with:
the displacement lies along the second eigendirection, the joint lowering
of developed-medium and woody wetland, which the absolute prior of this
run rates at lambda = 0.68 and the +/-30% prior the paper assigns at 2.99.
Then Sec 6.3's 2.9% and Sec 6.4's "best-informed classes" are both true
and the three-class result (which spans v_0, v_1, v_2 at +/-30%) follows.
Question B2 asks whether to compute the +/-30% projection outright.

### A5. WRONG, and a consequence UNSUPPORTED. The fifteen-class +/-30% run was not "flat", so the matched misfit is not evidence of a floor

**Where.** tab:threefifteen caption l.1920-1922: "the fifteen-class runs
and the three-class +/-30% run were stopped by the wall clock with the
objective flat". tab:ladder caption l.1590-1593: the two fifteen-class
fields "agree in misfit to four figures (J_mis = 615.4), which suggests a
floor that roughness does not set". Sec 6.3 l.1652-1653: "A wider prior
reaches the same fit sooner".

**Why it fails.** `logs/o62/o62_c15_sa0.30.log` l.68-71: TAO f 1.000 ->
0.822 -> 0.795 -> 0.782, a 1.7% fall on the last iteration, with the
projected-gradient residual 0.157 -> 0.061 -> 0.119 -> 0.352, rising. That
run was descending when the wall stopped it. The sigma_n run is flat
(0.01% on its last iteration, gradient norm down 8.5x, RESULTS
l.2039-2045) and the three-class +/-30% run nearly so (0.810 -> 0.804 ->
0.803, `o62_c3_sa0.30.log` l.69-72). A still-descending run's misfit
matching a converged run's to four figures is where the clock fell, not a
floor; the next iteration of the +/-30% run would have gone below 615.4.

**Fix.** Caption: "the sigma_n fifteen-class run and the three-class
+/-30% run were stopped with the objective flat; the fifteen-class +/-30%
run was stopped after three iterations with the objective still falling
1.7% per iteration." Drop "which suggests a floor" or rest it on the
sigma_n run alone. "reaches the same fit sooner" -> "had reached the same
misfit when its budget ran out".

### A6. WRONG. The sensitivity-matrix gradient does not agree with the adjoint "to within 4% on every class"

**Where.** Sec 6.4 l.1720-1725.

**Why it fails.** The per-class S-implied gradient is printed in
`logs/o65/o65_spectrum_iid_no24.txt` l.72; the adjoint gradient at the
lookup is `logs/o62/o62_g_c15_sa0.30.txt` (dJ/dn times n_prior gives
alpha units). Relative misses: barren 31 38%, mixed forest 43 25%,
emergent wetland 95 14%, pasture 81 8.3%, developed-open 21 6.7%,
deciduous 41 5.1%, shrub 52 4.9%, grassland 71 4.4%. Within 4%: 11, 22,
23, 24, 42, 82, 90 (seven of fifteen). The three leading classes are
within 4% (3.8%, 1.1%, 1.6%) and the largest absolute miss is 2.8 units on
pasture (-31.3 vs -34.1) against leading entries of 90.7 and 66.2.
RESULTS l.2684-2686 makes the same "every class" claim and lists pasture's
8% miss in the same sentence.

**Fix.** "within 4% on the three leading classes and within 3 units in
alpha on every class; the relative misses reach 38% on classes whose
gradient is under 6 units."

### A7. MISLEADING. The half-domain experiment mixes two currencies and the "45x" has a near-zero denominator

**Where.** Sec 6.2 l.1510-1512 and tab:halves caption l.1503-1505: "Each
half alone makes the fit worse, and the two together help by 45x the sum
of the parts: a positive interaction, not two competing signs."

**Why it fails.** By MAE both halves worsen (+0.0118, +0.0080; sum
+0.0198) and together they help by -0.0574: no "45x" exists in MAE. By
objective the developed half IMPROVES (-21.5) and the rest worsens (+19.2):
two competing signs, and 45x = 104.5 / 2.3 is a ratio to a sum that nearly
cancels. The sentence takes "worse" from MAE and "45x" from J.

**Fix.** State both currencies: "by MAE each half alone is worse and
together they remove 0.057 m; by objective the developed half alone gains
21.5 of the 104.5 units the two gain together." Drop the multiple.

### A8. MISLEADING. The 35% saturation cited as the mark spectrum's caveat is the gauge objective's

**Where.** Sec 6.4 l.1690-1692: "along the one step we measured against its
linear prediction (Section crossobs) the objective moved 35% of the
predicted amount."

**Why it fails.** -7354 / -21283 is the GAUGE objective along the gauge
step. The MARK objective along the same step, printed at l.2066-2068, is
+87 / +171 = 51%. The mark spectrum's own linearization test is the 51%.
This was flagged in `plans/trace-2026-09-17/trace_E.md` l.21 and is still
in the text.

**Fix.** Cite 51% (or both, each labelled). Sec 6.5 l.2069-2071 "the
saturation Section spectrum cites" then follows.

### A9. MISLEADING. The acknowledgement claims an answer the paper withdrew

**Where.** l.2365-2368: "The cross-observable comparison of Section
crossobs answers his question of whether the two observables are
complementary, redundant or opposed."

**Why it fails.** Emil's question (decision list l.198-212) is about the
overlap of the two informative subspaces. That overlap was computed in o65
and RETRACTED in o66 (RESULTS l.2787-2792: "a valid gauge Gauss-Newton
spectrum has not been computed"). Sec 6.5 gives a sign comparison on three
classes at the lookup, which is what survives, and Sec 6.4 l.1730-1731
says why there is no gauge spectrum. Emil reads this section.

**Fix.** "Section crossobs gives the part of his question that needs no
gauge sensitivity matrix: the two gradients' signs at the lookup. The
subspace comparison waits on a gauge sensitivity that converges under step
refinement, which Section spectrum reports we do not have."

### A10. MISLEADING. The absolute-prior "signature" is not shown by the example given

**Where.** Sec 2.3 l.741-747: "The iterates show that signature: ... the
three classes driven to the floor were the large-n ones ... and the one
driven up ... was the small-n developed-low (0.090), an ordering by prior
value."

**Why it fails.** The mechanism claimed is that a small-n class buys a
large fractional excursion cheaply. The classes that moved are the three
LARGEST-n, which the absolute prior penalises most per fractional move,
and developed-low at 0.090 is the fourth largest, not small. The fractional
prior sends developed-medium and woody wetland to the same floor
(tab:threefifteen), so the floor is the data's pull on the informative
classes under either prior, not the absolute prior's pricing.

**Fix.** Drop "The iterates show that signature" and the example, or
replace with the measured comparison: the first step at +/-30% put seven
classes on bounds against four under the absolute prior, with the same two
classes on the floor either way.

### A11. MISLEADING. "a calibration drives them to those bounds as well"

**Where.** Intro l.149-153.

**Why it fails.** The first step of the fifteen-class +/-30% run put seven
classes on bounds (RESULTS l.2451-2453); at its (wall-stopped) end, one of
the twelve unconstrained classes, shrub, is on a bound (RESULTS
l.2453-2454: developed-low 0.179 = 2x, developed-high 0.127, pasture 0.062,
woody wetland 0.063, none at a bound). Sec 6.4 l.1862-1864 states it
correctly as the first step.

**Fix.** "and the first step of a calibration drives seven of the fifteen
classes to those bounds".

### A12. MISLEADING. "the misfit's unconstrained minimum"

**Where.** Sec 6.3 l.1615-1616: "The latter [alpha = 0.30] is the misfit's
unconstrained minimum and reaches n = 0.008".

**Why it fails.** tab:alpha's caption l.1440-1441: "J falls monotonically
toward lower roughness with no interior minimum"; 0.30 is where the
implicit solve stops converging.

**Fix.** "the lowest point of the scan the solver reaches".

### A13. MISLEADING. "any value a calibration reports for them is the prior"

**Where.** tab:learned caption l.1887-1889.

**Why it fails.** Shrub (52, narrowing <=1% and 5%) sits on the floor at
the end of the fifteen-class +/-30% run (tab:ladder l.1579). The paper's
own l.1963-1965 says the box, not the Gaussian prior, does the
constraining.

**Fix.** "is set by the prior or by the bounds of the search, not by the
data."

### A14. UNCLEAR. The 2.4 ms limit is not the median-depth cell's

**Where.** Sec 4 l.898-901: "(99.9% wet, median depth 6 mm) the
friction-rate limit is Delta t <~ 2.4 ms".

**Why.** 2.4 ms = 1 / 420 s^-1, the rate the stiffest cells reach (RESULTS
l.1206-1208: 90k cells violate the limit at 0.25 s). A reader applying the
formula two lines above at 6 mm depth and sheet-flow velocity gets a limit
of order a second, not milliseconds, and concludes the number is wrong.

**Fix.** "the stiffest cells reach a friction rate of 420 s^-1, so Delta t
<~ 2.4 ms, and 90k cells violate the limit at 0.25 s".

### A15. UNCLEAR. Two FD statements read as contradictory

**Where.** Sec 2.2.2 l.453-455 "the FD value is stable across a decade of
probe step" and l.466-469 "As the probe shrinks, the FD converges to the
adjoint direction by direction".

**Why.** If the FD is stable across a decade it is not converging to
anything; the two statements are presumably about different probe ranges
and directions (domain-wide vs single-class), but the text does not say,
and a referee checking the gradient argument will stop here.

**Fix.** Name the probe range and direction for each statement.

### A16. UNCLEAR. tab:crossobs mixes the total objective with "J over 134 records"

**Where.** tab:crossobs l.2084-2086 and caption l.2092-2094.

**Why.** 24,007 is J_total (misfit 23,959 plus 47.5 of prior,
`plans/o63-gauge-weight-audit.md` l.49-50) and 30,333 is the TAO start
objective of the o62 field, which carries its 15.4 of prior; 31,313 has no
prior term. The RMSE column is unaffected at two decimals (2.837 vs 2.839).

**Fix.** Label the column as the total objective or subtract the prior
terms.

### A17. UNCLEAR. The roughness slope 0.098 has no stated range

**Where.** Sec 7 l.2230-2232.

**Why.** 0.098 is the slope over alpha in [0.7, 1.0] (0.0294 / 0.3); the
whole scan gives 0.114 and the steep end [0.3, 0.45] gives 0.148, so "about
four times" is 2.6x to 4.0x depending on the range. The like-for-like 20%
pair (0.075 vs 0.020 m) carries the claim and is unambiguous.

**Fix.** "over the same width inside the prior, alpha in [0.7, 1]".

### A18. UNCLEAR. Three statements of the crest clustering disagree

**Where.** Sec 6.1 l.1354 "event hours 29-41"; Sec 6.2 l.1409-1411
"clustering in event hours 29-42 and 60-72 with an empty gap"; tab:bands
l.1369 "crest h48-72 (middle)".

**Why.** If the second cluster starts at hour 60 with an empty gap, no
mark crests at hours 48-60, which the table's label says some do.

**Fix.** State the measured crest-time range of the middle band and use
one window boundary throughout.

### A19. UNCLEAR. Which comparison "establishes" the 15%

**Where.** Sec 6.3 l.1614-1616: "The 15% is worth having, and the
comparison that establishes it is against alpha = 0.70".

**Why.** The 15% is against the lookup (0.7188 m); the alpha = 0.70
comparison establishes the 0.078 m advantage of redistribution. A reader
can compute 0.078 / 0.6894 = 11% and think that is the 15%.

**Fix.** "The comparison that shows the extra parameters earn their keep is
against alpha = 0.70".

### A20. MISLEADING (minor). Two changes attributed to one

**Where.** Sec 5 l.1024-1025 and fig:map caption l.1042-1044: "A properly
regularized run (beta = 10^-4, 686 gauges) reaches 19.4%".

**Why.** The weight rose tenfold AND the gauge count doubled; the sentence
credits the regularization.

**Fix.** "with ten times the Tikhonov weight and twice the gauges".

---

## B. Questions for the team

Format as in `plans/team-decision-list.md`: the measurement, the options,
what each option changes in the paper. Hydrology and data-assimilation
judgments are the coauthors'; the measurements are ours.

### B1. The outlet admitted water for 38 hours. What does the paper do with that?

**Measurement.** In the production 72-hour forward the free-outflow outlet
passes inflow from hour 1 to hour 38 (-8,522 m3/s at hour 3), about
193e6 m3 in all, 19% of the water stored at hour 29, and turns outward only
at hour 39; net over the event it is a source of about 157e6 m3
(`logs/stepA/stepA_mass_balance.log`; RESULTS l.2886-2891; integral in
A2). The mechanism is the one Step A found for the perimeter: the ghost
state copies the interior, so the edge passes whatever the interior
momentum sends, with no inflow guard. Where that inflow sits at hour 29,
and whether any reaches the upstream band, is unmeasured.

| option | what it changes in the paper | cost |
|---|---|---|
| (a) report it beside the divide attribution | one sentence in Sec 6.1, "boundary inflow" added to Sec 7's candidate list, Sec 4's "cure" qualified, contribution 5 keeps the divide and names the second mechanism | none |
| (b) first verify the sign and split the 193e6 m3 by reach (censored reach vs upstream band) from the hour-29 checkpoint with the Step A script | (a) with a number attached, and an answer on the 46 marks | login node, zero node-hours |
| (c) treat it as a configuration defect: an inflow-guarded transmissive outlet (driver work; the critical-outflow guard is what pinned Newton, so the guard must be smooth) and a new 72-hour forward | supersedes the baseline, the IC scan, the spectrum's sixteen forwards and every calibration (the HANDOFF scope warning: ~60 node-hours) | large |

Recommendation from the measurement: (b), then (a). Whether (c) is
required before submission is a hydrology call: is a numerical inflow of
19% of hour-29 storage, entering at the low end, something the paper can
report and bracket, or something that invalidates the hour-29 initial
condition?

### B2. At which prior width does the paper describe the calibrated displacement?

**Measurement.** The fifteen-class displacement lies along the second
eigenvector (developed-medium and woody wetland lowered together), which
the absolute prior rates lambda = 0.68 and the +/-30% prior 2.99 (A4). The
printed 2.9 / 85.6 / 11.5 split is at the absolute prior only.

| option | what it changes |
|---|---|
| (a) rewrite the Sec 6.4 paragraph at the absolute prior as it is | one paragraph; the displacement is in a direction that prior rates slightly below the data, which is consistent with Sec 6.3's equifinality sentence |
| (b) also project the o48 field on the +/-30% eigenbasis (`o58_gauss_newton.py` on the existing o61 dumps; post-processing, no run) and print both splits | adds the number that supports "set by the data" at the width the paper assigns, and makes the three-class design (v_0, v_1, v_2 at +/-30%) the visible reason the three classes were chosen |

### B3. Does the abstract keep "measurable from forward runs alone"?

**Measurement.** Scan 0.080 m (11%); calibration 0.107 m (15%); the gap
is the scan's direction, not its method (Sec 6.4).

| option | what it changes |
|---|---|
| (a) keep the claim with both numbers: the scan gives the scale, the calibration the value | abstract sentence, contribution 2, two sentences in Sec 6.2, one word in the conclusions (A1's fix) |
| (b) remove "forward runs alone" from the abstract and keep the scan as an estimate in Sec 6.2 | the second sentence of the abstract, and the "measurement we would most like others to copy" paragraph of the conclusions needs a different object |

This is also the frame of decision 1 (lead with the measurement): if the
headline is the measurement, the measurement's own limit (it under-read
the calibration by a third) has to be in the headline too.

### B4. Is the "floor" claim worth the iterations?

**Measurement.** The fifteen-class +/-30% run was stopped descending at
1.7% per iteration with a rising projected gradient (A5); its misfit
coincides with the sigma_n run's converged 615.4 by timing.

| option | what it changes |
|---|---|
| (a) drop "suggests a floor" and "reaches the same fit sooner" | one caption clause, one Sec 6.3 sentence |
| (b) two or three more iterations from `o62_p_c15_sa0.30.txt` | possibly tab:ladder's 0.6154 row and the claim; ~2 nodes x ~1.7 h per iteration, on hold with everything else |

### B5. Is +/-30% the prior, or the box?

**Measurement.** The Gaussian prior contributes 15 units at +/-30% and 10
at +/-50% against a misfit near 630; the box alpha in [0.3, 3] does the
constraining (l.1963-1965, RESULTS l.2482-2484). The "three combinations at
+/-30%" count is a property of a Gaussian width the calibrations did not
feel.

| option | what it changes |
|---|---|
| (a) say so once, plainly: the calibrations ran under a uniform prior on [0.3, 3]; the +/-30% count is the linearized statement for the width the team defends | one or two sentences in Sec 6.4 and the Sec 2.3 prior paragraph |
| (b) narrow the box to match the width (e.g. [0.4, 1.9] at two sigma) and rerun the three-class ladder | changes the three-class values that currently define "where the three land" |

Question for Donghui and Emil: which width is the one the paper should
defend as the lookup's uncertainty, and should the search bounds encode
it?

### B6. Open from before, unchanged

Decision 2 (Emil's within-mark hold-out, ~24 node-hours) is still the only
way to put a positive generalization number in the paper; the conclusions
l.2311-2315 state it as the remaining test. No new information.

---

## C. Checked and found sound

**Mathematics.**
- Eq. (eq:mom): the Manning term -g n^2 h^{-7/3} q |q| is the conservative
  form of g h n^2 |v| v / h^{4/3}; the parameter Jacobian l.358-359
  (-2 g n h^{-7/3} q |q|) is its n-derivative; the linearized drag rate
  g n^2 h^{-4/3} |v| (l.891-893) follows.
- Algorithm 2: the seed sigma^{-2} H^T r_K, the order (accumulate mu with
  the current lambda, then propagate, then add the jump when u_{s-1} is an
  observation), the index condition 1 <= k < K, and the outputs
  lambda = dJ/du_0, mu = dJ/dn are all correct for J = sum_k
  (1/2 sigma^2) ||H u_{km} - y_k||^2. Algorithm 1's g = mu + beta (n - n_0)
  matches Eq. (eq:sigman) with beta = sigma_n^{-2}.
- The argmax injection is the envelope-theorem derivative of max_t H u(t)
  where the argmax is locally constant (l.441-446); the Gauss-Newton
  construction differences the same peak values, so the two are consistent
  with each other.
- Sec 6.4: in prior-whitened coordinates the posterior precision is
  I + Gamma^{1/2} G Gamma^{1/2}; lambda > 1 means data precision exceeds
  prior precision in that direction; Rodgers' degrees of freedom
  sum lambda / (1 + lambda) is the trace of the averaging kernel; G =
  sigma^{-2} S^T W S and the gradient S^T W r / sigma^2 are the right
  Gauss-Newton objects for J_mis = (1/2 sigma^2) sum (peak - y)^2; lambda
  scales linearly in the number of like observations and as sigma^{-2}.
- RMSE = sigma sqrt(2 J / 134) (l.2094) is the correct inversion of the
  objective; 3.243, 2.839 and 3.19 m reproduce from 31,313, 24,007 and
  30,333.
- Eq. (eq:gradcost): 14.8 / 62.9 = 0.235 and 3,199 / 3,600 = 0.89; the sum
  2.13. 518,400 steps x 70 MB = 36 TB ("tens of terabytes"). 259,200 =
  72 x 3,600. 1 s / 2.4 ms = 417 ("a factor of 400"). sigma_n 0.015 on
  n = 0.027 and 0.16 is +/-56% and +/-9%. 8.8M / 15 = 5.9e5 ("almost six
  orders").
- The IC scan keeps velocity fixed under a common scaling of (h, hu, hv)
  (l.2201-2203): correct.

**Numbers traced to logs or RESULTS lines (exact unless noted).**
- Both spectra: every eigenvalue in Sec 6.4 l.1778 and l.1786, the gap
  4.1, dofs 2.20 and 3.4 (log 3.39), the scan overlaps 0.218 / 0.699, the
  leading-vector components l.1810-1812, and the full `fig_spectrum.tex`
  data, against `logs/o61/o61_spectrum_sigma_{n0.015,alpha0.30}.txt`.
- The count ladder 1/1/2/3/4 at 10/15/20/30/50% reproduces by scaling the
  +/-30% eigenvalues by (sigma/0.3)^2.
- tab:scaling: all five mark rows (71 -> 2, 108 -> 3, 324 -> 5, 1.2e5 ->
  15 from lambda_15 = 3.867e-4) and both sigma rows (0.10 -> 3, 0.05 -> 5)
  reproduce by hand from the absolute-prior eigenvalue list.
- tab:learned: all thirty entries against the two logs' per-class
  width tables; the caption's "ten" and "eight" counts.
- tab:threefifteen: every alpha, J_mis, J_prior, iteration count and MAE
  against RESULTS o59 (l.2069-2128, l.2145-2175) and o62 (l.2448-2480).
  126 / 151 = 84%, 8.8 and 33.5 units, 85% / 84% / 83% MAE shares, the
  18.7% and the 8.5x, the 9.2 sigma (log 9.16), woody wetland's 4.2 sigma.
- tab:ladder, tab:alpha, tab:halves, tab:ic and `fig_authority.tex`
  against RESULTS and each other (ic per-0.1 differences, 0.392 slope,
  the 20% pair 0.075 / 0.020, the 0.596 m extrapolation at 7.37e-4 m per
  unit J, the 0.6894 / 0.6895 pair at RESULTS l.1903).
- tab:crossobs and tab:gaugeresid against `logs/o63`, `logs/o64`,
  RESULTS o66 item 5: +171 predicted (-29.5 x 2 + 90.7 x 2 + 66.2 x 0.74)
  vs +87 measured (894.93 - 808.26, both misfit-only in the score logs);
  -7354 / -21283 = 35%; 0.4 m of 3.24; 82% / 13% / 4% / 0.6% / 0.4%.
- tab:gauges: the four bed elevations the 09-17 trace could not find
  (34.10, 34.60, 23.60, 14.50) are in `logs/o65/o65_gauge_base.txt.zb`;
  "71% of all gauge records" below bed reproduces exactly (331 of 465,
  71.2%) from `logs/o63/obs_turning_h29_41.txt` and the .zb file; 134 =
  48 + 48 + 20 + 14 + 4.
- Sec 6.4 argmax counts 156 / 690 (22.6%), 27 / 690 (3.9%), 10 pairs
  (1.4%), bin 2100-6900 steps = "more than half an hour".
- Sec 6.1: 35 of 6,198 nodes, 13-edge outlet, median 29.9 m, 37 marks
  +7.06 m, 71 marks 1.51 / +1.20 m, 2,364 / 324 / 62% / 108 (RESULTS o37,
  mesh forensics, HANDOFF).

**Withdrawn material.** No o65 spectrum number survives: grep finds no
75.6, no "98%", no 0.33 / 0.34 step, no demeaning, no AR(1), no "factor of
ten", no "physically impossible", no "converged in one". "complementary"
occurs only in the intro (a different sense) and the acknowledgement (A9).

**Labelling.** Secs 6.3-6.5 and the captions of tab:ladder,
tab:threefifteen, tab:learned, tab:scaling and tab:crossobs carry the
fit / linearized / held-out labels as the HANDOFF rule requires; the only
unlabelled crossover found is A8 (a gauge-objective number standing in for
the mark objective's).

**Logic that holds.** The FD-gate argument of Sec 2.2.2 rests on the
direction-by-direction convergence and the additivity of per-class gaps,
which are the right tests (A15 is about wording, not substance). The
three-class falsification test is a genuine prediction with a measured
remainder. The sign-based cross-observable argument is independent of the
observation weight, as stated. The IC-scan caveats (model state, not
observation; 20% illustrative) are stated where the number is used.
