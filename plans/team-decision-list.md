# Manning paper: where we are, and four decisions for the team

*Draft for Mark to send. Everything below is measured; run IDs and file
names are in plans/RESULTS-gpu-implicit.md. Last updated 2026-09-14
after o63.*

---

All the runs we were waiting on have finished. Three of them changed
what the paper can claim, so before we converge on a final draft I'd
like the group's decision on four points.

## The thesis, as it stands

1. RDycore is differentiated in place: exact assembled Jacobian,
   TSAdjoint sensitivities, implicit stepping, a GPU-resident gradient
   path, every derivative gated against finite differences in CI. An
   objective and its gradient cost about 2.1 forward solves regardless
   of the number of parameters.
2. Before running a calibration, we can measure what the parameter it
   would fit is able to explain -- from forward runs alone, with no
   optimizer involved. For Hurricane Harvey on the 30 m mesh scored
   against 46 surveyed high-water marks, Manning roughness can remove
   at most about 15% of a 0.72 m model-survey error. The initial
   condition can remove more: a 20% change in stored water is worth
   0.075 m against roughness's 0.020 m at the same 20% perturbation.
3. The spectrum of the calibration problem says why. The 46 marks
   determine about three combinations of the fifteen land-cover
   classes at a +/-30% prior; calibrating those three recovers 84% of
   what all fifteen achieve.
4. **New, and the reason for this email:** the three-parameter field
   does not survive a change of observable, and neither field is
   inside the land-cover table. Details below.
5. The downstream reach is a lake behind a 13-edge outlet. The water
   balance and the mesh, not the friction, are where the remaining
   error lives.

## What the last three runs measured

**(a) The prior width does not matter above 30%.** Three classes
(developed-low 22, developed-medium 23, woody wetland 90) calibrated on
the marks:

| prior width | mark MAE | iterations | 22 | 23 | 90 |
|---|---|---|---|---|---|
| absolute, sigma_n = 0.015 | 0.6274 m | 4 | 1.63x | 0.30x | 0.30x |
| +/-30% | 0.6290 m | 3 | 2.34x | 0.30x | 0.30x |
| +/-50% | 0.6295 m | 1 | 3.00x | 0.30x | 0.30x |

Developed-medium and woody wetland go to the bottom of the prior under
every width we ran, i.e. n = 0.036 and 0.029 -- below the lookup's own
value for developed open land (0.040). Developed-low goes wherever the
prior lets it while the MAE moves 0.002 m. What the marks determine is
a roughness the table does not contain; what they don't determine, the
prior sets.

**(b) The gauges see a different problem entirely.** Using the
above-bed mask (an observation counts only while the observed water
surface is above the model cell's bed), the gauge misfit at the NLCD
prior is **3.24 m RMSE**. The mark-calibrated fields lower it by 3-10%.
At 30 m the model cell holds the bank of an incised bayou while the
gauge hangs in the channel, so most of that 3 m is representation
error that no roughness value can remove.

**(c) Calibrate on the gauges and the model gets worse at the marks.**
Donghui's design, run as specified -- calibrate the three classes on
the above-bed gauge stage, validate on the 46 marks:

| field | 22 | 23 | 90 | gauge J | mark MAE |
|---|---|---|---|---|---|
| NLCD lookup | 0.090 | 0.120 | 0.098 | 31,313 | 0.7188 m |
| calibrated on GAUGES | 0.27 (3x) | 0.36 (3x) | 0.171 | 24,007 (-23%) | **0.7609 m** |
| calibrated on MARKS | 0.210 | 0.036 | 0.029 | 30,333 (-3%) | 0.6290 m |

The two observables put developed-medium a factor of ten apart, each at
an edge of the same prior. The gauge-calibrated field is *worse at the
marks than not calibrating at all*. The marks sit on floodplain cells
where the modelled peak is too high and ask for less friction; the
gauges sit on bank cells where the modelled stage is too low and ask
for more. Neither request is about friction.

We predicted this in writing before the run, from the sign of the gauge
gradient, and recorded the prediction in the repository first. That is
worth a sentence in the paper on its own.

## The four decisions

### 1. Does the paper lead with the three-parameter calibration, or with the reason there isn't one?

| option | what the abstract says | risk |
|---|---|---|
| **A. Lead with the measurement (recommended)** | roughness can explain 15% of this error; three parameters capture most of that; neither observable's answer is inside the table, so the error is elsewhere | a reviewer may want a "successful" calibration; we don't have one |
| B. Lead with the three-parameter field | a three-parameter calibration removes 12.5% of the survey error | we would be reporting as a result a field we then show does not transfer and is outside the prior |

I lean strongly to A: it is what we measured, it is the more useful
paper, and B invites exactly the review question we cannot answer.

### 2. How do we present a negative validation result?

The cross-observable test is now the paper's validation, and it fails.
Options: (a) report it in Section 6 as it stands, with the 3.24 m gauge
RMSE beside it so no one reads "calibrated on gauges" at face value;
(b) also add a hold-out within the marks (Emil's two spatial folds,
~24 node-hours) so the paper has an in-observable generalization
number as well; (c) drop the gauge test and report only the marks.

(c) is not honest given that we ran it. (a) is the minimum. (b) costs a
day of machine time and is the only thing that would let us say
anything positive about generalization -- **does the group want it?**

### 3. Is a 20% error in antecedent stored water plausible?

The paper's last section says the initial condition has more authority
than roughness (0.075 m vs 0.020 m at a 20% perturbation), and that
the next target is the water balance rather than the friction. That
rests on a 20% error in water stored at event hour 29, after a 29-hour
spin-up under radar rainfall, being physically plausible. Donghui has
said it is; I'd like that confirmed on the record, ideally with a
sentence we can cite.

### 4. Venue and timing.

GMD is the current target (abstract length is fine, a code-and-data
availability section is in, the marks come from the USGS STN Flood
Event Viewer API). The draft is 32 pages. Open placeholders: the
Zenodo DOI, and the archive location for the mesh, the checkpoint and
the rainfall. **Who mints the DOI, and where do the inputs live?**

## Responses received

### Donghui (2026-09-14)

**On (c), the ordering argument.** Donghui reads the gauge-calibrated
field as physically consistent because it raises both developed
classes and keeps developed-medium (0.36) above developed-low (0.27),
whereas the mark-calibrated field inverts them (0.036 vs 0.210), and
medium-intensity development should be rougher than low.

What the artifacts say about that:

- The gauge field's ordering is the lookup's, not the gauges'. Both
  classes sit at 3x their prior (0.12 and 0.09), the same relative
  bound, and a common factor cannot reorder them. The gauges did not
  determine that 23 > 22; the box did. The marks' inversion IS
  data-driven (23 to the floor under three different priors, 22 up).
  So "physically consistent" is true of the gauge field in exactly the
  sense it is true of the prior.
- The 3x itself is not the gauges' verdict either: it is where one
  projected quasi-Newton step landed under an over-tight observation
  weight (survey-grade sigma, 134 autocorrelated records counted as
  independent). See `plans/o63-gauge-weight-audit.md`. At an honest
  weight the gauges move the classes less; the direction (up) and the
  disagreement with the marks on 23 are what is robust.
- Donghui's criticism of the mark field is the paper's own
  (RESULTS o62 reading 2: the three-class field is not defensible for
  developed/wetland ground). The two claims are about different things:
  his is ordering, the paper's is magnitude. Both hold.

**Question to put back to Donghui:** is n = 0.36 for medium-intensity
developed land (3x the lookup's 0.12) a value you would defend in a
table? The paper currently says the gauge field is "outside the
land-cover table"; if a hydrologist would accept 0.36, that sentence
weakens and Sec 6.1 must say so.

**On decision 1.** "It makes sense to reduce 15% of the uncertainty by
calibrating the Manning coefficient": Donghui wants the calibration
reported as worth having. That is compatible with option A (lead with
the measurement, the three parameters capture most of the 15%) and
argues against dropping the calibration from the abstract. Record as a
lean toward keeping the 15% prominent, not a vote for B.

**On decision 2.** Agrees the two observables' error sources differ,
and reads the cross results as we do (gauges do not help marks; marks
barely help gauges, -3%). No opinion given on Emil's hold-out (2b).
Still open.

**On decision 3.** "20% is possible, but we cannot quantify without a
long-term simulation" (one to two months of spin-up before Harvey).
Citation offered: Kiang et al. 2018, WRR, doi:10.1029/2018WR022708,
"A Comparison of Methods for Streamflow Uncertainty Estimation". That
paper measures stage-discharge rating-curve uncertainty (observed
streamflow), which is a different quantity from antecedent stored
water; it supports "20% is a normal scale of hydrologic uncertainty at
high flow", not the stored-water number itself. For the paper: state
the 20% as an illustrative perturbation whose plausibility a domain
coauthor accepts but has not quantified, cite Kiang for the scale of
streamflow uncertainty, and name the spin-up run as the way to
quantify it. Do not write "a 20% error in stored water is plausible"
as a measured fact.

**On decision 4.** No response from Donghui. **Mark's ruling
(2026-09-14): closed.** Gautam is the PI and prefers GMD, so GMD is
assumed. Timing is not a decision: this is a new configuration of
authors and the effort goes into the paper, not the venue or the
calendar. The DOI and archive-location placeholders stay on the to-do
list as logistics, not as an open decision.

### Emil (2026-09-15)

Supports computing the gauge spectrum and comparing its informative
directions with the marks', "to assess whether the two sources
constrain complementary roughness combinations using the same setup."

What that adds to the plan: Emil's question is about the overlap of
the two informative subspaces, not only the count. The analysis must
therefore report, from the same S construction and prior, (i) the
gauge eigenvalue count at a defensible weighting, (ii) the angle
between the leading eigenvectors, and (iii) the projection of each
observable's informative subspace onto the other's, so "complementary"
(orthogonal), "redundant" (aligned), or "opposed" (aligned, opposite
gradient sign) can be stated as numbers. Same 16 forwards; all three
are post-processing. Two coauthors now support the run; no objections.

Not addressed by Emil: decision 2(b), his own hold-out proposal. Still
open.

## What we'd do next, once you've decided

- Add a per-gauge residual dump to the driver (one-line change, ~3
  node-hours to rerun) so we can say *which* gauges carry the 3 m: the
  two reservoir gauges would point at the water balance, the two
  main-stem gauges at the channel geometry. I'd like this before the
  gauge result goes out to reviewers.
- Emil's hold-out, if the group wants it (decision 2).
- A final pass over Sections 2-5 is done; Section 6 gets restructured
  once decision 1 lands.

### Donghui on n = 0.36 (2026-09-15)

Asked whether 0.36 for developed-medium is defensible in a table:
"a little bit high, but is reasonable... the bottom line is to have
Manning n in consistent magnitude among developed land with different
intensity."

**What this settles.** The paper's Sec 6.1 sentence "a calibration
reported against the gauges alone would show a 23% misfit reduction and
a **physically impossible field**" (line 1239) is contradicted by our
own domain coauthor and must go. The gauge field's magnitudes are
acceptable.

**What it does not settle, because the criterion has four rungs, not
two.** The NLCD developed ladder is 21 open 0.040, 22 low 0.090, 23
medium 0.120, 24 high 0.160 -- monotone in intensity. The three-class
design freezes 21 and 24 at the lookup and moves only 22 and 23, so any
large move breaks the ladder somewhere:

| field | 21 open | 22 low | 23 med | 24 high | ladder |
|---|---|---|---|---|---|
| NLCD lookup | 0.040 | 0.090 | 0.120 | 0.160 | monotone |
| gauge-calibrated (o63) | 0.040 | **0.27** | **0.36** | 0.160 | breaks at 23->24: high is *less* rough than low |
| mark-calibrated, 3 class (o62) | 0.040 | **0.210** | **0.036** | 0.160 | breaks at 22->23: medium below *open* |
| mark-calibrated, 15 class (o62) | 0.059 | 0.179 | 0.036 | 0.127 | breaks at 22->23 |
| gauge, offset removed (o65 GN) | 0.040 | 0.030 | 0.041 | 0.160 | 22 and 23 ordered, but both below open |

Donghui compared 22 against 23 and found the gauge field consistent. On
the full ladder it is not: developed-high, pinned at 0.160 because it is
outside the active set, ends up rougher than nothing and smoother than
both classes below it. By his own stated bottom line no field we have
produced is consistent -- which is the paper's "neither field is inside
the table", now on a sharper criterion than "the values are too low".

**Question back to Donghui:** does the consistency requirement include
developed-high? In the three-class design 21 and 24 are frozen at the
lookup, so 22 = 0.27 sits beside 24 = 0.16. If that ordering matters,
the physically admissible calibration is one that moves the developed
classes *together*, which is a constraint none of our runs imposed and
which the spectrum says the marks cannot resolve anyway (the
developed-low direction has lambda = 1.13, data and prior contributing
almost equally).

**Interaction with o65, and why this makes the gauge result stronger
evidence, not weaker.** o65 showed the gauge calibration is absorbing a
4.9 m model-high bias at Buffalo Bayou at Houston, 82% of the gauge
misfit, in a reach the drainage analysis already flagged. Donghui's
ruling means that fit produces roughness values a reviewer would wave
through. A plausible-looking field obtained by soaking up a drainage
defect is the more dangerous outcome, and it is exactly the cautionary
case the paper should report: the misfit reduction is real, the field
looks reasonable, and both come from a bug.

**Replacement for the Sec 6.1 sentence** (claim only, not the section):

> A calibration reported against the gauges alone would show a $23\%$
> misfit reduction and roughness values a reviewer would accept ---
> $0.27$ and $0.36$ on developed-low and developed-medium, high but
> within argument, and in the lookup's own order. That is what makes it
> worth reporting. Neither the reduction nor the plausibility of the
> field is evidence about friction.

(The $23\%$ should still become the RMSE statement, $3.24 \to 2.84$\,m,
per the denominator cleanup.)

### CORRECTION to the o65 entry below (o66, 2026-09-15 evening)

The gauge sensitivity matrix turned out not to be valid at a 5% step, so
**the eigenvalue counts, the subspace overlap and the demeaning sign
claim in the next section are withdrawn** (details in RESULTS o66). Do
not send those numbers. What replaces them is simpler, needs no spectrum,
and says the same thing more directly:

**99.5% of the gauge misfit is a constant per-gauge offset.** Splitting
J at the NLCD prior into each gauge's mean residual plus the variation
about it gives 31145 offset and 168 shape, i.e. RMSE 3.243 m overall and
**0.238 m in hydrograph shape alone**. Buffalo Bayou at Houston is
+4.90 m biased and carries 82% of the misfit. The model tracks the shape
of every hydrograph to a quarter of a metre and has the level wrong by
metres. A constant several-metre bias held for twelve hours is not
something roughness produces, so the gauge observable at 30 m is a
datum/storage measurement with a half-percent roughness-relevant part,
and the o63 calibration spent its three parameters on the other 99.5%.

Emil's question is therefore answered differently: not "the informative
directions overlap" but "the gauges carry almost no roughness
information at this resolution, and what little they carry agrees with
the marks on developed-medium only". The sign disagreement between the
two observables stands, because it comes from the adjoint, not from the
spectrum.

One thing improved rather than withdrawn: the same test applied to the
MARK columns reproduces the adjoint gradient to within 4% on every class,
so Sec 6.4's construction now has an independent check it never had.

### o65 answers Emil's question (2026-09-15) -- SUPERSEDED, see the correction above

Sixteen forwards, same construction as Sec 6.4, gauge observable
(`logs/o65/`, RESULTS o65). Class 24 excluded (its +5% column is a
>1 m stage response at Houston, not a sensitivity; central differences
pending in o66).

- **Not complementary.** 98% of the marks' informative subspace (3
  combinations at +/-30%) lies inside the gauges' (5 at the survey
  weighting). The gauges see what the marks see, plus pasture/hay.
- **Opposed in sign along both leading directions** -- at o63's
  weighting.
- **The opposition is the per-gauge offset.** Remove each gauge's mean
  residual (offset as nuisance parameter, hydrograph shape only) and the
  two gradients have the same sign and the gauges also send the
  developed classes DOWN. The disagreement is the 4.9 m bias at Buffalo
  Bayou at Houston (82% of the gauge misfit), i.e. the ponded reach.
- **How many combinations the gauges determine at their own error:**
  5 at 0.15 m iid, 1 at 1 m, 0 at 2 m and beyond.
- The per-gauge residual exists now: model HIGH by 4.90 (Houston), 1.94
  (Katy), 1.97 (Fulshear), 1.20 (Bear Ck); low by 0.61 (Langham).

For the paper this replaces "the two observables disagree" with "the
two observables agree about roughness once the downstream bias is
removed, and the bias is not a roughness error" -- a stronger and
cleaner statement of thesis item 5. Sec 6.1's mechanism paragraph must
be rewritten (model is too HIGH at the gauges). Emil's hold-out
(decision 2b) is still open.
