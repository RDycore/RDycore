# o63 audit: what the gauge calibration's weight does and does not decide

*2026-09-14. Adversarial check of the cross-observable inference before
it goes to the coauthors. Every number below traces to a file in
`logs/o62`, `logs/o63`, `logs/o64` or to `plans/RESULTS-gpu-implicit.md`;
the few inferences that do not are marked UNTRACED. Arithmetic script:
scratchpad `decomp.py` (reproduced at the end).*

## Inputs, traced

| quantity | value | source |
|---|---|---|
| gauge J at the NLCD prior, 3 active classes | 31313.32 | `o63_g_gauge_c3_sa0.30.txt` header = `o64_g_gauge_prior.txt` |
| gauge J_tot after 1 TAO iteration | 24006.56 | `o63_p_gauge_c3_sa0.30.txt` header; TAO F 0.766657 x J0 |
| solution n (22, 23, 90) | 0.27, 0.36, 0.17054 | same file; alpha = 3.000, 3.000, 1.7402 |
| gauge dJ/dn at the prior (22, 23, 90) | -44620, -53129, -6903 | `o64_g_gauge_prior.txt` |
| mark dJ/dn at the same prior (22, 23, 90) | -328, +756, +675 | `o62_g_c3_sa0.30.txt` |
| gauge dJ/dn at the o62 mark-solution (22, 23, 90) | -18849, -41090, **+6991** | `o64_g_gauge_c3_sa0.30.txt` |
| prior | beta/2 sum (alpha-1)^2, beta = 11.11 | `adjoint_test.c:1184`; check: o62 c3 J_tot 649.31 - prior 15.37 = 633.94 = score-log J_mis |
| observation error | sigma = 0.15 m, iid, 134 obs | `plans/campaigns/o63_gauge_validation.sh` (`-adjoint_obs_error 0.15`) |
| TAO | BLMVM, armijo, tolerances 1e-12, max_it 12 | `adjoint_test.c:2289-2294`, script |
| exit | 124 (300-min timeout) after 1 accepted iteration | `o63_slurm_58241452.out` |
| residual | 0.2416 -> 7.97e-4 (scaled by 1/J0) | same |
| marks MAE, prior / gauge-field | 0.7188 / 0.7609; J 808.26 / 894.93 | `o62_c3_sa0.30.log`, `o63_score_gauge_c3_sa0.30.log` |
| forward-only wall, 2 nodes | 22.5 min (02:27:22 -> 02:49:49) | `o63_slurm_58241452.out` |
| forward+adjoint wall, 2 nodes | 51 min | `o64_slurm_58242097_0.out` |

**The 134 kept observations**, reconstructed from `obs_turning_h29_41.txt`
and the cell beds in the paper's Table `tab:gauges` (exact match, 134):

| gauge | records in window | kept | observed depth above cell bed |
|---|---|---|---|
| Buffalo Bayou nr Katy | 48 | 48 | 2.31-2.78 m |
| Buffalo Bayou at Houston | 48 | 48 | 3.78-5.84 m |
| Langham Ck nr Addicks (reservoir) | 20 | 20 | 2.51-3.02 m |
| Buffalo Bayou nr Fulshear | 48 | 14 | 0.00-0.09 m |
| Bear Ck nr Barker (reservoir) | 4 | 4 | 0.14-0.21 m |

Five gauges, not four; the two main-stem gauges carry 96 of 134 (72%).
The paper's "four gauges that hold their water for the whole window"
is true of the records that exist but Langham has 20 and Bear Ck 4.

## Q1. Decomposition, and at what weight the field stays home

At sigma = 0.15:

| point | J_mis | prior term | RMSE |
|---|---|---|---|
| NLCD prior | 31313.3 | 0.00 | 3.243 m |
| o63 solution | 23959.1 | 47.49 | 2.837 m |

The prior is 1/505 of the misfit at the solution. J-value parity
(J_mis = prior at the o63 point) needs sigma = 3.4 m, but J values do
not locate the minimizer; gradients do.

Gradients in alpha at the prior (dJ/dalpha_k = n_prior,k dJ/dn_k):
gauges (22, 23, 90) = (-4016, -6376, -676); marks = (-29.5, +90.7,
+66.2). The prior's restoring force beta(alpha-1) is 3.3 at one
sigma_alpha and 22.2 at the bound. So at sigma = 0.15 the gauge pull
exceeds the prior's force at the bound by 181x (22), 287x (23), 30x (90).

**sigma alone does not bring the field back.** Under iid weighting the
pull equals the prior-at-bound at sigma = 2.0 m (22), 2.5 m (23),
0.83 m (90), and the field sits within one sigma_alpha only for
sigma >= 5.2 / 6.6 / 2.1 m. A secant-quadratic model along the observed
step (first-order predicted decrease -21283, measured -7354: 35% of
linear, so 0.5 d'Hd = 13929) refines this:

| sigma (m) | N_eff | alpha_22 = alpha_23 | alpha_90 |
|---|---|---|---|
| 0.15 | 134 | 2.52 | 1.56 |
| 1 | 134 | 2.33 | 1.49 |
| 2 | 134 | 1.95 | 1.35 |
| 3 | 134 | 1.65 | 1.24 |
| 5 | 134 | 1.32 | 1.12 |
| 1 | 15 | 1.65 | 1.24 |
| 2 | 15 | 1.24 | 1.09 |
| 2 | 5 | 1.09 | 1.03 |

Reading: at a representation-dominated sigma of 1-3 m, with the 134
records still counted as independent, the gauges still move
developed-medium 2-4 sigma_alpha from the lookup. It is the product
of the two mis-specifications (sigma AND independence: 134/0.15^2 =
5956 m^-2 of precision, against an honest ~15/2^2 = 4 m^-2, a factor
~1500) that keeps the field inside the prior. Caveat: two points on
one line through a demonstrably non-quadratic, non-monotone objective
(the gauge gradient on 90 flips sign between the prior and the o62
field). Order of magnitude only; Q5 replaces this model.

## Q2. What is sigma-independent: CONFIRMED, with two qualifications

At the prior the prior-gradient is zero, so g_tot = g_mis, and any
uniform reweighting (sigma, a uniform N_eff correction) multiplies g_mis
by a positive scalar. Direction and every sign are exactly invariant.
The sign disagreement on 23 (gauges -53129, marks +756) and on 90
(-6903, +675), and the agreement on 22 (both negative), are not
functions of sigma. Neither is the first-order effect of the gauge step
on the marks: g_marks . dalpha_gauge = (-29.5)(2) + 90.7(2) + 66.2(0.74)
= +171 J-units (measured +86.7), positive for any step length in that
direction. "Calibrating on the gauges worsens the marks" is a
statement about a direction, and it survives any sigma.

Qualification 1 (per-gauge weights): the gradient is a sum over five
gauges. Different sigma or N_eff per gauge (Houston at 3.8-5.8 m depth
vs Katy at 2.3-2.8; 96 of 134 from two gauges) is NOT a uniform
rescaling and could change the sum's sign if one gauge carries it.
There is no per-gauge decomposition in any artifact. UNTRACED, and
what the per-gauge residual dump (already on the to-do list) or the S
matrix of Q5 settles.

Qualification 2 (which class): the robust disagreement is on
developed-medium (23, the largest class). At the o62 mark-solution
(alpha_23 = 0.3) the gauge gradient on 23 is still -41090: the gauges
want more 23 everywhere the marks went. On woody wetland (90) the
disagreement holds at the prior only: at the mark-solution the gauge
gradient on 90 is +6991 (wants LESS), and under the gauges 90 stopped
at 1.74x, interior. "The same two classes to opposite bounds" is one
class robustly, one class at the prior only.

## Q3. Independence

Five smooth 15-minute hydrographs over 12 h. If the residual is a
per-gauge offset, N_eff ~ 1 per gauge (5); offset + trend + curvature,
~3 per gauge (15). Fulshear (0-9 cm depth) and Bear Ck (4 records)
carry almost nothing. Data weight inflated 9-27x; equivalent iid
sigma 0.45-0.78 m before any representation error. As a uniform
factor it changes nothing in Q2 and enters Q1 as the N_eff column:
alone (sigma 0.15, N_eff 15) it still leaves alpha_23 ~ 2.5; with
sigma >= 2 m it brings the field within one sigma_alpha.

The principled treatment of a representation-dominated residual is not
a larger sigma but a per-gauge offset nuisance parameter (project out
each gauge's mean residual). What remains is hydrograph shape, which is
what roughness physically controls. In Q5's construction this is free:
W = P (per-gauge demeaning) in place of I.

## Q4. Convergence: NOT defensible as "converged"

- TAO's tolerances are 1e-12; TAO never declared convergence. The run
  was killed at 300 min (exit 124) while computing iteration 2: after
  iteration 1 there were > 3 h left, enough for two forward+adjoints,
  and no monitor line appeared. Consistent with a line search repeatedly
  projecting trial points back onto the same corner and shrinking.
- Residual 7.97e-4 x J0 = 25 J-units per unit alpha is the norm of the
  PROJECTED gradient. With 22 and 23 on the upper bound their components
  are zeroed by the projection whenever the gradient points outward; the
  residual is then essentially |dJ_tot/dalpha_90|. The prior force on 90
  at alpha 1.74 is 8.2, so 90 is a near-stationary INTERIOR point: the
  misfit gradient on 90 fell from -676 to about -8 +/- 25 over the
  step. That is the one class where the data-prior balance is visible,
  and it says the misfit is nearly flat in 90 there.
- No gradient exists at the solution (the grad dump is start-point
  only), so whether 22/23 press against the bound (KKT) cannot be
  checked. The secant-quadratic model puts the unconstrained line
  minimum at t* = 0.76 (alpha_22 = alpha_23 = 2.5); Armijo with
  c1 = 1e-4 needs 2.1 J-units of decrease and got 7307, so it accepts
  t = 1 either way. "Exactly on the bound" is where BLMVM's first
  projected step landed, not a verdict of the data. The same is true of
  o62 sa0.50's "Residual 0, converged in one iteration": a projected
  gradient of exactly zero at a corner is what corners do.
- What survives regardless: every point on the step between t = 0.5 and
  1 has classes 22 and 23 at >= 2x the lookup (> 3 sigma_alpha), on the
  opposite side from the marks. "Well outside the prior, opposite side"
  holds; "3.0x" and "at the bound" carry no weight.

Paper wording: "one accepted quasi-Newton iteration in the five-hour
budget, stopped by the wall clock; two classes were projected onto the
upper bound, the third is interior."

## Q5. The gauge spectrum: yes, and it is the cheapest answer

Construction: perturb class k by 5% (as o58/o61), one forward each,
record modelled WSE at 13 gauges x 48 times (624 x 15 S matrix; mask,
weights and sigma become post-processing). Then for ANY sigma, any
W (iid, AR(1) blocks, per-gauge weights, the above-bed mask, per-gauge
demeaning P), H = Gamma^{1/2} S^T W S Gamma^{1/2} / sigma^2 is a
15 x 15 eigenproblem run offline. One set of forwards answers Q1-Q3
for every weighting choice, gives the per-gauge rows that settle Q2's
qualification 1, gives the angle between the gauge and mark leading
eigenvectors (the principled form of "opposite directions", replacing
two optimizer anecdotes), and with the measured gradient gives a
Gauss-Newton prediction of the sigma-honest solution in place of the
secant model above. The gauge observable has no argmax, so the
peak-time caveat of Sec 6.4 does not arise. The base forward's series
IS the per-gauge residual dump the to-do list asks for.

Cost: 16 forwards x 2 nodes x 22.5 min = 12 node-hours of wall
(reserve 35-min slots: 19 node-hours), as a 16-task job array on
m4267_g, submitted from a login shell. GPU charge factor on top. The
4-forward version (base + 22/23/90) is ~3.5 node-hours but gives only
the 3 x 3 block: enough for "do the gauges constrain THESE three at an
honest weight", not the 15-class spectrum comparable to Sec 6.4, and
it misses developed-high (24), whose gauge gradient is the largest of
all (+72214). Recommend the full 16.

Driver: the real-observation gauge path has no model-series writer;
`WriteObsTable` (`adjoint_test.c:219`) exists and is used only by the
two twin paths (lines 1955, 2562). A `-adjoint_obs_model_dump <file>`
option in the real-obs path plus a forward-only exit (`-tao_max_it 0`
errors out; `-adjoint_classes_grad_only` exists but pays an adjoint,
51 min vs 22) is ~10-15 lines. Build a gpu11.

## Sec 6.1: what survives

Paragraph "Calibrating on the gauges instead":

1. "The four gauges that hold their water..." -- survives with a fix:
   five gauges, 134 records, 72% from the two main-stem gauges.
2. "We calibrated the same three classes on the above-bed gauge stage
   (134 observations...)" -- survives; add sigma = 0.15 m, the survey
   grade, records treated as independent.
3. "The answer is that it does not, and the failure is directional
   rather than marginal" -- survives.
4. "...developed-low and developed-medium to the 3x upper bound..."
   -- numbers right; "to the bound" as a finding does NOT survive (Q4).
5. "The two observables place the same class a factor of ten apart,
   each at an edge of the same prior." -- does NOT survive. Ten is
   3.0/0.3, both box edges, both set by the box. Replace with the sign
   statement (Q2).
6. "Scored on the marks ... 0.7609 vs 0.7188 ... worse than not
   calibrating at all, while lowering the gauge misfit by 23%" --
   numbers survive; needs "at this weight", and the first-order
   direction argument makes it sigma-independent. Drop 23% (J, quadratic)
   for the RMSE number the next paragraph already gives.

Paragraph "The mechanism is the geometry":

7. "At a mark ... the misfit gradient asks for less conveyance; at a
   gauge on a bank cell the modelled stage is too low ... so it asks
   for more." -- does NOT survive, twice. (a) Sign: the marks' gradient
   on 23 and 90 is POSITIVE, so the marks ask for less roughness, i.e.
   MORE conveyance. (b) "the modelled stage is too low" at the gauges
   is measured nowhere (no per-gauge residual exists), and the same
   subsection's first paragraph says modelled WSE is 1-11 m ABOVE
   observed at all 13 gauges -- a sentence from commit c0ab9162
   (2026-08-24, before the above-bed work) that I could not trace to
   any log. Both cannot hold for Katy and Houston, where the observed
   surface is 2.3-5.8 m above the cell bed. dJ_gauge/dn_23 < 0 is
   consistent with a too-low local stage (more friction holds water on
   the cell) AND with a too-high, too-early main-stem stage (more
   upstream friction delays inflow). State as hypothesis or measure it.
8. "3.24 m RMSE ... the best field we obtained leaves 2.84 m, so
   roughness spans 12%" -- numbers survive (3.243, 2.837); "best" ->
   "the field one projected step reached".
9. "A calibration reported against the gauges alone would show a 23%
   misfit reduction and a physically impossible field" -- does NOT
   survive. Donghui (2026-09-15) rules 0.36 for developed-medium "a
   little bit high, but reasonable", so "physically impossible" is
   contradicted by our own domain coauthor. Replacement text and the
   four-rung ladder check (both fields break it, at different rungs)
   are in the decision list. Prefer 3.24 -> 2.84 m to the 23%.

Paragraph "We report this as the paper's validation result": survives.
Add that the two observables also agree on developed-low (both raise
it), which makes the disagreement specific rather than generic.

Table `tab:crossobs`: survives as the record of what ran; caption must
add sigma = 0.15 m, 134 records treated as independent, one iteration
stopped by the wall clock, and "0.27 and 0.36 were projected onto the
bound" for "are at it". Also the first sentence of Sec 6.1 (line 1195,
"1-11 m above") must be re-sourced or removed before the coauthors read
the mechanism paragraph.

## UPDATE 2026-09-15 (o65 base dump): the untraced sentence is settled

The per-gauge residual now exists (`logs/o65/o65_gauge_base.txt` minus
the obs table). Model minus observed at the kept records: Houston
+4.90 m, Katy +1.94, Fulshear +1.97, Bear Ck +1.20, Langham/Addicks
-0.61. The model is too HIGH at the gauges, as Sec 6.1's first
paragraph says; the mechanism paragraph's "modelled stage is too low
... asks for more" is wrong on the measurement as well as on the sign.
Item 7 above therefore does not survive in either half, and the
replacement mechanism sentence should read: at the gauges the model is
metres too high and the gradient asks for more upstream friction,
which slows runoff into the main stem; Buffalo Bayou at Houston, 4.9 m
high, carries ~82% of the gauge misfit and sits in the reach the o37
drainage analysis identified. The gauge calibration was fitting the
lake with friction. Qualification 1 of Q2 (one gauge carrying the sign)
is now the expected case, not a hypothetical: the gauge gradient IS
essentially the Houston gauge's.

## The strongest defensible claim

Mark's candidate: "at 30 m neither observable constrains roughness to
anything the land-cover table contains, and they fail in opposite
directions." Second half: supported, and sigma-independent (opposite
gradient signs on developed-medium at the lookup; the gauge direction
worsens the marks at first order). First half: supported for the marks
(three priors, same floor, o62) but NOT for the gauges, where the
honest alternative is that they do not constrain roughness at all --
a different claim from "constrain it to something outside the table".
Until the gauge spectrum exists, the claim is: the marks determine a
roughness the table does not contain; the gauges, at the survey's
error and counted as independent, ask for the opposite; whether the
gauges at their own error and correlation constrain roughness at all
is what the spectrum decides.

## Replacement text (sentences 4-6 and 7; not the section)

> The gauge calibration moves all three classes upward, and in a single
> projected quasi-Newton step two of them reach the edge of the
> admissible box ($3\times$; woody wetland stops at $1.74\times$),
> where the mark calibration had put developed-medium and woody wetland
> on the $0.3\times$ floor. How far that step runs is set by the
> observation weight, and the weight is not the gauges' own: the gauge
> objective used the survey grade $\sigma = 0.15$\,m and treated 134
> fifteen-minute records from five gauges as independent, against a
> residual of $3.24$\,m RMSE that is representation error at this
> resolution. What does not depend on the weight is the direction. At
> the lookup itself the two objectives' gradients on developed-medium
> have opposite sign ($\partial J/\partial n = -5.3\times10^{4}$ for the
> gauges, $+7.6\times10^{2}$ for the marks, the prior contributing
> nothing at its own centre), and no rescaling of the observation error
> or of the number of independent records can change a sign; the two
> agree on developed-low, which both raise. The step the gauges choose
> also worsens the marks at first order
> ($\nabla J_{\rm marks}\cdot\Delta\alpha_{\rm gauge} = +171$ against a
> measured $+87$), so a shorter step in the same direction would degrade
> the marks less, not differently. Scored on the marks, the
> gauge-calibrated field reaches $0.7609$\,m against the uncalibrated
> prior's $0.7188$: \emph{calibrating on the gauges at this weight makes
> the model worse at the marks than not calibrating at all}.

> At a mark on the floodplain the modelled peak is too high and the
> misfit gradient asks for less roughness; at the gauges it asks for
> more, which is consistent either with a modelled stage below the
> observed one on the bank cell or with too early an arrival on the main
> stem, and the per-gauge residual that would separate the two is not
> yet measured. Neither request is about friction.

## Secondary: passages that predate o63, and the denominators

- Sec 5 (~line 1470) "out-of-sample skill has to be tested by
  cross-validation within the upstream band instead" -> "...and the
  out-of-sample test this paper reports is against a second observable
  (Section~\ref{sec:hwm}); cross-validation within the band is the
  remaining in-observable option."
- Sec 6.3 (~1741) "Out-of-sample validation has to be cross-validation
  within the upstream band" -> "The out-of-sample test we have is the
  cross-observable one of Section~\ref{sec:hwm}, and it fails;
  cross-validation within the upstream band is the remaining
  in-observable option, because the middle band..."
- Conclusions (~2177) "the calibration then reached 15%" -> "reached
  15% in sample"; (~2186) "First, cross-validate the calibration
  reported here" -> lead with "The one out-of-sample test we ran,
  against the stream gauges, failed in sign as well as in size; the
  remaining test is a hold-out within the upstream band." The
  conclusions currently do not mention the validation at all.
- Denominators. Currently: 15% (0.107 of 0.72 m, fifteen classes),
  12.5% (0.090 of 0.7188, three classes), 84 / 85 / 83% (three-class
  share of the fifteen-class reduction, by MAE at +/-30%, by MAE at the
  absolute prior, by MAE at +/-50%; line 1942's 84% reads as an
  objective share), 23% (gauge J), 12% (gauge RMSE). One currency:
  metres of MAE at the marks. "How much of the error": fraction of the
  prior's 0.7188 m (15% fifteen classes, 12.5% three, +5.9% for the
  gauge field). "Three vs fifteen": fraction of the fifteen-class
  0.107 m, quoted once at the paper's prior (84%); drop the 85% and
  83% variants and every objective-based share. Gauges: metres RMSE
  (3.24 -> 2.84, 12%) and never the 23% of J, which a reader will set
  beside 12.5% and conclude the gauge calibration did better. Lines:
  87, 91, 146, 212, 1244-45, 1753, 1942, 1949-50, 1981, 2065, 2177, and
  Sec 6.1's 23% / 12%.

## Arithmetic (decomp.py)

```python
sig=0.15; N=134; beta=1/0.30**2; J0=31313.32; Jtot1=24006.56
nprior={22:0.09,23:0.12,90:0.098}
g_n={22:-44619.87548,23:-53129.23959,90:-6902.503731}      # o64_g_gauge_prior.txt
n1={22:0.27,23:0.36,90:0.1705411928}                          # o63_p_gauge_c3_sa0.30.txt
gm_n={22:-327.8729023,23:755.9332114,90:675.3149776}         # o62_g_c3_sa0.30.txt
a1={k:n1[k]/nprior[k] for k in n1}
prior1=0.5*beta*sum((a1[k]-1)**2 for k in a1)                # 47.49
Jmis1=Jtot1-prior1                                           # 23959.1
g_a={k:nprior[k]*g_n[k] for k in g_n}                        # -4016, -6376, -676
d={k:a1[k]-1 for k in a1}; gd=sum(g_a[k]*d[k] for k in d)    # -21283
c=(Jmis1-J0)-gd                                              # 13929 = 0.5 d'Hd
dd=sum(v*v for v in d.values())
def tstar(sigma_m,neff): w=(sig/sigma_m)**2*neff/N; return -gd*w/(2*(c*w+0.5*beta*dd))
```
