> **SUPERSEDED.** Read `plans/PROJECT-STATE.md` for current status.
> This file is kept for history. Claims here that are now known to be
> wrong are listed at the top of that document.

# Session Handoff — August 26, 2026 (afternoon)

Branch `adams/gpu-implicit`, committed and pushed. Laptop tree clean, **22**
adjoint/calibration ctests pass. PM builds: **`build-claude-gpu7`** (current
for calibration) and **`build-claude-gpu8`** (adds
`-adjoint_classes_grad_only`). Never rebuild a build directory a running or
queued job launches from — that is why there are eight.

**Supersedes** `session-handoff-2026-08-26.md` (the morning file). Read
`plans/PROJECT-STATE-2026-08-26.md` for the standing picture, but note that
its Section 4 — "we have never completed a successful calibration" — **is no
longer true**, and its roughness-authority numbers have been superseded.

---

## 1. The headline: the hole is closed, and the answer moved

The morning's Section 3 asked for a calibration that runs to completion.
There is now a calibrated real-data number, and it clears the bar it was
built to clear:

| field | MAE | vs NLCD | defensible |
| --- | --- | --- | --- |
| NLCD lookup (uncalibrated) | 0.7188 | — | yes |
| uniform alpha 0.70 (prior-consistent optimum) | 0.6894 | −0.029 | yes |
| uniform alpha 0.30 (unregularized floor) | 0.6392 | −0.080 | no, n to 0.008 |
| **calibrated, 15 classes, 2 iterations** | **0.6242** | **−0.095** | mostly; 1 class pinned |

Against the honest one-parameter competitor (alpha 0.70, not the
indefensible 0.30) the calibration wins by **0.065 m**, so spatial
redistribution reaches authority global scaling cannot. Roughness
authority revises from "~11% of the error, only at indefensible values"
to **13% at mostly defensible ones**.

**And the initial condition beats it.** Scaling antecedent storage gives
`dMAE` per unit fractional change of **0.392** against roughness's
**0.098** — four times the authority, from a perturbation far easier to
defend (a 20% water error after a 29-hour spin-up under radar rainfall).
Monotone, no interior optimum, and *convex* where roughness is concave:
the model does not hold the wrong distribution of water at hour 29, it
holds too much. Treat this as a **diagnosis, not a knob** (Section 5).

## 2. Why the calibration had never worked

A scaling defect, not the adjoint — the FD gates always passed. BLMVM's
first trial is `x - g`, and with Manning *n* as the variable
`|g| = 1554` where `n ~ 0.05` put every trial on the lower bound, where
the solver provably fails.

`o48_smoke.sh` reproduces the zero-step failure in **five minutes** on
the full production wiring (600-step window). That control is the most
reusable thing built today: any future "the optimizer will not move"
question costs five minutes, not twelve hours.

## 3. Driver additions (all gated, all in `driver/adjoint_test.c`)

| option | what it does |
| --- | --- |
| `-adjoint_classes_relative` | optimize `alpha = n/n_prior`, objective scaled by `1/J(start)`; both O(1) so the unit quasi-Newton step is a ~10% roughness change. Bounds become `alpha in [0.3, 3]`, which also excludes the region where the implicit solve fails. |
| `-adjoint_sigma_n <s>` | `beta = 1/s^2` from a prior std dev on *n*. |
| `-adjoint_sigma_alpha <s>` | the same prior as a **fraction** of the prior value. Prefer this — see Section 6. |
| `-adjoint_classes_grad_dump` | `(code, n, dJ/dn)` at the start point, before TAO. |
| `-adjoint_classes_grad_only` | exit after that dump: one objective+gradient per run, for assembling a Hessian by columns. |
| `-adjoint_ic_scale <a>` | scale the whole restart state (velocities unchanged) — the IC analogue of the roughness scan. |
| `-adjoint_hwm_eval_only` + `-adjoint_classes_init` | score a dumped class table for ONE forward instead of a forward+adjoint. |
| per-iteration dump | `-adjoint_classes_dump` now rewrites after every TAO iteration, so a queue wall is a stopping rule rather than a lost slot. |
| `-adjoint_jred_gate` | now applies to the classes branch too. |

## 4. Measurements that changed decisions

- **n16 is 2.42x faster than n4** and all of n4/n8/n16 reproduce
  `J = 8.082566e+02, MAE 0.7188` to every printed digit. Production runs
  on `-N 4 / srun -n 16 -G 16`. This *corrects* a note written earlier
  the same day claiming the mesh does not scale past one node — it does
  not scale to 64, but it scales usefully to 16.
- **armijo beats More-Thuente.** Lower objective from a *smaller* step,
  falling gradient (−6.3% vs +48%), half the pinning. The mechanism:
  More-Thuente's expansion buys 15.8 units of misfit for 50.5 units of
  prior violation. Production uses `-tao_ls_type armijo`.
- **The production gradient reshuffles the reduced-parameter design.**
  Ranked by `|dJ/dn| * n_prior`: 23 (28.8%), 90 (21.0%), 81 (10.8%),
  22 (9.4%). Pasture moves 7th->3rd from the 2-hour proxy, so designing
  `o50` on the short window would have picked the wrong third parameter.
- **The sigma_n balance predicted MAE 0.6895 at alpha 0.70; measured
  0.6894.**

## 5. Two corrections I made mid-session, both worth knowing

- **Iteration 1's 91%-of-the-domain pinning is a transient**, not the
  minimizer. Iteration 2 pulled six of seven classes off the bounds and
  cut the prior term 4x. It was the first quasi-Newton step with `H0 = I`
  and no curvature history. The residual fell monotonically throughout
  (0.1575 -> 0.1483 -> 0.1150), which was the better signal to weight.
- **"The line search changes the path, not the minimizer" was too
  pessimistic.** Asymptotically true; within the ~10 iterations a slot
  buys, armijo is materially better, not merely slower.

## 6. The prior is mis-specified, and the fix does not dissolve the problem

A single `sigma_n = 0.015` over an NLCD table spanning `n = 0.027` to
`0.16` asserts ±56% for barren and ±9% for developed-high, and because
the penalty scales as `n_prior^2` a small-*n* class buys a large
*fractional* excursion cheaply. The iterates show exactly that: every
class driven to the **upper** bound was small-*n* (pasture 0.038,
developed-open 0.040, developed-low 0.090) and every class driven to the
**lower** bound was large-*n* (developed-high 0.160, developed-med 0.120,
shrub 0.115, woody wetland 0.098). An ordering by prior value, not by
hydrology.

`-adjoint_sigma_alpha` penalizes fractional departure instead. **The
tension survives**: holding the first step's excursion needs
`sigma_alpha ~ 0.15`. One must believe the NLCD values are good to ±15%
for the calibration to stay physical; allow ±30% and the data goes
somewhere indefensible. *That* is the result, and it is question 1 for
the domain team.

## 7. Paper state

`papers/manning-calibration/`, **28 pp**, builds clean, no undefined
references. New since this morning:

- **S7.4** the authority measurement (alpha scan, half-domain,
  equifinality connection to Beven)
- **S7.5** the calibrated result and the ladder above
- **S7.6** the initial-condition scan
- **S5** the scaling fix, and absolute-vs-fractional prior
- **Abstract and conclusions** rewritten to lead with the measurement
- **"Questions we need domain guidance on"** — five questions, each with
  what we measured, what we need, and what changes on the answer
- **Citations added after a literature check**: Pujol et al. 2024
  (DassFlow2D — closest prior work; variational HWM assimilation with
  adjoint-guided parameterization, which narrows our novelty claim),
  MITgcm/ECCO, Beven equifinality, Funke et al. wet/dry adjoint

## 8. In flight

| job | what |
| --- | --- |
| `57627058` | 12-hr batch calibration, warm-started from o53 it2, armijo — **still pending all day** |
| o55 (interactive) | same continuation, started 14:27, ~2 iterations before its 17:00 wall |
| o56 (interactive) | Hessian pilot, started 14:39, ~100 min |

## 9. What o56 is for, and the risk it carries

Every "the observable constrains about ONE roughness degree of freedom"
statement rests on scans. The eigenvalues of the prior-preconditioned
Gauss-Newton Hessian answer it directly — each eigenvalue above 1 is a
direction where data beats prior, so the count IS the supported
parameter number, and the eigenvectors say which *combinations*.

**It may challenge rather than confirm the claim.** If the production
spectrum shows 3–4 supported directions instead of ~1, S7.4 needs
softening to "two or three, the leading one dominant". That stays
consistent with everything measured — the alpha scan bounds *authority*,
not *dimension* — but it is a real possibility.

The analysis carries a free end-to-end check: the dumped gradient
includes the Tikhonov term, so the whitened prior is exactly the
identity and an unconstrained direction **must** return `lambda = 1`. On
the Houston twin the five weakest return 1.0011–1.0096. **If that
self-check drifts, do not believe the spectrum.** It already caught one
real error — the first version reported the posterior spectrum as though
it were the misfit one and counted 15 supported directions where there
are 7.

o56's output is also **prior-independent**: the misfit Hessian does not
depend on sigma, only the whitening does, so its 16 gradients can be
re-whitened under any prior without rerunning anything.

## 10. Next, in order

1. Read o56's spectrum; adjust S7.4's dimension claim to match.
2. Converge the calibration (o55 / `57627058`), then cross-validate
   **within** cluster A — the middle band carries the drainage defect.
3. Rerun the calibration under `-adjoint_sigma_alpha` once the domain
   team gives a fractional width.
4. `o50_calibrate_active.sh 6 23,90,81` — the few-parameter run, aimed by
   the production gradient. Deliberately excludes 22 to avoid the 22/23
   compensating pair that blew up in o52.
5. The initial condition / forcing, which the o54 scan says carries more
   of this residual than friction does. A 4D-Var over `u0` is where a
   careful design pass earns its keep (background covariance, positivity
   of *h*, wet/dry front motion under an IC control, component scaling
   across *h* and *hu*) — the authority scan was the cheap part.
