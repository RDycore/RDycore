# Next-phase strategy: what to demonstrate, and how to get unblocked

Written 2026-08-24 after a step-back with Mark. Supersedes the *ordering*
in `campaign-wednesday.md` (that document remains the record of the
meeting agenda and the measured results). Read `RESULTS-gpu-implicit.md`
for the engineering history.

## Working model with the scientists (Mark's call)

They are busy, and **results get them engaged more than questions do**.
So: press on as best we can, absorb their comments as they arrive, and
do not gate work on answers. Corollaries:

- Questions stay in the paper's open-questions section as a *record*,
  not as a blocking queue. Where we have acted, say so and invite
  correction (that section is already written this way).
- Prefer decisions we can defend ourselves and reverse cheaply.
- Check previous work first — Donghui Xu's papers, the OFMmesh pipeline,
  the RDycore EMS paper — before inventing an approach.

## The goal, stated plainly

**Demonstrate data-driven modeling inside RDycore**, on a problem people
care about: Hurricane Harvey, inferring Manning roughness. The
differentiator is not the algorithm but the setting — every competitor
(Inunda, Hydrograd, AegirJAX, CaMa-Flood-GPU) rewrote a *reduced* solver
inside an ML framework; we made the production Earth-system component
differentiable. That claim only lands if we calibrate against **real
data** and report a number comparable to theirs.

Benchmark to beat or match, from the closest competitor: **Inunda
(PyTorch, local-inertial) reports 0.67 m MAE against high-water marks on
a Hurricane Harvey hindcast in Harris County.** That is the number to
put ours beside.

## The two blockers, reassessed

### Blocker A — critical-outflow BC pin. OURS TO FIX. Do it.

Evidence is complete (o22): the BC is solely responsible, the drag is
exonerated, and the tolerance crutch is a treadmill (o23: rtol 3e-3
bought exactly 8 steps). Rain-forced NLCD runs are impossible until it
is fixed.

Mark's read, which I agree with: the scientists are unlikely to be
adamant about the form of a **free-outflow** boundary — it is a
numerical convenience, not a physical claim about the Buffalo Bayou
outlet. So implement a defensible smoothing ourselves, put it behind a
flag with the current behaviour as default, document the choice, and
let them correct it later. Do not wait.

Design sketch (to be verified against the literature first):
- The switch is at `uperp = 0`: below it the code zeroes both states (a
  wall), above it it imposes the critical ghost. The flux therefore
  jumps by the full wet-onto-dry Roe flux, O(g h^2 / 2).
- Candidate fixes: (a) blend the two branches over a small |uperp| band;
  (b) a Froude-limited ghost that degrades continuously to the wall
  state; (c) simply use the interior state as the ghost (zero-gradient /
  transmissive outflow) — the standard "free outflow" that is continuous
  by construction and probably what most codes do.
- (c) is the least clever and most likely to be both correct and
  acceptable — check what RDycore's CEED path and the literature do
  before building (b).
- Whatever lands must be differentiated in the Jacobian (host + device)
  and pass the FD gates, exactly as the drag fix did.

### Blocker B — gauges sit in channels the 30 m mesh cannot resolve. CHANGE THE OBSERVABLE.

The scientists may not have an answer here either, and the honest
diagnosis is that **stage in an incised channel is the wrong observable
for a 30 m mesh** — not that our model is wrong. Rather than fight the
datum, use data the mesh CAN represent.

Ranked by fit to what we already know:

1. **High-water marks (HWMs).** 2,100+ USGS-surveyed peak water-surface
   elevations across southeast Texas. Why they fit:
   - They record the **peak**, and o21A measured that model-vs-observed
     agreement is best exactly at peak inundation (bias collapsing from
     5–11 m to 0.6–3 m over 12 hr).
   - Marks are surveyed on structures and debris lines, i.e. typically
     on the **floodplain** — the surface a 30 m cell actually represents
     — rather than down in a channel.
   - They are the field's benchmark currency (Inunda's 0.67 m MAE), so
     the result is directly comparable.
   - A peak-WSE misfit needs no hydrograph timing, which sidesteps the
     clock/timing sensitivity entirely.
   - Caveat to check: HWM uncertainty is typically 0.1–0.3 m depending
     on flag quality; USGS publishes a quality code per mark. Filter.
2. **Flood extent / inundation boundary** (satellite + aerial). A binary
   wet/dry observable — arguably the thing a coarse mesh represents
   *best*, and completely immune to the channel-datum problem. Misfit
   options: cell-wise agreement, or a contour/critical-success-index
   score. Needs a differentiable surrogate (wet fraction rather than a
   hard threshold) to have a usable gradient — that is a real design
   question worth thinking about carefully.
3. **Harris County Flood Control District gauges.** A denser network
   than USGS alone in this domain. Same channel-datum caveat applies, so
   these help mainly by increasing coverage where the model is genuinely
   wet.
4. Stage at the USGS gauges, restricted per-gauge-and-per-time to where
   the model cell is genuinely wet AND the observation stands above the
   cell bed. Keeps the existing machinery; likely leaves too few
   usable pairs on its own, but is a free add-on to any of the above.

**HWMs are the recommended primary.** They convert a blocked problem
into the one the field already agrees on how to score.

## What this changes about the observability picture

The o25i result (13 gauges recover 15 NLCD classes to only 66%) was
partly an artifact of a 20-second window — see the narrowed claim in
`campaign-wednesday.md`. With HWMs the calculus changes completely:
~2,100 marks is a far denser and more spatially distributed observation
set than 13 gauges, and it samples land-cover classes the gauge network
never touches. The identifiability question should be re-asked against
the HWM set rather than treated as settled.

## Suggested order of work

1. **Free-outflow BC** (unblocks everything rain-forced). Verify the
   literature/CEED convention, implement behind a flag, FD-gate it,
   confirm the o22 controls now pass with critical-outflow replaced.
2. **Acquire and map the HWM dataset** to Turning-mesh cells, exactly as
   `map_gauges_to_cells.py` did for gauges (the CRS work is already
   done: EPSG:32610). QC against the mesh bed elevation — a mark below
   its cell's bed is the same trap as before, and we should measure how
   often that happens BEFORE building a misfit on it.
3. **Peak-WSE misfit** in the driver against HWMs; a new observation
   mode alongside gauges/classes.
4. **Re-ask identifiability** with the HWM set: which NLCD classes are
   observable now?
5. Only then the long-window rain-forced calibration.

Steps 1 and 2 are independent and can proceed in parallel; step 2 needs
no solver at all and is pure data work, so it is the safest thing to
hand to a session while the BC work is in flight.

## Open questions I cannot answer without checking

- Does the USGS HWM archive for Harvey cover the *Turning* domain
  (Buffalo Bayou / Whiteoak Bayou) densely, or is it concentrated
  elsewhere in southeast Texas? Must be checked before committing.
- What is RDycore's existing convention for outflow BCs on the CEED
  path — is there already a transmissive option we should match?
- Donghui Xu's papers: what observables and metrics does that line of
  work use for Harvey? Match where sensible so results are comparable.
