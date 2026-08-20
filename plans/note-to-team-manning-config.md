# Note to the team: designing the Manning calibration test configuration

## Where we are

**Time integration: we are using ARK-IMEX**, for two reasons that happen to
coincide:

1. It is *implicit on friction*, which is what the Harvey state requires. The
   spun-up 30 m state is 99.9% wet with a **median depth of 6.4 mm**. The
   friction rate `tb = g n^2 h^{-4/3} |v|` reaches 420/s there, so a fully
   explicit drag would need `dt < 2.4 ms` against the CFL-limited production
   step of 0.25 s (90k cells violate `dt*tb > 1` at 0.25 s, and the run
   diverges within a few steps). This is source stiffness, not an
   implementation artifact -- we verified that regularizing the drag velocities
   leaves the trajectory identical to six digits.
2. It is *differentiable*. The existing `semi_implicit` treatment is stable for
   the same reason (it caps the friction impulse at the available momentum) but
   embeds `dt` and the flux divergence in the right-hand side, so it admits no
   well-defined `df/du` and cannot carry an adjoint. ARK-IMEX integrates the
   same physical drag implicitly while keeping a clean Jacobian.

So ARK-IMEX is the differentiable member of the family the production
configuration already belongs to.

**The code runs at reasonable speed on the 2.93M-cell `Turning_30m` mesh.** One
ARK-IMEX step at the production `dt = 0.25 s` costs 6.1 s on 6 laptop cores
(down from 56.5 s after we right-sized the implicit Jacobian: the friction
Jacobian is per-cell block diagonal but was being allocated with the full flux
stencil). The remaining cost is the explicit flux stages, i.e. real work.
Rough scaling to Perlmutter CPU nodes: **~880 core-hours per forward solve for
a 6 h window**, ~5,300 for 36 h.

Two useful side effects of moving to 30 m: the **Addicks and Barker embankments
are resolved** (crest median 33.5 m at Addicks, no gaps; at 1 km they were
smoothed to a ~26 m sill that leaked through both reservoirs), and **21 USGS
gauges fall inside the mesh, 18 with Harvey stage series**.

## The next question: what should the Manning test configuration be?

The design has four axes. Our recommendation is in bold; we are looking for
your judgment, especially on the window and the observation set.

### 1. Parameterization

| option | # params | comment |
|---|---|---|
| uniform n | 1 | baseline/sanity only |
| **NLCD land-cover classes** | **~18** | **recommended**; physical, prior already exists |
| channel / floodplain split | 2-4 | simplest structure, less physical |
| per-cell + Tikhonov toward NLCD | 2.9M | most flexible, but see below |

We recommend calibrating **per-class values (or multipliers) on the NLCD
classes**, not a per-cell field, for three reasons:

- **Identifiability.** Reynolds numbers over the 2.90M wet cells: 40.6% are
  laminar (Re < 500) but hold only 2.9% of the water; 38.2% are turbulent
  (Re > 2000) and hold 92.3%. Only **10.5% of wet cells -- carrying 76% of the
  volume -- are in the regime (Re > 2000, h > 5 cm) where Manning's n is a
  physical roughness** rather than an effective sheet-flow parameter. A per-cell
  field spends millions of parameters where n is neither physical nor visible to
  a gauge.
- **Cost.** Iteration count, not parameter count, sets the bill: each gradient
  is a forward plus adjoint solve. Our 1 km per-cell runs needed 300-2000
  L-BFGS iterations; ~18 parameters should converge in tens. At 30 m that is the
  difference between feasible and not.
- **Conditioning.** At 1 km with 2,746 parameters against sparse gauges we hit
  textbook semiconvergence -- the fit improved while the parameters drifted to
  their bounds, and early stopping was doing the regularizing. Eighteen
  parameters against 18 gauges x many observation times is a well-posed problem.

**NLCD is natively 30 m**, so on this mesh it maps 1:1 with no averaging -- at
1 km it had to be block-averaged 33x33, which smears an urban surface into a
meaningless mean. Donghui's `OFMmesh ex2/code/Step03_Process_Manning.m` already
carries the 18-class table (n from 0.027 barren to 0.160 developed-high).

*Related observation, not a criticism:* the Harvey example config uses a uniform
**n = 0.015**, which is a smooth-channel value 2-10x below every developed or
vegetated NLCD class. We assume this is a placeholder chosen for lack of a
better map. If so, that is encouraging: it means the expected correction is
large and mostly one-signed, which is a far easier signal to recover from 18
gauges than small scattered adjustments -- and it means there is currently no
data-derived roughness for this domain, which is the gap this work fills.

### 2. Observation window

Cost scales directly with the number of time steps (CFL fixes `dt = 0.25 s`), so
this is the main lever we have.

- **6 h around the peak** -- ~880 core-hours per forward, cheapest
- **24-36 h covering the rising limb** -- ~3,500-5,300 core-hours per forward
- full event -- not affordable per gradient

Rather than guess which hours matter, we will measure it: one adjoint sweep
returns `dJ/dn` for observations at each time, so we can plot the sensitivity of
the misfit to roughness against observation time and read off directly which
part of the flood -- the rise, the peak, or the recession -- actually constrains
n, and which hours we would be paying for with no return. We will run that at
1 km first (minutes, not node-hours) and confirm the pattern at 30 m.

Our expectation is that the rise carries most of it, since n controls flow
resistance and therefore how fast the wave arrives, whereas peak level is set
mostly by rainfall volume. **If your experience says otherwise, tell us** --
particularly whether the recession should be included, since drainage rate is
also roughness-controlled and recessions are long (i.e. expensive).

### 3. Observations

- 18 in-mesh gauges have Harvey stage series; we compare water-surface
  elevation (the in-mesh gauges publish stage as ft NAVD88 directly).
- **Do we include the reservoir-affected gauges?** At 1 km we had to exclude 8
  of them because the dams were smoothed away. At 30 m the embankments are
  resolved, so impoundment is physical -- but gate operations are still not
  modeled, so the post-Aug-28 release hydrographs remain unmatchable. Our
  inclination: include them for the impoundment phase, exclude after releases
  begin.
- Discharge series (USGS 00060) and Harvey high-water marks are also available;
  HWMs would constrain the floodplain where gauges do not.

### 4. Forcing

Five spatially distributed rainfall products are staged (MRMS, Daymet, IMERG,
MSWEP, NLDAS). **Which is trusted for this domain?** Forcing error competes
directly with roughness error in the misfit -- calibrating n against the wrong
rain field just moves the rainfall bias into n.

## Proposed sequence

1. **30 m twin**: truth = NLCD-derived field, start from uniform 0.015, recover
   the 18 class values. Establishes identifiability and real cost with no data
   ambiguity.
2. **30 m real-gauge calibration**: same 18 parameters, NLCD map as prior,
   against the Harvey stage series.
3. Per-cell only afterwards, as a regularized refinement on top of the class
   solution, if step 2 shows the class structure is too coarse.

## What we need from you

- The NLCD-derived Manning map for this domain (or a pointer -- otherwise we
  will build it from NLCD 2021 and the class table above).
- Preferred rainfall product.
- A sanity check on the window: we will compute which hours of the flood
  constrain n from the adjoint sensitivity, but we would like to know if that
  disagrees with your experience.
- Your view on the reservoir-gauge question.
