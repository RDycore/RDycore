# Manning calibration on the 30 m Harvey mesh -- four questions

The adjoint code now runs at reasonable speed on the 2.9M-cell mesh. Before we
spend real compute, we would like to agree on the experiment.

**1. Calibrate per land-cover class, or per cell?** We suggest per class (~18
numbers, not 2.9M). We have 18 gauges in the domain; at 1 km, per-cell fitting
went wrong -- the fit kept improving while the roughness values drifted to their
limits. Also, median depth in the spun-up state is 6 mm, and Manning's n only
means much in deeper, faster water: ~10% of the wet cells, but 76% of the water
and what the gauges see. NLCD is natively 30 m, so it maps 1:1 onto this mesh,
and Donghui's script already has the class-to-roughness table -- that gives us
both a starting point and something sensible to stay near.

**2. Which rainfall product?** MRMS, Daymet, IMERG, MSWEP, NLDAS are all staged.
If the rain is wrong, calibration will quietly bury that error in the roughness.

**3. How long a window, and where?** We measured this instead of guessing -- the
adjoint gives the sensitivity of each hour's gauge readings to roughness. Twelve
hours at 1 km, once with rain falling, once just draining: **while water is
rising, roughness information accumulates ~6x faster than during drain-down, and
it accelerates** (the first 6 h give only 28% of the 12 h total), whereas
draining saturates (first 6 h = 69%). So: put the window on the rise and through
the peak, do not cut it short there, and do not pay for long recessions. **Does
that match your experience?**

**4. Can we use the reservoir gauges?** At 1 km we dropped the 8 near Addicks
and Barker -- the coarse mesh flattened the embankments and water flowed
straight through. At 30 m the dams hold and the reservoirs fill properly. Gate
operations are still unmodelled, so the releases from Aug 28 are unmatchable.
We would use those gauges up to the releases and drop them after.

**A bug you will want:** rain set in the YAML `sources:` block is silently
ignored -- an index/ID mix-up in `InitSourceConditions` means it is looked up,
not found, and quietly dropped. This affects any `grid_region_id` other than 0,
so effectively all configs, in both drivers. It went unnoticed because the
Harvey runs supply rain through the file-based path, which passes the ID
correctly. One-line fix on our branch; PR whenever you want it.

**Also:** the Harvey config uses n = 0.015 everywhere, a smooth-channel value
well below any developed or vegetated land-cover value (0.04-0.16). We assume a
placeholder -- if so that is good news, since the correction we are looking for
is then large and mostly one-directional, far easier to see with 18 gauges.

**Asks:** the land-cover roughness map if one exists (otherwise we build it),
your preferred rainfall product, and your views on 3 and 4.

## Meanwhile: what we are running

So we do not stall waiting, we are proceeding with our own best answers. These
are provisional -- tell us where you disagree and we will rerun.

1. **Parameterization: NLCD land-cover classes**, ~18 values, calibrated as
   multipliers on the class table, with that table as both the starting guess
   and the reference we regularize toward. We will build the 30 m NLCD map
   ourselves unless you already have one.
2. **Rainfall: MRMS.** It is the product the Harvey example case documents, and
   it is radar-based at the right resolution. Easy to swap.
3. **Window: 12 hours on the rising limb, ending at the peak**, based on the
   sensitivity result above. We will confirm the same pattern holds at 30 m
   before committing to the long runs.
4. **Gauges: all 18, but truncated at the start of the reservoir releases**, so
   the impoundment is used and the unmodelled gate operations are not.

We are also running a comparison of ARK-IMEX against the semi-implicit friction
you use in production, over a short Harvey window. Both integrate the same drag;
we need to know they agree, because calibration has to run on the differentiable
one and the result is only useful to you if it transfers.
