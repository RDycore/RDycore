# Note to the coauthors announcing the restructured draft (2026-09-17)

*Pasteable plain text, per the agreed format. Everything below is
measured unless it says otherwise. The draft is on Overleaf now; the
`manning-calibration-changes.pdf` beside it shows every change since the
version you last read (10 September) in red and blue.*

---

All,

A restructured draft is on Overleaf. The changes PDF next to it marks
everything that moved since the 10 September version, so you do not have
to re-read the whole thing.

**The result is unchanged.** Roughness explains at most about 15% of a
0.72 m model-versus-survey error on the 46 upstream high-water marks.
If the NLCD lookup is trusted to +/-30%, the marks outweigh it on about
three combinations of the fifteen classes, and calibrating just those
three recovers 84% of what all fifteen achieve. Where those three land
is the same under every prior we ran, and it is below the table's own
range for developed ground. The stream gauges cannot arbitrate: 99.5%
of their misfit at the lookup is a constant level error at each gauge,
while the model tracks hydrograph shape to 0.24 m. So the error lives in
the water balance and the mesh, not in the friction.

**What the rewrite did.** Four things, no new runs.

- The downstream ponding is now attributed rather than left as a list of
  candidate causes: a closed catchment-divide perimeter under a storm
  that exceeded the divide. That is Donghui's reading, and it matches
  our own mesh forensics (35 of 6,198 boundary nodes on the bounding
  box, so a delineated watershed; the outlet at the perimeter's low
  point). Some pond statistics that we could not trace to a logged run
  were cut rather than kept.
- The peak-water-surface observable and its adjoint moved from the
  results into Methods, where the rest of the derivative machinery is.
- A style pass on the introduction and the last three sections.
- Every number in the paper re-traced against the run logs. About
  thirty corrections; none changes a conclusion, but several sharpened
  a claim that was slightly over-stated. Examples: the gauge
  calibration moves two of the three classes against the marks, not all
  three; the level error is one to five metres at four of the five
  gauges, not at every gauge; the largest-survey row of the scaling
  table was computed from a pilot spectrum and is now the production
  one.

Emil, the draft now carries your labelling rule throughout: every
number reads as exactly one of a linearized uncertainty at the prior, an
achieved in-sample fit, or held-out skill. There is an acknowledgement
of your review in the paper; tell me if you would rather it read
differently, or not appear.

Five things I need from you.

---

**1. Emil, the within-mark hold-out. Worth 24 node-hours?**

The paper has no held-out skill number except the cross-observable test,
and that one fails in both directions:

| field | scored on marks | scored on gauges |
|---|---|---|
| NLCD lookup (uncalibrated) | 0.7188 m | 3.24 m RMSE |
| calibrated on the gauges | 0.7609 m (worse than no calibration) | 2.84 m |
| calibrated on the marks | 0.6290 m | 3.19 m |

Your proposal was two spatial folds within the 46 marks: calibrate on
one, score on the other, both ways. About 24 node-hours. It is the only
in-observable generalization number we could report.

*Our recommendation: do it if the reviewers are likely to ask for
generalization, skip it if the cross-observable failure is enough. We
lean toward doing it, because a reviewer who sees only a failed
cross-observable test may read it as a problem with the gauges rather
than with the marks' reach.*

---

**2. Donghui, the developed ladder. Does consistency include
developed-high?**

You compared developed-low against developed-medium and found the gauge
field consistent. On the full four-rung ladder no field we have produced
stays monotone, because the three-class design freezes developed-open
and developed-high at the lookup:

| field | 21 open | 22 low | 23 med | 24 high | ladder |
|---|---|---|---|---|---|
| NLCD lookup | 0.040 | 0.090 | 0.120 | 0.160 | monotone |
| calibrated on gauges | 0.040 | 0.27 | 0.36 | 0.160 | breaks at med->high |
| calibrated on marks, 3 classes | 0.040 | 0.210 | 0.036 | 0.160 | breaks at low->med |
| calibrated on marks, 15 classes | 0.059 | 0.179 | 0.036 | 0.127 | breaks at low->med |

If developed-high has to stay above the others, then the admissible
calibration is one that moves the developed classes together. No run of
ours imposed that, and the spectrum says the marks could not resolve it
anyway: the developed-low direction has an eigenvalue of 1.13, so data
and prior contribute about equally there.

*The paper does not need a ruling. It reports where each field sits
against the table and leaves admissibility to the land-cover
literature. But if you want the stronger criterion stated, say so and
we will add a sentence.*

---

**3. Donghui, your question about opening all the boundary edges.**

Short answer: it is a yaml-only change, no code. The untagged perimeter
is already collected into one auto-generated boundary, so naming it and
binding an outflow condition to it is a two-block edit.

Two cautions before we do it.

*Free outflow is transmissive, with no elevation threshold.* The ghost
state copies the interior state, so the edge passes the interior flux in
whichever direction the momentum points. A catchment divide slopes
inward, so the generic perimeter cell has inward momentum: an open
perimeter would drain the ponded reach and also manufacture inflow along
the rim. The rim is the upstream band, where all 46 calibration marks
are.

*The physically right condition is a weir at the divide, and it has to
be a flux formula rather than a switch.* Q = C_w L (eta - z_div)^{3/2}
above the divide elevation and zero below. Both the flux and its first
derivative vanish at the threshold, so Newton sees a continuous residual
and Jacobian. An "if above then open else wall" switch instead jumps by
the full wet-onto-dry flux, which is exactly the bug we hit in August:
the old critical-outflow outlet pinned the nonlinear solver at forward
step 4 of one test and step 498 of a 600-step control, at every drag
regularization we tried. Replacing it with the transmissive outlet was a
cure, not an improvement. A threshold switch on roughly 6,000 perimeter
edges, with cells crossing it continuously as the flood rises, would be
that same jump on 500 times the edges.

Two cheap steps if you want them, neither needing new code, and neither
affecting any number in the paper:

- Sum the would-be transmissive flux over the perimeter edges from an
  existing checkpoint and split it by sign. That tells us how much
  inward leakage an open perimeter would buy. Costs no allocation.
- One 72-hour forward with the perimeter open, about two node-hours, to
  see whether the 37 never-cresting marks drain and whether the 46
  upstream marks move. If the 37 drain and the 46 hold still, that
  confirms your mechanism at no cost to the paper.

*Our recommendation: treat an open perimeter as a diagnostic to report,
not as a configuration change for this paper. If it moves the upstream
water balance, every production number is measured on a superseded
setup, roughly 60 node-hours to redo.*

---

**4. Two citations we are missing.**

- The paper says RDycore is validated at up to 471M cells. We have no
  reference for that. Is there a paper or report to cite?
- For the overtopping reading in item 3: is there a reference for
  Harvey's flow crossing this watershed divide, or more generally for
  the limits of closed watershed-delineated domains under extreme
  events? Right now the paper states it as our reading rather than as a
  sourced fact.

---

**5. The archive. Where does it live?**

Two placeholders remain in Code and Data Availability: the Zenodo
identifier, which is minted at submission, and the location of the mesh,
the initial-condition checkpoint and the rainfall. The marks, the stage
records and the rainfall product are all public and cited. Whose
repository should hold the domain files, and is there a size limit we
should plan around?

---

**Gautam** -- there are three comment threads of yours on Overleaf from
31 August and 1 September that I cannot read, because Overleaf keeps
margin comments outside the document and they never reach git. They also
predate the whole rewrite. If anything in them is still live, could you
re-raise it in the text as a comment, or just send it to me?

Mark
