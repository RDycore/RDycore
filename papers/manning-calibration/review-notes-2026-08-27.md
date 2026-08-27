# Abstract review, 2026-08-27 (Mark Adams, PDF annotations)

Extracted from the annotated `manning-calibration.pdf` before that file was
overwritten by a rebuild. Each entry is the highlighted text and the comment
attached to it, in document order.

The through-line: **name the agent performing the action, drop coined
shorthand, split elliptical constructions, and do not let a sentence leave a
non-expert with the wrong conclusion.**

| # | highlighted | comment | status |
| --- | --- | --- | --- |
| 1 | "surveyed high-water marks recover the land-cover roughness classes covering 83% of a 2.93M-cell Hurricane Harvey domain," | I think it should be 'WITH surveyed HWM the land-cover ... are recovered' | fixed |
| 2 | "where the real stream-gauge network recovers almost none." | then fix the parallelism here | fixed |
| 3 | "and only at roughness 30% of published values; within defensible values, three centimetres." | this is hard to parse | fixed |
| 4 | "authority" | use simpler language for the abstract so that it is easily understood | fixed — word removed from abstract |
| 5 | "reaches" | also use simpler language, eg, "improves accuracy over a single scale" | fixed |
| 6 | "one roughness parameter," | it is not clear why the 46 marks support ONE roughness parameter. maybe 'support' is ambiguous. You then say the data can not influence 7 roughness parameters. | fixed — "support" removed; the one-vs-seven distinction now stated explicitly |
| 7 | "Applying the identical measurement to the initial condition finds" | Now ICs come in and they are a better thing to optimize, true ? | fixed — abstract now answers this: diagnosis, not a knob |
| 8 | "The same evaluation isolates a downstream reach the model cannot drain" | this is "with ICs" using the same evaluation method, use simpler language that is clear: I am not sure this is correct but something like "improve the match at downstream locations the roughness can not effect [reach might be OK here]." | fixed — the misreading was real; subject now named ("Comparing the model against the marks also exposed...") |

## Still open: the same problem in the body

The comments are about the abstract but the issue is global. The body still
leans on shorthand that a non-expert has to decode:

- **"authority"** — the paper's own coinage for "how much of the model's error
  a parameter can remove". Used in prose, in `\label{sec:authority}`, and in
  "the authority figure". Section 8.2 is already titled in plain language
  ("How much of the error can roughness explain?"), so the heading and the
  vocabulary disagree.
- **"support"** (as in "the marks support one parameter") — ambiguous exactly
  as noted above, and it appears in Section 8.4, Table `tab:scaling`
  ("Supported" column header), and the run log.
- **"reach"** — acceptable as river terminology (noted in comment 8), but the
  paper also uses "reach" as a verb for authority ("authority a single scale
  cannot reach"), so the same word carries two meanings on the same page.
