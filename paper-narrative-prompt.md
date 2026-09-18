Whole-paper narrative pass on papers/manning-calibration/manning-calibration.tex
(34 pp, builds clean with latexmk; everything committed; Overleaf in sync
at 6f1070e). Branch adams/gpu-implicit. The tree is clean except for the
root-level paper-*-prompt.md files, which are mine.

READ FIRST, in this order:
  1. plans/PAPER-SESSION-HANDOFF.md, the top section. It records what the
     09-17 restructure and the 09-18 edits did, the workflow rules, and
     what is open. It is authoritative over the .tex.
  2. ~/Codes/plasma-kinetics/paper/house-style.md, the house style.
  3. plans/team-decision-list.md, the last four entries, for Emil's
     labelling rule and the offset/shape ruling.
  4. Memory: manning-paper-session-start-here,
     no-coauthor-discussion-in-the-paper,
     vagueness-that-is-correct-at-every-zoom.

THE WORK. Read the whole paper once, for narrative: does each section
earn the next, is the thesis stated once and carried through, does the
momentum survive to the conclusions. Then go paragraph by paragraph, in
order, applying the checks below and fixing what you can. I am reading
in parallel and will send comments through you, so expect
interruptions. The abstract and the introduction were done on 09-18;
start at Section 2 and fix what you can get to first.

THE CHECKS. Each of these was learned on 09-18 by getting it wrong:
1. Undefined terms at first use. A coauthor read "land-cover table" as
   the NLCD map (cell -> class) when it means the lookup (class ->
   Manning value); "the survey" and "the marks" were used before
   anything said what they were. Every term a non-specialist could
   misread gets one clause at first use, no tutorial.
2. Four facts doing the work of one. The abstract's gauge passage
   carried four facts for one claim and needed two. For every
   paragraph, ask which facts the claim rests on; move the rest to
   where their table is. Headline numbers only in prose.
3. Name the mechanism. "Frees the twelve combinations" hid what
   happens; "leaves them with nothing to hold them but the bounds of
   the search, and a calibration drives them to those bounds" is the
   mechanism. A verb the reader cannot expand into what occurs is a
   defect.
4. Vague-but-true stays; vague-enabling-false goes. "SNES driven by the
   same Jacobian" stays, because it is true at every level of detail.
   "The whole gradient path runs on GPUs" went, because a reader could
   conclude the forward model was not. The test: what does a reader who
   does not know the mechanism conclude?
5. Give verbs their objects. "The stream gauges cannot arbitrate" read
   as a fragment; "arbitrate between the survey and the table" did not.
6. Restate the goal where a consequence is drawn. "Does not help"
   became "does not increase how much of the error roughness can
   explain", echoing the paper's question in its own words.
7. Hitting a constraint is a plausibility failure, and the reader must
   be able to see it as one. The 15% itself is reached with
   developed-medium on the floor of the search, below anything the
   table assigns to developed ground. Wherever a value sits on a bound,
   say so and say what that means.
8. Cut tangents that break momentum. The "digital twin" gloss went
   because Sec 5 defines the term where it is used. A parenthetical the
   argument does not need is a cost. Use judgement.
9. Plain words in the abstract and introduction. "In sample" became
   "the marks it was fit to". The technical term stays in the body
   where Sec 6.5 defines it.
10. Consistency, two kinds. Internal: a fix in one place must be applied
    everywhere the same fact appears (CPU node -> socket was fixed in
    the abstract and left wrong in the conclusions). Against the data:
    every number traces to logs/ or plans/RESULTS-gpu-implicit.md; the
    09-17 row-by-row audits are in plans/trace-2026-09-17/. Do not
    invent or interpolate a number; flag what cannot be traced.

RULINGS, so you do not have to ask:
- Rearrange freely. "support" is allowed. "authority", "ceiling" and
  "defensible" are defined terms (Sec 6.2); use them in that sense only
  and not before the definition.
- No coauthor discussion in the text. No \textcolor{red}. Open questions
  come to me in chat and go into the handoff. The two \textbf{[...]}
  placeholders in Code and Data Availability are not discussion; they
  stay.
- Emil's three labels: every number is a linearized uncertainty at the
  prior, an achieved in-sample fit, or held-out skill. The only held-out
  numbers are the two cross-observable scores; the gauge-calibrated
  field is worse at the marks, and the mark-calibrated field improves
  the gauges by 0.05 m of 3.24. Never write "each is worse".
- The offset/shape split is a decomposition of the residual, not of
  roughness information. Never "roughness cannot produce the offset".
- The mark sensitivities are one 5% one-sided check, 4% off the
  adjoint, not shown to converge under step refinement. Keep saying so.
- Feng et al. 2026 is cited in Sec 7 for event-scale deposition. It does
  not support the overtopping claim in Sec 6.1, which stands as our
  reading with no citation; that is settled.
- Zenodo DOI and archive location stay as placeholders.
- No runs, no scheduling. Perlmutter work is on hold.

WORKFLOW:
- I route my own edits through you and do not touch Overleaf while you
  work. Other coauthors DO edit Overleaf directly (Gautam has, twice).
  Before every push: git fetch, check for incoming commits, merge;
  never force.
- Every edit: build with latexmk and check for undefined refs; commit
  on adams/gpu-implicit; rebuild the latexdiff changes PDF against
  363971e (recipe in the overleaf memory, using old_trim.tex and the
  normalised math); push to Overleaf with a [Claude] subject. Verify
  the push against the live remote, not the local clone.
- Ask me about minor, non-structural points as they come up; do not
  batch them.
- Sign commits as the model doing the work.
