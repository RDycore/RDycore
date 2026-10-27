---
name: gate-claims
description: Verify that every gate a commit message claims actually catches what it says — put the fix back to its base revision and confirm exactly the named tests fail. Use when reviewing a branch before pushing, when a commit says a test was "shown failing with this reverted", or when asked to "check the gates" or "does this test actually fail".
argument-hint: <branch | commit-ref | empty for HEAD>
---

`SRC` is `$ARGUMENTS` if given, else `HEAD`; the range is `origin/main..SRC`.

A gate that cannot fail is the recurring defect in this project — four have been
found. A commit message that names a gate is making a testable claim, and nothing
tests it unless someone does this.

## Find the claims

    git log --reverse --format='%H %s' origin/main..SRC

For each commit, read its message and extract every test name it names. Cross
them against the real test list so option names are not mistaken for tests:

    cd <build-dir> && ctest -N | grep -oE '[a-z0-9_]+' | sort -u

## Test each claim

For each commit that names a gate:

1. **Put the changed files back to their base revision — exactly.**

   The hard rule is the *property*, not the mechanism: every file you roll back
   must end up **byte-identical to the base revision**. **Never by
   pattern-matching or by hand-editing the lines back.** A partial revert
   produces a *false* finding that the claim is overstated, which is worse than
   not checking, because it reads like diligence. It has happened: a
   hand-written pattern revert missed one call site in one file and then an
   entire second file, and the wrong conclusion was drawn both times —
   including a drafted reply to a reviewer conceding an error that did not
   exist.

   Three routes, in order of preference:

   **(a) `git revert` at the tip**, on a scratch branch, where it applies:

       git branch -f gate-check-tmp SRC && git checkout gate-check-tmp
       git revert --no-edit <commit>
       git diff --stat origin/main gate-check-tmp -- <files the commit touched>

   **(b) At the commit itself**, when `git revert` conflicts. On a stacked
   series it usually will — measured on `adams/amr-demo`, **all seven**
   mid-series commits conflicted at the tip, so (a) was unavailable for every
   one of them. Check out C and roll back only its **code** files, leaving C's
   **test** files in place, so the gate still exists and still has its
   instrument:

       git checkout --detach <commit>
       git checkout <commit>^ -- <code files only>

   Then assert completeness, which is exact and is the whole point:

       git diff <commit>^ -- <each file>   # must be EMPTY

   A file-level checkout cannot miss a call site, which is what the ban on
   pattern-matching is really protecting against. Separate code from tests
   first: the gate's instrument usually lives in the driver and must survive,
   or the gate fails for the wrong reason. A file the commit *created* is
   rolled back with `git rm`; a file it only *added lines to* still exists at
   the parent, so check it out rather than deleting it.

   **(c) Equivalence, when the commit is not separable at all** — typically
   because it introduces the API its own driver calls, so reverting the library
   breaks the build. Inject a perturbation the gate *claims* to catch and
   confirm it fails: a rank-dependent criterion for a "same on any ranks" gate,
   a 1e-13 relative nudge in the no-op path for a bit-identical gate. Say in
   the report that the claim was tested by equivalence, not by revert.

2. **Rebuild**, both arches if the change could be PETSc-version sensitive:

       PETSC_DIR=~/Codes/petsc-release PETSC_ARCH=arch-macosx-gnu-rdycore-kokkos-O \
         cmake --build build-upstream -j8

3. **Run the named gates and a margin around them**, serially. `ctest -j` breaks
   the `swe_roe_ex2b_*_chkpt_*` chains, which share filenames. With route (b),
   run them at the commit *before* rolling anything back too: a gate that does
   not pass there is telling you about the build, not the claim.

4. **Restore**: `git checkout SRC` for route (a), `git checkout <commit> -- .`
   for route (b). Rebuild, and confirm the suite is green again before moving
   on. A stale binary against a switched tree produces a failure that is not
   real — this has also happened twice.

Work in a scratch `git worktree` with its own build directory rather than in the
main checkout. Reverts and rebuilds then never leave the real tree dirty or its
binaries stale, and the main build stays usable for anything else. Configure it
the same way (`-DENABLE_FORTRAN=OFF` if the reference build has it off, and
`git submodule update --init` first, or `external/` is empty).

## What each outcome means

| outcome | verdict |
|---|---|
| exactly the named gates fail | the claim holds |
| named gates fail **and others too** | the claim understates; list the extra gates, and say whether they are independent coverage or the same mechanism |
| a named gate **passes** | the revert was incomplete, or the gate guards a *different* property than the commit says — **check the revert first, then find what the gate does hold, before concluding anything** |
| nothing fails | the gate cannot fail. This is a finding, at Style or above |

A gate that passes under a complete revert is usually still a real gate, just
attributed to the wrong mechanism. Run it down before reporting: turn each
candidate mechanism off in turn and see which one makes it fail. Measured
example — `amr_lake_at_rest_does_not_refine` is named as the gate for an
absolute indicator floor, but setting that floor to zero does **not** fail it;
removing the `err_max > 0.0` guard does. The gate is sound and the *commit
message* is wrong, which is a Style finding, not a missing gate.

A gate that passes on one rank but fails on three is not a contradiction: a
single-rank run has no ghost cells, so a whole class of bug cannot occur there.
Name the arm rather than the family when it matters.

## Report

One row per claim: commit, gates named, gates that actually failed, verdict —
and **which route (a/b/c) was used**, because a claim tested by equivalence is
weaker evidence than one tested by revert. Then any gate in the suite that no
commit claims. Note that this is a build-and-run cycle per commit, so say which
claims you tested and which you did not rather than implying the whole series
was covered.

Also spot-check any **measured numbers** the messages quote — cell-count
sequences, error norms, imbalance ratios. They are claims too, they cost one run
each, and a number measured at commit C must be re-measured at C, not at the
tip, because later commits move it.
