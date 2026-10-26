---
name: gate-claims
description: Verify that every gate a commit message claims actually catches what it says — revert the fix with git revert and confirm exactly the named tests fail. Use when reviewing a branch before pushing, when a commit says a test was "shown failing with this reverted", or when asked to "check the gates" or "does this test actually fail".
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

1. **Produce the revert with `git revert`**, on a scratch branch:

       git branch -f gate-check-tmp SRC && git checkout gate-check-tmp
       git revert --no-edit <commit>

   **Never by pattern-matching or by hand-editing the lines back.** This is the
   one hard rule here. A partial revert produces a *false* finding that the claim
   is overstated, which is worse than not checking, because it reads like
   diligence. It has happened: a hand-written pattern revert missed one call site
   in one file and then an entire second file, and the wrong conclusion was drawn
   both times — including a drafted reply to a reviewer conceding an error that
   did not exist.

   Confirm the revert is complete before trusting it:

       git diff --stat origin/main gate-check-tmp -- <files the commit touched>

   If `git revert` conflicts, the commit is not independently revertible — say so
   and test it another way rather than resolving the conflict by hand.

2. **Rebuild**, both arches if the change could be PETSc-version sensitive:

       PETSC_DIR=~/Codes/petsc-release PETSC_ARCH=arch-macosx-gnu-rdycore-kokkos-O \
         cmake --build build-upstream -j8

3. **Run the named gates and a margin around them**, serially. `ctest -j` breaks
   the `swe_roe_ex2b_*_chkpt_*` chains, which share filenames.

4. **Restore**: `git checkout SRC`, rebuild, and confirm the suite is green again
   before moving on. A stale binary against a switched tree produces a failure
   that is not real — this has also happened twice.

## What each outcome means

| outcome | verdict |
|---|---|
| exactly the named gates fail | the claim holds |
| named gates fail **and others too** | the claim understates; list the extra gates, and say whether they are independent coverage or the same mechanism |
| a named gate **passes** | either the claim is wrong, or the revert was incomplete — **check the revert before concluding the claim is wrong** |
| nothing fails | the gate cannot fail. This is a finding, at Style or above |

A gate that passes on one rank but fails on three is not a contradiction: a
single-rank run has no ghost cells, so a whole class of bug cannot occur there.
Name the arm rather than the family when it matters.

## Report

One row per claim: commit, gates named, gates that actually failed, verdict. Then
any gate in the suite that no commit claims, and any claim that could not be
tested because the commit would not revert cleanly. Note that reverting is a
build-and-run cycle per commit, so say which claims you tested and which you did
not rather than implying the whole series was covered.
