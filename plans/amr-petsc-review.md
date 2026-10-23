# AMR / PETSc practice review: findings to start from

Written 2026-10-22 at the end of the session that built the AMR demonstration
on `adams/amr-demo`. These are things noticed while working, with the evidence
attached, so a review session can start by confirming or dismissing them rather
than by searching. Nothing here has been fixed.

## Where the code is

| | |
|---|---|
| Branch | `adams/amr-demo` (13 commits on `main`, pushed) |
| Library | `src/rdyamr.c`, `src/rdyamr_adaptor.c`, `src/rdyamr_transfer.c`, `include/private/rdyamrimpl.h` |
| Driver | `driver/amr.c` |
| Cases | `driver/tests/amr/{channel_wave,dam_break}_amr.yaml`, meshes from `share/meshes/make_swe_tri_mesh.py` |
| Docs | `docs/user/example-cases/amr/index.md` |
| Related | PR #407 (two AMR bug fixes, required on PETSc 3.26.0), branch `adams/cmake-mpi-includes-optional` (no PR yet) |

Builds verified: macOS against the petsc-claude fork (3.26 dev) and against
upstream release 3.25.5; Perlmutter against the project's
`petsc_v3.26.0` (`load-modules-pm-gpu.sh`, `-DENABLE_FORTRAN=OFF`). All three
give 9/9 AMR tests and identical cell counts and volumes.

## A. PETSc usage worth a second opinion

0. **Fixed 2026-10-22, after the list below was first written** (commit
   b3126651): the taggers are now configured through options with
   `DMAdaptorSetOptionsPrefix`, not by reaching into the adaptor. The bug that
   exposed it is worth remembering -- `PetscObjectSetOptionsPrefix` does not
   prefix a DMAdaptor's taggers, so every `-rdy_amr_refine_vec_tagger_*` option
   in the code and the test suite was silently ignored, and the bit-identical
   gate was comparing a refined mesh against an unrefined one. `h5diff` exits 0
   when datasets are not comparable, so it never noticed. Gates that cannot
   fail are the thing to look for here.

1. **Private header dependency, now down to one use.**
   `src/rdyamr_adaptor.c` includes `petsc/private/dmadaptorimpl.h` for one
   thing only: setting `ops->computeerrorindicator` in the registered type's
   create function. The transfer uses the public
   `DMAdaptorSetTransferFunction`; the taggers are configured through options.
   `DMAdaptorRegister` is public but the indicator it installs has no public
   setter, so that is the one thing to ask PETSc for.
   `src/rdyamr.c` also includes `petsc/private/dmpleximpl.h`, which predates
   this work and is **not needed** -- it compiles with plain `petscdmplex.h`.
2. **Unchecked PETSc calls.** `InitBoundaries(rdy)` is called bare in
   `RDyPerformAMR` (`src/rdyamr.c:998`); its errors print and execution
   continues. That is how a misconfigured transform stayed invisible for a
   while during this work -- the run kept going with wrong boundary edges.
   Also bare: `PetscCalloc1` at `src/rdyamr.c:56`, `PetscFree` at `:389`, and
   the `PetscSectionGetNumFields`/`GetComponentName`/`SetComponentName` calls
   around `:793-842`.
3. **`RDyCreate` uses `PetscNew`** (`src/rdycore.c:161`), so an `RDy` carries a
   `PETSCHEADER` that is never initialised: `classid` 0, `bops` NULL, reference
   count 0. It therefore cannot be composed on a PETSc object, passed to
   `PetscObjectReference`, or validated. The AMR code works around it with a
   `PetscContainer`. Either initialise the header with `PetscHeaderCreate` and
   make `RDy` a real `PetscObject`, or drop the header so the pretence is
   visible.
4. **Options prefixes.** RDycore sets `-adapt_dm_adaptor` and the
   `adapt_`-prefixed transform options itself (`RDyAMRCreateAdaptor`), because
   `DMAdaptor` appends `adapt_` to the mesh's prefix around `DMAdaptLabel`.
   That works but is surprising; check whether a cleaner expression exists, and
   whether RDycore should set global options at all from a library routine.
5. **Dead debug blocks.** Two `if (0) { ... }` sections in `RDyPerformAMR`
   (`src/rdyamr.c:789`, `:809`) print section field names.
6. **`MatMultResuse`** (`src/rdyamr.c`) -- spelling, and it is a matrix-matrix
   product rather than a mult.
7. **Get/Restore pairing** was checked by hand in the new code but not
   systematically: `DMGetLocalVector`/`DMGetGlobalVector`, `MatGetRow`,
   `VecGetArray*`, `ISGetIndices`. Worth a pass with fresh eyes.
8. **Two meshes alive at once.** `RDyPerformAMR` now builds the new `RDyMesh`
   before destroying the old one, because the conservative transfer needs both
   geometries. Check the peak-memory implication on a large mesh.

## B. Correctness gaps, in priority order

1. **The transfer is not well balanced.** It carries `h`, so on a sloping bed a
   lake at rest does not survive an adapt -- refining gives children the
   parent's `h` while their own `z` differ, and coarsening averages `h` against
   a parent `z` that is not the mean. Fix: transfer `eta = h + z` and rebuild
   `h = max(eta - z_new, 0)`. **No AMR test has a sloping bed**: both demo
   meshes have `ZC` exactly 0, and `amr_dx1` has 2 mm of relief over 10 m and
   checks only exit status. The gate to add is exact degeneration -- still
   water over a slope stays still through several adapts -- with a no-AMR
   baseline first, since the scheme's own well-balancedness (`well_balancing:
   HR`, `RDyMeshOverride2DProjection`) has to be established separately.
2. **Wet/dry untested.** Averaging a wet cell with a dry one is the Harvey
   regime; nothing here exercises it.
3. **No load balancing after an adapt.** Refinement concentrates on the
   feature, so one rank ends up owning most of the fine cells.
4. **Adapt cost unmeasured.** Every adapt destroys and recreates the TS, the
   operator, the boundaries and the regions. The demo cases are too small to
   time meaningfully.
5. **Refinement granularity is one level per base cell**, and levels move one
   step per adapt in either direction.

## C. Verification already in place

`ctest -R amr` is 9 tests: both cases on 1 and 3 ranks, with and without
coarsening, plus `amr_off_is_bitwise`, which runs the adapt machinery with a
tagger that marks nothing and h5diffs against the plain driver.
`tools/postprocessing/amr_dam_break.py` evaluates Stoker's solution
independently (`error`), and draws the mesh (`plot`, `panels`, `movie`).

Classes claimed so far: bit-identical for the adapt-with-nothing-marked gate;
machine precision with the mechanism for the water volume across an adapt
(area-weighted transfer, only summation order changes, <= 8.2e-15 relative);
physics rate for the Stoker comparison. See [[verification-exactness-classes]].

## D. Draft questions for PETSc -- VET BEFORE SENDING

These are to be asked of the DMPlex/DMAdaptor developers, so each one costs
somebody's attention. Check every one against the current PETSc source before
it goes out, and drop any that already have a public answer. State plainly
which ones you dropped and why. We are NOT asking about DMForest or p4est:
those are for non-conforming AMR, which RDycore cannot use -- `RDyEdges` stores
exactly two cells per edge, so a hanging node has nowhere to go.

1. **The ask: a public setter for a registered DMAdaptor type's error
   indicator.** `DMAdaptorRegister` is public, but a create function has to
   write `adaptor->ops->computeerrorindicator`, which needs
   `petsc/private/dmadaptorimpl.h`. The transfer has
   `DMAdaptorSetTransferFunction`, so this looks like an omission. *Vet:*
   confirm there is still no public route in current main, and that nothing
   else in RDycore's AMR needs that header any more.
2. **What is the supported way to get each refined cell's parent?** We walk
   the saved transform (`DMPlexGetTransform`, `DMPlexTransformCellTransform`,
   `DMPlexTransformGetTargetPoint`), which broke when `DMPlexSetTransform`
   began clearing the transform's base DM. All of these are "Level:
   developer". *Vet:* look for a public parent/child or interpolation API
   between a mesh and its refinement -- `DMCreateInterpolation`,
   `DMPlexTransformGetChart`, the tree interface -- before asking.
3. **Should label adaptation of a plex have a default transfer?** With
   `DM_ADAPTATION_LABEL` and no transfer set, `DMAdaptorTransferSolution`
   calls `DMForestTransferVec` and crashes on a plex. *Vet:* confirm on
   current main; this may be worth an MR rather than a question.
4. **Is there a conservative transfer between two plexes derived from a common
   base?** Ours is Sutherland-Hodgman clipping in `src/rdyamr_transfer.c`.
   *Vet:* check `DMPlexTransferVecTree`, `DMPlexComputeClementInterpolant`,
   and anything under supermesh/intersection (`DMPlexIntersect`?) first. If
   PETSc has it, we should delete our clipping and use it.
