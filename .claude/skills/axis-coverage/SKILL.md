---
name: axis-coverage
description: Check whether a change is tested on every configuration axis RDycore varies — operator backend, rank count, build type, index size, mesh type, input source, option combinations, run duration. Use when reviewing a branch or MR, when adding a test case, or when asked "is this tested?" or "what configurations does this cover?".
argument-hint: <branch | commit-ref | empty for HEAD>
---

`SRC` is `$ARGUMENTS` if given, else `HEAD`; compare against `origin/main`.

Six of the eight bugs fixed in the AMR series were configuration-shaped: the code
was correct on the axis it was developed on and wrong on another. The suite is
meant to be a cross-product, not a list.

## The axes

| axis | values | crossed by the suite today |
|---|---|---|
| operator backend | `basic`, `ceed` | **yes** |
| timing mode | plain, `-preload` | partly |
| rank count | 1, 2, 3 | thinly — np3 is rare |
| build type | debug, optimized | **no** — one build per job |
| PETSc index size | int32, int64 | **no** — a build axis |
| mesh type | uniform tri, graded tri, quad, mixed elements, `.exo` DEM | **no** |
| input source | elevation from mesh geometry vs `grid.cell_elevation` file | **no** |
| option combinations | `-refine_at_start`, `-refine_allow_coarsening`, `-rdy_amr_balance 0` | partly |
| run duration | long enough to reach quiescence or a no-op adapt | **no** |

Confirm the "crossed today" column rather than trusting it:

    cd <build-dir> && ctest -N | grep -oE '_(basic|ceed|preload)$|_np_[0-9]+' | sort | uniq -c

## What to do

For each axis, ask two questions and answer both from the diff and the test
registrations:

1. **Does this change interact with the axis?** A change to an operator interacts
   with the backend axis. A change to cell ownership, a halo or a partition
   interacts with rank count. A per-cell cap or limit interacts with mesh type. A
   field read from a file interacts with input source. Anything with a steady
   state or a quiescent limit interacts with duration.
2. **Is there a test on the far side?** Not "is there a test" — is there one in
   the configuration where the interaction bites.

Specific checks that have each caught a real bug:

- **A new case needs a `_ceed` arm.** Seconds to run.
- **A new case needs `np_3` if it touches the mesh, ownership or a halo.** np 1
  has no ghosts at all and np 2 cannot make a three-way partition corner, so
  neither can distinguish "owned cells lead the chart" from "they do not".
- **A cap, a limit or a per-cell quantity must be run on a graded mesh**
  (`graded_40x2_tri.msh`). A uniform mesh cannot tell a per-cell limit from a
  global one — this is exactly why a refinement-cap bug survived.
- **A field on cells should be run from a file as well as from geometry**
  (`grid.cell_elevation`). Different code path.
- **A steady state must be run long enough to reach it.** The coarse-DM cycle
  needed an adapt that refined nothing, and no committed case went that far.
- **A new `PetscAssert`, `PetscCheck` or argument check needs a debug build.**
  `PetscAssert` compiles to `PetscAssume` without `PETSC_USE_DEBUG`, which is the
  configuration RDycore ships — so an unchecked check is the default.

## Report

One line per axis: interacts yes/no, covered yes/no/partly, and where the
covering test is. Report an uncovered interacting axis as a **coverage gap**, not
a finding, unless a concrete failure on it can be named — then it is a finding
with that trigger. Do not report gaps on axes the change cannot reach.
