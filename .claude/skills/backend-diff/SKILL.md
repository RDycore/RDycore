---
name: backend-diff
description: Run the same RDycore case through the PETSc operator and the libCEED operator on the same mesh and require them to agree, and compare a serial run against a parallel one by order-independent checksums. Use when reviewing a change to an operator, a flux, a source term or the mesh, or when asked to "compare the backends" or "check serial against parallel".
argument-hint: <build-dir> [case.yaml]
---

Two independent implementations of the same mathematics must agree. RDycore has
one such pair — `src/swe/swe_petsc.c` and the CEED suboperators — and **no
standing test compares their answers.**

## Backend differential

    bash .claude/skills/backend-diff/backend_diff.sh <build-dir> <tag>

Runs one AMR case twice on the same mesh and steps, once with `-ceed /cpu/self`
and once without, on 3 ranks, and compares the final depth field.

Expected when correct: identical cell counts and a relative difference at machine
precision — measured **3.3e-16** on the channel case. A failure looks like this,
from a branch with the CEED flux-offset bug:

    cells: basic 952, ceed 912
    FAIL: the two backends produced DIFFERENT MESHES -- cannot even compare values

The two backends diverging enough to choose different adapted meshes is itself
the signal; the gate trips on cell count before it reaches a tolerance.

**Blind spot, and it is large:** this sees only a defect present in exactly one
of the two implementations. Seven of the eight bugs fixed in the AMR series are
in shared code — the mesh, the transfer, the criterion — where both backends
compute the same wrong answer. Run it, report it, do not rely on it.

## Serial against parallel, by order-independent checksums

    python3 .claude/skills/backend-diff/moments.py <serial-run-dir> <parallel-run-dir>

Each run directory is one that contains `output/*.xmf`. The checksums are global
reductions, so they do not depend on how the mesh is numbered or divided — no
natural ordering and no gather to rank 0 is needed.

Measured on the channel case, serial against 3 ranks under CEED:

    sum h*A      0.00e+00      sum h*y*A    1.89e-16
    sum h^2*A    0.00e+00      max h        1.11e-16
    sum h*x*A    0.00e+00

Three agree bit-for-bit and two to one ulp, so hold this at ~1e-14.

**Do not use `sum(h*area)` alone.** It is invariant under a permutation of values
among cells — each interior face still contributes `+F` to one cell and `−F` to
another however they are misdirected — so global conservation survives the exact
bug you are hunting. That is why `-refine_check_volume` never saw the CEED flux
bug. The weighted moments break the symmetry; keep `sum h*A` only as a control
that should stay at zero.

## Before trusting a result

**Rebuild first.** A `git checkout` does not rebuild, and a stale binary against
a switched tree gives a failure that is not real. This script reads the binary in
`<build-dir>`, not the source tree, and it has already caught the author of this
skill out once: it correctly reported a failure that came from a leftover binary.
Check that `git log -1` and the build agree before reading anything into a FAIL.

## Interpreting a failure

A mismatch here does not say which implementation is wrong. Serial is usually the
reference, because on one rank there are no ghost cells and several whole classes
of indexing bug cannot occur. Compare the suspect site against its counterpart in
the other backend before concluding; disagreement between the two on the same
quantity is itself the finding. If the suspicion is an index or ownership
mismatch, run `index-map-audit` next.
