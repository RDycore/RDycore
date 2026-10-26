---
name: index-map-audit
description: Audit every use of RDycore's index-mapping arrays (local_to_owned, owned_to_local, edge id maps, the base-cell interpolation matrices) against the size and numbering of the object each one indexes. Use when the mesh, ownership or partition layout changes, when reviewing an MR that touches operators or AMR, or when asked to "audit the index maps" or "check the cell numbering".
argument-hint: <commit-ref | empty for HEAD>
---

`REV` is `$ARGUMENTS` if given, else `HEAD`. Read that revision from git
(`git show REV:<path>`, `git grep -n <pat> REV`, `git ls-tree -r --name-only REV`)
rather than the working tree, which may be on another branch. Say which `REV` you
audited.

## Why this exists

An ordering invariant in this codebase is **reified** as a named permutation
array. That makes it auditable mechanically, without knowing which change broke
it — which matters because the invariant behind the 2026-10 CEED flux bug was
written down in exactly one place: the commit that fixed it. A diff-based review
cannot find that class of bug at all, because the defect sits in a file the
branch never touches. This audit can, and when first run it found that bug plus
two more mismatches of the same class that were pre-existing on `main`.

## The numbering spaces

Established by reading allocations, not names. Re-verify if the mesh code
changes.

| Tag | Range | What is laid out in it |
|---|---|---|
| **L** — local cell | `[0, mesh->num_cells)`, = DMPlex point − `cStart` | every `RDyCells`/`RDyEdges`/`RDyVertices` array; every DM **local** vector (`u_local`, the `f_local` behind `op->ceed.rhs`, `work_local`, `locX`); `grad_*` |
| **O** — owned cell | `[0, mesh->num_owned_cells)`, the rank of the cell among `is_owned` cells **in increasing L order** | every DM **global** vector (`u_global`, `f_global`, `flux_divergence`, `primitive_variables*`, `src_*`); `external_sources`; `material_properties`; `OperatorData` arrays; the strided CEED vectors of the **source** suboperators; `amr.refine_cell`; the adaptor's `ind` |
| **E** — edge spaces | `E_L = [0,num_edges)`; `E_int = [0,num_internal_edges)`; `E_bnd = [0,boundary.num_edges)` | `edges->*` are `E_L`; `RiemannStateData`/`RiemannEdgeData` are `E_int`/`E_bnd`; `internal_edge_ids`/`boundary_edge_ids`/`boundary->edge_ids` map the ordinals into `E_L` |
| **B** — base cell | `[0, amr.num_base_cells)`, base-DM local cells, ghosts included | `amr.base_level`, `amr.base_cell_area`, `base_marked` |

**The load-bearing fact:** `O == L` **if and only if the owned cells lead the
cell chart.** They do after `DMPlexDistribute` + `DMPlexDistributeOverlap`, where
`local_to_owned` is literally the identity. They do **not** on a mesh refined
since: `DMPlexTransformGetTargetPoint` groups new points by refine type, so the
children of unmarked ghost cells precede those of marked owned cells. Every
verdict turns on this, and it is why a bug here is invisible on one rank and on
any unadapted mesh.

## The maps to audit

Enumerate them freshly — do not trust this list to be complete:

    git grep -nE '\b[a-z_]+_to_[a-z_]+\b' REV -- src include

Known as of 2026-10: `local_to_owned` (L→O, and for ghosts continues into
`[num_owned, num_cells)`, which is **not** a position in any vector),
`owned_to_local` (O→L), `dirichlet_bc_to_data_idx`, and the
`BaseToCurrentMat{1Dof,NDof}` matrices (row = fine L, col = coarse L or B).
Same role, different spelling: `region->owned_cell_global_ids` (region ordinal →
**O**, despite the name), `region->cell_local_ids` (→ L),
`edges->internal_edge_ids` / `boundary_edge_ids` / `boundary->edge_ids` (→ `E_L`),
`data2mesh_idx`.

## The one question, per use site

> **What is the size and numbering of the object being indexed here, and does the
> codomain of the map being applied match it?**

A result in numbering B must only index something laid out in B. A result used to
index something laid out in A is a mismatch.

To answer, **follow the indexed object to its allocation** — the size argument of
its creation call, its stride or block size, and which numbering that size
corresponds to. Never infer from the variable's name: `owned_cell_global_ids`
holds owned indices, and `op->ceed.rhs` is local-sized while `op->ceed.sources`
is owned-sized.

For a CEED restriction, the numbering is fixed by the `lsize` argument of
`CeedElemRestrictionCreate` and by which PETSc Vec the active vector is pointed
at in `src/operator.c`, not by the offsets array's name.

## Report

List every **mismatch**: `file:line`, the two numberings, why they differ, what
would make it right, and whether it is reachable (does an adapted multi-rank mesh
exist on this path?). Also list the sites **checked and found correct**, one line
each naming the numbering of the indexed object, so coverage is visible — a
mismatch list with no denominator is not an audit.

Where a site cannot be resolved from the source, say so and name the missing
evidence. Do not dismiss a mismatch because it looks longstanding: two of the
three found on the first run were pre-existing on `main`.

A useful cross-check when a verdict is unclear: the PETSc and CEED backends
implement the same mathematics, so compare the site against its counterpart in
`src/swe/swe_petsc.c` or `src/operator_sources_ceed.c`. Disagreement between the
two backends on the same quantity is itself the finding.
