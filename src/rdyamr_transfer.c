/// Conservative transfer of a cell-centred field between two adapted meshes.
///
/// When AMR may coarsen, the new mesh is rebuilt from the base mesh rather than
/// refined from the current one, so the two meshes are different triangulations
/// of the same domain and neither is a refinement of the other. The transfer
/// below is therefore geometric: a new cell takes the area-weighted average of
/// the old cells it overlaps,
///
///     U_new[n] = sum_o |n ^ o| U_old[o] / |n|,
///
/// which preserves sum(U * area) exactly, reproduces a constant field exactly,
/// and degenerates to injection where the new mesh refines the old and to
/// averaging where it coarsens it.
///
/// Candidate pairs are restricted to cells descended from the same base cell:
/// both meshes tile every base cell, the per-rank interpolation matrices mean a
/// base cell's descendants stay on one rank, and the number of descendants is
/// small, so the search is local and cheap.
///
/// The old side of the transfer is a packed array rather than an `RDyMesh`:
/// per base cell, its descendants' polygons and values, laid out under a
/// `PetscSection` over the base mesh's cells. That is what lets the old cells
/// arrive from another rank -- `DMPlexDistributeField()` moves exactly this
/// shape along a base-mesh migration -- and the clipping needs nothing else.

#include <petscdmplex.h>
#include <private/rdyamrimpl.h>
#include <private/rdycoreimpl.h>
#include <private/rdymeshimpl.h>

#define MAX_CLIP_VERTICES 16  // a triangle clipped by a triangle has at most 6

/// Twice the signed area of a polygon, by the shoelace formula: positive when
/// the vertices run counter-clockwise.
static inline PetscReal Shoelace2(const PetscReal *x, const PetscReal *y, PetscInt n) {
  PetscReal s = 0.0;
  for (PetscInt i = 0, j = n - 1; i < n; j = i++) s += (x[j] - x[i]) * (y[j] + y[i]);
  return s;
}

/// Area of the intersection of two convex polygons, by Sutherland-Hodgman
/// clipping of `subject` against each edge of `clip`.
static PetscErrorCode ConvexIntersectionArea(const PetscReal *sx, const PetscReal *sy, PetscInt ns, const PetscReal *cx, const PetscReal *cy,
                                             PetscInt nc, PetscReal *area) {
  PetscFunctionBegin;
  PetscReal px[MAX_CLIP_VERTICES], py[MAX_CLIP_VERTICES];
  PetscReal qx[MAX_CLIP_VERTICES], qy[MAX_CLIP_VERTICES];
  PetscInt  np = ns;

  // clipping a convex polygon adds at most one vertex per clip edge; dropping
  // one instead would silently shrink the overlap and lose mass
  PetscCheck(ns + nc <= MAX_CLIP_VERTICES, PETSC_COMM_SELF, PETSC_ERR_SUP,
             "a %" PetscInt_FMT "-gon clipped by a %" PetscInt_FMT "-gon can exceed %d vertices", ns, nc, MAX_CLIP_VERTICES);

  for (PetscInt i = 0; i < ns; ++i) {
    px[i] = sx[i];
    py[i] = sy[i];
  }

  // walk the clip polygon counter-clockwise, so "inside" is to the left of
  // every edge; a clockwise polygon is walked in reverse
  PetscBool clip_cw = (PetscBool)(Shoelace2(cx, cy, nc) < 0.0);

  for (PetscInt e = 0; e < nc && np; ++e) {
    PetscInt  e0 = clip_cw ? (nc - 1 - e) : e;
    PetscInt  e1 = clip_cw ? (e0 == 0 ? nc - 1 : e0 - 1) : (e + 1) % nc;
    PetscReal ax = cx[e0], ay = cy[e0], bx = cx[e1], by = cy[e1];
    PetscInt  nq = 0;

    for (PetscInt i = 0, j = np - 1; i < np; j = i++) {
      PetscReal sidei = (bx - ax) * (py[i] - ay) - (by - ay) * (px[i] - ax);
      PetscReal sidej = (bx - ax) * (py[j] - ay) - (by - ay) * (px[j] - ax);
      if (sidei >= 0.0) {
        if (sidej < 0.0) {  // entering: add the crossing
          PetscReal t = sidej / (sidej - sidei);
          qx[nq]      = px[j] + t * (px[i] - px[j]);
          qy[nq]      = py[j] + t * (py[i] - py[j]);
          ++nq;
        }
        qx[nq] = px[i];
        qy[nq] = py[i];
        ++nq;
      } else if (sidej >= 0.0) {  // leaving: add the crossing
        PetscReal t = sidej / (sidej - sidei);
        qx[nq]      = px[j] + t * (px[i] - px[j]);
        qy[nq]      = py[j] + t * (py[i] - py[j]);
        ++nq;
      }
    }
    np = nq;
    for (PetscInt i = 0; i < np; ++i) {
      px[i] = qx[i];
      py[i] = qy[i];
    }
  }

  *area = (np < 3) ? 0.0 : 0.5 * PetscAbsReal(Shoelace2(px, py, np));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Gathers a cell's vertex coordinates projected on the x-y plane, returning
/// the vertex count.
static PetscErrorCode CellPolygon(RDyMesh *mesh, PetscInt c, PetscReal *x, PetscReal *y, PetscInt *np) {
  PetscFunctionBegin;
  RDyCells *cells = &mesh->cells;
  PetscInt  n     = cells->num_vertices[c];
  PetscCheck(n <= MAX_CLIP_VERTICES, PETSC_COMM_SELF, PETSC_ERR_SUP, "cell %" PetscInt_FMT " has %" PetscInt_FMT " vertices, more than %d", c, n,
             MAX_CLIP_VERTICES);
  for (PetscInt v = 0; v < n; ++v) {
    PetscInt vid = cells->vertex_ids[cells->vertex_offsets[c] + v];
    x[v]         = mesh->vertices.points[vid].X[0];
    y[v]         = mesh->vertices.points[vid].X[1];
  }
  *np = n;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// For each local cell of `mesh` (ghosts included, matching the rows of the
/// interpolation matrices and the layout of the local solution vectors), the
/// base cell it descends from: the single nonzero in that cell's row.
static PetscErrorCode BaseCellOfEachCell(RDyMesh *mesh, Mat base_to_mesh, PetscInt **base_of) {
  PetscFunctionBegin;
  PetscCall(PetscMalloc1(mesh->num_cells, base_of));
  if (!base_to_mesh) {  // the mesh is the base mesh
    for (PetscInt o = 0; o < mesh->num_cells; ++o) (*base_of)[o] = o;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  for (PetscInt o = 0; o < mesh->num_cells; ++o) {
    PetscInt           ncols;
    const PetscInt    *cols;
    const PetscScalar *vals;
    PetscCall(MatGetRow(base_to_mesh, o, &ncols, &cols, &vals));
    PetscCheck(ncols >= 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "cell %" PetscInt_FMT " descends from no base cell", o);
    (*base_of)[o] = cols[0];
    PetscCall(MatRestoreRow(base_to_mesh, o, &ncols, &cols, &vals));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Groups owned cells by base cell: `start[b]` .. `start[b+1]` index `cells`.
static PetscErrorCode GroupByBaseCell(PetscInt ncells, const PetscInt *base_of, PetscInt nbase, PetscInt **start, PetscInt **cells) {
  PetscFunctionBegin;
  PetscCall(PetscCalloc1(nbase + 1, start));
  for (PetscInt c = 0; c < ncells; ++c) {
    // a base index out of range would otherwise write outside start[] and
    // corrupt the heap rather than say anything; after a balance the base mesh
    // and the interpolation matrix have to be the ones that belong together
    PetscCheck(base_of[c] >= 0 && base_of[c] < nbase, PETSC_COMM_SELF, PETSC_ERR_PLIB,
               "cell %" PetscInt_FMT " claims base cell %" PetscInt_FMT " of %" PetscInt_FMT, c, base_of[c], nbase);
    (*start)[base_of[c] + 1]++;
  }
  for (PetscInt b = 0; b < nbase; ++b) (*start)[b + 1] += (*start)[b];

  PetscInt *fill;
  PetscCall(PetscCalloc1(nbase, &fill));
  PetscCall(PetscMalloc1(ncells, cells));
  for (PetscInt c = 0; c < ncells; ++c) {
    PetscInt b                        = base_of[c];
    (*cells)[(*start)[b] + fill[b]++] = c;
  }
  PetscCall(PetscFree(fill));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Packs, per base cell, the polygon and the `ndof` values of each cell
///        of `mesh` that descends from it.
///
///        The section is over the base mesh's cell points, so the result is in
///        the shape `DMPlexDistributeField()` moves along a base-mesh
///        migration: a rank that is given a base cell is given its old
///        descendants with it.
///
/// @param [in] mesh         the mesh the solution is on
/// @param [in] dm_base      the base mesh, for the section's chart
/// @param [in] base_to_mesh base-to-mesh interpolation matrix (1 dof), or NULL
///                          if `mesh` is the base mesh
/// @param [in] ndof         components per cell
/// @param [in] u_local      the solution, by local cell
/// @param [out] sec         the layout, created here
/// @param [out] packed      the data, created here
PetscErrorCode RDyAMRPackCellsByBaseCell(RDyMesh *mesh, DM dm_base, Mat base_to_mesh, PetscInt ndof, Vec u_local, PetscSection *sec,
                                         PetscScalar **packed) {
  PetscFunctionBegin;

  PetscInt cStart, cEnd;
  PetscCall(DMPlexGetHeightStratum(dm_base, 0, &cStart, &cEnd));
  const PetscInt nbase  = cEnd - cStart;
  const PetscInt stride = RDyAMRPackStride(ndof);

  PetscInt *base_of, *start, *cells;
  PetscCall(BaseCellOfEachCell(mesh, base_to_mesh, &base_of));
  PetscCall(GroupByBaseCell(mesh->num_cells, base_of, nbase, &start, &cells));

  PetscCall(PetscSectionCreate(PETSC_COMM_SELF, sec));
  PetscCall(PetscSectionSetChart(*sec, cStart, cEnd));
  for (PetscInt b = 0; b < nbase; ++b) PetscCall(PetscSectionSetDof(*sec, b + cStart, (start[b + 1] - start[b]) * stride));
  PetscCall(PetscSectionSetUp(*sec));

  PetscInt total;
  PetscCall(PetscSectionGetStorageSize(*sec, &total));
  PetscCall(PetscCalloc1(total, packed));

  const PetscScalar *u;
  PetscCall(VecGetArrayRead(u_local, &u));
  for (PetscInt b = 0; b < nbase; ++b) {
    PetscInt off;
    PetscCall(PetscSectionGetOffset(*sec, b + cStart, &off));
    for (PetscInt i = start[b]; i < start[b + 1]; ++i) {
      PetscInt     c = cells[i];
      PetscReal    x[MAX_CLIP_VERTICES], y[MAX_CLIP_VERTICES];
      PetscInt     nv;
      PetscScalar *slot = *packed + off + (i - start[b]) * stride;
      PetscCall(CellPolygon(mesh, c, x, y, &nv));
      PetscCheck(nv <= RDY_AMR_PACK_MAX_V, PETSC_COMM_SELF, PETSC_ERR_SUP,
                 "cell %" PetscInt_FMT " has %" PetscInt_FMT " vertices, more than the %d a packed cell holds", c, nv, RDY_AMR_PACK_MAX_V);
      slot[0] = (PetscScalar)nv;
      for (PetscInt v = 0; v < nv; ++v) {
        slot[1 + 2 * v]     = x[v];
        slot[1 + 2 * v + 1] = y[v];
      }
      for (PetscInt d = 0; d < ndof; ++d) slot[1 + 2 * RDY_AMR_PACK_MAX_V + d] = u[ndof * c + d];
    }
  }
  PetscCall(VecRestoreArrayRead(u_local, &u));

  PetscCall(PetscFree(base_of));
  PetscCall(PetscFree(start));
  PetscCall(PetscFree(cells));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief The conservative transfer: every cell of `new_mesh` takes the
///        area-weighted average of the packed old cells under the same base
///        cell that it overlaps.
///
/// @param [in] new_mesh    the mesh the solution is going to
/// @param [in] dm_base     the base mesh, for the section's chart
/// @param [in] base_to_new base-to-new interpolation matrix (1 dof)
/// @param [in] ndof        components per cell
/// @param [in] sec         the packed layout, over the base mesh's cells
/// @param [in] packed      the packed old cells
/// @param [out] u_new      the solution on the new mesh, by local cell
PetscErrorCode RDyAMRTransferFromPacked(RDyMesh *new_mesh, DM dm_base, Mat base_to_new, PetscInt ndof, PetscSection sec, const PetscScalar *packed,
                                        Vec u_new) {
  PetscFunctionBegin;

  PetscInt cStart, cEnd;
  PetscCall(DMPlexGetHeightStratum(dm_base, 0, &cStart, &cEnd));
  const PetscInt nbase  = cEnd - cStart;
  const PetscInt stride = RDyAMRPackStride(ndof);

  PetscInt *base_of_new, *new_start, *new_cells;
  PetscCall(BaseCellOfEachCell(new_mesh, base_to_new, &base_of_new));
  PetscCall(GroupByBaseCell(new_mesh->num_cells, base_of_new, nbase, &new_start, &new_cells));

  PetscScalar *un;
  PetscCall(VecGetArray(u_new, &un));

  for (PetscInt b = 0; b < nbase; ++b) {
    PetscInt off, dof;
    PetscCall(PetscSectionGetOffset(sec, b + cStart, &off));
    PetscCall(PetscSectionGetDof(sec, b + cStart, &dof));
    const PetscInt nold = dof / stride;

    for (PetscInt in = new_start[b]; in < new_start[b + 1]; ++in) {
      PetscInt  n = new_cells[in];
      PetscReal nx[MAX_CLIP_VERTICES], ny[MAX_CLIP_VERTICES];
      PetscInt  nn;
      PetscCall(CellPolygon(new_mesh, n, nx, ny, &nn));
      // the area of the same x-y projection the clipping works in, not
      // cells->areas, which on a bed with relief is the area of the tilted
      // cell and would scale every weight by the tilt
      PetscReal new_area = 0.5 * PetscAbsReal(Shoelace2(nx, ny, nn));
      PetscReal covered  = 0.0;

      for (PetscInt d = 0; d < ndof; ++d) un[ndof * n + d] = 0.0;

      for (PetscInt io = 0; io < nold; ++io) {
        const PetscScalar *slot = packed + off + io * stride;
        PetscReal          ox[MAX_CLIP_VERTICES], oy[MAX_CLIP_VERTICES];
        PetscInt           no = (PetscInt)PetscRealPart(slot[0]);
        for (PetscInt v = 0; v < no; ++v) {
          ox[v] = PetscRealPart(slot[1 + 2 * v]);
          oy[v] = PetscRealPart(slot[1 + 2 * v + 1]);
        }
        PetscReal area;
        PetscCall(ConvexIntersectionArea(nx, ny, nn, ox, oy, no, &area));
        if (area <= 0.0) continue;
        covered += area;
        for (PetscInt d = 0; d < ndof; ++d) un[ndof * n + d] += (area / new_area) * slot[1 + 2 * RDY_AMR_PACK_MAX_V + d];
      }

      PetscCheck(PetscAbsReal(covered - new_area) <= 1e-10 * new_area, PETSC_COMM_SELF, PETSC_ERR_PLIB,
                 "cell %" PetscInt_FMT " of the new mesh is %g%% covered by the old mesh; the two are not triangulations of the same base cell", n,
                 (double)(100.0 * covered / new_area));
    }
  }

  PetscCall(VecRestoreArray(u_new, &un));
  PetscCall(PetscFree(base_of_new));
  PetscCall(PetscFree(new_start));
  PetscCall(PetscFree(new_cells));
  PetscFunctionReturn(PETSC_SUCCESS);
}
