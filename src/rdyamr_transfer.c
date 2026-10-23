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

/// @brief Builds the conservative old-to-new transfer for a cell-centred field
///        of `ndof` components per cell: row block n, column block o, weight
///        |n ^ o| / |n|.
/// @param [in] old_mesh      the mesh the solution is on
/// @param [in] new_mesh      the mesh it is going to
/// @param [in] base_to_old   base-to-old interpolation matrix (1 dof)
/// @param [in] base_to_new   base-to-new interpolation matrix (1 dof)
/// @param [in] nbase         number of base cells on this rank
/// @param [in] ndof          components per cell
/// @param [out] transfer     the conservative transfer, created here (sequential, per rank)
PetscErrorCode RDyAMRCreateTransferMatrix(RDyMesh *old_mesh, RDyMesh *new_mesh, Mat base_to_old, Mat base_to_new, PetscInt nbase, PetscInt ndof,
                                          Mat *transfer) {
  PetscFunctionBegin;

  PetscInt *base_of_old, *base_of_new, *old_start, *old_cells, *new_start, *new_cells;
  PetscCall(BaseCellOfEachCell(old_mesh, base_to_old, &base_of_old));
  PetscCall(BaseCellOfEachCell(new_mesh, base_to_new, &base_of_new));
  PetscCall(GroupByBaseCell(old_mesh->num_cells, base_of_old, nbase, &old_start, &old_cells));
  PetscCall(GroupByBaseCell(new_mesh->num_cells, base_of_new, nbase, &new_start, &new_cells));

  // a new cell overlaps only cells under the same base cell, and the deepest
  // allowed level bounds how many that can be
  PetscInt max_overlap = 1;
  for (PetscInt b = 0; b < nbase; ++b) max_overlap = PetscMax(max_overlap, old_start[b + 1] - old_start[b]);

  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, ndof * new_mesh->num_cells, ndof * old_mesh->num_cells, max_overlap, NULL, transfer));
  PetscCall(MatSetBlockSize(*transfer, ndof));

  for (PetscInt b = 0; b < nbase; ++b) {
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

      for (PetscInt io = old_start[b]; io < old_start[b + 1]; ++io) {
        PetscInt  o = old_cells[io];
        PetscReal ox[MAX_CLIP_VERTICES], oy[MAX_CLIP_VERTICES];
        PetscInt  no;
        PetscCall(CellPolygon(old_mesh, o, ox, oy, &no));
        PetscReal area;
        PetscCall(ConvexIntersectionArea(nx, ny, nn, ox, oy, no, &area));
        if (area <= 0.0) continue;
        covered += area;
        for (PetscInt d = 0; d < ndof; ++d) {
          PetscCall(MatSetValue(*transfer, ndof * n + d, ndof * o + d, area / new_area, INSERT_VALUES));
        }
      }

      PetscCheck(PetscAbsReal(covered - new_area) <= 1e-10 * new_area, PETSC_COMM_SELF, PETSC_ERR_PLIB,
                 "cell %" PetscInt_FMT " of the new mesh is %g%% covered by the old mesh; the two are not triangulations of the same base cell", n,
                 (double)(100.0 * covered / new_area));
    }
  }

  PetscCall(MatAssemblyBegin(*transfer, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*transfer, MAT_FINAL_ASSEMBLY));

  PetscCall(PetscFree(base_of_old));
  PetscCall(PetscFree(base_of_new));
  PetscCall(PetscFree(old_start));
  PetscCall(PetscFree(old_cells));
  PetscCall(PetscFree(new_start));
  PetscCall(PetscFree(new_cells));
  PetscFunctionReturn(PETSC_SUCCESS);
}
