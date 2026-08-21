// SWE RHS Jacobian: matrix creation, TS registration, and the
// finite-difference-coloring baseline implementation.
//
// The Jacobian of the SWE right-hand side f(u) is block-sparse with 3x3 blocks
// on the cell-adjacency graph: each interior edge (i,j) contributes blocks to
// rows i and j, and local source terms (Manning drag, bed slope) contribute to
// diagonal blocks only. The matrix is used by TSAdjoint (transpose actions in
// the backward sweep) and by implicit/IMEX time stepping; explicit runs never
// evaluate it, so registering it has no effect on existing configurations.
//
// This file currently provides:
//   * RegisterSWERHSJacobian(): allocates the matrix from the DM's adjacency
//     (a slight superset of edge adjacency, since rdy->dm uses closure-based
//     adjacency) and registers the configured Jacobian method with the TS.
//   * JACOBIAN_FD: a finite-difference coloring Jacobian, computed by applying
//     MatFDColoring to the registered TS RHS function. This is the
//     verification baseline that analytic blocks are tested against, and it
//     remains available as `numerics.jacobian: fd`.
// The analytic method (JACOBIAN_ANALYTIC) is added incrementally: source-term
// diagonal blocks first, then Roe flux edge blocks.

#include <petscdmplex.h>
#include <private/rdycoreimpl.h>
#include <private/rdymeshimpl.h>
#include <private/rdysweimpl.h>

#include "swe_roe_flux_jacobian_petsc.h"

// Wrapper with the calling convention MatFDColoringApply expects for its
// function: the "sctx" slot carries the TS, and the user context carries the
// RDy instance, whose rhs_jac_time field holds the time at which the TS
// requested the Jacobian.
static PetscErrorCode RHSFunctionForFDColoring(TS ts, Vec u, Vec f, void *ctx) {
  RDy rdy = ctx;
  PetscFunctionBegin;
  PetscCall(TSComputeRHSFunction(ts, rdy->rhs_jac_time, u, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// TSRHSJacobianFn implementing numerics.jacobian: fd -- fills the
// preallocated Jacobian by finite-difference coloring of the RHS function.
static PetscErrorCode SWERHSJacobianFD(TS ts, PetscReal t, Vec u, Mat J, Mat P, void *ctx) {
  RDy rdy = ctx;
  PetscFunctionBegin;

  // lazily create the coloring context from the preallocated pattern
  if (!rdy->rhs_jac_fd_coloring) {
    MatColoring coloring;
    ISColoring  is_coloring;
    PetscCall(MatColoringCreate(P, &coloring));
    PetscCall(MatColoringSetDistance(coloring, 2));
    PetscCall(MatColoringSetType(coloring, MATCOLORINGSL));
    PetscCall(MatColoringSetFromOptions(coloring));
    PetscCall(MatColoringApply(coloring, &is_coloring));
    PetscCall(MatColoringDestroy(&coloring));

    PetscCall(MatFDColoringCreate(P, is_coloring, &rdy->rhs_jac_fd_coloring));
    PetscCall(MatFDColoringSetFunction(rdy->rhs_jac_fd_coloring, (MatFDColoringFn *)RHSFunctionForFDColoring, rdy));
    PetscCall(MatFDColoringSetFromOptions(rdy->rhs_jac_fd_coloring));
    PetscCall(MatFDColoringSetUp(P, is_coloring, rdy->rhs_jac_fd_coloring));
    PetscCall(ISColoringDestroy(&is_coloring));
  }

  rdy->rhs_jac_time = t;
  PetscCall(MatFDColoringApply(P, rdy->rhs_jac_fd_coloring, u, ts));

  if (J != P) {
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------------------------------------------
// COO assembly for the analytic Jacobian. The sparsity pattern is fixed by
// mesh topology: per interior edge, up to four 3x3 blocks -- (l,l),(l,r) when
// l is owned and (r,r),(r,l) when r is owned -- plus one (l,l) block per owned
// boundary edge and one (c,c) source block per owned cell. The pattern is
// registered once with MatSetPreallocationCOO (repeated index pairs are
// summed by MatSetValuesCOO, reproducing the former ADD_VALUES semantics),
// and every assembly fills one flat values array in the SAME loop order:
// state-dependent skips (dry/dry edges, non-contributing outflow) write zero
// blocks instead of skipping, so index and value cursors always agree. This
// replaces ~4 MatSetValuesBlockedLocal calls per edge (the dominant cost of
// assembly at scale) with a single MatSetValuesCOO, and replaces the
// closure-adjacency preallocation superset with the exact FV pattern.
// ---------------------------------------------------------------------------

// appends the COO indices of the 3x3 block (brow, bcol) (local block indices,
// ghosted cell numbering) to coo_i/coo_j at *cursor via the DM's scalar
// local-to-global map: 9 scalar index pairs, or -- when blocked COO is in
// effect (MatCOOUseBlockIndices, BAIJ types on the petsc-claude fork) -- one
// block index pair
static PetscErrorCode AppendBlockIndices(ISLocalToGlobalMapping l2g, PetscBool blocked, PetscInt brow, PetscInt bcol, PetscInt *coo_i,
                                         PetscInt *coo_j, PetscCount *cursor) {
  PetscInt lrow[3] = {3 * brow, 3 * brow + 1, 3 * brow + 2}, grow[3];
  PetscInt lcol[3] = {3 * bcol, 3 * bcol + 1, 3 * bcol + 2}, gcol[3];
  PetscFunctionBegin;
  PetscCall(ISLocalToGlobalMappingApply(l2g, 3, lrow, grow));
  PetscCall(ISLocalToGlobalMappingApply(l2g, 3, lcol, gcol));
  if (blocked) {
    // cells carry contiguous 3-dof blocks and rank offsets are multiples of 3,
    // so the global block index is the first dof's scalar index / 3
    PetscCheck(grow[0] % 3 == 0 && gcol[0] % 3 == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "global dof index not 3-aligned in blocked COO pattern");
    coo_i[*cursor] = grow[0] / 3;
    coo_j[*cursor] = gcol[0] / 3;
    ++(*cursor);
  } else {
    for (PetscInt i = 0; i < 3; ++i)
      for (PetscInt j = 0; j < 3; ++j) {
        coo_i[*cursor] = grow[i];
        coo_j[*cursor] = gcol[j];
        ++(*cursor);
      }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// creates rdy->rhs_jac with the exact FV sparsity and registers the COO
// pattern; allocates the reusable values buffer
static PetscErrorCode CreateAnalyticJacobianCOO(RDy rdy) {
  PetscFunctionBegin;

  RDyMesh  *mesh  = &rdy->mesh;
  RDyCells *cells = &mesh->cells;
  RDyEdges *edges = &mesh->edges;

  PetscInt n_local;
  PetscCall(VecGetLocalSize(rdy->u_global, &n_local));
  PetscCall(MatCreate(rdy->comm, &rdy->rhs_jac));
  PetscCall(MatSetSizes(rdy->rhs_jac, n_local, n_local, PETSC_DETERMINE, PETSC_DETERMINE));
  // honor -dm_mat_type (e.g. aijkokkos/baijkokkos for device solves): the DM's
  // mat type defaults to MATAIJ, so CPU behavior is unchanged. The prefixed
  // MatSetFromOptions additionally allows -rhs_jac_mat_type to override the
  // type of THIS matrix alone (a bare -mat_type would also be picked up by
  // every DMCreateMatrix, e.g. the FD twin in the jacobian tests)
  MatType mat_type;
  PetscCall(DMGetMatType(rdy->dm, &mat_type));
  PetscCall(MatSetType(rdy->rhs_jac, mat_type));
  PetscCall(MatSetBlockSize(rdy->rhs_jac, 3));
  PetscCall(MatSetOptionsPrefix(rdy->rhs_jac, "rhs_jac_"));
  PetscCall(MatSetFromOptions(rdy->rhs_jac));

  ISLocalToGlobalMapping l2g;
  PetscCall(DMGetLocalToGlobalMapping(rdy->dm, &l2g));

  // BAIJ matrix types on the petsc-claude fork assemble by blocked COO:
  // coo_i/coo_j are block indices (one pair per 3x3 block) and MatSetValuesCOO
  // consumes one dense row-major 3x3 block per entry -- exactly the values
  // layout the assembly loop already produces, so only the pattern differs.
  // Only the types that implement blocked COO may opt in: the Kokkos/CUDA
  // block classes and the MPIBAIJ host twin. Plain SEQBAIJ has no blocked-COO
  // path -- it IGNORES the flag and would misread block indices as scalar --
  // so it (and any type on a stock PETSc build) uses scalar COO.
  PetscBool blocked = PETSC_FALSE;
#if RDY_HAVE_MAT_COO_BLOCK_INDICES
  {
    MatType type;
    PetscCall(MatGetType(rdy->rhs_jac, &type));
    if (strstr(type, "baijkokkos") || strstr(type, "baijcuda") || !strcmp(type, MATMPIBAIJ)) {
      PetscCall(MatCOOUseBlockIndices(rdy->rhs_jac, PETSC_TRUE));
      blocked = PETSC_TRUE;
    }
  }
#endif

  // count blocks (same loop structure as the fill; only topology matters)
  PetscCount nblocks = 0;
  for (PetscInt e = 0; e < mesh->num_internal_edges; ++e) {
    PetscInt edge_id = edges->internal_edge_ids[e];
    PetscInt l       = edges->cell_ids[2 * edge_id];
    PetscInt r       = edges->cell_ids[2 * edge_id + 1];
    if (r == -1) continue;
    if (cells->is_owned[l]) nblocks += 2;
    if (cells->is_owned[r]) nblocks += 2;
  }
  for (PetscInt b = 0; b < rdy->num_boundaries; ++b) {
    RDyBoundary boundary = rdy->boundaries[b];
    for (PetscInt e = 0; e < boundary.num_edges; ++e) {
      PetscInt l = edges->cell_ids[2 * boundary.edge_ids[e]];
      if (cells->is_owned[l]) nblocks += 1;
    }
  }
  for (PetscInt c = 0; c < mesh->num_cells; ++c) {
    if (cells->is_owned[c]) nblocks += 1;
  }

  PetscCount ncoo = blocked ? nblocks : 9 * nblocks;  // index pairs registered with MatSetPreallocationCOO
  PetscInt  *coo_i, *coo_j;
  PetscCall(PetscMalloc2(ncoo, &coo_i, ncoo, &coo_j));
  PetscCount cursor = 0;
  for (PetscInt e = 0; e < mesh->num_internal_edges; ++e) {
    PetscInt edge_id = edges->internal_edge_ids[e];
    PetscInt l       = edges->cell_ids[2 * edge_id];
    PetscInt r       = edges->cell_ids[2 * edge_id + 1];
    if (r == -1) continue;
    if (cells->is_owned[l]) {
      PetscCall(AppendBlockIndices(l2g, blocked, l, l, coo_i, coo_j, &cursor));
      PetscCall(AppendBlockIndices(l2g, blocked, l, r, coo_i, coo_j, &cursor));
    }
    if (cells->is_owned[r]) {
      PetscCall(AppendBlockIndices(l2g, blocked, r, r, coo_i, coo_j, &cursor));
      PetscCall(AppendBlockIndices(l2g, blocked, r, l, coo_i, coo_j, &cursor));
    }
  }
  for (PetscInt b = 0; b < rdy->num_boundaries; ++b) {
    RDyBoundary      boundary = rdy->boundaries[b];
    RDyConditionType bc_type  = rdy->boundary_conditions[b].flow->type;
    PetscCheck(bc_type == CONDITION_DIRICHLET || bc_type == CONDITION_REFLECTING || bc_type == CONDITION_CRITICAL_OUTFLOW, rdy->comm,
               PETSC_ERR_SUP, "numerics.jacobian: analytic does not support this boundary condition type yet");
    for (PetscInt e = 0; e < boundary.num_edges; ++e) {
      PetscInt l = edges->cell_ids[2 * boundary.edge_ids[e]];
      if (!cells->is_owned[l]) continue;
      PetscCall(AppendBlockIndices(l2g, blocked, l, l, coo_i, coo_j, &cursor));
    }
  }
  for (PetscInt c = 0; c < mesh->num_cells; ++c) {
    if (!cells->is_owned[c]) continue;
    PetscCall(AppendBlockIndices(l2g, blocked, c, c, coo_i, coo_j, &cursor));
  }
  PetscCheck(cursor == ncoo, rdy->comm, PETSC_ERR_PLIB, "COO pattern cursor mismatch: %" PetscCount_FMT " != %" PetscCount_FMT, cursor, ncoo);

  PetscCall(MatSetPreallocationCOO(rdy->rhs_jac, ncoo, coo_i, coo_j));
  PetscCall(PetscFree2(coo_i, coo_j));

  // the values buffer is 9 scalars per block in both modes (row-major blocks
  // in pattern order), so the assembly fill and its final cursor check are
  // mode-independent
  rdy->rhs_jac_ncoo = 9 * nblocks;
  PetscCall(PetscMalloc1(rdy->rhs_jac_ncoo, &rdy->rhs_jac_coo_v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// TSRHSJacobianFn implementing numerics.jacobian: analytic -- assembles the
// block-sparse Jacobian from closed-form per-edge flux blocks and per-cell
// source blocks, mirroring the loops (and wet/dry branches) of
// ApplyInteriorFlux and ApplySourceExplicit.
//
// The flux blocks are exact (hand-written forward-mode differential of the
// Roe flux, entropy-fix branches included) and boundary-edge contributions
// are assembled for reflecting, Dirichlet, and critical-outflow conditions,
// so the assembled matrix matches finite differences of the full RHS to FD
// accuracy on interior and boundary rows alike. Values are gathered into the
// COO buffer laid out by CreateAnalyticJacobianCOO (identical loop order;
// state-dependent skips write zeros) and set with one MatSetValuesCOO.
static PetscErrorCode SWERHSJacobianAnalytic(TS ts, PetscReal t, Vec u_global, Mat J, Mat P, void *ctx) {
  RDy rdy = ctx;
  (void)t;  // the RHS state dependence is autonomous (forcing is state-independent)
  PetscFunctionBegin;

  RDyMesh  *mesh  = &rdy->mesh;
  RDyCells *cells = &mesh->cells;
  RDyEdges *edges = &mesh->edges;

  const PetscReal tiny_h  = rdy->config.physics.flow.tiny_h;
  const PetscReal h_anuga = rdy->config.physics.flow.h_anuga_regular;

  PetscScalar *v      = rdy->rhs_jac_coo_v;
  PetscCount   cursor = 0;

  // ghosted state
  Vec u_local;
  PetscCall(DMGetLocalVector(rdy->dm, &u_local));
  PetscCall(DMGlobalToLocalBegin(rdy->dm, u_global, INSERT_VALUES, u_local));
  PetscCall(DMGlobalToLocalEnd(rdy->dm, u_global, INSERT_VALUES, u_local));
  const PetscScalar *u_ptr;
  PetscCall(VecGetArrayRead(u_local, &u_ptr));

  // interior-edge flux blocks: f_l += -len/A_l F, f_r += +len/A_r F, so
  //   J[l][*] -= len/A_l dF/du_*, J[r][*] += len/A_r dF/du_*
  // (each rank computes shared edges too and writes only its owned rows,
  // mirroring the RHS convention)
  for (PetscInt e = 0; e < mesh->num_internal_edges; ++e) {
    PetscInt edge_id = edges->internal_edge_ids[e];
    PetscInt l       = edges->cell_ids[2 * edge_id];
    PetscInt r       = edges->cell_ids[2 * edge_id + 1];
    if (r == -1) continue;

    PetscInt nblocks = (cells->is_owned[l] ? 2 : 0) + (cells->is_owned[r] ? 2 : 0);

    const PetscScalar *consL = &u_ptr[3 * l];
    const PetscScalar *consR = &u_ptr[3 * r];
    if (consL[0] < tiny_h && consR[0] < tiny_h) {  // RHS skips dry/dry edges
      PetscCall(PetscArrayzero(&v[cursor], 9 * nblocks));
      cursor += 9 * nblocks;
      continue;
    }

    PetscReal dFdUL[3][3], dFdUR[3][3];
    PetscReal uL[3] = {consL[0], consL[1], consL[2]}, uR[3] = {consR[0], consR[1], consR[2]};
    PetscCall(SWERoeFluxJacobian(uL, uR, edges->sn[edge_id], edges->cn[edge_id], tiny_h, h_anuga, dFdUL, dFdUR));

    PetscReal len = edges->lengths[edge_id];
    PetscReal wl = -len / cells->areas[l], wr = len / cells->areas[r];

    if (cells->is_owned[l]) {
      for (PetscInt i = 0; i < 3; ++i)
        for (PetscInt j = 0; j < 3; ++j) v[cursor + 3 * i + j] = wl * dFdUL[i][j];
      cursor += 9;
      for (PetscInt i = 0; i < 3; ++i)
        for (PetscInt j = 0; j < 3; ++j) v[cursor + 3 * i + j] = wl * dFdUR[i][j];
      cursor += 9;
    }
    if (cells->is_owned[r]) {
      for (PetscInt i = 0; i < 3; ++i)
        for (PetscInt j = 0; j < 3; ++j) v[cursor + 3 * i + j] = wr * dFdUR[i][j];
      cursor += 9;
      for (PetscInt i = 0; i < 3; ++i)
        for (PetscInt j = 0; j < 3; ++j) v[cursor + 3 * i + j] = wr * dFdUL[i][j];
      cursor += 9;
    }
  }

  // boundary-edge blocks: F(u_in, u_ghost(u_in)) contributes
  //   J[l][l] += -len/A_l (dF/dqL + dF/dqR dG) P_L
  // where G is the ghost-state map of the BC type in primitive variables
  // (reflecting: linear mirror; Dirichlet: constant ghost, dG = 0;
  // critical outflow: closed-form, with the code's inflow branch giving a
  // state-independent zero flux). Mirrors ApplyBoundaryFlux exactly.
  for (PetscInt b = 0; b < rdy->num_boundaries; ++b) {
    RDyBoundary      boundary = rdy->boundaries[b];
    RDyConditionType bc_type  = rdy->boundary_conditions[b].flow->type;

    PetscScalar *bv_ptr = NULL;
    if (bc_type == CONDITION_DIRICHLET) PetscCall(VecGetArray(rdy->operator->petsc.boundary_values[b], &bv_ptr));

    for (PetscInt e = 0; e < boundary.num_edges; ++e) {
      PetscInt edge_id = boundary.edge_ids[e];
      PetscInt l       = edges->cell_ids[2 * edge_id];
      if (!cells->is_owned[l]) continue;
      PetscCount block_at = cursor;  // this edge's (l,l) block in the COO layout
      cursor += 9;

      const PetscScalar *consL_s = &u_ptr[3 * l];
      PetscReal          consL[3] = {PetscRealPart(consL_s[0]), PetscRealPart(consL_s[1]), PetscRealPart(consL_s[2])};
      PetscReal          sn = edges->sn[edge_id], cn = edges->cn[edge_id];
      PetscReal          len = edges->lengths[edge_id];
      PetscReal          wl  = -len / cells->areas[l];

      PetscReal qL[3], PL[3][3];
      PetscCall(SWEReconstructPrimitiveWithJacobian(consL, tiny_h, h_anuga, qL, PL));

      // ghost primitive state and its map dG = dqR/dqL (primitive space)
      PetscReal qR[3], G[3][3] = {{0}};
      PetscBool contribute = PETSC_TRUE;
      switch (bc_type) {
        case CONDITION_DIRICHLET: {
          PetscReal consR[3] = {PetscRealPart(bv_ptr[3 * e]), PetscRealPart(bv_ptr[3 * e + 1]), PetscRealPart(bv_ptr[3 * e + 2])};
          PetscReal PR_unused[3][3];
          PetscCall(SWEReconstructPrimitiveWithJacobian(consR, tiny_h, h_anuga, qR, PR_unused));
          // ghost independent of the interior state: dG stays zero
        } break;
        case CONDITION_REFLECTING: {
          PetscReal d1 = Square(sn) - Square(cn), d2 = 2.0 * sn * cn;
          qR[0]   = qL[0];
          qR[1]   = qL[1] * d1 - qL[2] * d2;
          qR[2]   = -qL[1] * d2 - qL[2] * d1;
          G[0][0] = 1.0;
          G[1][1] = d1;
          G[1][2] = -d2;
          G[2][1] = -d2;
          G[2][2] = -d1;
        } break;
        case CONDITION_CRITICAL_OUTFLOW: {
          PetscReal uperp = qL[1] * cn + qL[2] * sn;
          if (uperp < 0.0) {
            contribute = PETSC_FALSE;  // code zeroes BOTH states: flux is identically zero
          } else {
            PetscReal q = qL[0] * uperp;  // = h |uperp| for uperp >= 0
            qR[0]       = PetscPowReal(Square(q) / GRAVITY, 1.0 / 3.0);
            PetscReal vel = PetscSqrtReal(GRAVITY * qR[0]);
            qR[1]         = vel * cn;
            qR[2]         = vel * sn;
            if (q > 1e-14) {
              // dq/dqL = (uperp, h cn, h sn); dhR = (2/3)(hR/q) dq; dvel = g dhR / (2 vel)
              PetscReal dq[3]   = {uperp, qL[0] * cn, qL[0] * sn};
              PetscReal hR_coef = (2.0 / 3.0) * qR[0] / q;
              PetscReal v_coef  = (vel > 0.0) ? 0.5 * GRAVITY / vel : 0.0;
              for (PetscInt j = 0; j < 3; ++j) {
                G[0][j] = hR_coef * dq[j];
                G[1][j] = cn * v_coef * G[0][j];
                G[2][j] = sn * v_coef * G[0][j];
              }
            }
          }
        } break;
        default:
          PetscCheck(PETSC_FALSE, rdy->comm, PETSC_ERR_SUP, "numerics.jacobian: analytic does not support this boundary condition type yet");
      }
      if (!contribute || (qL[0] < tiny_h && qR[0] < tiny_h)) {  // zero flux / RHS skips dry/dry boundary edges
        PetscCall(PetscArrayzero(&v[block_at], 9));
        continue;
      }

      // columns: conservative direction e_j -> primitive dir = P_L e_j,
      // ghost dir = G (P_L e_j); exact differential gives the column
      for (PetscInt j = 0; j < 3; ++j) {
        PetscReal dirL[3] = {PL[0][j], PL[1][j], PL[2][j]};
        PetscReal dirR[3];
        for (PetscInt i = 0; i < 3; ++i) dirR[i] = G[i][0] * dirL[0] + G[i][1] * dirL[1] + G[i][2] * dirL[2];
        PetscReal dF[3];
        PetscCall(SWERoeFluxDifferentialPrim(qL, qR, sn, cn, dirL, dirR, dF));
        for (PetscInt i = 0; i < 3; ++i) v[block_at + 3 * i + j] = wl * dF[i];
      }
    }
    if (bc_type == CONDITION_DIRICHLET) PetscCall(VecRestoreArray(rdy->operator->petsc.boundary_values[b], &bv_ptr));
  }

  // per-cell source blocks: full (friction + bed slope) for the explicit
  // source treatment; bed slope only for ARK-IMEX, whose friction lives in
  // the implicit part (SWEIJacobianFriction)
  PetscBool    friction_in_rhs = (rdy->config.physics.flow.source.method == SOURCE_EXPLICIT) ? PETSC_TRUE : PETSC_FALSE;
  PetscScalar *mat_props_ptr;
  PetscCall(VecGetArray(rdy->operator->petsc.material_properties, &mat_props_ptr));
  for (PetscInt c = 0; c < mesh->num_cells; ++c) {
    if (!cells->is_owned[c]) continue;
    PetscInt owned = cells->local_to_owned[c];

    PetscReal n_manning = mat_props_ptr[NUM_MATERIAL_PROPERTIES * owned + MATERIAL_PROPERTY_MANNINGS];
    PetscReal cons[3]   = {u_ptr[3 * c + 0], u_ptr[3 * c + 1], u_ptr[3 * c + 2]};
    PetscReal D[3][3];
    if (friction_in_rhs) {
      PetscCall(SWESourceJacobian(cons, n_manning, cells->dz_dx[c], cells->dz_dy[c], tiny_h, D));
    } else {
      for (PetscInt i = 0; i < 3; ++i)
        for (PetscInt j = 0; j < 3; ++j) D[i][j] = 0.0;
      D[1][0] = -GRAVITY * cells->dz_dx[c];
      D[2][0] = -GRAVITY * cells->dz_dy[c];
    }

    for (PetscInt i = 0; i < 3; ++i)
      for (PetscInt j = 0; j < 3; ++j) v[cursor + 3 * i + j] = D[i][j];
    cursor += 9;
  }
  PetscCall(VecRestoreArray(rdy->operator->petsc.material_properties, &mat_props_ptr));

  PetscCall(VecRestoreArrayRead(u_local, &u_ptr));
  PetscCall(DMRestoreLocalVector(rdy->dm, &u_local));

  PetscCheck(cursor == rdy->rhs_jac_ncoo, rdy->comm, PETSC_ERR_PLIB, "COO value cursor mismatch: %" PetscCount_FMT " != %" PetscCount_FMT, cursor,
             rdy->rhs_jac_ncoo);
  PetscCall(MatSetValuesCOO(P, v, INSERT_VALUES));
  if (J != P) {
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// TSRHSJacobianPFn: dF/dp for p = per-cell Manning n. Only the explicit drag
// term depends on n:  S_q = -g n^2 h^{-7/3} q |q|, so
//   dS_q/dn = -2 g n h^{-7/3} q |q|
// -- two nonzeros per cell (momentum rows), zero for dry or motionless cells.
// Rows follow the global state layout (contiguous owned dofs); column c is
// the owned-cell parameter index with the same distribution as the rows.
static PetscErrorCode SWERHSJacobianP(TS ts, PetscReal t, Vec u_global, Mat Jacp, void *ctx) {
  RDy rdy = ctx;
  (void)t;
  PetscFunctionBegin;

  const PetscReal tiny_h = rdy->config.physics.flow.tiny_h;

  PetscCall(MatZeroEntries(Jacp));

  PetscInt rstart, rend, cstart, cend;
  PetscCall(MatGetOwnershipRange(Jacp, &rstart, &rend));
  PetscCall(MatGetOwnershipRangeColumn(Jacp, &cstart, &cend));

  const PetscScalar *u_ptr;
  PetscCall(VecGetArrayRead(u_global, &u_ptr));
  PetscScalar *mat_props_ptr;
  PetscCall(VecGetArray(rdy->operator->petsc.material_properties, &mat_props_ptr));

  PetscInt num_owned = (rend - rstart) / 3;
  for (PetscInt o = 0; o < num_owned; ++o) {
    PetscReal h = u_ptr[3 * o], hu = u_ptr[3 * o + 1], hv = u_ptr[3 * o + 2];
    if (h < tiny_h) continue;
    PetscReal m = PetscSqrtReal(Square(hu) + Square(hv));
    if (m == 0.0) continue;

    PetscReal n_manning = mat_props_ptr[NUM_MATERIAL_PROPERTIES * o + MATERIAL_PROPERTY_MANNINGS];
    PetscReal coeff     = -2.0 * GRAVITY * n_manning * PetscPowReal(h, -7.0 / 3.0) * m;

    PetscInt col = cstart + o;
    PetscInt row = rstart + 3 * o + 1;
    PetscCall(MatSetValue(Jacp, row, col, coeff * hu, INSERT_VALUES));
    row = rstart + 3 * o + 2;
    PetscCall(MatSetValue(Jacp, row, col, coeff * hv, INSERT_VALUES));
  }

  PetscCall(VecRestoreArray(rdy->operator->petsc.material_properties, &mat_props_ptr));
  PetscCall(VecRestoreArrayRead(u_global, &u_ptr));

  PetscCall(MatAssemblyBegin(Jacp, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Jacp, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------------------------------------------
// ARK-IMEX friction: the Manning drag is the stiff implicit part.
//   IFunction  F = Udot - S_fric(u)      (per-cell local; no ghosts needed)
//   IJacobian  dF/du = shift I - dS_fric/du   (block-diagonal 3x3)
//   IJacobianP dF/dn = -dS_fric/dn            (2 nonzeros per cell)
// ---------------------------------------------------------------------------

static PetscErrorCode SWEIFunctionFriction(TS ts, PetscReal t, Vec u_global, Vec u_dot, Vec f, void *ctx) {
  RDy rdy = ctx;
  (void)t;
  PetscFunctionBegin;

  const PetscReal tiny_h = rdy->config.physics.flow.tiny_h;

  const PetscScalar *u_ptr, *udot_ptr;
  PetscScalar       *f_ptr;
  PetscCall(VecGetArrayRead(u_global, &u_ptr));
  PetscCall(VecGetArrayRead(u_dot, &udot_ptr));
  PetscCall(VecGetArray(f, &f_ptr));
  OperatorData mat_props;  // backend-agnostic (PETSc and CEED operators)
  PetscCall(GetOperatorDomainMaterialProperties(rdy->operator, &mat_props));

  PetscInt n_local;
  PetscCall(VecGetLocalSize(u_global, &n_local));
  PetscInt num_owned = n_local / 3;
  for (PetscInt o = 0; o < num_owned; ++o) {
    PetscReal h = PetscRealPart(u_ptr[3 * o]), hu = PetscRealPart(u_ptr[3 * o + 1]), hv = PetscRealPart(u_ptr[3 * o + 2]);
    PetscReal tbx = 0.0, tby = 0.0;
    if (h >= tiny_h) {
      PetscReal m = PetscSqrtReal(Square(hu) + Square(hv));
      if (m > 0.0) {
        PetscReal n_manning = mat_props.values[MATERIAL_PROPERTY_MANNINGS][o];
        PetscReal tb        = GRAVITY * Square(n_manning) * PetscPowReal(h, -7.0 / 3.0) * m;
        tbx                 = tb * hu;
        tby                 = tb * hv;
      }
    }
    // F = Udot - S_fric, S_fric = (0, -tbx, -tby)
    f_ptr[3 * o + 0] = udot_ptr[3 * o + 0];
    f_ptr[3 * o + 1] = udot_ptr[3 * o + 1] + tbx;
    f_ptr[3 * o + 2] = udot_ptr[3 * o + 2] + tby;
  }

  PetscCall(RestoreOperatorDomainMaterialProperties(rdy->operator, &mat_props));
  PetscCall(VecRestoreArrayRead(u_global, &u_ptr));
  PetscCall(VecRestoreArrayRead(u_dot, &udot_ptr));
  PetscCall(VecRestoreArray(f, &f_ptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SWEIJacobianFriction(TS ts, PetscReal t, Vec u_global, Vec u_dot, PetscReal shift, Mat J, Mat P, void *ctx) {
  RDy rdy = ctx;
  (void)t;
  (void)u_dot;
  PetscFunctionBegin;

  const PetscReal tiny_h = rdy->config.physics.flow.tiny_h;

  // P is block diagonal with exactly one 3x3 block per owned cell, and every
  // block is INSERTed below, so no MatZeroEntries is needed.
  const PetscScalar *u_ptr;
  PetscCall(VecGetArrayRead(u_global, &u_ptr));
  OperatorData mat_props;
  PetscCall(GetOperatorDomainMaterialProperties(rdy->operator, &mat_props));

  PetscInt n_local, row_start;
  PetscCall(VecGetLocalSize(u_global, &n_local));
  PetscCall(MatGetOwnershipRange(P, &row_start, NULL));
  PetscInt block_start = row_start / 3;  // global block row of this rank's first cell
  PetscInt num_owned   = n_local / 3;
  for (PetscInt o = 0; o < num_owned; ++o) {
    PetscReal cons[3]   = {PetscRealPart(u_ptr[3 * o]), PetscRealPart(u_ptr[3 * o + 1]), PetscRealPart(u_ptr[3 * o + 2])};
    PetscReal n_manning = mat_props.values[MATERIAL_PROPERTY_MANNINGS][o];
    PetscReal D[3][3];
    PetscCall(SWEFrictionJacobian(cons, n_manning, tiny_h, D));

    PetscReal block[9];
    for (PetscInt i = 0; i < 3; ++i)
      for (PetscInt j = 0; j < 3; ++j) block[3 * i + j] = ((i == j) ? shift : 0.0) - D[i][j];
    PetscInt brow = block_start + o;
    PetscCall(MatSetValuesBlocked(P, 1, &brow, 1, &brow, block, INSERT_VALUES));
  }

  PetscCall(RestoreOperatorDomainMaterialProperties(rdy->operator, &mat_props));
  PetscCall(VecRestoreArrayRead(u_global, &u_ptr));

  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  if (J != P) {
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// the ARK-IMEX explicit part carries no Manning dependence: dG/dn = 0
// (the ARKIMEX adjoint applies both dF/dp and dG/dp unconditionally)
static PetscErrorCode SWERHSJacobianPZero(TS ts, PetscReal t, Vec u_global, Mat Jacp, void *ctx) {
  (void)ts;
  (void)t;
  (void)u_global;
  (void)ctx;
  PetscFunctionBegin;
  PetscCall(MatZeroEntries(Jacp));
  PetscCall(MatAssemblyBegin(Jacp, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Jacp, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// dF/dn = -dS_fric/dn = +2 g n h^{-7/3} m (hu, hv) in the momentum rows
static PetscErrorCode SWEIJacobianPFriction(TS ts, PetscReal t, Vec u_global, Vec u_dot, PetscReal shift, Mat Jacp, void *ctx) {
  RDy rdy = ctx;
  (void)t;
  (void)u_dot;
  (void)shift;
  PetscFunctionBegin;

  const PetscReal tiny_h = rdy->config.physics.flow.tiny_h;

  PetscCall(MatZeroEntries(Jacp));

  PetscInt rstart, rend, cstart, cend;
  PetscCall(MatGetOwnershipRange(Jacp, &rstart, &rend));
  PetscCall(MatGetOwnershipRangeColumn(Jacp, &cstart, &cend));

  const PetscScalar *u_ptr;
  PetscCall(VecGetArrayRead(u_global, &u_ptr));
  OperatorData mat_props;
  PetscCall(GetOperatorDomainMaterialProperties(rdy->operator, &mat_props));

  PetscInt num_owned = (rend - rstart) / 3;
  for (PetscInt o = 0; o < num_owned; ++o) {
    PetscReal h = PetscRealPart(u_ptr[3 * o]), hu = PetscRealPart(u_ptr[3 * o + 1]), hv = PetscRealPart(u_ptr[3 * o + 2]);
    if (h < tiny_h) continue;
    PetscReal m = PetscSqrtReal(Square(hu) + Square(hv));
    if (m == 0.0) continue;

    PetscReal n_manning = mat_props.values[MATERIAL_PROPERTY_MANNINGS][o];
    PetscReal coeff     = 2.0 * GRAVITY * n_manning * PetscPowReal(h, -7.0 / 3.0) * m;

    PetscInt col = cstart + o;
    PetscCall(MatSetValue(Jacp, rstart + 3 * o + 1, col, coeff * hu, INSERT_VALUES));
    PetscCall(MatSetValue(Jacp, rstart + 3 * o + 2, col, coeff * hv, INSERT_VALUES));
  }

  PetscCall(RestoreOperatorDomainMaterialProperties(rdy->operator, &mat_props));
  PetscCall(VecRestoreArrayRead(u_global, &u_ptr));

  PetscCall(MatAssemblyBegin(Jacp, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Jacp, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Registers the stiff-friction implicit part (IFunction/IJacobian/IJacobianP)
/// for the ARK-IMEX temporal method. Requires the PETSc operator backend.
PetscErrorCode RegisterSWEIMEXFriction(RDy rdy) {
  PetscFunctionBegin;

  // works with both operator backends: the explicit side uses Jeff Johnson's
  // friction-free source Q-functions under CEED (PR #359), and the implicit
  // friction part runs on host arrays (device backends incur a per-stage
  // host transfer; a CEED Q-function IFunction is the follow-up)
  // Friction couples nothing across cells, so the implicit Jacobian is exactly
  // block diagonal: one 3x3 block per cell. Allocate exactly that, rather than
  // DMCreateMatrix (whose flux-stencil sparsity is ~10x larger and all zeros
  // here) -- on the 2.9M-cell Harvey mesh the oversized matrix made
  // MatZeroEntries/assembly/ILU dominate the step cost.
  // AIJ with block size 3 (not BAIJ, which is less well maintained): 3 nonzeros
  // per row, all inside the row's own 3x3 diagonal block, and no off-diagonal.
  PetscInt n_local;
  PetscCall(VecGetLocalSize(rdy->u_global, &n_local));
  PetscCall(MatCreate(rdy->comm, &rdy->imex_ijac));
  PetscCall(MatSetSizes(rdy->imex_ijac, n_local, n_local, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetType(rdy->imex_ijac, MATAIJ));
  PetscCall(MatSetBlockSize(rdy->imex_ijac, 3));
  PetscCall(MatSeqAIJSetPreallocation(rdy->imex_ijac, 3, NULL));
  PetscCall(MatMPIAIJSetPreallocation(rdy->imex_ijac, 3, NULL, 0, NULL));
  PetscCall(PetscObjectSetName((PetscObject)rdy->imex_ijac, "swe_imex_friction_ijac"));
  PetscCall(TSSetIFunction(rdy->ts, NULL, SWEIFunctionFriction, rdy));
  PetscCall(TSSetIJacobian(rdy->ts, rdy->imex_ijac, rdy->imex_ijac, SWEIJacobianFriction, rdy));

  // With one block per row, a per-rank direct solve of the block-diagonal
  // system is exact and cheap, so skip the Krylov iteration entirely.
  // NB: PCPBJACOBI would be the natural choice here but is UNSAFE for a
  // repeatedly re-assembled matrix: MatInvertBlockDiagonal() caches the
  // inverted blocks, and MatAssemblyEnd_Seq{AIJ,BAIJ} only clears that cache
  // after an early return taken whenever the nonzero structure is unchanged --
  // so PCPBJACOBI silently preconditions every later Newton step with the
  // FIRST step's blocks (reproducer: plans/ex_baij_invertblockdiag_stale.c).
  // Set before TSSetFromOptions runs, so -ksp_type/-pc_type still win.
  {
    SNES snes;
    KSP  ksp;
    PC   pc;
    PetscCall(TSGetSNES(rdy->ts, &snes));
    PetscCall(SNESGetKSP(snes, &ksp));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(KSPSetType(ksp, KSPPREONLY));
    PetscCall(PCSetType(pc, PCBJACOBI));
  }

  // parameter dependence lives in the implicit part
  {
    PetscInt n_local;
    PetscCall(VecGetLocalSize(rdy->u_global, &n_local));
    PetscInt num_owned = n_local / 3;
    PetscCall(MatCreateAIJ(rdy->comm, n_local, num_owned, PETSC_DETERMINE, PETSC_DETERMINE, 1, NULL, 0, NULL, &rdy->rhs_jac_p));
    PetscCall(PetscObjectSetName((PetscObject)rdy->rhs_jac_p, "swe_imex_friction_dn"));
    PetscCall(TSSetIJacobianP(rdy->ts, rdy->rhs_jac_p, SWEIJacobianPFriction, rdy));
    // the ARKIMEX adjoint also consumes an explicit-part dG/dp
    PetscCall(MatCreateAIJ(rdy->comm, n_local, num_owned, PETSC_DETERMINE, PETSC_DETERMINE, 0, NULL, 0, NULL, &rdy->imex_jacp_rhs));
    PetscCall(PetscObjectSetName((PetscObject)rdy->imex_jacp_rhs, "swe_imex_dn_rhs_zero"));
    PetscCall(MatAssemblyBegin(rdy->imex_jacp_rhs, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(rdy->imex_jacp_rhs, MAT_FINAL_ASSEMBLY));
    PetscCall(TSSetRHSJacobianP(rdy->ts, rdy->imex_jacp_rhs, SWERHSJacobianPZero, rdy));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Allocates the RHS Jacobian matrix and registers the Jacobian method
/// selected by numerics.jacobian with rdy->ts. Called from InitSolver; a
/// no-op contract: callers must check numerics.jacobian != JACOBIAN_NONE.
PetscErrorCode RegisterSWERHSJacobian(RDy rdy) {
  PetscFunctionBegin;

  PetscCheck(!CeedEnabled(), rdy->comm, PETSC_ERR_SUP,
             "numerics.jacobian requires the PETSc operator backend (run without -ceed)");
  PetscCheck(rdy->config.physics.flow.source.method == SOURCE_EXPLICIT || rdy->config.physics.flow.source.method == SOURCE_ARK_IMEX,
             rdy->comm, PETSC_ERR_SUP,
             "numerics.jacobian requires physics.flow.source.method: explicit or ark_imex -- the semi-implicit "
             "and xq2018 treatments embed dt in the RHS and admit no well-defined Jacobian");

  // second-order (MUSCL) guards: the analytic Jacobian differentiates the
  // first-order flux path only, and an inconsistent Jacobian silently corrupts
  // TSAdjoint gradients (the discrete adjoint applies the supplied Jacobian
  // transpose as the exact linearization, unlike forward Newton, which merely
  // converges slower). Runs after CreateOperator, so the -second_order CLI
  // override is already folded into the config.
  if (rdy->config.numerics.second_order) {
    PetscCheck(rdy->config.numerics.jacobian != JACOBIAN_ANALYTIC, rdy->comm, PETSC_ERR_SUP,
               "numerics.jacobian: analytic differentiates the first-order flux path and is inconsistent with "
               "numerics.second_order (MUSCL); use jacobian: fd (simplicial meshes only) or disable second_order");
    // FD coloring is built from the closure-adjacency pattern, which covers the
    // two-edge-hop MUSCL stencil only on simplicial meshes (any two edges of a
    // triangle share a vertex); on quads the opposite-edge coupling is missing
    // from the pattern and the colored Jacobian is silently wrong.
    PetscInt  max_cell_vertices = 0;
    RDyCells *cells             = &rdy->mesh.cells;
    for (PetscInt i = 0; i < rdy->mesh.num_cells; i++) {
      if (cells->is_owned[i] && cells->num_vertices[i] > max_cell_vertices) max_cell_vertices = cells->num_vertices[i];
    }
    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &max_cell_vertices, 1, MPIU_INT, MPI_MAX, rdy->comm));
    PetscCheck(max_cell_vertices <= 3, rdy->comm, PETSC_ERR_SUP,
               "numerics.jacobian: fd with numerics.second_order requires a triangular mesh -- the Jacobian "
               "sparsity pattern does not cover the MUSCL stencil on non-simplicial cells");
  }

  switch (rdy->config.numerics.jacobian) {
    case JACOBIAN_FD:
      // FD coloring works from the DM's closure-adjacency pattern (a superset
      // of the FV edge pattern -- and the full MUSCL stencil on triangles)
      PetscCall(DMCreateMatrix(rdy->dm, &rdy->rhs_jac));
      PetscCall(PetscObjectSetName((PetscObject)rdy->rhs_jac, "swe_rhs_jacobian"));
      PetscCall(MatSetOption(rdy->rhs_jac, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
      PetscCall(TSSetRHSJacobian(rdy->ts, rdy->rhs_jac, rdy->rhs_jac, SWERHSJacobianFD, rdy));
      break;
    case JACOBIAN_ANALYTIC:
      // exact FV pattern, COO-preallocated (see CreateAnalyticJacobianCOO)
      PetscCall(CreateAnalyticJacobianCOO(rdy));
      PetscCall(PetscObjectSetName((PetscObject)rdy->rhs_jac, "swe_rhs_jacobian"));
      PetscCall(TSSetRHSJacobian(rdy->ts, rdy->rhs_jac, rdy->rhs_jac, SWERHSJacobianAnalytic, rdy));
      break;
    default:
      PetscCheck(PETSC_FALSE, rdy->comm, PETSC_ERR_PLIB, "RegisterSWERHSJacobian called with numerics.jacobian == none");
  }

  // parameter Jacobian df/dn (p = per-cell Manning): rows follow the global
  // state layout, one parameter column per owned cell, 2 nonzeros per column.
  // With ARK-IMEX the n-dependence sits in the implicit part instead, and
  // RegisterSWEIMEXFriction registers TSSetIJacobianP.
  if (rdy->config.physics.flow.source.method == SOURCE_EXPLICIT) {
    PetscInt n_local;
    PetscCall(VecGetLocalSize(rdy->u_global, &n_local));
    PetscInt num_owned = n_local / 3;
    PetscCall(MatCreateAIJ(rdy->comm, n_local, num_owned, PETSC_DETERMINE, PETSC_DETERMINE, 1, NULL, 0, NULL, &rdy->rhs_jac_p));
    PetscCall(PetscObjectSetName((PetscObject)rdy->rhs_jac_p, "swe_rhs_jacobian_dn"));
    PetscCall(TSSetRHSJacobianP(rdy->ts, rdy->rhs_jac_p, SWERHSJacobianP, rdy));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Frees Jacobian-related resources (safe to call when none were created).
PetscErrorCode DestroySWERHSJacobian(RDy rdy) {
  PetscFunctionBegin;
  if (rdy->rhs_jac_fd_coloring) PetscCall(MatFDColoringDestroy(&rdy->rhs_jac_fd_coloring));
  if (rdy->rhs_jac) PetscCall(MatDestroy(&rdy->rhs_jac));
  if (rdy->rhs_jac_coo_v) PetscCall(PetscFree(rdy->rhs_jac_coo_v));
  if (rdy->rhs_jac_p) PetscCall(MatDestroy(&rdy->rhs_jac_p));
  if (rdy->imex_ijac) PetscCall(MatDestroy(&rdy->imex_ijac));
  if (rdy->imex_jacp_rhs) PetscCall(MatDestroy(&rdy->imex_jacp_rhs));
  PetscFunctionReturn(PETSC_SUCCESS);
}
