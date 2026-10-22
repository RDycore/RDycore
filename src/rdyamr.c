#include <muParserDLL.h>
#include <petsc/private/dmpleximpl.h> /*I      "petscdmplex.h"   I*/
#include <petscdmceed.h>
#include <petscdmplex.h>
#include <petscsys.h>
#include <private/rdycoreimpl.h>
#include <private/rdydmimpl.h>
#include <private/rdymathimpl.h>
#include <private/rdysweimpl.h>

static PetscErrorCode CreateAdaptLabelInternal(RDy rdy, DMLabel *adaptLabel) {
  /* PetscMPIInt rank; */
  DMLabel  label;
  PetscInt cStart, cEnd, c;

  PetscFunctionBegin;
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Adaptation Label", adaptLabel));
  label = *adaptLabel;

  DM dm = rdy->dm;
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));

  RDyMesh  *mesh  = &rdy->mesh;
  RDyCells *cells = &mesh->cells;

  for (c = cStart; c < cEnd; ++c) {
    PetscReal x, y;

    PetscInt icell = c - cStart;
    if (cells->is_owned[icell]) {
      x = cells->centroids[icell].X[0];
      y = cells->centroids[icell].X[1];
      if ((x >= 0.0 && x < 1.) && (y >= 3. && y <= 4.)) {
        PetscCall(DMLabelSetValue(label, c, DM_ADAPT_REFINE));
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Saves information regarding refinement of owned cells.
/// @param rdy         A RDy struct
/// @param size        The number of owned cells
/// @param refine_cell True/False array indicating if the cell should be refined
/// @return PETSC_SUCESS on success
PetscErrorCode RDyMarkOwnedCellsForAMR(RDy rdy, const PetscInt size, const PetscBool *refine_cell) {
  PetscFunctionBegin;

  rdy->amr.cells_marked_for_refinement = PETSC_TRUE;

  // check if the size of the array is equal to the number of owned cells
  PetscAssert(rdy->mesh.num_owned_cells == size, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "The size of array is not equal to the number of owned cells");

  // copy the data
  PetscCalloc1(size, &rdy->amr.refine_cell);
  for (PetscInt icell = 0; icell < size; ++icell) {
    rdy->amr.refine_cell[icell] = refine_cell[icell];
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Computes a per-owned-cell refinement indicator from the solution: the
///        largest jump in water-surface elevation (h + z) across the cell's
///        internal faces. Free surface rather than depth, so that water at rest
///        over a varying bed indicates nothing; a dam, a bore and a wet/dry
///        front all show up as a jump, including at step 0 before anything moves.
/// @param [in] rdy       the dycore
/// @param [out] indicator a Vec over owned cells, created here
static PetscErrorCode ComputeSurfaceJumpIndicator(RDy rdy, Vec *indicator) {
  PetscFunctionBegin;

  RDyMesh  *mesh  = &rdy->mesh;
  RDyCells *cells = &mesh->cells;
  RDyEdges *edges = &mesh->edges;

  PetscCall(VecCreateMPI(rdy->comm, mesh->num_owned_cells, PETSC_DETERMINE, indicator));
  PetscCall(VecZeroEntries(*indicator));

  // the local vector carries ghost cells, so a face on the partition boundary
  // sees the same jump from either side
  PetscCall(DMGlobalToLocal(rdy->dm, rdy->u_global, INSERT_VALUES, rdy->u_local));
  const PetscScalar *u;
  PetscCall(VecGetArrayRead(rdy->u_local, &u));
  PetscScalar *ind;
  PetscCall(VecGetArray(*indicator, &ind));

  PetscInt ndof = 3;  // h, hu, hv
  for (PetscInt e = 0; e < mesh->num_internal_edges; ++e) {
    PetscInt edge_id = edges->internal_edge_ids[e];
    PetscInt l = edges->cell_ids[2 * edge_id], r = edges->cell_ids[2 * edge_id + 1];
    if (l < 0 || r < 0) continue;

    PetscReal eta_l = u[ndof * l] + cells->centroids[l].X[2];
    PetscReal eta_r = u[ndof * r] + cells->centroids[r].X[2];
    PetscReal jump  = PetscAbsReal(eta_l - eta_r);

    for (PetscInt i = 0; i < 2; ++i) {
      PetscInt c = (i == 0) ? l : r;
      if (!cells->is_owned[c]) continue;
      PetscInt owned = cells->local_to_owned[c];
      if (jump > PetscRealPart(ind[owned])) ind[owned] = jump;
    }
  }

  // weight by cell area, so the indicator estimates the cell's CONTRIBUTION to
  // the error rather than the steepness at its faces. For a first-order scheme
  // the depth error of a cell is O(|jump|), so its share of an area-weighted
  // norm is O(|jump| * area). Refining on the unweighted jump chases the shock
  // alone and leaves the error, which a wide smooth region can dominate,
  // largely untouched (measured: it barely beat the unrefined mesh).
  PetscBool weight_by_area = PETSC_TRUE;
  PetscCall(PetscOptionsGetBool(NULL, "rdy_amr_", "-weight_by_area", &weight_by_area, NULL));
  if (weight_by_area) {
    for (PetscInt c = 0; c < mesh->num_cells; ++c) {
      if (!cells->is_owned[c]) continue;
      ind[cells->local_to_owned[c]] *= cells->areas[c];
    }
  }

  PetscCall(VecRestoreArray(*indicator, &ind));
  PetscCall(VecRestoreArrayRead(rdy->u_local, &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Spreads the indicator outward by `layers` cell layers, so that cells
///        the feature will reach before the next adapt are refined with it.
///        Without this the mesh lags: a front crossing several coarse cells per
///        coupling interval is resolved only after it has already passed.
///
///        The spread runs through the solution DM's halo exchange (component 0
///        of a 3-dof work vector carries the indicator), so it crosses
///        partition boundaries and gives the same result on any rank count.
static PetscErrorCode DilateIndicator(RDy rdy, Vec indicator, PetscInt layers) {
  PetscFunctionBegin;
  if (layers <= 0) PetscFunctionReturn(PETSC_SUCCESS);

  RDyMesh  *mesh  = &rdy->mesh;
  RDyCells *cells = &mesh->cells;
  RDyEdges *edges = &mesh->edges;

  Vec work_global, work_local;
  PetscCall(DMGetGlobalVector(rdy->dm, &work_global));
  PetscCall(DMGetLocalVector(rdy->dm, &work_local));

  for (PetscInt pass = 0; pass < layers; ++pass) {
    const PetscScalar *ind;
    PetscScalar       *w;
    PetscCall(VecZeroEntries(work_global));
    PetscCall(VecGetArrayRead(indicator, &ind));
    PetscCall(VecGetArray(work_global, &w));
    for (PetscInt o = 0; o < mesh->num_owned_cells; ++o) w[3 * o] = ind[o];
    PetscCall(VecRestoreArray(work_global, &w));
    PetscCall(VecRestoreArrayRead(indicator, &ind));

    PetscCall(DMGlobalToLocal(rdy->dm, work_global, INSERT_VALUES, work_local));

    const PetscScalar *wl;
    PetscScalar       *out;
    PetscCall(VecGetArrayRead(work_local, &wl));
    PetscCall(VecGetArray(indicator, &out));
    for (PetscInt e = 0; e < mesh->num_internal_edges; ++e) {
      PetscInt edge_id = edges->internal_edge_ids[e];
      PetscInt l = edges->cell_ids[2 * edge_id], r = edges->cell_ids[2 * edge_id + 1];
      if (l < 0 || r < 0) continue;
      if (cells->is_owned[l]) {
        PetscInt o = cells->local_to_owned[l];
        if (PetscRealPart(wl[3 * r]) > PetscRealPart(out[o])) out[o] = wl[3 * r];
      }
      if (cells->is_owned[r]) {
        PetscInt o = cells->local_to_owned[r];
        if (PetscRealPart(wl[3 * l]) > PetscRealPart(out[o])) out[o] = wl[3 * l];
      }
    }
    PetscCall(VecRestoreArray(indicator, &out));
    PetscCall(VecRestoreArrayRead(work_local, &wl));
  }

  PetscCall(DMRestoreLocalVector(rdy->dm, &work_local));
  PetscCall(DMRestoreGlobalVector(rdy->dm, &work_global));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Marks cells for refinement from the solution itself. The indicator is
///        the water-surface jump of ComputeSurfaceJumpIndicator; which cells it
///        selects is decided by a PETSc VecTagger under the "rdy_amr_" prefix,
///        so the criterion is a runtime choice:
///
///          -rdy_amr_vec_tagger_type relative -rdy_amr_vec_tagger_box 0.5,1  (default)
///          -rdy_amr_vec_tagger_type cdf      -rdy_amr_vec_tagger_box 0.95,1 (worst 5%)
///          -rdy_amr_vec_tagger_type absolute -rdy_amr_vec_tagger_box 0.1,1e30
///
///        The default is deliberately scale-free. An absolute jump threshold
///        disables itself: halving a cell halves the jump across its faces, so
///        a front that was over the threshold drops under it after one adapt
///        and the mesh stops following it. Relative and CDF boxes measure each
///        cell against the current field instead.
///
///        Refinement is capped at -rdy_amr_max_levels (default 3) levels below
///        the mesh AMR started from, since at a genuine discontinuity the
///        indicator never runs out of signal.
///
/// @param [in] rdy      the dycore
/// @param [out] nmarked number of cells marked across all ranks (may be NULL)
/// @return PETSC_SUCCESS on success
PetscErrorCode RDyMarkCellsForAMRBySurfaceJump(RDy rdy, PetscInt *nmarked) {
  PetscFunctionBegin;

  Vec indicator;
  PetscCall(ComputeSurfaceJumpIndicator(rdy, &indicator));

  // a front crosses several cells between adapts, so refine a band ahead of it
  PetscInt buffer_layers = 2;
  PetscCall(PetscOptionsGetInt(NULL, "rdy_amr_", "-buffer_layers", &buffer_layers, NULL));
  PetscCall(DilateIndicator(rdy, indicator, buffer_layers));

  // cells already at the finest allowed level cannot be marked, so zero their
  // indicator before the tagger sees it (a relative or CDF box would otherwise
  // keep selecting them)
  PetscInt max_levels = 3;
  PetscCall(PetscOptionsGetInt(NULL, "rdy_amr_", "-max_levels", &max_levels, NULL));
  PetscReal min_area = rdy->amr.base_min_cell_area / PetscPowReal(2.0, (PetscReal)max_levels);
  {
    PetscScalar *ind;
    PetscCall(VecGetArray(indicator, &ind));
    for (PetscInt c = 0; c < rdy->mesh.num_cells; ++c) {
      if (!rdy->mesh.cells.is_owned[c]) continue;
      if (rdy->mesh.cells.areas[c] < min_area) ind[rdy->mesh.cells.local_to_owned[c]] = 0.0;
    }
    PetscCall(VecRestoreArray(indicator, &ind));
  }

  VecTagger tagger;
  PetscCall(VecTaggerCreate(rdy->comm, &tagger));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)tagger, "rdy_amr_"));
  PetscCall(VecTaggerSetType(tagger, VECTAGGERRELATIVE));
  VecTaggerBox box = {.min = 0.5, .max = 1.0};
  PetscCall(VecTaggerRelativeSetBox(tagger, &box));
  PetscCall(VecTaggerSetFromOptions(tagger));
  PetscCall(VecTaggerSetUp(tagger));

  IS        tagged;
  PetscBool listed;
  PetscCall(VecTaggerComputeIS(tagger, indicator, &tagged, &listed));
  PetscCheck(listed, rdy->comm, PETSC_ERR_SUP, "The -rdy_amr_tagger_type in use cannot list the cells it tags");

  PetscBool *refine_cell;
  PetscCall(PetscCalloc1(rdy->mesh.num_owned_cells, &refine_cell));

  // the tagger lists RANK-LOCAL indices into the indicator, not global ones
  // (PETSc's own TS ex11 consumes them the same way)
  PetscInt        n;
  const PetscInt *idx;
  PetscCall(ISGetLocalSize(tagged, &n));
  PetscCall(ISGetIndices(tagged, &idx));
  for (PetscInt i = 0; i < n; ++i) {
    PetscCheck(idx[i] >= 0 && idx[i] < rdy->mesh.num_owned_cells, PETSC_COMM_SELF, PETSC_ERR_PLIB,
               "tagged index %" PetscInt_FMT " outside this rank's %" PetscInt_FMT " owned cells", idx[i], rdy->mesh.num_owned_cells);
    refine_cell[idx[i]] = PETSC_TRUE;
  }
  PetscCall(ISRestoreIndices(tagged, &idx));

  PetscCall(RDyMarkOwnedCellsForAMR(rdy, rdy->mesh.num_owned_cells, refine_cell));

  if (nmarked) PetscCallMPI(MPI_Allreduce(&n, nmarked, 1, MPIU_INT, MPI_SUM, rdy->comm));

  PetscCall(PetscFree(refine_cell));
  PetscCall(ISDestroy(&tagged));
  PetscCall(VecTaggerDestroy(&tagger));
  PetscCall(VecDestroy(&indicator));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Creates a label to refine cells based on the data previously provided by call
///        to RDyMarkOwnedCellsForAMR
/// @param rdy        RDy struct
/// @param adaptLabel A DMLabel marks appropriate locally cells for refinement
/// @return PETSC_SUCESS on success
static PetscErrorCode CreateAdaptLabelFromMarkedCells(RDy rdy, DMLabel *adaptLabel) {
  PetscFunctionBegin;
  DMLabel label;

  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Adaptation Label", adaptLabel));
  label = *adaptLabel;

  PetscInt cStart, cEnd;
  PetscCall(DMPlexGetHeightStratum(rdy->dm, 0, &cStart, &cEnd));

  RDyMesh  *mesh  = &rdy->mesh;
  RDyCells *cells = &mesh->cells;

  for (PetscInt c = 0; c < mesh->num_owned_cells; ++c) {
    if (rdy->amr.refine_cell[c]) {
      PetscInt idx = cells->owned_to_local[c] + cStart;
      PetscCall(DMLabelSetValue(label, idx, DM_ADAPT_REFINE));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Creates matrices for the interpolation between coarse and fine meshes.
/// @param odm          The coarse DM that was refined
/// @param dm           The fine DM
/// @param CoarseToFine Matrix for interpolating local Vec from coarse to fine grid
/// @param FineToCoarse Matrix for interpolating local Vec from fine to coarse grid
/// @return PETSC_SUCESS on success
static PetscErrorCode ConstructRefineTree(DM odm, DM dm, Mat CoarseToFineMatNDof, Mat FineToCoarseMatNDof, Mat CoarseToFineMat1Dof,
                                          Mat FineToCoarseMat1Dof) {
  DMPlexTransform tr;
  PetscInt        cStart, cEnd, bs, Istart, Jstart, Istart_1Dof, Jstart_1Dof;
  PetscScalar     val = 1.0;

  PetscFunctionBegin;
  PetscCall(MatGetBlockSize(CoarseToFineMatNDof, &bs));
  PetscCall(DMPlexGetTransform(dm, &tr));
  if (!tr) PetscFunctionReturn(PETSC_SUCCESS);
  // the saved transform no longer holds its base DM (PETSc clears it to break a reference cycle),
  // so the coarse DM must be passed in
  PetscCall(DMPlexGetHeightStratum(odm, 0, &cStart, &cEnd));

  PetscCall(MatGetOwnershipRange(CoarseToFineMatNDof, &Istart, NULL));
  PetscCall(MatGetOwnershipRangeColumn(CoarseToFineMatNDof, &Jstart, NULL));

  PetscCall(MatGetOwnershipRange(CoarseToFineMat1Dof, &Istart_1Dof, NULL));
  PetscCall(MatGetOwnershipRangeColumn(CoarseToFineMat1Dof, &Jstart_1Dof, NULL));

  for (PetscInt c = cStart; c < cEnd; ++c) {
    DMPolytopeType  ct;
    DMPolytopeType *rct;
    PetscInt       *rsize, *rcone, *rornt;
    PetscInt        Nct, dim, pNew = 0;

    PetscCall(DMPlexGetCellType(odm, c, &ct));
    dim = DMPolytopeTypeGetDim(ct);
    PetscCall(DMPlexTransformCellTransform(tr, ct, c, NULL, &Nct, &rct, &rsize, &rcone, &rornt));
    for (PetscInt n = 0; n < Nct; ++n) {
      if (DMPolytopeTypeGetDim(rct[n]) != dim) continue;
      for (PetscInt r = 0; r < rsize[n]; ++r) {
        PetscCall(DMPlexTransformGetTargetPoint(tr, ct, rct[n], c, r, &pNew));
        for (PetscInt i = 0; i < bs; i++) {
          PetscCall(MatSetValue(CoarseToFineMatNDof, Istart + bs * (pNew - 0) + i, Jstart + bs * (c - cStart) + i, val, INSERT_VALUES));
          PetscCall(MatSetValue(FineToCoarseMatNDof, Jstart + bs * (c - cStart) + i, Istart + bs * (pNew - 0) + i, 1.0 / rsize[n], INSERT_VALUES));
        }

        PetscCall(MatSetValue(CoarseToFineMat1Dof, Istart + (pNew - 0), Jstart + (c - cStart), val, INSERT_VALUES));
        PetscCall(MatSetValue(FineToCoarseMat1Dof, Jstart + (c - cStart), Istart + (pNew - 0), 1.0 / rsize[n], INSERT_VALUES));
      }
    }
  }

  PetscCall(MatSetFromOptions(CoarseToFineMatNDof));
  PetscCall(MatAssemblyBegin(CoarseToFineMatNDof, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(CoarseToFineMatNDof, MAT_FINAL_ASSEMBLY));
  PetscCall(MatViewFromOptions(CoarseToFineMatNDof, NULL, "-adapt_c2f_mat_view"));

  PetscCall(MatSetFromOptions(FineToCoarseMatNDof));
  PetscCall(MatAssemblyBegin(FineToCoarseMatNDof, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(FineToCoarseMatNDof, MAT_FINAL_ASSEMBLY));
  PetscCall(MatViewFromOptions(FineToCoarseMatNDof, NULL, "-adapt_f2c_mat_view"));

  PetscCall(MatSetFromOptions(CoarseToFineMat1Dof));
  PetscCall(MatAssemblyBegin(CoarseToFineMat1Dof, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(CoarseToFineMat1Dof, MAT_FINAL_ASSEMBLY));
  PetscCall(MatViewFromOptions(CoarseToFineMat1Dof, NULL, "-adapt_c2f_1dof_mat_view"));

  PetscCall(MatSetFromOptions(FineToCoarseMat1Dof));
  PetscCall(MatAssemblyBegin(FineToCoarseMat1Dof, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(FineToCoarseMat1Dof, MAT_FINAL_ASSEMBLY));
  PetscCall(MatViewFromOptions(FineToCoarseMat1Dof, NULL, "-adapt_f2c_1dof_mat_view"));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AdaptMesh(RDy rdy, const PetscInt bs, DM *dm_fine, Mat *CoarseToFineMatNDof, Mat *FineToCoarseMatNDof, Mat *CoarseToFineMat1Dof,
                                Mat *FineToCoarseMat1Dof) {
  PetscFunctionBeginUser;

  DM       dm = rdy->dm;
  DM       dmAdapt;
  DMLabel  adaptLabel;
  PetscInt d_nz = 4, ccStart, ccEnd, fcStart, fcEnd;
  char     opt[128];

  PetscCall(DMViewFromOptions(dm, NULL, "-adapt_pre_dm_view"));

  if (!rdy->amr.cells_marked_for_refinement) {
    PetscCall(CreateAdaptLabelInternal(rdy, &adaptLabel));
  } else {
    PetscCall(CreateAdaptLabelFromMarkedCells(rdy, &adaptLabel));
    rdy->amr.cells_marked_for_refinement = PETSC_FALSE;
    PetscFree(rdy->amr.refine_cell);
  }

  PetscCall(DMPlexSetSaveTransform(dm, PETSC_TRUE));
  PetscCall(DMAdaptLabel(dm, adaptLabel, &dmAdapt));  // DMRefine
  PetscCall(DMLabelDestroy(&adaptLabel));

  PetscCheck(dmAdapt, PETSC_COMM_WORLD, PETSC_ERR_USER, "Refinement failed.");

  PetscCall(PetscObjectSetName((PetscObject)dmAdapt, "Adapted Mesh"));
  PetscCall(PetscSNPrintf(opt, 128, "-adapt_dm_view"));
  PetscCall(DMViewFromOptions(dmAdapt, NULL, opt));

  // make interpolation matrix
  PetscCall(DMPlexGetHeightStratum(dm, 0, &ccStart, &ccEnd));
  PetscCall(DMPlexGetHeightStratum(dmAdapt, 0, &fcStart, &fcEnd));

  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, bs * (fcEnd - fcStart), bs * (ccEnd - ccStart), d_nz, NULL, CoarseToFineMatNDof));
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, bs * (ccEnd - ccStart), bs * (fcEnd - fcStart), d_nz, NULL, FineToCoarseMatNDof));
  PetscCall(MatSetBlockSize(*CoarseToFineMatNDof, bs));
  PetscCall(MatSetBlockSize(*FineToCoarseMatNDof, bs));

  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, (fcEnd - fcStart), (ccEnd - ccStart), d_nz, NULL, CoarseToFineMat1Dof));
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, (ccEnd - ccStart), (fcEnd - fcStart), d_nz, NULL, FineToCoarseMat1Dof));
  PetscCall(MatSetBlockSize(*CoarseToFineMat1Dof, 1));
  PetscCall(MatSetBlockSize(*FineToCoarseMat1Dof, 1));

  PetscCall(ConstructRefineTree(dm, dmAdapt, *CoarseToFineMatNDof, *FineToCoarseMatNDof, *CoarseToFineMat1Dof, *FineToCoarseMat1Dof));

  *dm_fine = dmAdapt;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Given the coarse-to-fine mesh matrix, determine the mapping from coarse cells to fine cells.
/// @param rdy_coarse       RDy struct corresponding to the coarse mesh
/// @param CoarseToFine     Matrix to remap data from coarse to fine mesh
/// @param X_coarse         A local Vec on the coarse mesh
/// @param X_fine           A local Vec on the fine mesh
/// @param fineIsOwned      For all fine cells, whether the cell is owned by this rank
/// @param fineToCoarseMap  For all fine cells, the corresponding coarse cell ID
/// @param coarseToFineMap  For all coarse cells, the corresponding fine cell IDs
/// @param coarseNumFine    For all coarse cells, the number of fine cells
/// @param coarseOffsetFine Offset for each coarse cell in the coarseToFineMap
/// @return PETSC_SUCESS on success
static PetscErrorCode DetermineCoarseToFineCellMapping(RDy rdy_coarse, Mat CoarseToFineMatNDof, Vec X_coarse, Vec X_fine, PetscInt *fineIsOwned,
                                                       PetscInt *fineToCoarseMap, PetscInt *coarseToFineMap, PetscInt *coarseNumFine,
                                                       PetscInt *coarseOffsetFine) {
  PetscFunctionBeginUser;

  RDyMesh  *mesh  = &rdy_coarse->mesh;
  RDyCells *cells = &mesh->cells;

  PetscInt ndof;
  PetscCall(VecGetBlockSize(X_coarse, &ndof));

  // create Vec to mark owned cells
  Vec X_coarse_owned, X_fine_owned;
  PetscCall(VecDuplicate(X_coarse, &X_coarse_owned));
  PetscCall(VecDuplicate(X_fine, &X_fine_owned));

  PetscScalar *x_ptr_coarse, *x_ptr_coarse_owned, *x_ptr_fine, *x_ptr_fine_owned;

  // - Put the local cell ID in the first position of X_coarse
  //   and -1 in the rest of the positions
  // - Put 1 or 0 in the first position of X_coarse_owned to indicate
  //   if the cell is owned or not, and -1 in the rest of the positions
  PetscCall(VecGetArray(X_coarse, &x_ptr_coarse));
  PetscCall(VecGetArray(X_coarse_owned, &x_ptr_coarse_owned));

  for (PetscInt c = 0; c < mesh->num_cells; ++c) {
    x_ptr_coarse[c * ndof]       = c;
    x_ptr_coarse_owned[c * ndof] = cells->is_owned[c];

    // put -1 in the rest of the positions
    for (PetscInt idof = 1; idof < ndof; idof++) {
      x_ptr_coarse[c * ndof + idof]       = -1;
      x_ptr_coarse_owned[c * ndof + idof] = -1;
    }
  }
  PetscCall(VecRestoreArray(X_coarse, &x_ptr_coarse));
  PetscCall(VecRestoreArray(X_coarse_owned, &x_ptr_coarse_owned));

  // Multiply the coarse-to-fine matrix with the coarse vector
  // to get the fine vector
  PetscCall(MatMult(CoarseToFineMatNDof, X_coarse, X_fine));
  PetscCall(MatMult(CoarseToFineMatNDof, X_coarse_owned, X_fine_owned));

  PetscInt numFine, numCoarse;
  PetscCall(VecGetLocalSize(X_fine, &numFine));
  PetscCall(VecGetLocalSize(X_coarse, &numCoarse));
  numFine   = (PetscInt)numFine / ndof;
  numCoarse = (PetscInt)numCoarse / ndof;

  // For each fine cell, determine the corresponding coarse cell
  // and keep track of the number of fine cells corresponding to each coarse cell
  PetscCall(VecGetArray(X_fine, &x_ptr_fine));
  PetscCall(VecGetArray(X_fine_owned, &x_ptr_fine_owned));

  for (PetscInt c = 0; c < numFine; c++) {
    // get the local cell ID of the coarse cell
    PetscInt coarse_cell_id = (PetscInt)x_ptr_fine[c * ndof];

    fineToCoarseMap[c] = coarse_cell_id;
    fineIsOwned[c]     = (PetscInt)x_ptr_fine_owned[c * ndof];
    coarseNumFine[coarse_cell_id]++;
  }

  // Before creating the coarseToFineMap, we need to create the offsets
  // for each coarse cell. The offsets are the number of fine cells
  // corresponding to each coarse cell. The offsets are used to
  // create the coarseToFineMap.
  coarseOffsetFine[0] = 0;
  for (PetscInt c = 1; c < numCoarse; c++) {
    coarseOffsetFine[c]  = coarseOffsetFine[c - 1] + coarseNumFine[c - 1];
    coarseNumFine[c - 1] = 0;
  }
  coarseNumFine[numCoarse - 1] = 0;

  // Now traverse the fine vector again and save the fine cell ID
  // for each coarse cell in the coarseToFineMap.
  for (PetscInt c = 0; c < numFine; c++) {
    // get the natural cell ID
    PetscInt coarse_cell_id = (PetscInt)x_ptr_fine[c * ndof];

    PetscInt offset   = coarseOffsetFine[coarse_cell_id];
    PetscInt num_fine = coarseNumFine[coarse_cell_id];

    coarseToFineMap[offset + num_fine] = c;

    // increment the number of fine cells for this coarse cell
    coarseNumFine[coarse_cell_id]++;
  }
  PetscCall(VecRestoreArray(X_fine, &x_ptr_fine));
  PetscCall(VecRestoreArray(X_fine_owned, &x_ptr_fine_owned));

  // clean up
  PetscCall(VecDestroy(&X_coarse_owned));
  PetscCall(VecDestroy(&X_fine_owned));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief For all the regions in the coarse mesh, create the corresponding regions in the fine mesh.
/// @param rdy_coarse       RDy struct corresponding to the coarse mesh
/// @param numFine          Number of owned cells in the fine mesh
/// @param fineIsOwned      For all fine cells, whether the cell is owned by this rank
/// @param coarseToFineMap  For all coarse cells, the corresponding fine cell IDs
/// @param coarseNumFine    For all coarse cells, the number of fine cells that the coarse cell was split into
/// @param coarseOffsetFine For all coarse cells, the offset in the coarseToFineMap for the coarse cell
/// @param num_regions      Number of regions for the fine mesh
/// @param regions          Regions for the fine mesh
/// @return PETSC_SUCESS on success
static PetscErrorCode CreateRefinedRegionsFromCoarseRDy(RDy rdy_coarse, PetscInt numFine, PetscInt *fineIsOwned, PetscInt *coarseToFineMap,
                                                        PetscInt *coarseNumFine, PetscInt *coarseOffsetFine, PetscInt *num_regions,
                                                        RDyRegion **regions) {
  PetscFunctionBeginUser;

  RDyMesh  *mesh  = &rdy_coarse->mesh;
  RDyCells *cells = &mesh->cells;

  *num_regions = rdy_coarse->num_regions;

  PetscCall(PetscCalloc1(*num_regions, regions));

  // For fine cells, determine the global IDs of owned cells, which depends on if a local cell
  // is owned or not.
  PetscInt *fine_LocaltoGlobalIDs, count = 0;
  PetscCall(PetscCalloc1(numFine, &fine_LocaltoGlobalIDs));
  for (PetscInt c = 0; c < numFine; c++) {
    if (fineIsOwned[c]) {
      fine_LocaltoGlobalIDs[c] = count;
      count++;
    } else {
      fine_LocaltoGlobalIDs[c] = -1;
    }
  }

  for (PetscInt r = 0; r < rdy_coarse->num_regions; ++r) {
    RDyRegion *region_coarse = &rdy_coarse->regions[r];
    RDyRegion *region_fine   = (*regions + r);

    region_fine->id = region_coarse->id;
    strcpy(region_fine->name, region_coarse->name);

    if (region_coarse->num_owned_cells > 0) {
      //
      // Determine the global IDs of "owned" cells in the fine region
      //

      // 1. Determine the number of owned fine cells in the fine region by
      //    traversing the coarse region and counting the number of fine cells
      //    corresponding to each "owened" coarse cell.
      PetscInt num_owned_fine_cells = 0;
      for (PetscInt c = 0; c < region_coarse->num_owned_cells; ++c) {
        PetscInt coarse_global_id = region_coarse->owned_cell_global_ids[c];
        PetscInt coarse_local_id  = cells->owned_to_local[coarse_global_id];
        num_owned_fine_cells += coarseNumFine[coarse_local_id];
      }

      // 2. Allocate memory
      region_fine->num_owned_cells = num_owned_fine_cells;
      PetscCall(PetscCalloc1(region_fine->num_owned_cells, &region_fine->owned_cell_global_ids));

      // 3. Fill the global IDs of the owned cells in the fine region.
      //    The coarseOffsetFine and coarseToFineMap use "local IDs".
      //    So, the global ID of the corase cell must be converted to local ID
      //    before using the coarseOffsetFine and coarseToFineMap.
      count = 0;
      for (PetscInt c = 0; c < region_coarse->num_owned_cells; ++c) {
        PetscInt coarse_global_id = region_coarse->owned_cell_global_ids[c];
        PetscInt coarse_local_id  = cells->owned_to_local[coarse_global_id];
        PetscInt offset           = coarseOffsetFine[coarse_local_id];

        for (PetscInt j = 0; j < coarseNumFine[coarse_local_id]; j++) {
          PetscInt fine_local_id                    = coarseToFineMap[offset + j];
          region_fine->owned_cell_global_ids[count] = fine_LocaltoGlobalIDs[fine_local_id];
          count++;
        }
      }

      //
      // Determine the local IDs of "local" cells in the fine region
      //

      // 1. Determine the number of local fine cells in the fine region by
      //    traversing the coarse region and counting the number of fine cells
      //    corresponding to each coarse cell.
      PetscInt num_local_fine_cells = 0;
      for (PetscInt c = 0; c < region_coarse->num_local_cells; ++c) {
        PetscInt coarse_local_id = region_coarse->cell_local_ids[c];
        num_local_fine_cells += coarseNumFine[coarse_local_id];
      }

      // 2. Allocate memory
      region_fine->num_local_cells = num_local_fine_cells;
      PetscCall(PetscCalloc1(region_fine->num_local_cells, &region_fine->cell_local_ids));

      // 3. Fill the local IDs of the fine cells in the fine region.
      count = 0;
      for (PetscInt c = 0; c < region_coarse->num_local_cells; ++c) {
        PetscInt coarse_local_id = region_coarse->cell_local_ids[c];
        PetscInt offset          = coarseOffsetFine[coarse_local_id];

        for (PetscInt j = 0; j < coarseNumFine[coarse_local_id]; j++) {
          region_fine->cell_local_ids[count] = coarseToFineMap[offset + j];
          count++;
        }
      }

    } else {
      region_fine->num_local_cells = 0;
      region_fine->num_owned_cells = 0;
    }
  }

  PetscCall(PetscFree(fine_LocaltoGlobalIDs));

  PetscFunctionReturn(PETSC_SUCCESS);
}

extern PetscErrorCode InitOperator(RDy rdy);
extern PetscErrorCode InitMaterialProperties(RDy rdy);
extern PetscErrorCode InitSolver(RDy rdy);
extern PetscErrorCode InitDirichletBoundaryConditions(RDy rdy);
extern PetscErrorCode InitSourceConditions(RDy rdy);

/// @brief Computes A * B, and updates A or B with the result.
/// @param A Matrix
/// @param B Matrix
/// @return PETSC_SUCESS on success
static PetscErrorCode MatMultResuse(Mat *A, Mat *B, PetscBool update_B) {
  PetscFunctionBeginUser;
  Mat tmp;

  PetscCall(MatMatMult(*A, *B, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &tmp));

  if (update_B) {
    PetscCall(MatDestroy(B));
    PetscCall(MatDuplicate(tmp, MAT_COPY_VALUES, B));
  } else {
    PetscCall(MatDestroy(A));
    PetscCall(MatDuplicate(tmp, MAT_COPY_VALUES, A));
  }

  PetscCall(MatDestroy(&tmp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyEnableAMR(RDy rdy) {
  PetscFunctionBeginUser;

  PetscInt nstep;
  PetscCall(RDyGetStep(rdy, &nstep));
  PetscAssert(nstep == 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ,
              "It seems the model has already taken few time steps. AMR cannot be enabled only during initialization.");

  rdy->amr.is_refinement_on                = PETSC_TRUE;
  rdy->amr.last_refinement_level_outputted = -1;

  PetscReal local_min = PETSC_MAX_REAL;
  for (PetscInt c = 0; c < rdy->mesh.num_cells; ++c) {
    if (rdy->mesh.cells.is_owned[c]) local_min = PetscMin(local_min, rdy->mesh.cells.areas[c]);
  }
  PetscCallMPI(MPI_Allreduce(&local_min, &rdy->amr.base_min_cell_area, 1, MPIU_REAL, MPI_MIN, rdy->comm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyPerformAMR(RDy rdy) {
  Mat      CoarseToFineMatNDof, FineToCoarseMatNDof;
  Mat      CoarseToFineMat1Dof, FineToCoarseMat1Dof;
  DM       dm_fine;
  Vec      U_coarse_local, U_fine_local;
  PetscInt ndof_coarse;
  PetscFunctionBeginUser;
  PetscCall(VecGetBlockSize(rdy->u_global, &ndof_coarse));

  /* Adapt */
  PetscCall(AdaptMesh(rdy, ndof_coarse, &dm_fine, &CoarseToFineMatNDof, &FineToCoarseMatNDof, &CoarseToFineMat1Dof, &FineToCoarseMat1Dof));
  PetscCall(DMLocalizeCoordinates(dm_fine));
  PetscCall(DMViewFromOptions(dm_fine, NULL, "-dm_fine_view"));
  PetscCall(DMSetCoarseDM(dm_fine, rdy->dm));
  PetscCall(DMGetCoordinatesLocalSetUp(dm_fine));

  if (0) {
    PetscSection sec;
    PetscCall(DMGetLocalSection(rdy->dm, &sec));
    PetscInt nfields;
    PetscSectionGetNumFields(sec, &nfields);
    printf("++++++++++++++++++++++++++++++++++++\n");
    printf("About coarse DM: \n");
    printf("nfields = %d\n", (int)nfields);
    for (PetscInt f = 0; f < nfields; f++) {
      PetscInt ncomp;
      PetscCall(PetscSectionGetFieldComponents(sec, f, &ncomp));
      printf("  field = %d; num_component = %d\n", (int)f, (int)ncomp);
      for (PetscInt c = 0; c < ncomp; c++) {
        const char *comp_name;
        PetscSectionGetComponentName(sec, f, c, &comp_name);
        printf("    field = %d; component = %d; comp_name = %s\n", (int)f, (int)c, comp_name);
      }
    }
  }

  if (0) {
    PetscSection sec;
    PetscCall(DMGetLocalSection(dm_fine, &sec));
    PetscInt nfields;
    PetscSectionGetNumFields(sec, &nfields);
    printf("++++++++++++++++++++++++++++++++++++\n");
    printf("About refined DM: \n");
    printf("nfields = %d\n", (int)nfields);
    for (PetscInt f = 0; f < nfields; f++) {
      PetscInt ncomp;
      PetscCall(PetscSectionGetFieldComponents(sec, f, &ncomp));
      printf("  field = %d; num_component = %d\n", (int)f, (int)ncomp);
      for (PetscInt c = 0; c < ncomp; c++) {
        const char *comp_name;
        PetscSectionGetComponentName(sec, f, c, &comp_name);
        printf("    field = %d; component = %d; comp_name = %s\n", (int)f, (int)c, comp_name);
      }
    }
    printf("++++++++++++++++++++++++++++++++++++\n");
  }

  {
    PetscSection coarse_sec, fine_sec;
    PetscCall(DMGetLocalSection(rdy->dm, &coarse_sec));
    PetscCall(DMGetLocalSection(dm_fine, &fine_sec));
    PetscInt nfields;
    PetscSectionGetNumFields(coarse_sec, &nfields);
    for (PetscInt f = 0; f < nfields; f++) {
      PetscInt ncomp;
      PetscCall(PetscSectionGetFieldComponents(coarse_sec, f, &ncomp));
      for (PetscInt c = 0; c < ncomp; c++) {
        const char *comp_name;
        PetscSectionGetComponentName(coarse_sec, f, c, &comp_name);
        PetscSectionSetComponentName(fine_sec, f, c, comp_name);
      }
    }
  }

  // create a local vector for the coarse mesh
  PetscCall(VecDuplicate(rdy->u_local, &U_coarse_local));

  // determine the mapping of cells from fine to coarse mesh
  PetscCall(DMCreateLocalVector(dm_fine, &U_fine_local));

  PetscInt *fineToCoarseMap, *fineIsOwned, *coarseToFineMap, *coarseNumFine, *coarseOffsetFine;

  PetscInt numFine, numCoarse;

  PetscCall(VecGetLocalSize(U_coarse_local, &numCoarse));
  PetscCall(PetscCalloc1(numCoarse, &coarseNumFine));
  PetscCall(PetscCalloc1(numCoarse, &coarseOffsetFine));
  for (PetscInt c = 0; c < numCoarse; c++) {
    coarseNumFine[c] = 0;
  }

  PetscCall(VecGetLocalSize(U_fine_local, &numFine));
  PetscCall(PetscCalloc1(numFine, &fineIsOwned));
  PetscCall(PetscCalloc1(numFine, &fineToCoarseMap));
  PetscCall(PetscCalloc1(numFine, &coarseToFineMap));

  // determine the mapping of cells from coarse to fine mesh
  PetscCall(DetermineCoarseToFineCellMapping(rdy, CoarseToFineMatNDof, U_coarse_local, U_fine_local, fineIsOwned, fineToCoarseMap, coarseToFineMap,
                                             coarseNumFine, coarseOffsetFine));

  // create data structure for the refined regions from existing coarse regions
  RDyRegion *refined_regions = NULL;
  PetscInt   num_regions;

  PetscMPIInt rank;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm_fine), &rank));
  PetscCall(
      CreateRefinedRegionsFromCoarseRDy(rdy, numFine, fineIsOwned, coarseToFineMap, coarseNumFine, coarseOffsetFine, &num_regions, &refined_regions));
  PetscCall(PetscFree(fineIsOwned));
  PetscCall(PetscFree(coarseNumFine));
  PetscCall(PetscFree(coarseOffsetFine));
  PetscCall(PetscFree(coarseToFineMap));
  PetscCall(PetscFree(fineToCoarseMap));

  // make a copy of the old solution
  PetscCall(DMGlobalToLocal(rdy->dm, rdy->u_global, INSERT_VALUES, U_coarse_local));

  // destroy the coarse vectors
  PetscCall(RDyDestroyVectors(&rdy));

  // keep a copy of base DMs, otherwise destroy the coarse DMs
  if (!rdy->amr.num_refinements) {
    rdy->amr.dm_base      = rdy->dm;
    rdy->amr.dm_1dof_base = rdy->dm_1dof;

    PetscCall(MatDuplicate(CoarseToFineMatNDof, MAT_COPY_VALUES, &rdy->amr.BaseToCurrentMatNDof));
    PetscCall(MatDuplicate(FineToCoarseMatNDof, MAT_COPY_VALUES, &rdy->amr.CurrentToBaseMatNDof));
    PetscCall(MatDuplicate(CoarseToFineMat1Dof, MAT_COPY_VALUES, &rdy->amr.BaseToCurrentMat1Dof));
    PetscCall(MatDuplicate(FineToCoarseMat1Dof, MAT_COPY_VALUES, &rdy->amr.CurrentToBaseMat1Dof));

  } else {
    // destroy the coarse DM
    PetscCall(DMDestroy(&rdy->dm));
    PetscCall(DMDestroy(&rdy->dm_1dof));

    PetscCall(MatMultResuse(&CoarseToFineMatNDof, &rdy->amr.BaseToCurrentMatNDof, PETSC_TRUE));
    PetscCall(MatMultResuse(&rdy->amr.CurrentToBaseMatNDof, &FineToCoarseMatNDof, PETSC_FALSE));
    PetscCall(MatMultResuse(&CoarseToFineMat1Dof, &rdy->amr.BaseToCurrentMat1Dof, PETSC_TRUE));
    PetscCall(MatMultResuse(&rdy->amr.CurrentToBaseMat1Dof, &FineToCoarseMat1Dof, PETSC_FALSE));
  }

  // set the DM to be the refined DM
  rdy->dm      = dm_fine;
  rdy->flow_dm = rdy->dm;

  // create the auxiliary DM
  PetscCall(CreateAuxiliaryDMs(rdy));

  // create new refined vectors
  PetscCall(CreateVectors(rdy));

  // destroy and recreate regions
  for (PetscInt i = 0; i < rdy->num_regions; ++i) {
    PetscCall(DestroyRegion(&rdy->regions[i]));
  }
  if (rdy->regions) PetscFree(rdy->regions);

  // set region as the new refined regions
  rdy->num_regions = num_regions;
  rdy->regions     = refined_regions;

  // mark the mesh was refined
  rdy->amr.mesh_was_refined = PETSC_TRUE;
  rdy->amr.num_refinements++;

  // destroy and recreate mesh
  PetscCall(RDyMeshDestroy(rdy->mesh));
  PetscCall(RDyMeshCreateFromDM(rdy->dm, rdy->amr.num_refinements, &rdy->mesh));
  if (rdy->config.physics.flow.well_balancing == WELL_BALANCING_HR) {
    PetscCall(RDyMeshOverride2DProjection(&rdy->mesh));
  }
  if (rdy->config.grid.cell_elevation.file[0]) {
    PetscCall(OverrideCellElevation(rdy));
  }

  // initialize the refined solution from existing previous solution
  PetscCall(MatMult(CoarseToFineMatNDof, U_coarse_local, rdy->u_local));
  PetscCall(DMLocalToGlobal(rdy->dm, rdy->u_local, INSERT_VALUES, rdy->u_global));
  PetscCall(MatDestroy(&CoarseToFineMatNDof));
  PetscCall(MatDestroy(&FineToCoarseMatNDof));
  PetscCall(MatDestroy(&CoarseToFineMat1Dof));
  PetscCall(MatDestroy(&FineToCoarseMat1Dof));
  PetscCall(VecDestroy(&U_coarse_local));
  PetscCall(VecDestroy(&U_fine_local));

  // destroy the operator
  PetscCall(DestroyOperator(&rdy->operator));

  // destroy the boundaries and reallocate memory
  PetscCall(RDyDestroyBoundaries(&rdy));
  InitBoundaries(rdy);

  // reinitialize the operator
  PetscCall(InitOperator(rdy));

  // Re-wire OutputVar references since the operator was destroyed and recreated.
  {
    Operator *op                      = rdy->operator;
    rdy->prim_vars_output.petsc_inst  = op->primitive_variables;
    rdy->prim_vars_output.petsc_accum = op->primitive_variables_accum;
    rdy->prim_vars_output.ceed_inst   = op->ceed.primitive_variables;
    rdy->prim_vars_output.ceed_accum  = op->ceed.primitive_variables_accum;
    rdy->src_output.petsc_inst        = op->src_inst;
    rdy->src_output.petsc_accum       = op->src_accum;
    rdy->src_output.ceed_inst         = op->ceed.ceed_src_inst;
    rdy->src_output.ceed_accum        = op->ceed.ceed_src_accum;
  }

  // reinitialize material properties
  PetscCall(InitMaterialProperties(rdy));

  // save time and timestep from TS
  PetscReal time, dt;
  PetscInt  nstep;
  PetscCall(TSGetTime(rdy->ts, &time));
  PetscCall(TSGetTimeStep(rdy->ts, &dt));
  PetscCall(TSGetStepNumber(rdy->ts, &nstep));

  // destroy the TS and re-create it
  PetscCall(TSDestroy(&rdy->ts));
  PetscCall(InitSolver(rdy));

  // set the time and timstep in TS
  PetscCall(TSSetStepNumber(rdy->ts, nstep));
  PetscCall(TSSetTime(rdy->ts, time));
  PetscCall(TSSetTimeStep(rdy->ts, dt));

  // make sure any Dirichlet boundary conditions are properly specified
  PetscCall(InitDirichletBoundaryConditions(rdy));

  // initialize the source terms
  PetscCall(InitSourceConditions(rdy));

  PetscFunctionReturn(PETSC_SUCCESS);
}
