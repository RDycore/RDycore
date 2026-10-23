#include <muParserDLL.h>
#include <petscdmceed.h>
#include <petscdmplex.h>
#include <petscsys.h>
#include <private/rdyamrimpl.h>
#include <private/rdycoreimpl.h>
#include <private/rdydmimpl.h>
#include <private/rdylogimpl.h>
#include <private/rdymathimpl.h>
#include <private/rdysweimpl.h>

/// @brief Saves information regarding refinement of owned cells.
/// @param rdy         A RDy struct
/// @param size        The number of owned cells
/// @param refine_cell True/False array indicating if the cell should be refined
/// @return PETSC_SUCESS on success
PetscErrorCode RDyMarkOwnedCellsForAMR(RDy rdy, const PetscInt size, const PetscBool *refine_cell) {
  PetscFunctionBegin;

  rdy->amr.cells_marked_for_refinement = PETSC_TRUE;

  // PetscCheck, not PetscAssert: this is a caller's array length, and
  // PetscAssert compiles out of an optimized build, which is what RDycore runs
  PetscCheck(rdy->mesh.num_owned_cells == size, rdy->comm, PETSC_ERR_ARG_SIZ,
             "refine_cell has %" PetscInt_FMT " entries but the domain owns %" PetscInt_FMT " cells", size, rdy->mesh.num_owned_cells);

  // a second call before the next adapt would otherwise leak the first array
  PetscCall(PetscFree(rdy->amr.refine_cell));

  // copy the data
  PetscCall(PetscCalloc1(size, &rdy->amr.refine_cell));
  for (PetscInt icell = 0; icell < size; ++icell) {
    rdy->amr.refine_cell[icell] = refine_cell[icell];
  }

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
/// @param odm          The coarse DM that was refined; it cannot be recovered
///                     from the transform, whose base DM DMPlexSetTransform() clears
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

/// @brief Refines `dm` wherever `adaptLabel` marks a cell, and builds the four
///        interpolation matrices between it and the result. The label is
///        consumed (destroyed) here.
/// @param [in] dm         the mesh to refine
/// @param [in] adaptLabel cells marked DM_ADAPT_REFINE
/// @param [in] bs         components per cell for the NDof matrices
/// @param [out] dm_fine   the refined mesh
PetscErrorCode RDyAMRRefineByLabel(DM dm, DMLabel adaptLabel, const PetscInt bs, DM *dm_fine, Mat *CoarseToFineMatNDof, Mat *FineToCoarseMatNDof,
                                   Mat *CoarseToFineMat1Dof, Mat *FineToCoarseMat1Dof) {
  PetscFunctionBeginUser;

  DM dmAdapt;

  PetscCall(DMViewFromOptions(dm, NULL, "-adapt_pre_dm_view"));

  // This is PETSc's DMPlexTransformAdaptLabel() with the two settings RDycore's
  // AMR cannot do without stated in code rather than left to the options
  // database: simplex bisection, because RDyEdges holds exactly two cells per
  // edge and so cannot carry a hanging node, and labels that follow strata,
  // without which the refined mesh's boundary edges come back as the wrong mesh
  // points. DMPlexTransformSetFromOptions() leaves a type that is already set
  // alone, so -dm_plex_transform_* still overrides both.
  DMPlexTransform tr;
  PetscCall(DMPlexTransformCreate(PetscObjectComm((PetscObject)dm), &tr));
  PetscCall(PetscObjectSetName((PetscObject)tr, "Adapt Label Transform"));
  PetscCall(DMPlexTransformSetType(tr, DMPLEXREFINESBR));
  PetscCall(DMPlexTransformSetMatchStrata(tr, PETSC_TRUE));
  PetscCall(DMPlexTransformSetDM(tr, dm));
  PetscCall(DMPlexTransformSetFromOptions(tr));  // -dm_plex_transform_* still override
  PetscCall(DMPlexTransformSetActive(tr, adaptLabel));
  PetscCall(DMPlexTransformSetUp(tr));
  PetscCall(DMPlexTransformApply(tr, dm, &dmAdapt));
  PetscCall(DMCopyDisc(dm, dmAdapt));
  {
    DM cdm, rcdm;
    PetscCall(DMGetCoordinateDM(dm, &cdm));
    PetscCall(DMGetCoordinateDM(dmAdapt, &rcdm));
    PetscCall(DMCopyDisc(cdm, rcdm));
  }
  PetscCall(DMPlexTransformCreateDiscLabels(tr, dmAdapt));
  PetscCall(DMCopyDisc(dm, dmAdapt));
  PetscCall(DMPlexSetTransform(dmAdapt, tr));  // ConstructRefineTree reads it back
  PetscCall(DMPlexTransformDestroy(&tr));
  PetscCall(DMLabelDestroy(&adaptLabel));

  PetscCheck(dmAdapt, PETSC_COMM_WORLD, PETSC_ERR_USER, "Refinement failed.");

  PetscCall(PetscObjectSetName((PetscObject)dmAdapt, "Adapted Mesh"));
  PetscCall(DMViewFromOptions(dmAdapt, NULL, "-adapt_dm_view"));

  PetscCall(RDyAMRBuildInterpolationMatrices(dm, dmAdapt, bs, CoarseToFineMatNDof, FineToCoarseMatNDof, CoarseToFineMat1Dof, FineToCoarseMat1Dof));
  *dm_fine = dmAdapt;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Builds the four interpolation matrices between a mesh and the mesh
///        refined from it, from the transform the refinement saved.
PetscErrorCode RDyAMRBuildInterpolationMatrices(DM dm, DM dmAdapt, const PetscInt bs, Mat *CoarseToFineMatNDof, Mat *FineToCoarseMatNDof,
                                                Mat *CoarseToFineMat1Dof, Mat *FineToCoarseMat1Dof) {
  PetscFunctionBeginUser;
  const PetscInt d_nz = 4;  // one level of simplex bisection splits a cell into at most four
  PetscInt       ccStart, ccEnd, fcStart, fcEnd;

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
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMatMultReplace(Mat *, Mat *, PetscBool);

/// @brief Replaces the depth in a local solution vector with the free surface
///        h + z, so that an interpolation between two meshes carries a quantity
///        the bed does not enter. Component 0 of each cell block is the depth.
static PetscErrorCode DepthToSurface(RDyMesh *mesh, const PetscInt ndof, Vec u_local) {
  PetscFunctionBeginUser;
  PetscScalar *u;
  PetscCall(VecGetArray(u_local, &u));
  for (PetscInt c = 0; c < mesh->num_cells; ++c) u[ndof * c] += mesh->cells.centroids[c].X[2];
  PetscCall(VecRestoreArray(u_local, &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief The inverse of DepthToSurface() on the mesh the solution has landed
///        on. The clamp only bites where the transferred surface falls below
///        the new bed, which is a dry cell; there it adds water, so a transfer
///        that crosses a shoreline is not conservative.
static PetscErrorCode SurfaceToDepth(RDyMesh *mesh, const PetscInt ndof, Vec u_local) {
  PetscFunctionBeginUser;
  PetscScalar *u;
  PetscCall(VecGetArray(u_local, &u));
  for (PetscInt c = 0; c < mesh->num_cells; ++c) {
    u[ndof * c] = PetscMax(PetscRealPart(u[ndof * c]) - mesh->cells.centroids[c].X[2], 0.0);
  }
  PetscCall(VecRestoreArray(u_local, &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Puts the bed elevations the configuration names back on an adapted
///        mesh.
///
///        OverrideCellElevation() cannot be used after an adapt: it reads a
///        file holding one value per base cell into a vector sized for the mesh
///        in hand, and the two stopped matching at the first refinement. The
///        bed belongs to the base mesh, so the values are injected along the
///        base-to-current interpolation instead, which gives every descendant
///        of a base cell its elevation.
static PetscErrorCode OverrideCellElevationFromBase(RDy rdy) {
  PetscFunctionBeginUser;

  Vec global_base, local_base, local_current;
  PetscCall(RDyReadAMRScalarGlobalVecLevel0FromBinary(rdy, rdy->config.grid.cell_elevation.file, &global_base));
  PetscCall(DMCreateLocalVector(rdy->amr.dm_1dof_base, &local_base));
  PetscCall(DMGlobalToLocal(rdy->amr.dm_1dof_base, global_base, INSERT_VALUES, local_base));
  PetscCall(DMCreateLocalVector(rdy->dm_1dof, &local_current));
  PetscCall(MatMult(rdy->amr.BaseToCurrentMat1Dof, local_base, local_current));

  const PetscScalar *z;
  PetscCall(VecGetArrayRead(local_current, &z));
  RDyCells *cells = &rdy->mesh.cells;
  for (PetscInt c = 0; c < rdy->mesh.num_cells; ++c) cells->centroids[c].X[2] = PetscRealPart(z[c]);
  PetscCall(VecRestoreArrayRead(local_current, &z));

  PetscCall(VecDestroy(&global_base));
  PetscCall(VecDestroy(&local_base));
  PetscCall(VecDestroy(&local_current));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief The base cell a local cell of the mesh in hand descends from, or -1
///        where the interpolation holds no parent for it. Before the first
///        adapt the mesh IS the base mesh, so a cell is its own base cell.
///
///        Every descendant of a base cell has exactly one nonzero in its row of
///        BaseToCurrentMat1Dof, which is why column 0 is the answer.
PetscErrorCode RDyAMRBaseCellOfLocalCell(RDy rdy, PetscInt c, PetscInt *base) {
  PetscFunctionBeginUser;
  if (!rdy->amr.BaseToCurrentMat1Dof) {
    *base = c;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscInt           ncols;
  const PetscInt    *cols;
  const PetscScalar *vals;
  PetscCall(MatGetRow(rdy->amr.BaseToCurrentMat1Dof, c, &ncols, &cols, &vals));
  *base = ncols ? cols[0] : -1;
  PetscCall(MatRestoreRow(rdy->amr.BaseToCurrentMat1Dof, c, &ncols, &cols, &vals));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Updates each base cell's target refinement level from this adapt's
///        marks: a base cell with a marked descendant goes one level deeper,
///        one without comes one level back. One level per adapt in either
///        direction keeps the mesh from thrashing between two states.
static PetscErrorCode UpdateBaseLevels(RDy rdy, PetscInt max_levels) {
  PetscFunctionBeginUser;

  RDyMesh *mesh  = &rdy->mesh;
  PetscInt nbase = rdy->amr.num_base_cells;

  PetscBool *base_marked;
  PetscCall(PetscCalloc1(nbase, &base_marked));

  for (PetscInt c = 0; c < mesh->num_cells; ++c) {
    if (!mesh->cells.is_owned[c]) continue;
    PetscInt owned = mesh->cells.local_to_owned[c];
    if (!rdy->amr.refine_cell[owned]) continue;

    PetscInt base;
    PetscCall(RDyAMRBaseCellOfLocalCell(rdy, c, &base));
    if (base >= 0) base_marked[base] = PETSC_TRUE;
  }

  for (PetscInt b = 0; b < nbase; ++b) {
    if (base_marked[b]) {
      rdy->amr.base_level[b] = PetscMin(rdy->amr.base_level[b] + 1, max_levels);
    } else {
      rdy->amr.base_level[b] = PetscMax(rdy->amr.base_level[b] - 1, 0);
    }
  }

  PetscCall(PetscFree(base_marked));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Builds a mesh from the base mesh by refining each base cell to its
///        target level, level by level, and accumulates the interpolation
///        matrices between the base mesh and the result.
///
///        Refining from the base rather than from the current mesh is what
///        makes coarsening possible: a region the indicator has stopped
///        marking is simply built with fewer levels this time.
PetscErrorCode RDyAMRRebuildMeshFromBase(RDy rdy, const PetscInt bs, PetscInt max_levels, DM *dm_new, Mat *BaseToNewNDof, Mat *NewToBaseNDof,
                                         Mat *BaseToNew1Dof, Mat *NewToBase1Dof) {
  PetscFunctionBeginUser;

  DM  dm_base = rdy->amr.dm_base ? rdy->amr.dm_base : rdy->dm;
  DM  dm      = dm_base;
  Mat b2n_n = NULL, n2b_n = NULL, b2n_1 = NULL, n2b_1 = NULL;

  for (PetscInt level = 0; level < max_levels; ++level) {
    PetscInt cStart, cEnd;
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));

    DMLabel label;
    PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Adaptation Label", &label));
    PetscInt marked = 0;
    for (PetscInt c = cStart; c < cEnd; ++c) {
      PetscInt base = c - cStart;  // at level 0 the mesh IS the base mesh
      if (b2n_1) {
        PetscInt           ncols;
        const PetscInt    *cols;
        const PetscScalar *vals;
        PetscCall(MatGetRow(b2n_1, c - cStart, &ncols, &cols, &vals));
        base = ncols ? cols[0] : -1;
        PetscCall(MatRestoreRow(b2n_1, c - cStart, &ncols, &cols, &vals));
      }
      if (base >= 0 && rdy->amr.base_level[base] > level) {
        PetscCall(DMLabelSetValue(label, c, DM_ADAPT_REFINE));
        ++marked;
      }
    }

    PetscInt marked_global;
    PetscCallMPI(MPI_Allreduce(&marked, &marked_global, 1, MPIU_INT, MPI_SUM, rdy->comm));
    if (!marked_global) {
      PetscCall(DMLabelDestroy(&label));
      break;
    }

    DM  dm_next;
    Mat c2f_n, f2c_n, c2f_1, f2c_1;
    PetscCall(RDyAMRRefineByLabel(dm, label, bs, &dm_next, &c2f_n, &f2c_n, &c2f_1, &f2c_1));

    if (!b2n_n) {  // first level: the matrices already run from the base mesh
      b2n_n = c2f_n;
      n2b_n = f2c_n;
      b2n_1 = c2f_1;
      n2b_1 = f2c_1;
    } else {
      PetscCall(MatMatMultReplace(&c2f_n, &b2n_n, PETSC_TRUE));
      PetscCall(MatMatMultReplace(&n2b_n, &f2c_n, PETSC_FALSE));
      PetscCall(MatMatMultReplace(&c2f_1, &b2n_1, PETSC_TRUE));
      PetscCall(MatMatMultReplace(&n2b_1, &f2c_1, PETSC_FALSE));
      PetscCall(MatDestroy(&c2f_n));
      PetscCall(MatDestroy(&f2c_n));
      PetscCall(MatDestroy(&c2f_1));
      PetscCall(MatDestroy(&f2c_1));
    }

    if (dm != dm_base) PetscCall(DMDestroy(&dm));
    dm = dm_next;
  }

  if (!b2n_n) {  // nothing is refined this time: the new mesh is the base mesh
    PetscInt cStart, cEnd;
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
    PetscInt n = cEnd - cStart;
    PetscCall(MatCreateConstantDiagonal(PETSC_COMM_SELF, bs * n, bs * n, bs * n, bs * n, 1.0, &b2n_n));
    PetscCall(MatConvert(b2n_n, MATSEQAIJ, MAT_INPLACE_MATRIX, &b2n_n));
    PetscCall(MatDuplicate(b2n_n, MAT_COPY_VALUES, &n2b_n));
    PetscCall(MatCreateConstantDiagonal(PETSC_COMM_SELF, n, n, n, n, 1.0, &b2n_1));
    PetscCall(MatConvert(b2n_1, MATSEQAIJ, MAT_INPLACE_MATRIX, &b2n_1));
    PetscCall(MatDuplicate(b2n_1, MAT_COPY_VALUES, &n2b_1));
    PetscCall(PetscObjectReference((PetscObject)dm));  // the caller takes a reference
  }

  *dm_new        = dm;
  *BaseToNewNDof = b2n_n;
  *NewToBaseNDof = n2b_n;
  *BaseToNew1Dof = b2n_1;
  *NewToBase1Dof = n2b_1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Refines the current mesh where this adapt's marks ask for it.
static PetscErrorCode AdaptMesh(RDy rdy, const PetscInt bs, DM *dm_fine, Mat *CoarseToFineMatNDof, Mat *FineToCoarseMatNDof, Mat *CoarseToFineMat1Dof,
                                Mat *FineToCoarseMat1Dof) {
  PetscFunctionBeginUser;

  PetscCheck(rdy->amr.cells_marked_for_refinement, rdy->comm, PETSC_ERR_ARG_WRONGSTATE,
             "no cells are marked: call RDyMarkOwnedCellsForAMR() before every RDyPerformAMR()");

  DMLabel adaptLabel;
  PetscCall(CreateAdaptLabelFromMarkedCells(rdy, &adaptLabel));
  rdy->amr.cells_marked_for_refinement = PETSC_FALSE;
  PetscCall(PetscFree(rdy->amr.refine_cell));

  PetscCall(
      RDyAMRRefineByLabel(rdy->dm, adaptLabel, bs, dm_fine, CoarseToFineMatNDof, FineToCoarseMatNDof, CoarseToFineMat1Dof, FineToCoarseMat1Dof));
  PetscFunctionReturn(PETSC_SUCCESS);
}

extern PetscErrorCode InitOperator(RDy rdy);
extern PetscErrorCode InitMaterialProperties(RDy rdy);
extern PetscErrorCode InitSolver(RDy rdy);
extern PetscErrorCode InitDirichletBoundaryConditions(RDy rdy);
extern PetscErrorCode InitSourceConditions(RDy rdy);

/// @brief Computes A * B and replaces one of the two with it, destroying what
///        was there.
/// @param A        left factor
/// @param B        right factor
/// @param update_B replace B with the product, rather than A
/// @return PETSC_SUCESS on success
static PetscErrorCode MatMatMultReplace(Mat *A, Mat *B, PetscBool update_B) {
  PetscFunctionBeginUser;
  Mat tmp;

  PetscCall(MatMatMult(*A, *B, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &tmp));
  PetscCall(MatDestroy(update_B ? B : A));
  if (update_B) *B = tmp;
  else *A = tmp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyEnableAMR(RDy rdy) {
  PetscFunctionBeginUser;

  PetscInt nstep;
  PetscCall(RDyGetStep(rdy, &nstep));
  PetscCheck(nstep == 0, rdy->comm, PETSC_ERR_ARG_WRONGSTATE, "AMR must be enabled before the first step; the model has already taken %" PetscInt_FMT,
             nstep);

  rdy->amr.is_refinement_on                = PETSC_TRUE;
  rdy->amr.last_refinement_level_outputted = -1;

  // with coarsening, each adapt rebuilds the mesh from this one, so the level
  // every base cell is refined to has to be remembered between adapts
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-refine_allow_coarsening", &rdy->amr.allow_coarsening, NULL));

  // Balancing is on by default where it can act, which is the rebuild: a
  // refinement of the mesh in hand never touches the base mesh, so there is
  // nothing to repartition. -rdy_amr_balance 0 turns it off, which is what
  // the gate that compares a balanced run against an unbalanced one uses.
  PetscBool balance_set;
  rdy->amr.balance = PETSC_TRUE;
  PetscCall(PetscOptionsGetBool(NULL, "rdy_amr_", "-balance", &rdy->amr.balance, &balance_set));
  if (rdy->amr.balance && !rdy->amr.allow_coarsening) {
    // say so rather than quietly doing nothing: balancing rides on the rebuild
    if (balance_set) {
      RDyLogWarning(rdy, "-rdy_amr_balance has no effect without -refine_allow_coarsening: only a rebuild from the base mesh can repartition");
    }
    rdy->amr.balance = PETSC_FALSE;
  }
  rdy->amr.max_levels = 3;
  PetscCall(PetscOptionsGetInt(NULL, "rdy_amr_", "-max_levels", &rdy->amr.max_levels, NULL));
  rdy->amr.num_base_cells = rdy->mesh.num_cells;
  PetscCall(PetscCalloc1(rdy->amr.num_base_cells, &rdy->amr.base_level));

  // the depth a cell may be refined to is measured against its own base cell,
  // so the base areas are kept for as long as base_level is
  PetscCall(PetscCalloc1(rdy->amr.num_base_cells, &rdy->amr.base_cell_area));
  for (PetscInt c = 0; c < rdy->mesh.num_cells; ++c) rdy->amr.base_cell_area[c] = rdy->mesh.cells.areas[c];

  // The mesh in hand is the base mesh, and it is held from here rather than
  // from the end of the first adapt: a rebuild that rebalances replaces the
  // base mesh, and the first adapt would then overwrite it with this one.
  rdy->amr.dm_base      = rdy->dm;
  rdy->amr.dm_1dof_base = rdy->dm_1dof;
  PetscCall(PetscObjectReference((PetscObject)rdy->amr.dm_base));
  PetscCall(PetscObjectReference((PetscObject)rdy->amr.dm_1dof_base));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief The deepest refinement any base cell may reach, as
///        `-rdy_amr_max_levels` set it.
PetscErrorCode RDyGetAMRMaxLevels(RDy rdy, PetscInt *max_levels) {
  PetscFunctionBeginUser;
  PetscCheck(rdy->amr.is_refinement_on, rdy->comm, PETSC_ERR_ARG_WRONGSTATE, "AMR is not enabled, so there is no refinement limit to report");
  *max_levels = rdy->amr.max_levels;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief The area of the base cell each owned cell descends from, in the same
///        order as RDyGetOwnedCellAreas(). Dividing the two gives 2^level, so a
///        caller can check how deep the mesh has been refined.
PetscErrorCode RDyGetOwnedCellBaseAreas(RDy rdy, const PetscInt size, PetscReal *values) {
  PetscFunctionBeginUser;
  PetscCheck(rdy->amr.base_cell_area, rdy->comm, PETSC_ERR_ARG_WRONGSTATE, "AMR is not enabled, so no base mesh was recorded");
  PetscCheck(size == rdy->mesh.num_owned_cells, rdy->comm, PETSC_ERR_ARG_SIZ,
             "Wrong size (%" PetscInt_FMT ") for owned cells in domain (%" PetscInt_FMT ")", size, rdy->mesh.num_owned_cells);

  PetscInt count = 0;
  for (PetscInt c = 0; c < rdy->mesh.num_cells; ++c) {
    if (!rdy->mesh.cells.is_owned[c]) continue;
    PetscInt base;
    PetscCall(RDyAMRBaseCellOfLocalCell(rdy, c, &base));
    values[count++] = (base >= 0) ? rdy->amr.base_cell_area[base] : rdy->mesh.cells.areas[c];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Turns on AMR driven by RDycore's built-in refinement criterion: at
///        every adapt the water-surface-jump indicator is computed, thresholded
///        with a refine and a coarsen VecTagger, and the mesh adapted from the
///        marks. Call instead of RDyEnableAMR when the caller does not mark
///        cells itself.
/// @param [in] rdy the dycore
/// @return PETSC_SUCCESS on success
PetscErrorCode RDyEnableAMRBySurfaceJump(RDy rdy) {
  PetscFunctionBeginUser;
  PetscCall(RDyEnableAMR(rdy));
  rdy->amr.use_surface_jump = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyPerformAMR(RDy rdy) {
  Mat      CoarseToFineMatNDof, FineToCoarseMatNDof;
  Mat      CoarseToFineMat1Dof, FineToCoarseMat1Dof;
  DM       dm_fine;
  Vec      U_coarse_local = NULL;
  PetscInt ndof_coarse;
  PetscFunctionBeginUser;
  PetscCall(VecGetBlockSize(rdy->u_global, &ndof_coarse));

  // With coarsening the new mesh is built from the base mesh rather than
  // refined from the current one, so that regions the indicator has stopped
  // marking come back coarse. The solution then moves between two meshes that
  // are different triangulations of the same domain, which needs a
  // conservative transfer; refinement alone can inject.
  PetscBool rebuilt          = rdy->amr.allow_coarsening;
  Mat       SolutionTransfer = NULL;

  // the old mesh's cells, grouped by base cell, for the conservative transfer
  PetscSection old_sec    = NULL;
  PetscScalar *old_packed = NULL;

  // a rebuild has to know, before it adapts, what the mesh it is about to
  // replace maps to in the base mesh
  Mat base_to_old_1dof = rdy->amr.BaseToCurrentMat1Dof;
  if (base_to_old_1dof) PetscCall(PetscObjectReference((PetscObject)base_to_old_1dof));

  // The cells to adapt, as a label on the mesh in hand. Nothing is changed
  // yet: the levels the rebuild will use come next, and a rebuild that
  // rebalances has to know them before it moves the base mesh.
  DMLabel label = NULL;
  if (rdy->amr.use_surface_jump) {
    PetscCall(RDyAMRMarkBySurfaceJump(rdy, &label));

    PetscInt num_marked;
    PetscCall(DMLabelGetNumValues(label, &num_marked));
    if (!num_marked) {  // nothing was tagged anywhere: leave the mesh alone
      PetscCall(DMLabelDestroy(&label));
      PetscCall(MatDestroy(&base_to_old_1dof));
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  } else if (rebuilt) {
    PetscCheck(rdy->amr.cells_marked_for_refinement, rdy->comm, PETSC_ERR_USER,
               "-refine_allow_coarsening needs solution-driven marking (RDyMarkOwnedCellsForAMR) before every adapt");
  }

  if (rebuilt) {
    // Copy the old solution and pack it, by base cell, before anything moves:
    // the pack is what crosses to another rank when the base mesh is
    // repartitioned, and it has to be taken on the partition it is still on.
    PetscCall(VecDuplicate(rdy->u_local, &U_coarse_local));
    PetscCall(DMGlobalToLocal(rdy->dm, rdy->u_global, INSERT_VALUES, U_coarse_local));

    // The transfer carries the free surface, not the depth: a child cell's bed
    // elevation is not its parent's, so injecting h tilts a lake at rest. Mass
    // is still conserved exactly, because a triangle's bed is planar and
    // refinement splits it at edge midpoints, so sum(z * area) is the same on
    // both meshes.
    PetscCall(DepthToSurface(&rdy->mesh, ndof_coarse, U_coarse_local));

    DM dm_base_old = rdy->amr.dm_base ? rdy->amr.dm_base : rdy->dm;
    PetscCall(RDyAMRPackCellsByBaseCell(&rdy->mesh, dm_base_old, base_to_old_1dof, ndof_coarse, U_coarse_local, &old_sec, &old_packed));

    // the balance weights a base cell by the cells it is about to carry, so
    // it needs the level each one is coming from as well as going to
    PetscInt *old_level;
    PetscCall(PetscMalloc1(rdy->amr.num_base_cells, &old_level));
    PetscCall(PetscArraycpy(old_level, rdy->amr.base_level, rdy->amr.num_base_cells));

    // move each base cell to the level this adapt asks for
    if (label) {
      PetscCall(RDyAMRUpdateBaseLevelsFromLabel(rdy, label));
    } else {
      PetscCall(UpdateBaseLevels(rdy, rdy->amr.max_levels));
      rdy->amr.cells_marked_for_refinement = PETSC_FALSE;
      PetscCall(PetscFree(rdy->amr.refine_cell));
    }

    // Spread the work evenly before rebuilding: the partition follows the base
    // mesh, weighted by the cells each base cell is about to carry, and the
    // packed solution rides along with it.
    PetscCall(RDyAMRBalanceBaseMesh(rdy, dm_base_old, old_level, ndof_coarse, &old_sec, &old_packed));
    PetscCall(PetscFree(old_level));

    PetscCall(RDyAMRRebuildMeshFromBase(rdy, ndof_coarse, rdy->amr.max_levels, &dm_fine, &CoarseToFineMatNDof, &FineToCoarseMatNDof,
                                        &CoarseToFineMat1Dof, &FineToCoarseMat1Dof));
  } else if (label) {
    PetscCall(RDyAMRRefineFromLabel(rdy, label, &dm_fine, &CoarseToFineMatNDof, &FineToCoarseMatNDof, &CoarseToFineMat1Dof, &FineToCoarseMat1Dof));
  } else {
    PetscCall(AdaptMesh(rdy, ndof_coarse, &dm_fine, &CoarseToFineMatNDof, &FineToCoarseMatNDof, &CoarseToFineMat1Dof, &FineToCoarseMat1Dof));
  }
  PetscCall(DMLabelDestroy(&label));

  // a refinement of the mesh in hand: each new cell takes its parent's value
  if (!rebuilt) SolutionTransfer = CoarseToFineMatNDof;
  PetscCall(MatDestroy(&base_to_old_1dof));
  PetscCall(DMLocalizeCoordinates(dm_fine));
  PetscCall(DMViewFromOptions(dm_fine, NULL, "-dm_fine_view"));
  // Do not make the old mesh this one's coarse DM. A rebuild that refines
  // nothing returns the base mesh itself, which is already at the tail of that
  // chain, and SNESDestroy() walks it with no cycle guard.
  PetscCall(DMGetCoordinatesLocalSetUp(dm_fine));

  {
    PetscSection coarse_sec, fine_sec;
    PetscCall(DMGetLocalSection(rdy->dm, &coarse_sec));
    PetscCall(DMGetLocalSection(dm_fine, &fine_sec));
    PetscInt nfields;
    PetscCall(PetscSectionGetNumFields(coarse_sec, &nfields));
    for (PetscInt f = 0; f < nfields; f++) {
      PetscInt ncomp;
      PetscCall(PetscSectionGetFieldComponents(coarse_sec, f, &ncomp));
      for (PetscInt c = 0; c < ncomp; c++) {
        const char *comp_name;
        PetscCall(PetscSectionGetComponentName(coarse_sec, f, c, &comp_name));
        PetscCall(PetscSectionSetComponentName(fine_sec, f, c, comp_name));
      }
    }
  }

  // a refinement injects straight from the old local solution, which a rebuild
  // took and packed before the base mesh moved
  if (!rebuilt) {
    PetscCall(VecDuplicate(rdy->u_local, &U_coarse_local));
    PetscCall(DMGlobalToLocal(rdy->dm, rdy->u_global, INSERT_VALUES, U_coarse_local));

    // The transfer carries the free surface, not the depth: a child cell's bed
    // elevation is not its parent's, so injecting h tilts a lake at rest. Mass
    // is still conserved exactly, because a triangle's bed is planar and
    // refinement splits it at edge midpoints, so sum(z * area) is the same on
    // both meshes.
    PetscCall(DepthToSurface(&rdy->mesh, ndof_coarse, U_coarse_local));
  }

  // destroy the coarse vectors
  PetscCall(RDyDestroyVectors(&rdy));

  // the base DMs are held from RDyEnableAMR, so these are the last references
  PetscCall(DMDestroy(&rdy->dm));
  PetscCall(DMDestroy(&rdy->dm_1dof));

  if (rebuilt || !rdy->amr.num_refinements) {
    // the matrices run from the base mesh already -- replace rather than
    // compose. On the first adapt there is nothing yet to replace.
    PetscCall(MatDestroy(&rdy->amr.BaseToCurrentMatNDof));
    PetscCall(MatDestroy(&rdy->amr.CurrentToBaseMatNDof));
    PetscCall(MatDestroy(&rdy->amr.BaseToCurrentMat1Dof));
    PetscCall(MatDestroy(&rdy->amr.CurrentToBaseMat1Dof));
    PetscCall(MatDuplicate(CoarseToFineMatNDof, MAT_COPY_VALUES, &rdy->amr.BaseToCurrentMatNDof));
    PetscCall(MatDuplicate(FineToCoarseMatNDof, MAT_COPY_VALUES, &rdy->amr.CurrentToBaseMatNDof));
    PetscCall(MatDuplicate(CoarseToFineMat1Dof, MAT_COPY_VALUES, &rdy->amr.BaseToCurrentMat1Dof));
    PetscCall(MatDuplicate(FineToCoarseMat1Dof, MAT_COPY_VALUES, &rdy->amr.CurrentToBaseMat1Dof));
  } else {
    PetscCall(MatMatMultReplace(&CoarseToFineMatNDof, &rdy->amr.BaseToCurrentMatNDof, PETSC_TRUE));
    PetscCall(MatMatMultReplace(&rdy->amr.CurrentToBaseMatNDof, &FineToCoarseMatNDof, PETSC_FALSE));
    PetscCall(MatMatMultReplace(&CoarseToFineMat1Dof, &rdy->amr.BaseToCurrentMat1Dof, PETSC_TRUE));
    PetscCall(MatMatMultReplace(&rdy->amr.CurrentToBaseMat1Dof, &FineToCoarseMat1Dof, PETSC_FALSE));
  }

  // set the DM to be the refined DM
  rdy->dm      = dm_fine;
  rdy->flow_dm = rdy->dm;

  // create the auxiliary DM
  PetscCall(CreateAuxiliaryDMs(rdy));

  // create new refined vectors
  PetscCall(CreateVectors(rdy));

  // mark the mesh was refined
  rdy->amr.mesh_was_refined = PETSC_TRUE;
  rdy->amr.num_refinements++;

  // build the new mesh beside the old one: a rebuilt mesh is a different
  // triangulation of the same domain, and the conservative transfer between
  // them needs the geometry of both
  RDyMesh new_mesh;
  PetscCall(RDyMeshCreateFromDM(rdy->dm, rdy->amr.num_refinements, &new_mesh));
  PetscCall(RDyMeshDestroy(rdy->mesh));
  rdy->mesh = new_mesh;
  if (rdy->config.physics.flow.well_balancing == WELL_BALANCING_HR) {
    PetscCall(RDyMeshOverride2DProjection(&rdy->mesh));
  }
  if (rdy->config.grid.cell_elevation.file[0]) {
    PetscCall(OverrideCellElevationFromBase(rdy));
  }

  // the adapted mesh carries the "Cell Sets" label the regions are defined by,
  // so the regions are read off it exactly as they were at setup. Tracking
  // each new cell back to a cell of the old mesh would give the same answer
  // only while the two meshes are on the same partition.
  for (PetscInt i = 0; i < rdy->num_regions; ++i) PetscCall(DestroyRegion(&rdy->regions[i]));
  if (rdy->regions) PetscCall(PetscFree(rdy->regions));
  rdy->regions     = NULL;
  rdy->num_regions = 0;
  PetscCall(InitRegions(rdy));

  // initialize the solution on the new mesh from the old one: injection when
  // the mesh was refined from the current one, area-weighted averaging of the
  // packed old cells when it was rebuilt and some of it coarsened. Either way
  // the free surface crosses and the depth is rebuilt against the new mesh's bed.
  if (rebuilt) {
    PetscCall(RDyAMRTransferFromPacked(&rdy->mesh, rdy->amr.dm_base, rdy->amr.BaseToCurrentMat1Dof, ndof_coarse, old_sec, old_packed, rdy->u_local));
    PetscCall(PetscSectionDestroy(&old_sec));
    PetscCall(PetscFree(old_packed));
  } else {
    PetscCall(MatMult(SolutionTransfer, U_coarse_local, rdy->u_local));
  }
  PetscCall(SurfaceToDepth(&rdy->mesh, ndof_coarse, rdy->u_local));
  PetscCall(DMLocalToGlobal(rdy->dm, rdy->u_local, INSERT_VALUES, rdy->u_global));
  PetscCall(MatDestroy(&CoarseToFineMatNDof));
  PetscCall(MatDestroy(&FineToCoarseMatNDof));
  PetscCall(MatDestroy(&CoarseToFineMat1Dof));
  PetscCall(MatDestroy(&FineToCoarseMat1Dof));
  PetscCall(VecDestroy(&U_coarse_local));

  // destroy the operator
  PetscCall(DestroyOperator(&rdy->operator));

  // destroy the boundaries and reallocate memory. The conditions are indexed
  // by boundary, so they are rebuilt with them: which boundaries a rank holds
  // changes when the mesh is repartitioned.
  PetscCall(RDyDestroyBoundaries(&rdy));
  PetscCall(InitBoundaries(rdy));
  PetscCall(InitBoundaryConditions(rdy));

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
