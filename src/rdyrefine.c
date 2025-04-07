#include <muParserDLL.h>
#include <petsc/private/dmpleximpl.h> /*I      "petscdmplex.h"   I*/
#include <petscdmceed.h>
#include <petscdmplex.h>
#include <petscsys.h>
#include <private/rdycoreimpl.h>
#include <private/rdydmimpl.h>
#include <private/rdymathimpl.h>
#include <private/rdysweimpl.h>
typedef struct {
  PetscInt  adapt;  /* Flag for adaptation of the surface mesh */
  PetscBool metric; /* Flag to use metric adaptation, instead of tagging */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options) {
  PetscMPIInt size;

  PetscFunctionBeginUser;
  options->adapt  = 1;
  options->metric = PETSC_FALSE;
  PetscCallMPI(MPI_Comm_size(comm, &size));

  PetscOptionsBegin(comm, "", "Meshing Interpolation Test Options", "DMPLEX");
  PetscCall(PetscOptionsInt("-adapt", "Number of adaptation steps mesh", "ex10.c", options->adapt, &options->adapt, NULL));
  PetscCall(PetscOptionsBool("-metric", "Flag for metric refinement", "ex41.c", options->metric, &options->metric, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateAdaptLabel(DM dm, AppCtx *ctx, DMLabel *adaptLabel) {
  /* PetscMPIInt rank; */
  DMLabel  label;
  PetscInt cStart, cEnd, c;

  PetscFunctionBegin;
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Adaptation Label", adaptLabel));
  label = *adaptLabel;
  PetscCall(DMGetCoordinatesLocalSetUp(dm));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  for (c = cStart; c < cEnd; ++c) {
    PetscReal centroid[3], volume, x, y;

    PetscCall(DMPlexComputeCellGeometryFVM(dm, c, &volume, centroid, NULL));
    x = centroid[0];
    y = centroid[1];
    /* Headwaters are (0.0,0.25)--(0.1,0.75) */
    if ((x >= 0.0 && x < 1.) && (y >= 3. && y <= 4.)) {
      PetscCall(DMLabelSetValue(label, c, DM_ADAPT_REFINE));
      // PetscCall(PetscPrintf(PETSC_COMM_SELF, "refine: %" PetscInt_FMT "\n", c));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Creates matrices for the interpolation between coarse and fine meshes.
/// @param dm           The fine DM
/// @param CoarseToFine Matrix for interpolating local Vec from coarse to fine grid
/// @param FineToCoarse Matrix for interpolating local Vec from fine to coarse grid
/// @return PETSC_SUCESS on success
static PetscErrorCode ConstructRefineTree(DM dm, Mat CoarseToFine, Mat FineToCoarse) {
  DMPlexTransform tr;
  DM              odm;
  PetscInt        cStart, cEnd, bs, Istart, Jstart;
  PetscScalar     val = 1.0;

  PetscFunctionBegin;
  PetscCall(MatGetBlockSize(CoarseToFine, &bs));
  PetscCall(DMPlexGetTransform(dm, &tr));
  if (!tr) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMPlexTransformGetDM(tr, &odm));
  PetscCall(DMPlexGetHeightStratum(odm, 0, &cStart, &cEnd));
  PetscCall(MatGetOwnershipRange(CoarseToFine, &Istart, NULL));
  PetscCall(MatGetOwnershipRangeColumn(CoarseToFine, &Jstart, NULL));
  PetscMPIInt myrank, commsize;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &myrank));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm), &commsize));

  for (PetscInt i = 0; i < commsize; i++) {
    if (i == myrank) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n"));
      for (PetscInt c = cStart; c < cEnd; ++c) {
        DMPolytopeType  ct;
        DMPolytopeType *rct;
        PetscInt       *rsize, *rcone, *rornt;
        PetscInt        Nct, dim, pNew = 0;

        PetscCall(PetscPrintf(PETSC_COMM_SELF, "Rank %d; Istart = %d Cell %" PetscInt_FMT " produced new cells", myrank, Istart, c));
        PetscCall(DMPlexGetCellType(odm, c, &ct));
        dim = DMPolytopeTypeGetDim(ct);
        PetscCall(DMPlexTransformCellTransform(tr, ct, c, NULL, &Nct, &rct, &rsize, &rcone, &rornt));
        for (PetscInt n = 0; n < Nct; ++n) {
          if (DMPolytopeTypeGetDim(rct[n]) != dim) continue;
          for (PetscInt r = 0; r < rsize[n]; ++r) {
            PetscCall(DMPlexTransformGetTargetPoint(tr, ct, rct[n], c, r, &pNew));
            PetscCall(PetscPrintf(PETSC_COMM_SELF, " %" PetscInt_FMT, pNew));
            for (PetscInt i = 0; i < bs; i++) {
              //PetscCall(MatSetValue(CoarseToFine, Istart + bs * (pNew - 0) + i, Jstart + bs * (c - cStart) + i, val, INSERT_VALUES));
              //PetscCall(MatSetValue(FineToCoarse, Jstart + bs * (c - cStart) + i, Istart + bs * (pNew - 0) + i, 1.0 / rsize[n], INSERT_VALUES));
            }
          }
        }
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n"));
      }
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n"));
    }
    MPI_Barrier(PETSC_COMM_WORLD);
  }

  PetscCall(MatSetFromOptions(CoarseToFine));
  PetscCall(MatAssemblyBegin(CoarseToFine, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(CoarseToFine, MAT_FINAL_ASSEMBLY));
  PetscCall(MatViewFromOptions(CoarseToFine, NULL, "-adapt_c2f_mat_view"));

  PetscCall(MatSetFromOptions(FineToCoarse));
  PetscCall(MatAssemblyBegin(FineToCoarse, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(FineToCoarse, MAT_FINAL_ASSEMBLY));
  PetscCall(MatViewFromOptions(FineToCoarse, NULL, "-adapt_f2c_mat_view"));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AdaptMesh(DM dm, const PetscInt bs, DM *dm_fine, Mat *CoarseToFine, Mat *FineToCoarse, AppCtx *ctx) {
  PetscFunctionBeginUser;
  PetscCall(DMViewFromOptions(dm, NULL, "-adapt_pre_dm_view"));

  DM       dmAdapt;
  DMLabel  adaptLabel;
  PetscInt d_nz = 4, o_nz = 0, ccStart, ccEnd, fcStart, fcEnd;
  char     opt[128];

  PetscCall(CreateAdaptLabel(dm, ctx, &adaptLabel));

  PetscMPIInt myrank, commsize;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &myrank));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm), &commsize));
  if (!myrank) printf("Information about coarse DM\n");
  for (PetscInt i = 0; i < commsize; i++) {
    if (i == myrank) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "Rank %d: Cell_id, X, Y, Is_owned \n", myrank));
      PetscInt c_start, c_end;

      DMPlexGetHeightStratum(dm, 0, &c_start, &c_end);
      for (PetscInt c = c_start; c < c_end; ++c) {
        PetscReal area, centroid[3], normal[3];
        DMPlexComputeCellGeometryFVM(dm, c, &area, &centroid, &normal[0]);
        PetscInt gref, junkInt;
        PetscCall(DMPlexGetPointGlobal(dm, c, &gref, &junkInt));
        printf("%" PetscInt_FMT " %e %e %" PetscInt_FMT "\n", c, centroid[0], centroid[1], ((gref >= 0)));
      }
    }
    MPI_Barrier(PETSC_COMM_WORLD);
  }

  PetscCall(DMPlexSetSaveTransform(dm, PETSC_TRUE));
  PetscCall(DMAdaptLabel(dm, adaptLabel, &dmAdapt));  // DMRefine
  PetscCall(DMLabelDestroy(&adaptLabel));

  if (!myrank) printf("Information about REFINED DM\n");
  for (PetscInt i = 0; i < commsize; i++) {
    if (i == myrank) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "Rank %d: Cell_id, X, Y \n", myrank));
      PetscInt c_start, c_end;

      DMPlexGetHeightStratum(dmAdapt, 0, &c_start, &c_end);
      for (PetscInt c = c_start; c < c_end; ++c) {
        PetscReal area, centroid[3], normal[3];
        DMPlexComputeCellGeometryFVM(dmAdapt, c, &area, &centroid, &normal[0]);
        PetscInt gref, junkInt;
        //PetscCall(DMPlexGetPointGlobal(dmAdapt, c, &gref, &junkInt));
        printf("%" PetscInt_FMT " %e %e \n", c, centroid[0], centroid[1]);
      }
    }
    MPI_Barrier(PETSC_COMM_WORLD);
  }


  if (!dmAdapt) {
    PetscCheck(PETSC_TRUE, PETSC_COMM_WORLD, PETSC_ERR_USER, "Refinement failed.");
  }

  PetscCall(PetscObjectSetName((PetscObject)dmAdapt, "Adapted Mesh"));
  PetscCall(PetscSNPrintf(opt, 128, "-adapt_dm_view"));
  PetscCall(DMViewFromOptions(dmAdapt, NULL, opt));

  // make interpolation matrix
  PetscCall(DMPlexGetHeightStratum(dm, 0, &ccStart, &ccEnd));
  PetscCall(DMPlexGetHeightStratum(dmAdapt, 0, &fcStart, &fcEnd));
  //PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, bs * (fcEnd - fcStart), bs * (ccEnd - ccStart), d_nz, NULL, CoarseToFine));
  //PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, bs * (ccEnd - ccStart), bs * (fcEnd - fcStart), d_nz, NULL, FineToCoarse));
  PetscCall(MatCreateAIJ(PetscObjectComm((PetscObject)dm), bs * (fcEnd-fcStart), bs * (ccEnd-ccStart), PETSC_DETERMINE, PETSC_DETERMINE, d_nz, NULL, o_nz, NULL, CoarseToFine));
  PetscCall(MatCreateAIJ(PetscObjectComm((PetscObject)dm), bs * (ccEnd-ccStart), bs * (fcEnd-fcStart), PETSC_DETERMINE, PETSC_DETERMINE, d_nz, NULL, o_nz, NULL, FineToCoarse));
  PetscCall(MatSetBlockSize(*CoarseToFine, bs));
  PetscCall(MatSetBlockSize(*FineToCoarse, bs));
  PetscCall(ConstructRefineTree(dmAdapt, *CoarseToFine, *FineToCoarse));
  MPI_Barrier(PETSC_COMM_WORLD);
  exit(0);

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
static PetscErrorCode DetermineCoarseToFineCellMapping(RDy rdy_coarse, Mat CoarseToFine, Vec X_coarse, Vec X_fine, PetscInt *fineIsOwned,
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
  PetscCall(MatMult(CoarseToFine, X_coarse, X_fine));
  PetscCall(MatMult(CoarseToFine, X_coarse_owned, X_fine_owned));

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
/// @param numFine          Number of local cells in the fine mesh
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

  // For fine cells, determine the global IDs of local cells, which depends on if a local cell
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

PetscErrorCode RDyRefine(RDy rdy) {
  AppCtx   user;
  Mat      CoarseToFine, FineToCoarse;
  DM       dm_fine;
  Vec      U_coarse_local, U_fine_local;
  PetscInt ndof_coarse;
  PetscFunctionBeginUser;
  PetscCall(VecGetBlockSize(rdy->u_global, &ndof_coarse));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));

  /* Adapt */
  PetscCall(AdaptMesh(rdy->dm, ndof_coarse, &dm_fine, &CoarseToFine, &FineToCoarse, &user));
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
  PetscCall(DetermineCoarseToFineCellMapping(rdy, CoarseToFine, U_coarse_local, U_fine_local, fineIsOwned, fineToCoarseMap, coarseToFineMap,
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

  // destroy the coarse DMs
  PetscCall(DMDestroy(&rdy->dm));
  PetscCall(DMDestroy(&rdy->aux_dm));

  // set the DM to be the refined DM
  rdy->dm      = dm_fine;
  rdy->flow_dm = rdy->dm;

  // create the auxiliary DM
  PetscCall(CreateAuxiliaryDM(rdy));

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
  rdy->mesh_was_refined = PETSC_TRUE;
  rdy->num_refinements++;

  // destroy and recreate mesh
  PetscCall(RDyMeshDestroy(rdy->mesh));
  PetscCall(RDyMeshCreateFromDM(rdy->dm, rdy->num_refinements, &rdy->mesh));

  // initialize the refined solution from existing previous solution
  PetscCall(MatMult(CoarseToFine, U_coarse_local, rdy->u_local));
  PetscCall(DMLocalToGlobal(rdy->dm, rdy->u_local, INSERT_VALUES, rdy->u_global));
  PetscCall(MatDestroy(&CoarseToFine));
  PetscCall(MatDestroy(&FineToCoarse));
  PetscCall(VecDestroy(&U_coarse_local));
  PetscCall(VecDestroy(&U_fine_local));

  // destroy the operator
  PetscCall(DestroyOperator(&rdy->operator));

  // destroy the boundaries and reallocate memory
  PetscCall(RDyDestroyBoundaries(&rdy));
  InitBoundaries(rdy);

  // reinitialize the operator
  PetscCall(InitOperator(rdy));

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
