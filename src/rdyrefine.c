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
  PetscBool redistribute;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options) {
  PetscMPIInt size;

  PetscFunctionBeginUser;
  options->adapt  = 1;
  options->metric = PETSC_FALSE;
  options->redistribute = PETSC_FALSE;
  PetscCallMPI(MPI_Comm_size(comm, &size));

  PetscOptionsBegin(comm, "", "Meshing Interpolation Test Options", "DMPLEX");
  PetscCall(PetscOptionsInt("-adapt", "Number of adaptation steps mesh", "ex10.c", options->adapt, &options->adapt, NULL));
  PetscCall(PetscOptionsBool("-metric", "Flag for metric refinement", "ex41.c", options->metric, &options->metric, NULL));
  PetscCall(PetscOptionsBool("-redistribute", "Redistribute the adapted mesh", __FILE__, options->redistribute, &options->redistribute, NULL));
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

static PetscErrorCode CreateInterpolator(DM adm, DM ddm, const PetscInt bs, PetscSF sf, Mat *Interp)
{
  DM              dm;
  PetscDS         ds;
  PetscSection    das, as, s;
  DMPlexTransform tr;
  PetscInt       *rows, *cols, *Nc;
  PetscScalar    *vals;
  PetscInt        cStart, cEnd, m, n;

  PetscFunctionBegin;
  PetscCall(DMGetDS(adm, &ds));
  PetscCall(PetscDSGetComponents(ds, &Nc));
  PetscCall(PetscMalloc3(Nc[0] * 16, &rows, Nc[0], &cols, Nc[0] * Nc[0] * 16, &vals));
  PetscCall(PetscMemzero(vals, sizeof(PetscScalar) * Nc[0] * Nc[0] * 16));
  for (PetscInt r = 0; r < 16; ++r) for (PetscInt i = 0; i < Nc[0]; ++i) vals[(r * Nc[0] + i) * Nc[0] + i] = 1.0;
  PetscCall(DMPlexGetTransform(adm, &tr));
  PetscCall(DMGetGlobalSection(adm, &as));
  PetscCall(DMGetGlobalSection(ddm, &das));
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  PetscCall(DMGetGlobalSection(dm, &s));
  PetscCall(PetscObjectSetName((PetscObject)s, "section s-dm"));
  PetscCall(PetscSectionViewFromOptions(s, NULL, "-sec_view"));
  PetscCall(PetscObjectSetName((PetscObject)as, "section as-adm"));
  PetscCall(PetscSectionViewFromOptions(as, NULL, "-sec_view"));
  PetscCall(PetscObjectSetName((PetscObject)das, "section s-ddm"));
  PetscCall(PetscSectionViewFromOptions(das, NULL, "-sec_view"));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(MatCreate(PetscObjectComm((PetscObject)ddm), Interp));
  PetscCall(PetscSectionGetConstrainedStorageSize(as, &m));
  PetscCall(PetscSectionGetConstrainedStorageSize(s, &n));
  PetscCall(MatSetSizes(*Interp, m, n, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetUp(*Interp));
PetscMPIInt myrank;
PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)ddm), &myrank));
  for (PetscInt c = cStart; c < cEnd; ++c) {
    DMPolytopeType *rct, ct;
    PetscInt       *rsize, *rcone, *rornt;
    PetscInt        Nct, dim, off, nrow = 0;

    PetscCall(PetscSectionGetOffset(s, c, &off));
    for (PetscInt i = 0; i < Nc[0]; ++i) cols[i] = off + i;
    PetscCall(DMPlexGetCellType(dm, c, &ct));
    dim = DMPolytopeTypeGetDim(ct);
    PetscCall(DMPlexTransformCellTransform(tr, ct, c, NULL, &Nct, &rct, &rsize, &rcone, &rornt));
    for (PetscInt n = 0; n < Nct; ++n) {
      if (DMPolytopeTypeGetDim(rct[n]) != dim) continue;
      //PetscPrintf(PETSC_COMM_SELF,"\t\t[%d] %d.%d size = %d\n",myrank, (int)c, (int)n, (int)rsize[n]);
      for (PetscInt r = 0; r < rsize[n]; ++r) {
        PetscInt cNew;
        PetscCall(DMPlexTransformGetTargetPoint(tr, ct, rct[n], c, r, &cNew));
        PetscCall(PetscSectionGetOffset(as, cNew, &off));
        for (PetscInt i = 0; i < Nc[0]; ++i) rows[nrow + i] = off + i;
        nrow += Nc[0];
      }
    }
    PetscCall(MatSetValues(*Interp, nrow, rows, Nc[0], cols, vals, INSERT_VALUES));
  }
  PetscCall(PetscFree3(rows, cols, vals));
  PetscCall(MatAssemblyBegin(*Interp, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*Interp, MAT_FINAL_ASSEMBLY));
  if (sf) {
    Mat                In;
    IS                 isrow, iscol;
    const PetscSFNode *remote;
    const PetscInt    *rStarts;
    PetscInt          *rows;
    PetscInt           Nl, cStart;

    PetscCall(PetscSFViewFromOptions(sf, NULL, "-sf_view"));
    PetscCall(MatGetOwnershipRanges(*Interp, &rStarts));
    PetscCall(PetscSFGetGraph(sf, NULL, &Nl, NULL, &remote));
    PetscCall(PetscMalloc1(Nl, &rows));
    for (PetscInt l = 0; l < Nl; ++l) {
      rows[l] = remote[l].index + rStarts[remote[l].rank];
    }
    
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, Nl, rows, PETSC_OWN_POINTER, &isrow));
    PetscCall(MatGetOwnershipRangeColumn(*Interp, &cStart, NULL));
    PetscCall(ISCreateStride(PETSC_COMM_SELF, n, cStart, 1, &iscol));
    //PetscCall(ISSetBlockSize(iscol, bs));
    PetscCall(MatCreateSubMatrix(*Interp, isrow, iscol, MAT_INITIAL_MATRIX, &In));
    PetscCall(ISDestroy(&isrow));
    PetscCall(ISDestroy(&iscol));
    PetscCall(MatDestroy(Interp));
    *Interp = In;
    PetscCall(MatViewFromOptions(*Interp, NULL, "-interp_view"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AdaptMesh(DM dm, const PetscInt bs, DM *dm_fine, Mat *CoarseToFine, Mat *FineToCoarse, AppCtx *ctx) {
  DM       adm, ddm = NULL;;
  DMLabel  adaptLabel;
  PetscSF sf = NULL;
  char     opt[128];

  PetscFunctionBeginUser;
  PetscCall(PetscSNPrintf(opt, 128, "-adapt_dm_view"));
  PetscCall(DMViewFromOptions(dm, NULL, opt));
  PetscCall(CreateAdaptLabel(dm, ctx, &adaptLabel));

  // view - debug
  PetscMPIInt myrank, commsize;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &myrank));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm), &commsize));
  // if (!myrank) PetscPrintf(PETSC_COMM_SELF,"Information about coarse DM\n");
  /* for (PetscInt i = 0; i < commsize; i++) { */
  /*   if (i == myrank) { */
  /*     PetscInt c_start, c_end; */

  /*     DMPlexGetHeightStratum(dm, 0, &c_start, &c_end); */
  /*     PetscCall(PetscPrintf(PETSC_COMM_SELF, "Rank %d: Cell_id, X, Y, Is_owned. %d cells\n", myrank, (int)c_end)); */
  /*     for (PetscInt c = c_start; c < c_end; ++c) { */
  /*       PetscReal area, centroid[3], normal[3]; */
  /*       DMPlexComputeCellGeometryFVM(dm, c, &area, &centroid, &normal[0]); */
  /*       PetscInt gref, junkInt; */
  /*       PetscCall(DMPlexGetPointGlobal(dm, c, &gref, &junkInt)); */
  /*       PetscPrintf(PETSC_COMM_SELF,"%d] %" PetscInt_FMT "] %e %e %" PetscInt_FMT "\n", myrank, c, centroid[0], centroid[1], ((gref >= 0))); */
  /*     } */
  /*   } */
  /*   MPI_Barrier(PETSC_COMM_WORLD); */
  /* } */

  PetscCall(DMPlexSetSaveTransform(dm, PETSC_TRUE));
  PetscCall(DMAdaptLabel(dm, adaptLabel, &adm));  // DMRefine
  PetscCall(DMLabelDestroy(&adaptLabel));
  PetscCheck(adm, PETSC_COMM_WORLD, PETSC_ERR_USER, "Refinement failed.");

  PetscCall(PetscObjectSetName((PetscObject)adm, "Adapted Mesh - pre distribute"));
  PetscCall(DMViewFromOptions(adm, NULL, opt));

  // view - debug
  /* if (!myrank) PetscPrintf(PETSC_COMM_SELF,"Information about REFINED DM\n"); */
  /* for (PetscInt i = 0; i < commsize; i++) { */
  /*   if (i == myrank) { */
  /*     PetscCall(PetscPrintf(PETSC_COMM_SELF, "Rank %d: Cell_id, X, Y \n", myrank)); */
  /*     PetscInt c_start, c_end; */

  /*     DMPlexGetHeightStratum(adm, 0, &c_start, &c_end); */
  /*     for (PetscInt c = c_start; c < c_end; ++c) { */
  /*       PetscReal area, centroid[3], normal[3]; */
  /*       DMPlexComputeCellGeometryFVM(adm, c, &area, &centroid, &normal[0]); */
  /*       PetscInt gref, junkInt; */
  /*       //PetscCall(DMPlexGetPointGlobal(adm, c, &gref, &junkInt)); */
  /*       PetscPrintf(PETSC_COMM_SELF,"%d] %" PetscInt_FMT ") %e %e \n", myrank, c, centroid[0], centroid[1]); */
  /*     } */
  /*   } */
  /*   MPI_Barrier(PETSC_COMM_WORLD); */
  /* } */

  if (ctx->redistribute && commsize > 1) {
    PetscSF      fieldSF;
    PetscSection s, ds;
    PetscInt    *remoteOffsets;

    PetscCall(DMPlexDistribute(adm, 0, &sf, &ddm));
    PetscCheck(ddm, PETSC_COMM_WORLD, PETSC_ERR_USER, "Distribute failed.");
    PetscCall(DMGetLocalSection(adm, &s));
    PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)s), &ds));
    PetscCall(PetscSFDistributeSection(sf, s, &remoteOffsets, ds));

    PetscCall(PetscSFCreateSectionSF(sf, s, remoteOffsets, ds, &fieldSF));
    PetscCall(PetscFree(remoteOffsets));
    PetscCall(PetscSFDestroy(&sf));
    sf = fieldSF;
    PetscCall(PetscObjectSetName((PetscObject)ddm, "Adapted Mesh - post distribute"));
    PetscCall(DMViewFromOptions(ddm, NULL, opt));
  } else {
    PetscCall(PetscObjectReference((PetscObject)adm));
    ddm = adm;
  }

  PetscCall(CreateInterpolator(adm, ddm, bs, sf, CoarseToFine));
  PetscCall(PetscSFDestroy(&sf));
  PetscCall(MatSetBlockSize(*CoarseToFine, bs));
  PetscCall(MatTranspose(*CoarseToFine, MAT_INITIAL_MATRIX, FineToCoarse));
  PetscCall(MatViewFromOptions(*CoarseToFine, NULL, "-interp_view"));  

PetscPrintf(PETSC_COMM_SELF,"%d DONE \n", myrank);
MPI_Barrier(PETSC_COMM_WORLD);
exit(0);

  *dm_fine = adm;

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
  DM       dm_fine, dm_nolap;
  Vec      U_coarse_local, U_fine_local;
  PetscInt ndof_coarse;
  PetscMPIInt size;
  PetscFunctionBeginUser;
  PetscCall(VecGetBlockSize(rdy->u_global, &ndof_coarse));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));

  /* Adapt */
  //PetscCall(DMClone(dm, &dm_nolap));
  //PetscCall(DMCopyDisc(dm, dm_nolap));
  PetscCallMPI(MPI_Comm_size(rdy->comm, &size));
  if (size > 1) dm_nolap = rdy->no_overlap_dm;
  else dm_nolap = rdy->dm;
  PetscCall(AdaptMesh(dm_nolap, ndof_coarse, &dm_fine, &CoarseToFine, &FineToCoarse, &user));

  if (size > 1) {
    DM      dmOverlap;
    PetscSF sfOverlap, sfMigration, sfMigrationNew;

    PetscCall(DMDestroy(&dm_nolap));
    PetscCall(DMPlexGetMigrationSF(dm_fine, &sfMigration));
    PetscCall(DMPlexDistributeOverlap(dm_fine, 1, &sfOverlap, &dmOverlap));
    PetscCall(DMPlexRemapMigrationSF(sfOverlap, sfMigration, &sfMigrationNew));
    PetscCall(PetscSFDestroy(&sfOverlap));
    PetscCall(DMPlexSetMigrationSF(dmOverlap, sfMigrationNew));
    PetscCall(PetscSFDestroy(&sfMigrationNew));
    //PetscCall(DMDestroy(&dm_fine));
    PetscCall(DMDestroy(&rdy->no_overlap_dm));
    rdy->no_overlap_dm = dm_fine;
    //PetscCall(PetscObjectReference((PetscObject)dm_fine));
    dm_fine = dmOverlap;
  }
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
