#include <muParserDLL.h>
#include <petscdmceed.h>
#include <petscdmplex.h>
#include <petscsys.h>
#include <private/rdycoreimpl.h>
#include <private/rdydmimpl.h>
#include <private/rdymathimpl.h>
#include <private/rdysweimpl.h>

#include <petsc/private/dmpleximpl.h> /*I      "petscdmplex.h"   I*/
typedef struct {
  PetscInt adapt; /* Flag for adaptation of the surface mesh */
  PetscBool metric;  /* Flag to use metric adaptation, instead of tagging */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscMPIInt size;

  PetscFunctionBeginUser;
  options->adapt = 1;
  options->metric = PETSC_FALSE;
  PetscCallMPI(MPI_Comm_size(comm, &size));

  PetscOptionsBegin(comm, "", "Meshing Interpolation Test Options", "DMPLEX");
  PetscCall(PetscOptionsInt("-adapt", "Number of adaptation steps mesh", "ex10.c", options->adapt, &options->adapt, NULL));
  PetscCall(PetscOptionsBool("-metric", "Flag for metric refinement", "ex41.c", options->metric, &options->metric, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateAdaptLabel(DM dm, AppCtx *ctx, DMLabel *adaptLabel)
{
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
    if ((x >= 0.0 && x < 1.) && (y >= 0. && y <= 1.)) {
      PetscCall(DMLabelSetValue(label, c, DM_ADAPT_REFINE));
      //PetscCall(PetscPrintf(PETSC_COMM_SELF, "refine: %" PetscInt_FMT "\n", c));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ConstructRefineTree(DM dm, Mat CoarseToFine)
{
  DMPlexTransform tr;
  DM              odm;
  PetscInt        cStart, cEnd, bs, Istart, Jstart;
  PetscScalar val = 1.0;

  PetscFunctionBegin;
  PetscCall(MatGetBlockSize(CoarseToFine, &bs));
  PetscCall(DMPlexGetTransform(dm, &tr));
  if (!tr) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMPlexTransformGetDM(tr, &odm));
  PetscCall(DMPlexGetHeightStratum(odm, 0, &cStart, &cEnd));
  PetscCall(MatGetOwnershipRange(CoarseToFine, &Istart, NULL));
  PetscCall(MatGetOwnershipRangeColumn(CoarseToFine, &Jstart, NULL));
  for (PetscInt c = cStart; c < cEnd; ++c) {
    DMPolytopeType  ct;
    DMPolytopeType *rct;
    PetscInt       *rsize, *rcone, *rornt;
    PetscInt        Nct, dim, pNew = 0;

    //PetscCall(PetscPrintf(PETSC_COMM_SELF, "Cell %" PetscInt_FMT " produced new cells", c));
    PetscCall(DMPlexGetCellType(odm, c, &ct));
    dim = DMPolytopeTypeGetDim(ct);
    PetscCall(DMPlexTransformCellTransform(tr, ct, c, NULL, &Nct, &rct, &rsize, &rcone, &rornt));
    for (PetscInt n = 0; n < Nct; ++n) {
      if (DMPolytopeTypeGetDim(rct[n]) != dim) continue;
      for (PetscInt r = 0; r < rsize[n]; ++r) {
        PetscCall(DMPlexTransformGetTargetPoint(tr, ct, rct[n], c, r, &pNew));
        //PetscCall(PetscPrintf(PETSC_COMM_SELF, " %" PetscInt_FMT, pNew));
        for (PetscInt i = 0 ; i < bs ; i++) {
          PetscCall(MatSetValue(CoarseToFine, Istart + bs*(pNew - 0) + i, Jstart + bs*(c - cStart) + i, val, INSERT_VALUES));
        }
      }
    }
    //PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n"));
  }
  PetscCall(MatSetFromOptions(CoarseToFine));
  PetscCall(MatAssemblyBegin(CoarseToFine, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(CoarseToFine, MAT_FINAL_ASSEMBLY));
  PetscCall(MatViewFromOptions(CoarseToFine, NULL, "-adapt_mat_view"));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AdaptMesh(DM dm, const PetscInt bs, DM *dm_fine, Mat *CoarseToFine, AppCtx *ctx)
{
  DM              dmCur = dm;
  Mat             cToF[10];

  PetscFunctionBeginUser;
  PetscCall(DMViewFromOptions(dmCur, NULL, "-adapt_pre_dm_view"));
  for (PetscInt ilev = 0 ; ilev < ctx->adapt && ilev < 9 ; ilev++) {
    DM       dmAdapt;
    DMLabel  adaptLabel;
    Mat CoarseToFine;
    PetscInt d_nz = 4, o_nz = 0, ccStart, ccEnd, fcStart, fcEnd;
    char opt[128];

    PetscCall(CreateAdaptLabel(dmCur, ctx, &adaptLabel));
    PetscCall(DMLabelView(adaptLabel, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(DMAdaptLabel(dmCur, adaptLabel, &dmAdapt)); // DMRefine
    PetscCall(DMLabelDestroy(&adaptLabel));
    cToF[ilev] = cToF[ilev+1] = NULL;
    if (!dmAdapt) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "refine: level %" PetscInt_FMT "failed ???\n", ilev));
      break; // nothing refined?
    }
    PetscCall(PetscObjectSetName((PetscObject)dmAdapt, "Adapted Mesh"));
    PetscCall(PetscSNPrintf(opt, 128, "-adapt_dm_view_%d", (int)ilev));
    PetscCall(DMViewFromOptions(dmAdapt, NULL, opt));
    // make interpolation matrix
    PetscCall(DMPlexGetHeightStratum(dmCur, 0, &ccStart, &ccEnd));
    PetscCall(DMPlexGetHeightStratum(dmAdapt, 0, &fcStart, &fcEnd));
    PetscCall(MatCreateAIJ(PetscObjectComm((PetscObject)dm), bs*(fcEnd-fcStart), bs*(ccEnd-ccStart), PETSC_DETERMINE, PETSC_DETERMINE, d_nz, NULL, o_nz, NULL, &CoarseToFine));
    PetscCall(MatSetBlockSize(CoarseToFine, bs));
    PetscCall(ConstructRefineTree(dmAdapt, CoarseToFine));
    cToF[ilev] = CoarseToFine;
    if (dmCur != dm) PetscCall(DMDestroy(&dmCur));
    dmCur = dmAdapt;
  }
  *dm_fine = dmCur;
  PetscCall(PetscObjectSetName((PetscObject)*dm_fine, "refined"));
  PetscCall(DMViewFromOptions(*dm_fine, NULL, "-adapt_post_dm_view"));
  // make final interpoation matrix CoarseToFine
  if (!cToF[0]) *CoarseToFine = NULL;
  else if (!cToF[1]) *CoarseToFine = cToF[0]; // one level 
  else {
    Mat AA;
    for (int ii = 1; cToF[ii] ; ii++) {
      PetscCall(MatMatMult(cToF[ii], cToF[ii-1], MAT_INITIAL_MATRIX, PETSC_DETERMINE, &AA));
      PetscCall(MatViewFromOptions(cToF[ii], NULL, "-adapt_mat_view"));
      PetscCall(MatDestroy(&cToF[ii-1]));
      PetscCall(MatDestroy(&cToF[ii]));
      *CoarseToFine = cToF[ii] = AA;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateRefinedRegionsFromCoarseRDy(RDy rdy_coarse, PetscInt *num_regions, RDyRegion **regions) {
  PetscFunctionBeginUser;

  *num_regions = rdy_coarse->num_regions;

  PetscCall(PetscCalloc1(*num_regions, regions));

  for (PetscInt r = 0; r < rdy_coarse->num_regions; ++r) {
    RDyRegion *region_coarse = &rdy_coarse->regions[r];
    RDyRegion *region_fine   = regions[r];

    region_fine->id = region_coarse->id;
    strcpy(region_fine->name, region_coarse->name);

    PetscInt num_refined_cells   = 4;  // assuming homogeneous refinement over the entire domain
    region_fine->num_local_cells = region_coarse->num_local_cells * num_refined_cells;
    region_fine->num_owned_cells = region_coarse->num_owned_cells * num_refined_cells;

    if (region_fine->num_owned_cells > 0) {
      PetscCall(PetscCalloc1(region_fine->num_local_cells, &region_fine->owned_cell_global_ids));
      PetscCall(PetscCalloc1(region_fine->num_owned_cells, &region_fine->cell_local_ids));

      PetscInt count = 0;
      for (PetscInt i = 0; i < region_coarse->num_local_cells; ++i) {
        for (PetscInt j = 0; j < num_refined_cells; j++) {
          region_fine->owned_cell_global_ids[count] = region_coarse->owned_cell_global_ids[i] * num_refined_cells + j;
          region_fine->cell_local_ids[count]        = region_coarse->cell_local_ids[i] * num_refined_cells + j;
          count++;
        }
      }
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Project the initial condition from the coarse grid to the fine grid using
///        a projection matrix.
/// @param CoarseToFine Projection matrix from coarse to fine grid
/// @param X_coarse 
/// @param X_fine
/// @return 
static PetscErrorCode InitSolutionFromCoarseRDy(Mat CoarseToFine, Vec X_coarse, Vec X_fine) {
  PetscFunctionBeginUser;

  PetscInt n_local_coarse, ndof_coarse;
  PetscCall(VecGetLocalSize(X_coarse, &n_local_coarse));
  PetscCall(VecGetBlockSize(X_coarse, &ndof_coarse));

  PetscInt n_local_fine, ndof_fine;
  PetscCall(VecGetLocalSize(X_fine, &n_local_fine));
  PetscCall(VecGetBlockSize(X_fine, &ndof_fine));

  // PetscCheck(ndof_coarse == ndof_fine, comm, PETSC_ERR_USER, "The block size is not same");
  PetscInt ndof = ndof_fine;

  PetscScalar *x_ptr_coarse, *x_ptr_fine, *x_ptr;
  PetscCall(VecGetArray(X_coarse, &x_ptr_coarse));
  PetscCall(VecGetArray(X_fine, &x_ptr_fine));

  Vec coarse_1dof, fine_1dof;
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, n_local_coarse, PETSC_DECIDE, &coarse_1dof));
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, n_local_fine, PETSC_DECIDE, &fine_1dof));

  PetscInt ncells_fine = n_local_fine / ndof;
  PetscInt ncells_coarse = n_local_coarse / ndof;

  for (PetscInt idof = 0; idof < ndof; idof++) {

    // fill the coarse 1 dof vector
    PetscCall(VecGetArray(coarse_1dof, &x_ptr));
    for (PetscInt c = 0; c < ncells_coarse; c++) {
      x_ptr[c] = x_ptr_coarse[c * ndof + idof];
    }
    PetscCall(VecRestoreArray(coarse_1dof, &x_ptr));

    // project the coarse 1 dof vector to the fine 1 dof vector
    PetscCall(MatMult(CoarseToFine, coarse_1dof, fine_1dof));

    // fill the fine 1 dof vector
    PetscCall(VecGetArray(fine_1dof, &x_ptr));
    for (PetscInt c = 0; c < ncells_fine; c++) {
      x_ptr_fine[c * ndof + idof] = x_ptr[c];
    }
    PetscCall(VecRestoreArray(fine_1dof, &x_ptr));
  }

  PetscCall(VecRestoreArray(X_coarse, &x_ptr_coarse));
  PetscCall(VecRestoreArray(X_fine, &x_ptr_fine));

  PetscCall(VecDestroy(&coarse_1dof));
  PetscCall(VecDestroy(&fine_1dof));

  PetscFunctionReturn(PETSC_SUCCESS);
}

extern PetscErrorCode InitOperator(RDy rdy);
extern PetscErrorCode InitMaterialProperties(RDy rdy);
extern PetscErrorCode InitSolver(RDy rdy);
extern PetscErrorCode InitDirichletBoundaryConditions(RDy rdy);
extern PetscErrorCode InitSourceConditions(RDy rdy);

PetscErrorCode RDyRefine(RDy rdy) {
  AppCtx user;
  Mat CoarseToFine;
  DM dm_fine;
  Vec U_coarse;
  PetscInt ndof_coarse;
  PetscFunctionBeginUser;
  PetscCall(VecGetBlockSize(rdy->u_global, &ndof_coarse));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  /* Adapt */
  PetscCall(AdaptMesh(rdy->dm, ndof_coarse, &dm_fine, &CoarseToFine, &user));
  PetscCall(DMLocalizeCoordinates(dm_fine));
  /* PetscCall(PetscObjectSetName((PetscObject)dm_fine, "Mesh")); */
  /* PetscCall(DMSetFromOptions(dm_fine)); */
  PetscCall(DMViewFromOptions(dm_fine, NULL, "-dm_fine_view"));
  PetscCall(DMSetCoarseDM(dm_fine, rdy->dm));
  PetscCall(DMGetCoordinatesLocalSetUp(dm_fine));
  {
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

  {
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

  // Mat A;
  // Vec Ascale;
  // PetscCall(DMCreateInterpolation(rdy->dm, dm_fine, &A, &Ascale));

  // make a copy of the old solution
  PetscCall(VecDuplicate(rdy->u_global, &U_coarse));
  PetscCall(VecCopy(rdy->u_global, U_coarse));

  // create data structure for the refined regions from existing coarse regions
  RDyRegion *refined_regions = NULL;
  PetscInt   num_regions;
  PetscCall(CreateRefinedRegionsFromCoarseRDy(rdy, &num_regions, &refined_regions));

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

  // destroy and recreate mesh
  PetscCall(RDyMeshDestroy(rdy->mesh));
  PetscCall(RDyMeshCreateFromDM(rdy->dm, &rdy->mesh));

  // initialize the refined solution from existing previous solution
  PetscCall(InitSolutionFromCoarseRDy(CoarseToFine, U_coarse, rdy->u_global));
  PetscCall(MatDestroy(&CoarseToFine));
 
  // destroy the operator
  PetscCall(DestroyOperator(&rdy->operator));

  // save time and timestep from TS
  PetscReal time, dt;
  PetscInt  nstep;
  PetscCall(TSGetTime(rdy->ts, &time));
  PetscCall(TSGetTimeStep(rdy->ts, &dt));
  PetscCall(TSGetStepNumber(rdy->ts, &nstep));

  // destroy the boundaries and reallocate memory
  PetscCall(RDyDestroyBoundaries(&rdy));
  PetscCall(PetscCalloc1(rdy->num_boundaries, &rdy->boundaries));

  // reinitialize the operator
  PetscCall(InitOperator(rdy));

  // reinitialize material properties
  PetscCall(InitMaterialProperties(rdy));

  // reinitialize the solver
  PetscCall(InitSolver(rdy));

  // set the time and timstep in TS
  PetscCall(TSSetStepNumber(rdy->ts, nstep));
  PetscCall(TSSetTime(rdy->ts, time));
  PetscCall(TSSetTimeStep(rdy->ts, dt));

  // make sure any Dirichlet boundary conditions are properly specified
  PetscCall(InitDirichletBoundaryConditions(rdy));

  // initialize the source terms
  PetscCall(InitSourceConditions(rdy));

  // mark the mesh was refined
  rdy->mesh_was_refined = PETSC_TRUE;
  rdy->num_refinements++;

  PetscFunctionReturn(PETSC_SUCCESS);
}
