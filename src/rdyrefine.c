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
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  options->adapt = 1;

  PetscOptionsBegin(comm, "", "Meshing Interpolation Test Options", "DMPLEX");
  PetscCall(PetscOptionsInt("-adapt", "Number of adaptation steps mesh", "ex10.c", options->adapt, &options->adapt, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateDomainLabel(DM dm)
{
  DMLabel  label;
  PetscInt cStart, cEnd, c;

  PetscFunctionBeginUser;
  PetscCall(DMGetCoordinatesLocalSetUp(dm));
  PetscCall(DMCreateLabel(dm, "Cell Sets"));
  PetscCall(DMGetLabel(dm, "Cell Sets", &label));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  for (c = cStart; c < cEnd; ++c) {
    PetscReal centroid[3], volume, x, y;

    PetscCall(DMPlexComputeCellGeometryFVM(dm, c, &volume, centroid, NULL));
    x = centroid[0];
    y = centroid[1];
    /* Headwaters are (0.0,0.25)--(0.1,0.75) */
    if ((x >= 0.0 && x < 0.1) && (y >= 0.25 && y <= 0.75)) {
      PetscCall(DMLabelSetValue(label, c, 1));
      continue;
    }
    /* River channel is (0.1,0.45)--(1.0,0.55) */
    if ((x >= 0.1 && x <= 1.0) && (y >= 0.45 && y <= 0.55)) {
      PetscCall(DMLabelSetValue(label, c, 2));
      continue;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AdaptMesh(DM dm, const PetscInt bs, DM *dm_fine, Mat *CoarseToFine, AppCtx *ctx)
{
  DM              dmCur = dm, last_dm;
  DMLabel         label;
  IS              valueIS, vIS;
  PetscBool       hasLabel;
  const PetscInt *values;
  PetscReal      *volConst; /* Volume constraints for each label value */
  PetscReal       ratio;
  PetscInt        dim, Nv, v, cStart, cEnd, c;
  PetscBool       adapt;
  Mat             cToF[10];

  PetscFunctionBeginUser;
  if (!ctx->adapt) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMHasLabel(dmCur, "Cell Sets", &hasLabel));
  if (!hasLabel) PetscCall(CreateDomainLabel(dmCur));
  PetscCall(DMGetDimension(dmCur, &dim));
  ratio = PetscPowRealInt(0.5, dim);
  /* Get volume constraints */
  PetscCall(DMGetLabel(dmCur, "Cell Sets", &label));
  PetscCall(DMLabelGetValueIS(label, &vIS));
  PetscCall(ISDuplicate(vIS, &valueIS));
  PetscCall(ISDestroy(&vIS));
  /* Sorting ruins the label */
  PetscCall(ISSort(valueIS));
  PetscCall(ISGetLocalSize(valueIS, &Nv));
  PetscCall(ISGetIndices(valueIS, &values));
  PetscCall(PetscMalloc1(Nv, &volConst));
  for (v = 0; v < Nv; ++v) {
    char opt[128];

    volConst[v] = PETSC_MAX_REAL;
    PetscCall(PetscSNPrintf(opt, 128, "-volume_constraint_%d", (int)values[v]));
    PetscCall(PetscOptionsGetReal(NULL, NULL, opt, &volConst[v], NULL));
  }
  PetscCall(ISRestoreIndices(valueIS, &values));
  PetscCall(ISDestroy(&valueIS));
  /* Adapt mesh iteratively */
  PetscCall(PetscObjectSetName((PetscObject)dmCur, "coarse"));
  PetscCall(DMViewFromOptions(dmCur, NULL, "-adapt_pre_dm_view"));
  adapt = PETSC_TRUE;
  for (PetscInt ilev = 0 ; ilev < ctx->adapt && adapt && ilev < 9 ; ilev++) {
    DM       dmAdapt;
    DMLabel  adaptLabel;
    PetscInt nAdaptLoc[2], nAdapt[2];

    nAdaptLoc[0] = nAdaptLoc[1] = 0;
    nAdapt[0] = nAdapt[1] = 0;
    /* Adaptation is not preserving the domain label */
    PetscCall(DMHasLabel(dmCur, "Cell Sets", &hasLabel));
    if (!hasLabel) PetscCall(CreateDomainLabel(dmCur));
    PetscCall(DMGetLabel(dmCur, "Cell Sets", &label));
    PetscCall(DMLabelGetValueIS(label, &vIS));
    PetscCall(ISDuplicate(vIS, &valueIS));
    PetscCall(ISDestroy(&vIS));
    /* Sorting directly the label's value IS would corrupt the label so we duplicate the IS first */
    PetscCall(ISSort(valueIS));
    PetscCall(ISGetLocalSize(valueIS, &Nv));
    PetscCall(ISGetIndices(valueIS, &values));
    /* Construct adaptation label */
    PetscCall(DMLabelCreate(PETSC_COMM_SELF, "adapt", &adaptLabel));
    PetscCall(DMPlexGetHeightStratum(dmCur, 0, &cStart, &cEnd));
    for (c = cStart; c < cEnd; ++c) {
      PetscReal volume, centroid[3];
      PetscInt  value, vidx;

      PetscCall(DMPlexComputeCellGeometryFVM(dmCur, c, &volume, centroid, NULL));
      PetscCall(DMLabelGetValue(label, c, &value));
      if (value < 0) continue;
      PetscCall(PetscFindInt(value, Nv, values, &vidx));
      PetscCheck(vidx >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Value %" PetscInt_FMT " for cell %" PetscInt_FMT " does not exist in label", value, c);
      if (volume > volConst[vidx] || 1) { // enable
        PetscCall(DMLabelSetValue(adaptLabel, c, DM_ADAPT_REFINE));
        ++nAdaptLoc[0];
      }
      if (volume < volConst[vidx] * ratio && 0) { // disable
        PetscCall(DMLabelSetValue(adaptLabel, c, DM_ADAPT_COARSEN));
        ++nAdaptLoc[1];
      }
    }
    PetscCall(ISRestoreIndices(valueIS, &values));
    PetscCall(ISDestroy(&valueIS));
    PetscCallMPI(MPIU_Allreduce(&nAdaptLoc, &nAdapt, 2, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject)dmCur)));
    last_dm = dmCur;
    adapt = PETSC_FALSE;
    cToF[ilev] = cToF[ilev+1] = NULL;
    if (nAdapt[0]) {
      DM cdm,rcdm;
      Mat CoarseToFine;
      DMPlexTransform tr;
      PetscInt d_nz = 4, o_nz = 0, ccStart, ccEnd, cpStart, fcStart, fcEnd, r, col, i;
      PetscScalar val = 1.0;
      PetscCall(PetscInfo(dmCur, "Adapted mesh, marking %" PetscInt_FMT " cells for refinement, and %" PetscInt_FMT " cells for coarsening\n", nAdapt[0], nAdapt[1]));
      PetscCall(DMAdaptLabel(dmCur, adaptLabel, &dmAdapt)); // DMRefine
      PetscCall(DMCopyDisc(dmCur, dmAdapt));
      PetscCall(DMGetCoordinateDM(dmCur, &cdm));
      PetscCall(DMGetCoordinateDM(dmAdapt, &rcdm));
      PetscCall(DMCopyDisc(cdm, rcdm));
      //PetscCall(DMPlexTransformCreateDiscLabels(tr, dm_fine));
      ((DM_Plex *)(dmAdapt)->data)->useHashLocation = ((DM_Plex *)dmCur->data)->useHashLocation;
      PetscCall(PetscObjectSetName((PetscObject)dmAdapt, "adapting"));

      char opt[128];
      PetscCall(PetscSNPrintf(opt, 128, "-adapt_dm_view_%d", (int)ilev));
      PetscCall(DMViewFromOptions(dmAdapt, NULL, opt));
      PetscCall(DMSetCoarseDM(dmAdapt, dmCur));
      // create the transformation
      PetscCall(DMPlexTransformCreate(PETSC_COMM_SELF, &tr));
      PetscCall(DMPlexTransformSetType(tr, DMPLEXREFINEREGULAR));
      PetscCall(DMPlexTransformSetDM(tr, dmAdapt));
      PetscCall(DMPlexTransformSetFromOptions(tr));
      PetscCall(DMPlexTransformSetUp(tr));
      // make interpolation matrix
      PetscCall(DMPlexGetHeightStratum(dmCur, 0, &ccStart, &ccEnd));
      PetscCall(DMPlexGetHeightStratum(dmAdapt, 0, &fcStart, &fcEnd));
      PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, bs*(fcEnd-fcStart), bs*(ccEnd-ccStart), PETSC_DETERMINE, PETSC_DETERMINE, d_nz, NULL, o_nz, NULL, &CoarseToFine));
      PetscCall(MatSetBlockSize(CoarseToFine,bs));
      PetscCall(DMPlexGetChart(dmCur, &cpStart, NULL));
      for (PetscInt fc = fcStart; fc < fcEnd; fc++) {
        DMPolytopeType  ct, qct;
        PetscCall(DMPlexTransformGetSourcePoint(tr, fc, &ct, &qct, &col, &r));
        for (i = 0 ; i < bs ; i++) {
          PetscCall(MatSetValue(CoarseToFine, bs*(fc - fcStart) + i, bs*col + i, val, INSERT_VALUES));
        }
      }
      PetscCall(MatSetFromOptions(CoarseToFine));
      PetscCall(MatAssemblyBegin(CoarseToFine, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(CoarseToFine, MAT_FINAL_ASSEMBLY));
      PetscCall(MatViewFromOptions(CoarseToFine, NULL, "-adapt_mat_view"));
      PetscCall(DMPlexTransformDestroy(&tr));
      if (last_dm != dm) PetscCall(DMDestroy(&last_dm));
      last_dm = dmAdapt;
      dmCur = dmAdapt;
      cToF[ilev] = CoarseToFine;
      adapt = PETSC_TRUE;
    }
    PetscCall(DMLabelDestroy(&adaptLabel));
  }
  PetscCall(PetscFree(volConst));
  *dm_fine = last_dm;
  PetscCall(PetscObjectSetName((PetscObject)*dm_fine, "refined"));
  PetscCall(DMViewFromOptions(*dm_fine, NULL, "-adapt_post_dm_view"));
  // make final interpoation matrix CoarseToFine
  if (!cToF[0]) *CoarseToFine = NULL;
  else if (!cToF[1]) *CoarseToFine = cToF[0];
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
