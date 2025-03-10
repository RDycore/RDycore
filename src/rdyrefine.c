#include <muParserDLL.h>
#include <petscdmceed.h>
#include <petscdmplex.h>
#include <petscsys.h>
#include <private/rdycoreimpl.h>
#include <private/rdydmimpl.h>
#include <private/rdymathimpl.h>
#include <private/rdysweimpl.h>

#define MAX_COMP_NAME_LENGTH 20

static PetscErrorCode CreateRefinedRegionsFromCoarseRDy(RDy rdy_coarse, PetscInt *num_regions, RDyRegion **regions) {
  PetscFunctionBegin;

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

/// @brief Create the project matrix from the coarse grid to the fine grid. This assumes
///        that all the cells of the coarse grid have been refined.
/// @param dm_coarse    The coarse grid DM
/// @param dm_fine      The fine grid DM
/// @param CoarseToFine The projection matrix from coarse to fine grid
/// @return 
static PetscErrorCode CreateProjectionMatrix(DM dm_coarse, DM dm_fine, Mat *CoarseToFine) {
  PetscFunctionBegin;

  // get the local size of the vectors
  Vec U_coarse, U_fine;

  PetscCall(DMCreateGlobalVector(dm_coarse, &U_coarse));
  PetscCall(DMCreateGlobalVector(dm_fine, &U_fine));

  PetscInt row, col;
  PetscCall(VecGetLocalSize(U_coarse, &col));
  PetscCall(VecGetLocalSize(U_fine, &row));

  PetscCall(VecDestroy(&U_coarse));
  PetscCall(VecDestroy(&U_fine));

  // now create the projection matrix
  PetscInt d_nz = 4; // assuming all cells have been refined
  PetscInt o_nz = 0;

  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, row, col, PETSC_DETERMINE, PETSC_DETERMINE, d_nz, NULL, o_nz, NULL, CoarseToFine));

  for (PetscInt j = 0; j < col; j++) {
    for (PetscInt i = 0; i < d_nz; i++) {
      PetscInt row = j * d_nz + i;
      PetscInt col = j;
      PetscScalar val = 1.0;

      PetscCall(MatSetValue(*CoarseToFine, row, col, val, INSERT_VALUES));
    }
  }
  PetscCall(MatSetFromOptions(*CoarseToFine));
  PetscCall(MatAssemblyBegin(*CoarseToFine, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*CoarseToFine, MAT_FINAL_ASSEMBLY));

  if (0) MatView(*CoarseToFine, PETSC_VIEWER_STDOUT_WORLD);

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Project the initial condition from the coarse grid to the fine grid using
///        a projection matrix.
/// @param CoarseToFine Projection matrix from coarse to fine grid
/// @param X_coarse 
/// @param X_fine
/// @return 
static PetscErrorCode InitSolutionFromCoarseRDy(Mat CoarseToFine, Vec X_coarse, Vec X_fine) {
  PetscFunctionBegin;

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
  PetscFunctionBegin;

  // mark the grid cells for refinement
  PetscInt cStart, cEnd;
  DMLabel  label;

  PetscCall(DMPlexGetHeightStratum(rdy->dm, 0, &cStart, &cEnd));
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "adapt", &label));
  for (PetscInt c = cStart; c < cEnd; c++) {
    PetscCall(DMLabelSetValue(label, c, DM_ADAPT_REFINE));
  }

  // create a refined DM
  DM dm_fine;
  PetscCall(DMAdaptLabel(rdy->dm, label, &dm_fine));
  PetscCall(DMCopyDisc(rdy->dm, dm_fine));
  PetscCall(DMViewFromOptions(dm_fine, NULL, "-dm_fine_view"));

  {
    PetscSection sec;
    PetscCall(DMGetLocalSection(rdy->dm, &sec));
    PetscInt nfields;
    PetscSectionGetNumFields(sec, &nfields);
    printf("++++++++++++++++++++++++++++++++++++\n");
    printf("About coarse DM: \n");
    printf("nfields = %d\n", nfields);
    for (PetscInt f = 0; f < nfields; f++) {
      PetscInt ncomp;
      PetscCall(PetscSectionGetFieldComponents(sec, f, &ncomp));
      printf("  field = %d; num_component = %d\n", f, ncomp);
      for (PetscInt c = 0; c < ncomp; c++) {
        const char *comp_name;
        PetscSectionGetComponentName(sec, f, c, &comp_name);
        printf("    field = %d; component = %d; comp_name = %s\n", f, c, comp_name);
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
    printf("nfields = %d\n", nfields);
    for (PetscInt f = 0; f < nfields; f++) {
      PetscInt ncomp;
      PetscCall(PetscSectionGetFieldComponents(sec, f, &ncomp));
      printf("  field = %d; num_component = %d\n", f, ncomp);
      for (PetscInt c = 0; c < ncomp; c++) {
        const char *comp_name;
        PetscSectionGetComponentName(sec, f, c, &comp_name);
        printf("    field = %d; component = %d; comp_name = %s\n", f, c, comp_name);
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
  Vec U_coarse;
  PetscCall(VecDuplicate(rdy->u_global, &U_coarse));
  PetscCall(VecCopy(rdy->u_global, U_coarse));

  // create the mapping from coarse to fine
  Mat CoarseToFine;
  PetscCall(CreateProjectionMatrix(rdy->dm, dm_fine, &CoarseToFine));

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