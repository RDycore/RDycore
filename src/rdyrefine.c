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

static PetscErrorCode InitSolutionFromCoarseRDy(Vec X_coarse, Vec u_global) {
  PetscFunctionBegin;

  PetscInt n_local_coarse, ndof_coarse;
  PetscCall(VecGetLocalSize(X_coarse, &n_local_coarse));
  PetscCall(VecGetBlockSize(X_coarse, &ndof_coarse));

  PetscInt n_local_fine, ndof_fine;
  PetscCall(VecGetLocalSize(u_global, &n_local_fine));
  PetscCall(VecGetBlockSize(u_global, &ndof_fine));

  // PetscCheck(ndof_coarse == ndof_fine, comm, PETSC_ERR_USER, "The block size is not same");
  PetscInt ndof = ndof_fine;

  PetscScalar *x_ptr_coarse, *x_ptr_fine;
  PetscCall(VecGetArray(X_coarse, &x_ptr_coarse));
  PetscCall(VecGetArray(u_global, &x_ptr_fine));

  PetscInt num_refined_cells = 4;  // assuming homogeneous refinement over the entire domain

  PetscInt count = 0;
  for (PetscInt i = 0; i < n_local_coarse / ndof; i++) {
    for (PetscInt j = 0; j < num_refined_cells; j++) {
      for (PetscInt idof = 0; idof < ndof; idof++) {
        x_ptr_fine[count * ndof + idof] = x_ptr_coarse[i * ndof + idof];
      }
      count++;
    }
  }

  PetscCall(VecRestoreArray(X_coarse, &x_ptr_coarse));
  PetscCall(VecRestoreArray(u_global, &x_ptr_fine));

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

  // Mat A;
  // Vec Ascale;
  // PetscCall(DMCreateInterpolation(rdy->dm, dm_fine, &A, &Ascale));

  // make a copy of the old solution
  Vec U_coarse;
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
  PetscCall(InitSolutionFromCoarseRDy(U_coarse, rdy->u_global));

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