#include <muParserDLL.h>
#include <petscdmceed.h>
#include <petscdmplex.h>
#include <petscsys.h>
#include <private/rdycoreimpl.h>
#include <private/rdydmimpl.h>
#include <private/rdymathimpl.h>
#include <private/rdysweimpl.h>
#include <private/rdyoperatordataimpl.h>

#define MAX_COMP_NAME_LENGTH 20

/// This function creates one Section with 3 DOFs for SWE.
static PetscErrorCode CreateSectionForSWE(RDy rdy, PetscSection *sec) {
  PetscInt n_field                             = 1;
  PetscInt n_field_comps[1]                    = {3};
  char     comp_names[3][MAX_COMP_NAME_LENGTH] = {
      "Height",
      "MomentumX",
      "MomentumY",
  };

  PetscFunctionBeginUser;
  PetscCall(PetscSectionCreate(rdy->comm, sec));
  PetscCall(PetscSectionSetNumFields(*sec, n_field));
  PetscInt n_field_dof_tot = 0;
  for (PetscInt f = 0; f < n_field; ++f) {
    PetscCall(PetscSectionSetFieldComponents(*sec, f, n_field_comps[f]));
    for (PetscInt c = 0; c < n_field_comps[f]; ++c, ++n_field_dof_tot) {
      PetscCall(PetscSectionSetComponentName(*sec, f, c, comp_names[c]));
    }
  }

  // set the number of degrees of freedom in each cell
  PetscInt c_start, c_end;  // starting and ending cell points
  PetscCall(DMPlexGetHeightStratum(rdy->dm, 0, &c_start, &c_end));
  PetscCall(PetscSectionSetChart(*sec, c_start, c_end));
  for (PetscInt c = c_start; c < c_end; ++c) {
    for (PetscInt f = 0; f < n_field; ++f) {
      PetscCall(PetscSectionSetFieldDof(*sec, c, f, n_field_comps[f]));
    }
    PetscCall(PetscSectionSetDof(*sec, c, n_field_dof_tot));
  }
  PetscCall(PetscSectionSetUp(*sec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InitRegionsFromCoarseRDy(RDy rdy_coarse, RDy rdy_fine) {
  PetscFunctionBegin;

  rdy_fine->num_regions = rdy_coarse->num_regions;

  PetscCall(PetscCalloc1(rdy_fine->num_regions, &rdy_fine->regions));

  for (PetscInt r = 0; r < rdy_coarse->num_regions; ++r) {
    RDyRegion *region_coarse = &rdy_coarse->regions[r];
    RDyRegion *region_fine   = &rdy_fine->regions[r];

    region_fine->id = region_coarse->id;
    strcpy(region_fine->name, region_coarse->name);

    PetscInt num_refined_cells = 4; // assuming homogeneous refinement over the entire domain
    region_fine->num_cells = region_coarse->num_cells * num_refined_cells;

    if (region_fine->num_cells > 0) {
      PetscCall(PetscCalloc1(region_fine->num_cells, &region_fine->cell_ids));

      PetscInt count = 0;
      for (PetscInt i = 0; i < region_coarse->num_cells; ++i) {
        for (PetscInt j = 0; j < num_refined_cells; j++) {
          region_fine->cell_ids[count] = region_coarse->cell_ids[i] * num_refined_cells + j;
          count++;
        }
      }
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateRegionsFromCoarseRDy(RDy rdy_coarse, PetscInt *num_regions, RDyRegion **regions) {
  PetscFunctionBegin;

  *num_regions = rdy_coarse->num_regions;

  PetscCall(PetscCalloc1(*num_regions, regions));

  for (PetscInt r = 0; r < rdy_coarse->num_regions; ++r) {
    RDyRegion *region_coarse = &rdy_coarse->regions[r];
    RDyRegion *region_fine   = regions[r];

    region_fine->id = region_coarse->id;
    strcpy(region_fine->name, region_coarse->name);

    PetscInt num_refined_cells = 4; // assuming homogeneous refinement over the entire domain
    region_fine->num_cells = region_coarse->num_cells * num_refined_cells;

    if (region_fine->num_cells > 0) {
      PetscCall(PetscCalloc1(region_fine->num_cells, &region_fine->cell_ids));

      PetscInt count = 0;
      for (PetscInt i = 0; i < region_coarse->num_cells; ++i) {
        for (PetscInt j = 0; j < num_refined_cells; j++) {
          region_fine->cell_ids[count] = region_coarse->cell_ids[i] * num_refined_cells + j;
          count++;
        }
      }
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InitMaterialsFromCoarseRDy(RDy rdy_coarse, RDy rdy_fine) {
  PetscFunctionBegin;

  PetscCall(PetscCalloc1(rdy_fine->mesh.num_cells, &rdy_fine->materials_by_cell));

  PetscInt num_refined_cells = 4; // assuming homogeneous refinement over the entire domain

  PetscInt count = 0;
  for (PetscInt i = 0; i < rdy_coarse->mesh.num_cells; i++) {
    for (PetscInt j = 0; j < num_refined_cells; j++) {
      rdy_fine->materials_by_cell[count].manning = rdy_coarse->materials_by_cell[i].manning;
      count++;
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMaterialsFromCoarseRDy(RDy rdy_coarse, PetscInt *num_cells, RDyMaterial **materials_by_cell) {
  PetscFunctionBegin;

  PetscInt num_refined_cells = 4; // assuming homogeneous refinement over the entire domain

  *num_cells = rdy_coarse->mesh.num_cells * num_refined_cells;
  PetscCall(PetscCalloc1(*num_cells, materials_by_cell));

  PetscInt count = 0;
  for (PetscInt i = 0; i < rdy_coarse->mesh.num_cells; i++) {
    for (PetscInt j = 0; j < num_refined_cells; j++) {
      (*materials_by_cell)[count].manning = rdy_coarse->materials_by_cell[i].manning;
      count++;
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InitSolutionFromCoarseRDy(Vec X_coarse, RDy rdy_fine) {
  PetscFunctionBegin;

  PetscInt n_local_coarse, ndof_coarse;
  PetscCall(VecGetLocalSize(X_coarse, &n_local_coarse));
  PetscCall(VecGetBlockSize(X_coarse, &ndof_coarse));

  PetscInt n_local_fine, ndof_fine;
  PetscCall(VecGetLocalSize(rdy_fine->u_global, &n_local_fine));
  PetscCall(VecGetBlockSize(rdy_fine->u_global, &ndof_fine));

  PetscCheck(ndof_coarse == ndof_fine, rdy_fine->comm, PETSC_ERR_USER, "The block size is not same");
  PetscInt ndof = ndof_fine;

  PetscScalar *x_ptr_coarse, *x_ptr_fine;
  PetscCall(VecGetArray(X_coarse, &x_ptr_coarse));
  PetscCall(VecGetArray(rdy_fine->u_global, &x_ptr_fine));

  PetscInt num_refined_cells = 4; // assuming homogeneous refinement over the entire domain

  PetscInt count = 0;
  for (PetscInt i = 0; i < n_local_coarse/ndof; i++) {
    for (PetscInt j = 0; j < num_refined_cells; j++) {
      for (PetscInt idof = 0; idof < ndof; idof++) {
        x_ptr_fine[count * ndof + idof] = x_ptr_coarse[i * ndof + idof];
      }
      count++;
    }
  }

  PetscCall(VecRestoreArray(X_coarse, &x_ptr_coarse));
  PetscCall(VecRestoreArray(rdy_fine->u_global, &x_ptr_fine));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RDyDestroyRegions(RDy rdy) {
  PetscFunctionBegin;

  for (PetscInt i = 0; i < rdy->num_regions; ++i) {
    if (rdy->regions[i].cell_ids) {
      PetscFree(rdy->regions[i].cell_ids);
    }
  }
  if (rdy->regions) PetscFree(rdy->regions);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RDyDestroyBoundaries(RDy rdy) {
  PetscFunctionBegin;

  for (PetscInt i = 0; i < rdy->num_boundaries; ++i) {
    if (rdy->boundaries[i].edge_ids) {
      PetscFree(rdy->boundaries[i].edge_ids);
    }
  }
  if (rdy->boundaries) PetscFree(rdy->boundaries);

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RDyDestroyVecs(RDy rdy) {
  PetscFunctionBegin;

  if (rdy->petsc.sources) VecDestroy(&(rdy->petsc.sources));
  if (rdy->rhs) VecDestroy(&(rdy->rhs));
  if (rdy->u_global) VecDestroy(&(rdy->u_global));
  if (rdy->u_local) VecDestroy(&(rdy->u_local));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyRefine (RDy rdy_coarse) {
  PetscFunctionBegin;

  PetscInt cStart, cEnd;
  DMLabel label;

  PetscCall(DMPlexGetHeightStratum(rdy_coarse->dm, 0, &cStart, &cEnd));
  PetscCall(DMCreateLabel(rdy_coarse->dm, "adapt"));
  PetscCall(DMGetLabel(rdy_coarse->dm, "adapt", &label));
  for (PetscInt c = cStart; c < cEnd; c++) {
    PetscCall(DMLabelSetValue(label, c, DM_ADAPT_REFINE));
  }

  DM dm_fine;
  PetscCall(DMRefine(rdy_coarse->dm, PETSC_COMM_WORLD, &dm_fine));
  PetscCall(DMCopyDisc(rdy_coarse->dm, dm_fine));
  PetscCall(DMViewFromOptions(dm_fine, NULL, "-dm_fine_view"));

  Vec U_coarse;
  PetscCall(VecDuplicate(rdy_coarse->u_global, &U_coarse));
  PetscCall(VecCopy(rdy_coarse->u_global, U_coarse));

  RDyRegion *regions = NULL;
  PetscInt num_regions;
  PetscCall(CreateRegionsFromCoarseRDy(rdy_coarse, &num_regions, &regions));

  RDyMaterial *materials_by_cell = NULL;
  PetscInt num_cells;
  PetscCall(CreateMaterialsFromCoarseRDy(rdy_coarse, &num_cells, &materials_by_cell));

  /*  
  RDy rdy_fine;
  PetscCall(RDyCreate(rdy_coarse->global_comm, rdy_coarse->config_file, &rdy_fine));

  // note: default config values are specified in the YAML input schema!
  PetscCall(ReadConfigFile(rdy_fine));

  rdy_fine->log = stdout;

  // override parameters using command line arguments
  PetscCall(OverrideParameters(rdy_fine));

  // print configuration info
  PetscCall(PrintConfig(rdy_fine));

  // ++++++++++++++++++++++++++++++++++++++++++++++++++
  // Create DM
  RDyLogDebug(rdy_coarse, "FINE: Creating DMs...");
  rdy_fine->dm = dm_fine;
  PetscSection sec;
  PetscCall(CreateSectionForSWE(rdy_fine, &sec));
  // embed the section's data in our grid and toss the section
  PetscCall(DMSetLocalSection(rdy_fine->dm, sec));
  PetscCall(DMViewFromOptions(rdy_fine->dm, NULL, "-dm_fine_view"));
  // ++++++++++++++++++++++++++++++++++++++++++++++++++

  PetscCall(CreateAuxiliaryDM(rdy_fine));  // for diagnostics
  PetscCall(CreateVectors(rdy_fine));      // global and local vectors, residuals

  RDyLogDebug(rdy_coarse, "FINE: Initializing regions...");
  rdy_fine->num_regions = num_regions;
  rdy_fine->regions = regions;
  //PetscCall(InitRegionsFromCoarseRDy(rdy_coarse, rdy_fine));

  RDyLogDebug(rdy_coarse, "FINE: Initializing initial conditions and sources...");
  PetscCall(InitInitialConditions(rdy_fine));
  PetscCall(InitSources(rdy_fine));

  RDyLogDebug(rdy_coarse, "FINE: Creating FV mesh...");
  PetscCall(RDyMeshCreateFromDM(rdy_fine->dm, &rdy_fine->mesh));

  RDyLogDebug(rdy_coarse, "FINE Initializing boundaries and boundary conditions...");
  PetscCall(InitBoundaries(rdy_fine));
  PetscCall(InitBoundaryConditions(rdy_fine));

  // ++++++++++++++++++++++++++++++++++++++++++++++++++
  rdy_fine->materials_by_cell = materials_by_cell;
  //PetscCall(InitMaterialsFromCoarseRDy(rdy_coarse, rdy_fine));


  PetscCall(InitSolutionFromCoarseRDy(U_coarse, rdy_fine));
  // ++++++++++++++++++++++++++++++++++++++++++++++++++

  PetscCall(InitSWE(rdy_fine)); // Sets up CEED solvers/Operators

  PetscCall(InitDirichletBoundaryConditions(rdy_fine));

  CourantNumberDiagnostics *courant_num_diags = &rdy_fine->courant_num_diags;
  courant_num_diags->max_courant_num          = 0.0;
  courant_num_diags->global_edge_id           = -1;
  courant_num_diags->global_cell_id           = -1;
  courant_num_diags->is_set                   = PETSC_FALSE;

  while (!RDyFinished(rdy_fine)) {
    PetscCall(RDyAdvance(rdy_fine));
  }
 
 */

  PetscCall(RDyDestroyVecs(rdy_coarse));
  PetscCall(RDyDestroyBoundaries(rdy_coarse));
  PetscCall(PetscCalloc1(rdy_coarse->num_boundaries, &rdy_coarse->boundaries));

  PetscCall(DMDestroy(&rdy_coarse->aux_dm));
  PetscCall(DMDestroy(&rdy_coarse->dm));

  rdy_coarse->dm  = dm_fine;
  PetscSection sec;
  PetscCall(CreateSectionForSWE(rdy_coarse, &sec));
  // embed the section's data in our grid and toss the section
  PetscCall(DMSetLocalSection(rdy_coarse->dm, sec));
  PetscCall(DMViewFromOptions(rdy_coarse->dm, NULL, "-dm_fine_view"));

  PetscCall(CreateAuxiliaryDM(rdy_coarse));  // for diagnostics
  PetscCall(CreateVectors(rdy_coarse));      // global and local vectors, residuals

  PetscCall(RDyDestroyRegions(rdy_coarse));
  rdy_coarse->num_regions = num_regions;
  rdy_coarse->regions = regions;

  PetscCall(RDyMeshDestroy(rdy_coarse->mesh));
  PetscCall(RDyMeshCreateFromDM(rdy_coarse->dm, &rdy_coarse->mesh));

  PetscFree(rdy_coarse->materials_by_cell);
  rdy_coarse->materials_by_cell = materials_by_cell;

  PetscCall(InitSolutionFromCoarseRDy(U_coarse, rdy_coarse));

  PetscCall(DestroyOperators(rdy_coarse));

  // save time and timestep
  PetscReal time, dt;
  PetscCall(TSGetTime(rdy_coarse->ts, &time));
  PetscCall(TSGetTimeStep(rdy_coarse->ts, &dt));

  // create solvers
  PetscCall(CreateSolvers(rdy_coarse));

  // set time and timestep
  PetscCall(TSSetTime(rdy_coarse->ts, time));
  PetscCall(TSSetTimeStep(rdy_coarse->ts, dt));

  PetscCall(CreateOperators(rdy_coarse));

  PetscCall(InitDirichletBoundaryConditions(rdy_coarse));

  PetscFunctionReturn(PETSC_SUCCESS);
}