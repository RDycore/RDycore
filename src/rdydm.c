#include <petscdmceed.h>
#include <petscdmplex.h>
#include <private/rdycoreimpl.h>
#include <private/rdydmimpl.h>
#include <rdycore.h>

/// This function creates a Section appropriate for use by the given (primary) DM.
static PetscErrorCode CreateDMSection(DM dm, SectionFieldSpec fields) {
  PetscFunctionBeginUser;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));

  // Create a P0 discretization
  DMPolytopeType ct;
  PetscInt       dim, cStart, cEnd;

  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  if (cEnd > cStart) PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  else ct = DM_POLYTOPE_QUADRILATERAL;
  for (PetscInt f = 0; f < fields.num_fields; ++f) {
    PetscFE fe;

    PetscCall(PetscFECreateLagrangeByCell(PETSC_COMM_SELF, dim, fields.num_field_components[f], ct, 0, PETSC_DETERMINE, &fe));
    PetscCall(PetscObjectSetName((PetscObject)fe, fields.field_names[f]));
    PetscCall(DMAddField(dm, NULL, (PetscObject)fe));
    PetscCall(PetscFEDestroy(&fe));
  }
  PetscCall(DMCreateDS(dm));

  // Set field and component names
  PetscSection sec;

  PetscCall(DMGetLocalSection(dm, &sec));
  for (PetscInt f = 0; f < fields.num_fields; ++f) {
    PetscCall(PetscSectionSetFieldName(sec, f, fields.field_names[f]));
    for (PetscInt c = 0; c < fields.num_field_components[f]; ++c) {
      if (fields.field_component_names[f][c][0]) PetscCall(PetscSectionSetComponentName(sec, f, c, fields.field_component_names[f][c]));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CreateCellCenteredDMFromDM(DM dm, const SectionFieldSpec fields, DM *cc_dm) {
  PetscFunctionBegin;

  PetscCall(DMClone(dm, cc_dm));

  PetscCall(CreateDMSection(*cc_dm, fields));

  PetscInt refine_level;
  PetscCall(DMGetRefineLevel(dm, &refine_level));
  if (!refine_level) {
    // copy adjacency info from the original DM
    PetscSF sf_migration, sf_natural;
    PetscCall(DMPlexGetMigrationSF(dm, &sf_migration));
    PetscCall(DMPlexCreateGlobalToNaturalSF(*cc_dm, NULL, sf_migration, &sf_natural));
    PetscCall(DMSetNaturalSF(*cc_dm, sf_natural));
    PetscCall(DMSetUseNatural(*cc_dm, PETSC_TRUE));
    PetscCall(PetscSFDestroy(&sf_natural));
  }
  PetscSection section;
  PetscCall(DMGetLocalSection(*cc_dm, &section));
  PetscCall(PetscSectionViewFromOptions(section, NULL, "-aux_layout_view"));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// This function create the primary DM for RDycore. The Vec and Mat types are
/// set for CPU or GPUs. Must be called after CreateOperator() so that an
/// operator is available to create sections.
PetscErrorCode CreateDM(RDy rdy) {
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(rdy->comm, &size));

  PetscCall(DMCreate(rdy->comm, &rdy->dm));
  PetscCall(DMSetType(rdy->dm, DMPLEX));

  // if we're using CEED, set Vec and Mat types based on the selected backend
  if (CeedEnabled()) {
    VecType vec_type = NULL;
    PetscCall(GetCeedVecType(&vec_type));
    PetscCall(DMSetVecType(rdy->dm, vec_type));

    MatType mat_type = NULL;
    if (strstr(vec_type, VECCUDA)) mat_type = MATAIJCUSPARSE;
    else if (strstr(vec_type, VECKOKKOS)) mat_type = MATAIJKOKKOS;
    else mat_type = MATAIJ;
    PetscCall(DMSetMatType(rdy->dm, mat_type));
  }

  PetscCall(DMPlexDistributeSetDefault(rdy->dm, PETSC_FALSE));
  PetscCall(DMSetRefineLevel(rdy->dm, 0));
  PetscCall(DMSetFromOptions(rdy->dm));

  // name the grid
  PetscCall(PetscObjectSetName((PetscObject)rdy->dm, "grid"));

  // NOTE Need to create section before distribution, so that natural map can be created
  // create a section with (h, hu, hv) as degrees of freedom
  PetscCall(CreateDMSection(rdy->dm, rdy->soln_fields));

  // distribution phase
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)rdy->dm, "dist_"));
  PetscCall(DMPlexDistributeSetDefault(rdy->dm, PETSC_TRUE));
  PetscCall(DMSetFromOptions(rdy->dm));
  PetscCall(DMViewFromOptions(rdy->dm, NULL, "-dm_view"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)rdy->dm, NULL));

  // parallel refinement phase
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)rdy->dm, "ref_"));
  PetscCall(DMPlexDistributeSetDefault(rdy->dm, PETSC_FALSE));
  PetscCall(DMSetFromOptions(rdy->dm));
  PetscCall(DMViewFromOptions(rdy->dm, NULL, "-dm_view"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)rdy->dm, NULL));

  // Overlap meshes after refinement
  if (size > 1) {
    DM      dmOverlap;
    PetscSF sfOverlap, sfMigration, sfMigrationNew;

    PetscCall(DMPlexGetMigrationSF(rdy->dm, &sfMigration));
    PetscCall(DMPlexDistributeOverlap(rdy->dm, 1, &sfOverlap, &dmOverlap));
    PetscCall(DMPlexRemapMigrationSF(sfOverlap, sfMigration, &sfMigrationNew));
    PetscCall(PetscSFDestroy(&sfOverlap));
    PetscCall(DMPlexSetMigrationSF(dmOverlap, sfMigrationNew));
    PetscCall(PetscSFDestroy(&sfMigrationNew));
    PetscCall(DMDestroy(&rdy->dm));
    rdy->dm = dmOverlap;
  }

  // mark boundary edges so we can enforce reflecting BCs on them if needed
  {
    DMLabel boundary_edges;
    PetscCall(DMCreateLabel(rdy->dm, "boundary_edges"));
    PetscCall(DMGetLabel(rdy->dm, "boundary_edges", &boundary_edges));
    PetscCall(DMPlexMarkBoundaryFaces(rdy->dm, 1, boundary_edges));
  }

  // create parallel section and global-to-natural mapping
  if (size > 1) {
    PetscSF sfMigration, sfNatural;

    PetscCall(DMPlexGetMigrationSF(rdy->dm, &sfMigration));
    PetscCall(DMPlexCreateGlobalToNaturalSF(rdy->dm, NULL, sfMigration, &sfNatural));
    PetscCall(DMSetNaturalSF(rdy->dm, sfNatural));
    PetscCall(DMSetUseNatural(rdy->dm, PETSC_TRUE));
    PetscCall(PetscSFDestroy(&sfNatural));
  }

  // set grid adacency
  PetscCall(DMSetBasicAdjacency(rdy->dm, PETSC_TRUE, PETSC_TRUE));
  PetscCall(DMCreateDS(rdy->dm));

  PetscCall(DMViewFromOptions(rdy->dm, NULL, "-dm_view"));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// This function creates an auxiliary (secondary) DM
PetscErrorCode CreateAuxiliaryDM(RDy rdy) {
  PetscFunctionBegin;

  PetscCall(CreateCellCenteredDMFromDM(rdy->dm, rdy->diag_fields, &rdy->aux_dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// This function creates an auxiliary (secondary) DM
PetscErrorCode CreateFlowDM(RDy rdy) {
  PetscFunctionBegin;
  rdy->flow_fields = rdy->soln_fields;
  PetscCall(CreateCellCenteredDMFromDM(rdy->dm, rdy->flow_fields, &rdy->flow_dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief  This function creates a DM for sediments
/// @param rdy 
/// @return PETSC_SUCESS on success
PetscErrorCode CreateSedimentDM(RDy rdy) {
  PetscFunctionBegin;
  PetscInt num_sediment_class = rdy->config.physics.sediment.num_classes;

  rdy->sd_fields.num_fields = 1;
  rdy->sd_fields.num_field_components[0] = num_sediment_class;

  sprintf(rdy->sd_fields.field_names[0],"Sediments");
  for (PetscInt i = 0; i < num_sediment_class; i++) {
    sprintf(rdy->sd_fields.field_component_names[0][i],"Class_%d",i);
  }

  PetscCall(CreateCellCenteredDMFromDM(rdy->dm, rdy->sd_fields, &rdy->sd_dm));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CreateCombinedDM(RDy rdy) {
  PetscFunctionBegin;

  PetscInt num_sediment_class = rdy->config.physics.sediment.num_classes;

  rdy->soln_fields.num_fields = 1;
  rdy->soln_fields.num_field_components[0] = 3 + num_sediment_class;

  sprintf(rdy->soln_fields.field_names[0],"Solution");

  sprintf(rdy->soln_fields.field_component_names[0][0],"Height");
  sprintf(rdy->soln_fields.field_component_names[0][1],"MomentumX");
  sprintf(rdy->soln_fields.field_component_names[0][2],"MomentumY");

  for (PetscInt i = 0; i < num_sediment_class; i++) {
    sprintf(rdy->soln_fields.field_component_names[0][i + 3],"Class_%d",i);
  }

  DM combined_dm;
  PetscCall(CreateCellCenteredDMFromDM(rdy->dm, rdy->soln_fields, &combined_dm));

  PetscCall(DMDestroy(&rdy->dm));
  rdy->dm = combined_dm;

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Creates global and local solution vectors and residuals
PetscErrorCode CreateVectors(RDy rdy) {
  PetscFunctionBegin;

  PetscCall(DMCreateGlobalVector(rdy->dm, &rdy->u_global));
  PetscCall(VecDuplicate(rdy->u_global, &rdy->rhs));
  PetscCall(VecViewFromOptions(rdy->u_global, NULL, "-vec_view"));
  PetscCall(DMCreateLocalVector(rdy->dm, &rdy->u_local));

  // diagnostics are all piled into a single vector whose block size is the
  // total number of field components
  PetscCall(DMCreateGlobalVector(rdy->aux_dm, &rdy->diags_vec));

  if (rdy->config.physics.sediment.num_classes) {

    // Vecs for flow 
    PetscCall(DMCreateGlobalVector(rdy->flow_dm, &rdy->flow_u_global));
    PetscCall(DMCreateLocalVector(rdy->flow_dm, &rdy->flow_u_local));

    // Vecs for sediment
    PetscCall(DMCreateGlobalVector(rdy->sd_dm, &rdy->sd_u_global));
    PetscCall(DMCreateLocalVector(rdy->sd_dm, &rdy->sd_u_local));

  } else {
    // Point the flow Vecs to soln Vecs
    rdy->flow_u_global = rdy->u_global;
    rdy->flow_u_local  = rdy->u_local;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}
