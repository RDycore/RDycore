#include <petscdmceed.h>
#include <petscdmplex.h>
#include <private/rdycoreimpl.h>
#include <private/rdydmimpl.h>
#include <rdycore.h>

static PetscErrorCode RenameDMFields(DM dm, SectionFieldSpec fields) {
  PetscFunctionBeginUser;

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
  PetscCall(RenameDMFields(dm, fields));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CreateCellCenteredDMFromDM(DM dm, PetscInt refinement_level, const SectionFieldSpec fields, DM *cc_dm) {
  PetscFunctionBegin;

  PetscCall(DMClone(dm, cc_dm));

  PetscCall(CreateDMSection(*cc_dm, fields));

  if (!refinement_level) {
    // copy adjacency info from the original DM
    PetscSF sf_migration, sf_natural;
    PetscCall(DMPlexGetMigrationSF(dm, &sf_migration));
    PetscCall(DMPlexCreateGlobalToNaturalSF(*cc_dm, NULL, sf_migration, &sf_natural));
    PetscCall(DMSetNaturalSF(*cc_dm, sf_natural));
    PetscCall(DMSetUseNatural(*cc_dm, PETSC_TRUE));
    PetscCall(PetscSFDestroy(&sf_natural));
  } else {
    PetscCall(DMSetUseNatural(dm, PETSC_FALSE));
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
  PetscCall(DMPlexDistributeSetDefault(rdy->dm, PETSC_FALSE));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)rdy->dm, NULL));

  // parallel refinement phase
  PetscInt  pStart, pEnd, pStartNew, pEndNew;
  PetscBool refined;
  PetscCall(DMPlexGetChart(rdy->dm, &pStart, &pEnd));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)rdy->dm, "ref_"));
  PetscCall(DMSetFromOptions(rdy->dm));
  PetscCall(DMViewFromOptions(rdy->dm, NULL, "-dm_view"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)rdy->dm, NULL));
  PetscCall(DMPlexGetChart(rdy->dm, &pStartNew, &pEndNew));
  refined = (pStart == pStartNew) && (pEnd == pEndNew) ? PETSC_FALSE : PETSC_TRUE;

  // distribution phase
  if (refined) {
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)rdy->dm, "ref_dist_"));
    PetscCall(DMPlexDistributeSetDefault(rdy->dm, PETSC_TRUE));
    PetscCall(DMSetFromOptions(rdy->dm));
    PetscCall(DMViewFromOptions(rdy->dm, NULL, "-dm_view"));
    PetscCall(DMPlexDistributeSetDefault(rdy->dm, PETSC_FALSE));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)rdy->dm, NULL));
  }

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
  if (size > 1 && !refined) {
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

  // rename the fields in the distributed section
  PetscCall(RenameDMFields(rdy->dm, rdy->soln_fields));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// This function creates an auxiliary (secondary) DM
PetscErrorCode CreateAuxiliaryDMs(RDy rdy) {
  PetscFunctionBegin;

  PetscCall(CreateCellCenteredDMFromDM(rdy->dm, rdy->amr.num_refinements, rdy->field_diags, &rdy->dm_diags));
  PetscCall(CreateCellCenteredDMFromDM(rdy->dm, rdy->amr.num_refinements, rdy->field_1dof, &rdy->dm_1dof));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief This function creates a DM for tracers
/// @param rdy
/// @return PETSC_SUCCESS on success
PetscErrorCode CreateTracerDM(RDy rdy) {
  PetscFunctionBegin;
  PetscInt num_sediment_class = rdy->config.physics.sediment.num_classes;

  rdy->tracer_fields.num_fields              = 1;
  rdy->tracer_fields.num_field_components[0] = num_sediment_class;

  snprintf(rdy->tracer_fields.field_names[0], MAX_NAME_LEN, "Sediments");
  for (PetscInt i = 0; i < num_sediment_class; i++) {
    snprintf(rdy->tracer_fields.field_component_names[0][i], MAX_NAME_LEN, "Class_%" PetscInt_FMT, i);
  }

  PetscCall(CreateCellCenteredDMFromDM(rdy->dm, rdy->amr.num_refinements, rdy->tracer_fields, &rdy->tracer_dm));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CreateFlowDM(RDy rdy) {
  PetscFunctionBegin;

  rdy->flow_fields.num_fields              = 1;
  rdy->flow_fields.num_field_components[0] = 3;

  snprintf(rdy->flow_fields.field_names[0], MAX_NAME_LEN, "Solution");

  snprintf(rdy->flow_fields.field_component_names[0][0], MAX_NAME_LEN, "Height");
  snprintf(rdy->flow_fields.field_component_names[0][1], MAX_NAME_LEN, "MomentumX");
  snprintf(rdy->flow_fields.field_component_names[0][2], MAX_NAME_LEN, "MomentumY");

  PetscCall(CreateCellCenteredDMFromDM(rdy->dm, rdy->amr.num_refinements, rdy->flow_fields, &rdy->flow_dm));

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
  PetscCall(DMCreateGlobalVector(rdy->dm_diags, &rdy->vec_diags));
  PetscCall(DMCreateGlobalVector(rdy->dm_1dof, &rdy->vec_1dof));

  if (rdy->config.physics.sediment.num_classes) {
    // Vecs for flow
    PetscCall(DMCreateGlobalVector(rdy->flow_dm, &rdy->flow_global));
    PetscCall(DMCreateLocalVector(rdy->flow_dm, &rdy->flow_local));

    // Vecs for sediment
    PetscCall(DMCreateGlobalVector(rdy->tracer_dm, &rdy->tracer_global));
    PetscCall(DMCreateLocalVector(rdy->tracer_dm, &rdy->tracer_local));

  } else {
    // Point the flow Vecs to soln Vecs
    rdy->flow_global = rdy->u_global;
    rdy->flow_local  = rdy->u_local;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}
