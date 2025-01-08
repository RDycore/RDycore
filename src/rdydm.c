#include <petscdmceed.h>
#include <petscdmplex.h>
#include <private/rdycoreimpl.h>
#include <private/rdydmimpl.h>
#include <rdycore.h>

/// This function creates a Section appropriate for use by the given (primary) DM.
static PetscErrorCode CreateDMSection(DM dm, SectionFieldSpec fields, PetscSection *sec) {
  PetscFunctionBeginUser;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));

  PetscCall(PetscSectionCreate(comm, sec));
  PetscCall(PetscSectionSetNumFields(*sec, fields.num_fields));
  PetscInt dof_per_cell = 0;
  for (PetscInt f = 0; f < fields.num_fields; ++f) {
    PetscCall(PetscSectionSetFieldName(*sec, f, fields.field_names[f]));
    PetscCall(PetscSectionSetFieldComponents(*sec, f, fields.num_field_components[f]));
    dof_per_cell += fields.num_field_components[f];
    for (PetscInt c = 0; c < fields.num_field_components[f]; ++c) {
      if (fields.field_component_names[f][c][0]) {
        PetscCall(PetscSectionSetComponentName(*sec, f, c, fields.field_component_names[f][c]));
      }
    }
  }

  // set the number of degrees of freedom for each field in each cell
  PetscInt c_start, c_end;
  PetscCall(DMPlexGetHeightStratum(dm, 0, &c_start, &c_end));
  PetscCall(PetscSectionSetChart(*sec, c_start, c_end));
  for (PetscInt c = c_start; c < c_end; ++c) {
    for (PetscInt f = 0; f < fields.num_fields; ++f) {
      PetscCall(PetscSectionSetFieldDof(*sec, c, f, fields.num_field_components[f]));
    }
    PetscCall(PetscSectionSetDof(*sec, c, dof_per_cell));
  }
  PetscCall(PetscSectionSetUp(*sec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CreateCellCenteredDMFromDM(DM dm, const SectionFieldSpec fields, DM *cc_dm) {
  PetscFunctionBegin;

  PetscCall(DMClone(dm, cc_dm));

  PetscSection section;
  PetscCall(CreateDMSection(*cc_dm, fields, &section));
  PetscCall(DMSetLocalSection(*cc_dm, section));
  PetscCall(PetscSectionViewFromOptions(section, NULL, "-aux_layout_view"));
  PetscCall(PetscSectionDestroy(&section));  // NOTE: yes, here and not below!

  PetscInt refine_level;
  DMGetRefineLevel(dm, &refine_level);
  if (!refine_level) {
    // copy adjacency info from the original DM
    PetscSF sf_migration, sf_natural;
    PetscCall(DMPlexGetMigrationSF(dm, &sf_migration));
    PetscCall(DMPlexCreateGlobalToNaturalSF(*cc_dm, section, sf_migration, &sf_natural));
    PetscCall(DMPlexSetGlobalToNaturalSF(*cc_dm, sf_natural));
    PetscCall(PetscSFDestroy(&sf_natural));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// This function create the primary DM for RDycore. The Vec and Mat types are
/// set for CPU or GPUs. Must be called after CreateOperator() so that an
/// operator is available to create sections.
PetscErrorCode CreateDM(RDy rdy) {
  PetscSection sec;
  PetscMPIInt  size;

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
  if (!rdy->refine) {
    PetscCall(CreateDMSection(rdy->dm, rdy->soln_fields, &sec));
    // embed the section's data in our grid and toss the section
    PetscCall(DMSetLocalSection(rdy->dm, sec));
  }

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
  if (rdy->refine) {
    PetscCall(CreateDMSection(rdy->dm, rdy->soln_fields, &sec));
    PetscCall(DMSetLocalSection(rdy->dm, sec));
  } else if (size > 1) {
    PetscSF      sfMigration, sfNatural;
    PetscSection psec;
    PetscInt    *remoteOffsets;

    PetscCall(DMPlexGetMigrationSF(rdy->dm, &sfMigration));
    PetscCall(DMPlexCreateGlobalToNaturalSF(rdy->dm, sec, sfMigration, &sfNatural));
    PetscCall(DMPlexSetGlobalToNaturalSF(rdy->dm, sfNatural));
    PetscCall(PetscSFDestroy(&sfNatural));

    PetscCall(PetscSectionCreate(rdy->comm, &psec));
    PetscCall(PetscSFDistributeSection(sfMigration, sec, &remoteOffsets, psec));
    PetscCall(DMSetLocalSection(rdy->dm, psec));
    PetscCall(PetscFree(remoteOffsets));
    PetscCall(PetscSectionDestroy(&sec));
    PetscCall(PetscSectionDestroy(&psec));
  }
  PetscCall(PetscSectionDestroy(&sec));

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

  PetscFunctionReturn(PETSC_SUCCESS);
}
