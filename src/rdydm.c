#include <petscdmceed.h>
#include <petscdmplex.h>
#include <private/rdycoreimpl.h>
#include <rdycore.h>

// Maximum length of the name of a prognostic or diagnostic field component
#define MAX_COMP_NAME_LENGTH 20

/// Create a new cell-centered DM (cc_dm) from a given DM and adds a number of given
/// cell-centered fields as Sections in the new DM.
PetscErrorCode CloneAndCreateCellCenteredDM(DM dm, PetscInt n_cc_field, PetscInt n_cc_field_dof[n_cc_field], PetscInt max_field_name,
                                            char aux_field_names[n_cc_field][max_field_name], DM *cc_dm) {
  PetscFunctionBegin;

  PetscCall(DMClone(dm, cc_dm));

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));

  PetscSection aux_sec;
  PetscCall(PetscSectionCreate(comm, &aux_sec));
  PetscCall(PetscSectionSetNumFields(aux_sec, n_cc_field));
  PetscInt n_cc_field_dof_tot = 0;
  for (PetscInt f = 0; f < n_cc_field; ++f) {
    PetscCall(PetscSectionSetFieldName(aux_sec, f, &aux_field_names[f][0]));
    PetscCall(PetscSectionSetFieldComponents(aux_sec, f, n_cc_field_dof[f]));
    n_cc_field_dof_tot += n_cc_field_dof[f];
  }

  // set the number of auxiliary degrees of freedom in each cell
  PetscInt c_start, c_end;  // starting and ending cell points
  DMPlexGetHeightStratum(dm, 0, &c_start, &c_end);
  PetscCall(PetscSectionSetChart(aux_sec, c_start, c_end));
  for (PetscInt c = c_start; c < c_end; ++c) {
    for (PetscInt f = 0; f < n_cc_field; ++f) {
      PetscCall(PetscSectionSetFieldDof(aux_sec, c, f, n_cc_field_dof[f]));
    }
    PetscCall(PetscSectionSetDof(aux_sec, c, n_cc_field_dof_tot));
  }

  // embed the section's data in the auxiliary DM and toss the section
  PetscCall(PetscSectionSetUp(aux_sec));
  PetscCall(DMSetLocalSection(*cc_dm, aux_sec));
  PetscCall(PetscSectionViewFromOptions(aux_sec, NULL, "-aux_layout_view"));
  PetscCall(PetscSectionDestroy(&aux_sec));

  PetscInt refine_level;
  DMGetRefineLevel(dm, &refine_level);

  if (!refine_level) {
    // copy adjacency info from the primary DM
    PetscSF sf_migration, sf_natural;
    PetscCall(DMPlexGetMigrationSF(dm, &sf_migration));
    PetscCall(DMPlexCreateGlobalToNaturalSF(*cc_dm, aux_sec, sf_migration, &sf_natural));
    PetscCall(DMPlexSetGlobalToNaturalSF(*cc_dm, sf_natural));
    PetscCall(PetscSFDestroy(&sf_natural));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

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

/// This function create the primary DM for RDycore. The Vec and Mat types are
/// set for CPU or GPUs.
PetscErrorCode CreateDM(RDy rdy) {
  PetscSection sec;
  PetscMPIInt  size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(rdy->comm, &size));

  PetscCall(DMCreate(rdy->comm, &rdy->dm));
  PetscCall(DMSetType(rdy->dm, DMPLEX));

  // if we're using CEED, set Vec and Mat types based on the selected backend
  if (rdy->ceed_resource[0]) {
    CeedMemType mem_type_backend;
    PetscCallCEED(CeedGetPreferredMemType(rdy->ceed, &mem_type_backend));
    VecType vec_type = NULL;
    switch (mem_type_backend) {
      case CEED_MEM_HOST:
        vec_type = VECSTANDARD;
        break;
      case CEED_MEM_DEVICE: {
        const char *resolved;
        PetscCallCEED(CeedGetResource(rdy->ceed, &resolved));
        if (strstr(resolved, "/gpu/cuda")) vec_type = VECCUDA;
        else if (strstr(resolved, "/gpu/hip")) vec_type = VECKOKKOS;
        else if (strstr(resolved, "/gpu/sycl")) vec_type = VECKOKKOS;
        else vec_type = VECSTANDARD;
      }
    }
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
    PetscCall(CreateSectionForSWE(rdy, &sec));
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
    PetscCall(CreateSectionForSWE(rdy, &sec));
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

  PetscCall(DMViewFromOptions(rdy->dm, NULL, "-dm_view"));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// This function creates an auxillary (or secondary) DM
PetscErrorCode CreateAuxiliaryDM(RDy rdy) {
  PetscFunctionBegin;

  // create an auxiliary section with a diagnostic parameter.
  PetscInt n_cc_field             = 1;
  PetscInt n_cc_field_dof[1]      = {1};
  char     aux_field_names[1][20] = {"Parameter"};

  PetscCall(CloneAndCreateCellCenteredDM(rdy->dm, n_cc_field, n_cc_field_dof, 20, &aux_field_names[0], &rdy->aux_dm));

  PetscFunctionReturn(PETSC_SUCCESS);
}
