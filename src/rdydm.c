#include <petscdmceed.h>
#include <petscdmplex.h>
#include <private/rdycoreimpl.h>
#include <rdycore.h>

PetscErrorCode CloneAndCreateCellCenteredDM(DM dm, PetscBool is_dm_refined, PetscInt n_aux_field, PetscInt n_aux_field_dof[n_aux_field], PetscInt m,
                                            char aux_field_names[n_aux_field][m], DM *aux_dm) {
  PetscFunctionBegin;

  PetscCall(DMClone(dm, aux_dm));

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));

  PetscSection aux_sec;
  PetscCall(PetscSectionCreate(comm, &aux_sec));
  PetscCall(PetscSectionSetNumFields(aux_sec, n_aux_field));
  PetscInt n_aux_field_dof_tot = 0;
  for (PetscInt f = 0; f < n_aux_field; ++f) {
    PetscCall(PetscSectionSetFieldName(aux_sec, f, &aux_field_names[f][0]));
    PetscCall(PetscSectionSetFieldComponents(aux_sec, f, n_aux_field_dof[f]));
    n_aux_field_dof_tot += n_aux_field_dof[f];
  }

  // set the number of auxiliary degrees of freedom in each cell
  PetscInt c_start, c_end;  // starting and ending cell points
  DMPlexGetHeightStratum(dm, 0, &c_start, &c_end);
  PetscCall(PetscSectionSetChart(aux_sec, c_start, c_end));
  for (PetscInt c = c_start; c < c_end; ++c) {
    for (PetscInt f = 0; f < n_aux_field; ++f) {
      PetscCall(PetscSectionSetFieldDof(aux_sec, c, f, n_aux_field_dof[f]));
    }
    PetscCall(PetscSectionSetDof(aux_sec, c, n_aux_field_dof_tot));
  }

  // embed the section's data in the auxiliary DM and toss the section
  PetscCall(PetscSectionSetUp(aux_sec));
  PetscCall(DMSetLocalSection(*aux_dm, aux_sec));
  PetscCall(PetscSectionViewFromOptions(aux_sec, NULL, "-aux_layout_view"));
  PetscCall(PetscSectionDestroy(&aux_sec));

  if (!is_dm_refined) {
    // copy adjacency info from the primary DM
    PetscSF sf_migration, sf_natural;
    PetscCall(DMPlexGetMigrationSF(dm, &sf_migration));
    PetscCall(DMPlexCreateGlobalToNaturalSF(*aux_dm, aux_sec, sf_migration, &sf_natural));
    PetscCall(DMPlexSetGlobalToNaturalSF(*aux_dm, sf_natural));
    PetscCall(PetscSFDestroy(&sf_natural));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}
