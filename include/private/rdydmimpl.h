#ifndef RDYDMIMPL_H
#define RDYDMIMPL_H

#include <petsc.h>
#include <rdycore.h>

PETSC_INTERN PetscErrorCode CloneAndCreateCellCenteredDM(DM dm, PetscBool is_dm_refined, PetscInt n_aux_field, PetscInt n_aux_field_dof[n_aux_field],
                                                         PetscInt m, char aux_field_names[n_aux_field][m], DM *aux_dm);
PETSC_INTERN PetscErrorCode CreateDM(RDy rdy);
PETSC_INTERN PetscErrorCode CreateAuxiliaryDM(RDy rdy);

#endif