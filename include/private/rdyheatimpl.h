#ifndef RDYHEATIMPL_H
#define RDYHEATIMPL_H

#include <petscsys.h>
#include <rdycore.h>

PETSC_INTERN PetscErrorCode RDyHeatCreate(RDy);
PETSC_INTERN PetscErrorCode RDyHeatDestroy(RDy);
PETSC_INTERN PetscErrorCode RDyHeatCaptureStarState(RDy);
PETSC_INTERN PetscErrorCode RDyHeatUpdateForcing(RDy, PetscReal);

#endif
