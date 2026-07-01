#ifndef RDYHEATIMPL_H
#define RDYHEATIMPL_H

#include <petscvec.h>
#include <private/rdyconfigimpl.h>
#include <private/rdymeshimpl.h>
#include <rdycore.h>

typedef struct {
  PetscReal* downwelling_shortwave;
  PetscReal* downwelling_longwave;
  PetscReal* wind_speed;
  PetscReal* air_temperature;
  PetscReal* specific_humidity;
} RDyHeatForcing;

struct _RDyHeat {
  RDyMesh*       mesh;
  RDyConfig*     config;
  PetscInt       heat_comp;
  PetscReal      dt;
  Vec            star_state;
  RDyHeatForcing forcing;
};

PETSC_INTERN PetscErrorCode RDyHeatCreate(RDy);
PETSC_INTERN PetscErrorCode RDyHeatDestroy(RDy);
PETSC_INTERN PetscErrorCode RDyHeatCaptureStarState(RDy);
PETSC_INTERN PetscErrorCode RDyHeatUpdateForcing(RDy, PetscReal);

#endif
