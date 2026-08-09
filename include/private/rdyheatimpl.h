#ifndef RDYHEATIMPL_H
#define RDYHEATIMPL_H

#include <ceed/ceed.h>
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
  PetscReal* direct_source;  // per-owned-cell direct Q_net override (W/m²); bypasses HeatQNet when use_direct_source is set
} RDyHeatForcing;

// CEED-backend state for the implicit atmospheric heat source step; unused (and
// left zeroed) when the PETSc backend is selected
typedef struct {
  CeedOperator ifunction_op;  // computes the implicit residual
  CeedOperator ijacobian_op;  // computes the (diagonal) implicit Jacobian

  CeedVector u;         // wraps the PETSc state vector during operator application
  CeedVector u_dot;     // wraps the PETSc state time derivative
  CeedVector residual;  // wraps the PETSc residual vector
  CeedVector diagonal;  // wraps diagonal_vec below
  CeedVector forcing;   // owned: per-cell atmospheric forcing, refreshed by UpdateCeedHeatForcing()

  Vec diagonal_vec;  // receives the Jacobian diagonal before MatDiagonalSet()

  CeedContextFieldLabel shift_label;                    // "time shift" on ijacobian_op
  CeedContextFieldLabel ifunction_direct_source_label;  // "use direct source" on ifunction_op
  CeedContextFieldLabel ijacobian_direct_source_label;  // "use direct source" on ijacobian_op
} RDyHeatCeed;

struct _RDyHeat {
  RDyMesh*       mesh;
  RDyConfig*     config;
  PetscInt       heat_comp;
  PetscReal      dt;
  RDyHeatForcing forcing;
  PetscBool      use_direct_source;  // when PETSC_TRUE, use direct_source instead of HeatQNet()
  RDyHeatCeed    ceed;
};

PETSC_INTERN PetscErrorCode RDyHeatCreate(RDy);
PETSC_INTERN PetscErrorCode RDyHeatDestroy(RDy);
PETSC_INTERN PetscErrorCode RDyHeatUpdateForcing(RDy, PetscReal);
PETSC_INTERN PetscErrorCode RDyHeatAdvance(RDy, PetscReal, PetscReal);

// CEED backend (heat_ceed.c)
PETSC_INTERN PetscErrorCode CreateCeedHeatOperators(RDy);
PETSC_INTERN PetscErrorCode DestroyCeedHeatOperators(RDy);
PETSC_INTERN PetscErrorCode UpdateCeedHeatForcing(RDy);
PETSC_INTERN PetscErrorCode HeatIFunctionCeed(TS, PetscReal, Vec, Vec, Vec, void*);
PETSC_INTERN PetscErrorCode HeatIJacobianCeed(TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void*);

#endif
