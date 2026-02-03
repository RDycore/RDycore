#ifndef RDYMODELIMPL_H
#define RDYMODELIMPL_H

#include <ceed/ceed.h>
#include <petsc/private/petscimpl.h>
#include <private/rdyconfigimpl.h>

//----------
// RDyModel
//----------

typedef struct {
  CeedQFunctionUser func;
  const char       *loc;
} CeedQFunctionAndLoc;

#define MAX_NUM_COMPONENTS (MAX_NUM_TRACERS + 1)

// This type contains a selection of functions and Q-functions that define the model equations
// solved by RDycore. It is used in tandem with the Operator type to calculate data within nonlinear
// solvers.
typedef struct RDyModel {
  union {
    struct {
      struct {
      } flow;
      struct {
      } tracers;
    } petsc;
    struct {
      struct {
        CeedQFunctionAndLoc interior_flux, boundary_flux, source;
      } flow;
      struct {
        CeedQFunctionAndLoc interior_flux, boundary_flux, sources[MAX_NUM_COMPONENTS];
      } tracers;
    } ceed;
  };
} RDyModel;

PETSC_INTERN PetscErrorCode ConfigureModel(RDyConfig *, RDyModel *);

#endif
