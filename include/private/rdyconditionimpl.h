#ifndef RDYCONDITIONIMPL_H
#define RDYCONDITIONIMPL_H

#include <petsc/private/petscimpl.h>
#include <private/rdyconfigimpl.h>

// This type defines a "condition" representing
// * an initial condition or source/sink associated with a region
// * a boundary condition associated with a boundary
typedef struct {
  // flow, sediment, salinity conditions (NULL for none)
  RDyFlowCondition     *flow;
  RDySedimentCondition *sediment;
  RDySalinityCondition *salinity;

  // value(s) associated with the condition
  PetscReal value;

  // was this boundary condition automatically generated and not explicitly
  // requested in the config file?
  PetscBool auto_generated;
} RDyCondition;

#endif
