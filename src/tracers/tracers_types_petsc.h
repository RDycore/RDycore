#ifndef TRACERS_TYPES_PETSC_H
#define TRACERS_TYPES_PETSC_H

#include <petscsys.h>

// GRAVITY is already defined in the PETSc SWE code, which we use
// static const PetscReal GRAVITY          = 9.806;   // gravitational acceleration [m/s^2]
static const PetscReal DENSITY_OF_WATER = 1000.0;  // [kg/m^3]

// riemann left and right states
typedef struct {
  PetscInt   num_states;         // number of states
  PetscInt   num_flow_comp;      // number of flow components
  PetscInt   num_tracers_comp;  // number of tracer components
  PetscReal *h, *hu, *hv, *hci;  // prognostic variables
  PetscReal *u, *v, *ci;         // diagnostic variables
} TracersRiemannStateData;

typedef struct {
  PetscInt   num_edges;          // number of edges
  PetscInt   num_flow_comp;      // number of flow components
  PetscInt   num_tracers_comp;  // number of tracer components
  PetscReal *cn, *sn;            // cosine and sine of the angle between edges and y-axis
  PetscReal *fluxes;             // fluxes through the edge
  PetscReal *amax;               // courant number on edges
} TracersRiemannEdgeData;

#endif  // TRACERS_PETSC_TYPES_H
