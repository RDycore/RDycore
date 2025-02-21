#ifndef SEDIMENT_PETSC_IMPL_H
#define SEDIMENT_PETSC_IMPL_H

#include <petscsys.h>

// GRAVITY is already defined in the PETSc SWE code, which we use
// static const PetscReal GRAVITY          = 9.806;   // gravitational acceleration [m/s^2]
static const PetscReal DENSITY_OF_WATER = 1000.0;  // [kg/m^3]

// riemann left and right states
typedef struct {
  PetscInt   num_states;         // number of states
  PetscInt   num_flow_comp;      // number of flow components
  PetscInt   num_sediment_comp;  // number of sediment components
  PetscReal *h, *hu, *hv, *hci;  // prognostic variables
  PetscReal *u, *v, *ci;         // diagnostic variables
} SedimentRiemannStateData;

typedef struct {
  PetscInt   num_edges;          // number of edges
  PetscInt   num_flow_comp;      // number of flow components
  PetscInt   num_sediment_comp;  // number of sediment components
  PetscReal *cn, *sn;            // cosine and sine of the angle between edges and y-axis
  PetscReal *fluxes;             // fluxes through the edge
  PetscReal *amax;               // courant number on edges
} SedimentRiemannEdgeData;

#endif  // SEDIMENT_PETSC_IMPL_H
