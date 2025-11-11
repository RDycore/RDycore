#ifndef SWE_TYPES_PETSC_H
#define SWE_TYPES_PETSC_H

#include <petscsys.h>

// gravitational acceleration [m/s/s]
static const PetscReal GRAVITY = 9.806;

//----------------
// Riemann Solver
//----------------

// riemann left and right states
typedef struct {
  PetscInt   num_states;   // number of states
  PetscReal *h, *hu, *hv;  // prognostic SWE variables
  PetscReal *u, *v;        // diagnostic SWE variables
} RiemannStateData;

typedef struct {
  PetscInt   num_edges;  // number of edges
  PetscReal *cn, *sn;    // cosine and sine of the angle between edges and y-axis
  PetscReal *fluxes;     // fluxes through the edge
  PetscReal *amax;       // courant number on edges
} RiemannEdgeData;

#endif  // SWE_TYPES_PETSC_H
