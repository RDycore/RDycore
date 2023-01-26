#ifndef RDYCOREIMPL_H
#define RDYCOREIMPL_H

#include <rdycore.h>
#include <private/rdymeshimpl.h>

#include <petsc/private/petscimpl.h>

/// This type serves as a "virtual table" containing function pointers that
/// define the behavior of the dycore.
typedef struct _RDyOps *RDyOps;
struct _RDyOps {
  /// Called by RDyCreate to allocate implementation-specific resources, storing
  /// the result in the given context pointer.
  PetscErrorCode (*create)(void**);
  /// Called by RDyDestroy to free implementation-specific resources.
  PetscErrorCode (*destroy)(void*);
};

/// an application context that stores data relevant to a simulation
struct _p_RDy {
  PETSCHEADER(struct _RDyOps);

  // Implementation-specific context pointer
  void *context;

  //------------------------------------------------------------------
  // TODO: The fields below are subject to change as we find our way!
  //------------------------------------------------------------------

  /// MPI communicator used for the simulation
  MPI_Comm comm;
  /// MPI rank of local process
  PetscInt rank;
  /// Number of processes in the communicator
  PetscInt nproc;
  /// filename storing input data for the simulation
  char filename[PETSC_MAX_PATH_LEN];

  //------------------------
  // Spatial discretization
  //------------------------

  /// PETSc (DMPlex) grid
  DM dm;

  /// mesh representing simulation domain
  RDyMesh mesh;

  //--------------
  // Timestepping
  //--------------

  /// total number of time steps
  PetscInt Nt;
  /// time step size
  PetscReal dt;
  /// index of current timestep
  PetscInt tstep;

  //-----------------
  // Simulation data
  //-----------------

  PetscInt  dof;
  Vec       B, localB;
  Vec       localX;
  PetscBool debug, save, add_building;
  PetscBool interpolate;
};

#endif
