#ifndef RDYCOREIMPL_H
#define RDYCOREIMPL_H

#include <rdycore.h>
#include <private/rdymeshimpl.h>

#include <petsc/private/petscimpl.h>

/// This type identifies available spatial discretization methods.
typedef enum {
  FV = 0, // finite volume method
  FE      // finite element method
} RDySpatial;

/// This type identifies available temporal discretization methods.
typedef enum {
  EULER = 0, // forward euler method
  RK4,       // 4th-order Runge-Kutta method
  BEULER     // backward euler method
} RDyTemporal;

/// This type identifies available Riemann solvers for horizontal flow.
typedef enum {
  ROE = 0, // Roe solver
  HLL      // Harten, Lax, van Leer solver
} RDyRiemann;

/// This type identifies available bed friction models.
typedef enum {
  CHEZY = 0, // Chezy model
  MANNING    // Manning model
} RDyBedFriction;

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

  //---------
  // Physics
  //---------

  PetscBool sediment;
  PetscBool salinity;

  RDyBedFriction bed_friction;

  //----------
  // Numerics
  //----------
  RDySpatial spatial;
  RDyTemporal temporal;
  RDyRiemann riemann;

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

  /// simulation time at which to end
  PetscReal final_time;
  /// Maximum number of time steps
  PetscInt max_step;
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
