#ifndef RDYCOREIMPL_H
#define RDYCOREIMPL_H

#include <rdycore.h>
#include <private/rdyconfigimpl.h>
#include <private/rdylogimpl.h>
#include <private/rdymeshimpl.h>

#include <petsc/private/petscimpl.h>

// This type defines a region consisting of cells identified by their local
// indices.
typedef struct {
  PetscInt   *cell_ids;
  PetscInt    num_cells;
} RDyRegion;

// This type defines a boundary consisting of edges identified by their local
// indices.
typedef struct {
  PetscInt   *edge_ids;
  PetscInt    num_edges;
} RDyBoundary;

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
} RDyCondition;

// This type serves as a "virtual table" containing function pointers that
// define the behavior of the dycore.
typedef struct _RDyOps *RDyOps;
struct _RDyOps {
  // Called by RDyCreate to allocate implementation-specific resources, storing
  // the result in the given context pointer.
  PetscErrorCode (*create)(void**);
  // Called by RDyDestroy to free implementation-specific resources.
  PetscErrorCode (*destroy)(void*);
};

// an application context that stores data relevant to a simulation
struct _p_RDy {
  PETSCHEADER(struct _RDyOps);

  // Implementation-specific context pointer
  void *context;

  //------------------------------------------------------------------
  // TODO: The fields below are subject to change as we find our way!
  //------------------------------------------------------------------

  // MPI communicator used for the simulation
  MPI_Comm comm;
  // MPI rank of local process
  PetscInt rank;
  // Number of processes in the communicator
  PetscInt nproc;
  // file storing input data for the simulation
  char config_file[PETSC_MAX_PATH_LEN];

  // configuration data read from config_file
  RDyConfig config;

  // PETSc (DMPlex) grid
  DM dm;

  // Auxiliary DM for diagnostics
  DM aux_dm;

  // mesh representing simulation domain
  RDyMesh mesh;

  // mesh regions
  PetscInt   num_regions;
  PetscInt  *region_ids;
  RDyRegion *regions;

  // mesh boundaries
  PetscInt     num_boundaries;
  PetscInt    *boundary_ids;
  RDyBoundary *boundaries;

  // initial conditions associated with mesh regions (1 per region)
  RDyCondition *initial_conditions;

  // sources (and sinks) associated with mesh regions (1 per region)
  RDyCondition *sources;

  // boundary conditions associated with mesh boundaries (1 per boundary)
  RDyCondition *boundary_conditions;

  // log file handle
  FILE *log;

  //--------------------------
  // Solver and solution data
  //--------------------------

  // time step size
  PetscReal dt;

  // index of current timestep
  PetscInt step;

  // time₋stepping solver
  TS ts;

  // solution vectors (global and local)
  Vec X, X_local;

  // residual vector
  Vec R;
};

PETSC_INTERN PetscErrorCode ReadConfigFile(RDy);
PETSC_INTERN PetscErrorCode PrintConfig(RDy);

// shallow water equations functions
PETSC_INTERN PetscErrorCode InitSWE(RDy);
PETSC_INTERN PetscErrorCode RHSFunctionSWE(TS, PetscReal, Vec, Vec, void*);

#endif
