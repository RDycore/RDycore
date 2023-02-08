#ifndef RDYCOREIMPL_H
#define RDYCOREIMPL_H

#include <rdycore.h>
#include <private/rdymeshimpl.h>

#include <petsc/private/petscimpl.h>

// the maximum number of regions that can be defined on a domain
#define MAX_NUM_REGIONS 128

// the maximum number of surfaces that can be defined on a domain
#define MAX_NUM_SURFACES 128

// This type identifies available spatial discretization methods.
typedef enum {
  SPATIAL_FV = 0, // finite volume method
  SPATIAL_FE      // finite element method
} RDySpatial;

// This type identifies available temporal discretization methods.
typedef enum {
  TEMPORAL_EULER = 0, // forward euler method
  TEMPORAL_RK4,       // 4th-order Runge-Kutta method
  TEMPORAL_BEULER     // backward euler method
} RDyTemporal;

// This type identifies available Riemann solvers for horizontal flow.
typedef enum {
  RIEMANN_ROE = 0, // Roe solver
  RIEMANN_HLLC     // Harten, Lax, van Leer Contact solver
} RDyRiemann;

// This type identifies available bed friction models.
typedef enum {
  BED_FRICTION_NOT_SET = 0,
  BED_FRICTION_NONE,
  BED_FRICTION_CHEZY,
  BED_FRICTION_MANNING
} RDyBedFriction;

// This type identifies a time unit.
typedef enum {
  TIME_MINUTES = 0,
  TIME_HOURS,
  TIME_DAYS,
  TIME_MONTHS,
  TIME_YEARS
} RDyTimeUnit;

// This type stores metadata for in-line quad meshes specified in config files.
typedef struct {
  PetscInt nx, ny;
  PetscReal xmin, xmax, ymin, ymax;
  char inactive_file[PETSC_MAX_PATH_LEN];
} RDyQuadMesh;

// This type specifies a "kind" of condition that indicates how that condition
// is to be enforced on a region or surface.
typedef enum {
  CONDITION_DIRICHLET = 0, // Dirichlet condition (value is specified)
  CONDITION_NEUMANN        // Neumann condition (derivative is specified)
} RDyConditionType;

// This type defines a "condition" representing
// * an initial condition or source/sink associated with a region
// * a boundary condition associated with a surface
typedef struct {
  // type of the condition
  RDyConditionType type;
  // value(s) associated with the condition
  PetscReal value;
} RDyCondition;

// This type defines a region consisting of cells identified by their local
// indices.
typedef struct {
  PetscInt *cell_ids;
  PetscInt  num_cells;
} RDyRegion;

// This type defines a surface consisting of edges identified by their local
// indices.
typedef struct {
  PetscInt *edge_ids;
  PetscInt  num_edges;
} RDySurface;

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
  // filename storing input data for the simulation
  char filename[PETSC_MAX_PATH_LEN];

  //---------
  // Physics
  //---------

  PetscBool sediment;
  PetscBool salinity;

  RDyBedFriction bed_friction;
  PetscReal bed_friction_coef;

  //----------
  // Numerics
  //----------
  RDySpatial spatial;
  RDyTemporal temporal;
  RDyRiemann riemann;

  //------------------------
  // Spatial discretization
  //------------------------

  // mesh file (not used when quad meshes are generated)
  char mesh_file[PETSC_MAX_PATH_LEN];

  // quad mesh metadata (not used when mesh files are read)
  RDyQuadMesh quadmesh;

  // PETSc (DMPlex) grid
  DM dm;

  // mesh representing simulation domain
  RDyMesh mesh;

  // mesh regions
  const char *region_names[MAX_NUM_REGIONS];
  RDyRegion   regions[MAX_NUM_REGIONS];

  // mesh surfaces
  const char *surface_names[MAX_NUM_REGIONS];
  RDySurface  surfaces[MAX_NUM_SURFACES];

  // initial conditions associated with mesh regions
  RDyCondition ics[MAX_NUM_REGIONS];
  PetscInt     num_ics;

  // sources (and sinks) associated with mesh regions
  RDyCondition sources[MAX_NUM_REGIONS];
  PetscInt     num_sources;

  // boundary conditions associated with mesh surfaces
  RDyCondition bcs[MAX_NUM_SURFACES];
  PetscInt     num_bcs;

  //--------------
  // Timestepping
  //--------------

  // simulation time at which to end
  PetscReal final_time;
  // Units in which final_time is expressed
  RDyTimeUnit time_unit;
  // Maximum number of time steps
  PetscInt max_step;

  //----------
  // Restarts
  //----------

  // Restart file format
  char restart_format[4];
  // Restart frequency (in steps)
  PetscInt restart_frequency;

  //---------
  // Logging
  //---------

  // Primary log file
  char log_file[PETSC_MAX_PATH_LEN];

  //-----------------
  // Simulation data
  //-----------------

  // time step size
  PetscReal dt;
  // index of current timestep
  PetscInt tstep;

  PetscInt  dof;
  Vec       B, localB;
  Vec       localX;
  PetscBool debug, save, add_building;
  PetscBool interpolate;
};

#endif
