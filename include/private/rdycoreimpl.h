#ifndef RDYCOREIMPL_H
#define RDYCOREIMPL_H

#include <rdycore.h>
#include <private/rdymeshimpl.h>

#include <petsc/private/petscimpl.h>

// the maximum number of regions that can be defined on a domain
#define MAX_NUM_REGIONS 128

// the maximum number of surfaces that can be defined on a domain
#define MAX_NUM_SURFACES 128

// the maximum number of flow conditions that can be defined for a simulation
#define MAX_NUM_FLOW_CONDITIONS 128

// the maximum number of sediment conditions that can be defined for a simulation
#define MAX_NUM_SEDIMENT_CONDITIONS 128

// the maximum number of salinity conditions that can be defined for a simulation
#define MAX_NUM_SALINITY_CONDITIONS 128

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

// This type identifies available physics flow modes.
typedef enum {
  FLOW_SWE = 0,
  FLOW_DIFFUSION
} RDyFlowMode;

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

// This type defines a flow-related condition.
typedef struct {
  const char       *name;
  RDyConditionType  type;
  PetscReal         water_flux;
} RDyFlowCondition;

// This type defines a sediment-related condition.
typedef struct {
  const char       *name;
  RDyConditionType  type;
  PetscReal         concentration;
} RDySedimentCondition;

// This type defines a salinity-related condition.
typedef struct {
  const char       *name;
  RDyConditionType  type;
  PetscReal         concentration;
} RDySalinityCondition;

// This type defines a "condition" representing
// * an initial condition or source/sink associated with a region
// * a boundary condition associated with a surface
typedef struct {
  // flow, sediment, salinity conditions (NULL for none)
  RDyFlowCondition     *flow;
  RDySedimentCondition *sediment;
  RDySalinityCondition *salinity;

  // value(s) associated with the condition
  PetscReal value;
} RDyCondition;

// This type defines a region consisting of cells identified by their local
// indices.
typedef struct {
  PetscInt   *cell_ids;
  PetscInt    num_cells;
} RDyRegion;

// This type defines a surface consisting of edges identified by their local
// indices.
typedef struct {
  PetscInt   *edge_ids;
  PetscInt    num_edges;
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
  // file storing input data for the simulation
  char config_file[PETSC_MAX_PATH_LEN];

  //---------
  // Physics
  //---------

  // flow and related parameterization(s)
  RDyFlowMode    flow_mode;
  RDyBedFriction bed_friction;
  PetscReal      bed_friction_coef;

  // sediment
  PetscBool sediment;

  // salinity
  PetscBool salinity;


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
  PetscInt  num_regions;
  PetscInt  region_ids[MAX_NUM_REGIONS];
  RDyRegion regions[MAX_NUM_REGIONS];

  // mesh surfaces
  PetscInt   num_surfaces;
  PetscInt   surface_ids[MAX_NUM_REGIONS];
  RDySurface surfaces[MAX_NUM_SURFACES];

  // Table of named sets of flow, sediment, and salinity conditions. Used by
  // initial/source/boundary conditions below.
  PetscInt             num_flow_conditions;
  RDyFlowCondition     flow_conditions[MAX_NUM_FLOW_CONDITIONS];
  PetscInt             num_sediment_conditions;
  RDySedimentCondition sediment_conditions[MAX_NUM_SEDIMENT_CONDITIONS];
  PetscInt             num_salinity_conditions;
  RDySalinityCondition salinity_conditions[MAX_NUM_SALINITY_CONDITIONS];

  // initial conditions (either a filename or a set of conditions associated
  // with mesh regions (1 per region)
  char         initial_conditions_file[PETSC_MAX_PATH_LEN];
  RDyCondition initial_conditions[MAX_NUM_REGIONS];

  // sources (and sinks) associated with mesh regions (1 per region)
  RDyCondition sources[MAX_NUM_REGIONS];

  // boundary conditions associated with mesh surfaces (1 per surface)
  RDyCondition boundary_conditions[MAX_NUM_SURFACES];

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
  char  log_file[PETSC_MAX_PATH_LEN];
  FILE *log;

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

// write a message to the primary log -- no need for PetscCall
#define RDyLog(rdy, ...) PetscCall(PetscFPrintf(rdy->comm, rdy->log, __VA_ARGS__))

#endif
