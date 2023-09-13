#ifndef RDYCOREIMPL_H
#define RDYCOREIMPL_H

#include <ceed/ceed.h>
#include <petsc/private/petscimpl.h>
#include <private/rdyconfigimpl.h>
#include <private/rdylogimpl.h>
#include <private/rdymeshimpl.h>
#include <rdycore.h>

// This type defines a region consisting of cells identified by their local
// indices.
typedef struct {
  PetscInt *cell_ids;
  PetscInt  num_cells;
} RDyRegion;

// This type defines a boundary consisting of edges identified by their local
// indices.
typedef struct {
  PetscInt  id;  // boundary identifier
  PetscInt *edge_ids;
  PetscInt  num_edges;
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

  // was this boundary condition automatically generated and not explicitly
  // requested in the config file?
  PetscBool auto_generated;
} RDyCondition;

// This type keeps track of accumulated time series data appended periodically
// to files.
typedef struct {
  // last step for which time series data was written
  PetscInt last_step;
  // fluxes on boundary edges
  struct {
    // numbers of local and global boundary edges on which fluxes are computed
    PetscInt num_local_edges, num_global_edges;
    // array of per-boundary offsets in local fluxes array below
    PetscInt *offsets;
    // local array of boundary fluxes
    struct {
      PetscReal water_mass;
      PetscReal x_momentum;
      PetscReal y_momentum;
    } * fluxes;
  } boundary_fluxes;
} RDyTimeSeriesData;

// This type serves as a "virtual table" containing function pointers that
// define the behavior of the dycore.
typedef struct _RDyOps *RDyOps;
struct _RDyOps {
  // Called by RDyCreate to allocate implementation-specific resources, storing
  // the result in the given context pointer.
  PetscErrorCode (*create)(void **);
  // Called by RDyDestroy to free implementation-specific resources.
  PetscErrorCode (*destroy)(void *);
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
  DM        dm;
  PetscBool refine;

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
  RDyBoundary *boundaries;

  // materials associated with mesh regions (1 per region)
  RDyMaterial *materials;
  // materials associated with individual (local) cells
  RDyMaterial *materials_by_cell;

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

  // fixed time step size (if specified)
  PetscReal dt;

  // time₋stepping solver
  TS ts;

  // solution vectors (global and local)
  Vec X, X_local;
  Vec Soln;

  // residual vector
  Vec R;

  // source-sink vector
  Vec water_src;

  // time series bookkeeping
  RDyTimeSeriesData time_series;

  //-------------------
  // Simulatіon output
  //-------------------
  PetscViewer           output_viewer;
  PetscViewerAndFormat *output_vf;

  //--------------
  // CEED support
  //--------------
  Ceed ceed;
  char ceed_resource[PETSC_MAX_PATH_LEN];
  // RHS operator (optional)
  struct {
    PetscBool water_src_updated;

    CeedOperator op_edges;
    CeedOperator op_src;
    CeedVector   u_local_ceed, u_ceed;
    CeedVector   f_ceed, s_ceed;
    CeedScalar   dt;

  } ceed_rhs;

  // Non-CEED solver data (must be cast to e. g. PetscRiemannDataSWE*)
  void *petsc_rhs;
};

PETSC_INTERN PetscErrorCode ReadConfigFile(RDy);
PETSC_INTERN PetscErrorCode PrintConfig(RDy);

// shallow water equations functions
PETSC_INTERN PetscErrorCode InitSWE(RDy);
PETSC_INTERN PetscErrorCode RHSFunctionSWE(TS, PetscReal, Vec, Vec, void *);

// output functions
PETSC_INTERN PetscErrorCode CreateOutputDir(RDy);
PETSC_INTERN PetscErrorCode GetOutputDir(RDy, char dir[PETSC_MAX_PATH_LEN]);
PETSC_INTERN PetscErrorCode DetermineOutputFile(RDy, PetscInt, PetscReal, const char *, char *);
PETSC_INTERN PetscErrorCode WriteXDMFOutput(TS, PetscInt, PetscReal, Vec, void *);

// time series
PETSC_INTERN PetscErrorCode InitTimeSeries(RDy);
PETSC_INTERN PetscErrorCode AccumulateBoundaryFluxes(RDy, RDyBoundary *, PetscInt ne, PetscReal[ne][3]);
PETSC_INTERN PetscErrorCode WriteTimeSeries(TS, PetscInt, PetscReal, Vec, void *);
PETSC_INTERN PetscErrorCode DestroyTimeSeries(RDy);

// utility functions
PETSC_INTERN const char *TimeUnitAsString(RDyTimeUnit);
PETSC_INTERN PetscReal   ConvertTimeToSeconds(PetscReal, RDyTimeUnit);
PETSC_INTERN PetscReal   ConvertTimeFromSeconds(PetscReal, RDyTimeUnit);

#endif
