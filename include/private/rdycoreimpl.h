#ifndef RDYCOREIMPL_H
#define RDYCOREIMPL_H

#include <ceed/ceed.h>
#include <petsc/private/petscimpl.h>
#include <private/rdyconfigimpl.h>
#include <private/rdylogimpl.h>
#include <private/rdymeshimpl.h>
#include <rdycore.h>

// Diagnostic structure that captures information about the conditions under
// which the maximum courant number is encountered. If you change this struct,
// update the call to MPI_Type_create_struct in InitMPITypesAndOps below.
typedef struct {
  PetscReal max_courant_num;  // maximum courant number
  PetscInt  global_edge_id;   // edge at which the max courant number was encountered
  PetscInt  global_cell_id;   // cell in which the max courant number was encountered
  PetscBool is_set;           // true if max_courant_num is set, otherwise false
} CourantNumberDiagnostics;

// This type defines a material with specific properties.
// (undefined properties are set to INVALID_INT/INVALID_REAL)
typedef struct {
  char      name[MAX_NAME_LEN + 1];
  PetscReal manning;  // Manning's coefficient [s/m**(1/3)]
} RDyMaterial;

// This type defines a region consisting of cells identified by their local
// indices.
typedef struct {
  char      name[MAX_NAME_LEN + 1];  // region name
  PetscInt  id;                      // region ID (as specified in mesh file)
  PetscInt *cell_ids;
  PetscInt  num_cells;
} RDyRegion;

// This type defines a boundary consisting of edges identified by their local
// indices.
typedef struct {
  char      name[MAX_NAME_LEN + 1];  // boundary name
  PetscInt  id;                      // boundary ID (as specified in mesh file)
  PetscInt  index;                   // index of boundary within RDycore boundary list
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
    // per-process numbers of local boundary edges on which fluxes are accumulated
    PetscInt *num_local_edges;
    // number of global boundary edges on which fluxes are accumulated
    PetscInt num_global_edges;
    // global flux metadata (global edge ID, boundary ID, BC type)
    PetscInt *global_flux_md;
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
  // global MPI communicator, used for ensemble analysis (equivalent to comm for
  // single simulations)
  MPI_Comm global_comm;
  // MPI rank of local process
  PetscMPIInt rank;
  // number of processes in the communicator
  PetscMPIInt nproc;
  // file storing input data for the simulation
  char config_file[PETSC_MAX_PATH_LEN];

  // index of the ensemble member for the local process
  PetscInt ensemble_member_index;

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
  RDyRegion *regions;

  // mesh boundaries
  PetscInt     num_boundaries;
  RDyBoundary *boundaries;

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

  // residual vector
  Vec R;

  // vector to store fluxes through internal and boundary edges
  Vec F_dup;

  // source-sink vector
  Vec swe_src;

  // time series bookkeeping
  RDyTimeSeriesData time_series;

  CourantNumberDiagnostics courant_num_diags;

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
    CeedOperator op_edges;
    CeedOperator op_src;
    CeedVector   u_local_ceed, u_ceed;
    CeedVector   f_ceed, s_ceed;
    CeedScalar   dt;
  } ceed_rhs;

  // Non-CEED solver data (must be cast to e. g. PetscRiemannDataSWE*)
  void *petsc_rhs;
};

// these are used by both the main (RDycore) driver and the MMS driver
PETSC_INTERN PetscErrorCode DetermineConfigPrefix(RDy, char *);
PETSC_INTERN PetscErrorCode ReadConfigFile(RDy);     // for RDycore driver only!
PETSC_INTERN PetscErrorCode ReadMMSConfigFile(RDy);  // for MMS driver only!
PETSC_INTERN PetscErrorCode InitBoundaries(RDy);
PETSC_INTERN PetscErrorCode InitRegions(RDy);
PETSC_INTERN PetscErrorCode OverrideParameters(RDy);
PETSC_INTERN PetscErrorCode PrintConfig(RDy);

// shallow water equations functions
PETSC_INTERN PetscErrorCode InitSWE(RDy);
PETSC_INTERN PetscErrorCode RHSFunctionSWE(TS, PetscReal, Vec, Vec, void *);

// output functions
PETSC_INTERN PetscErrorCode GetOutputDirectory(RDy, char dir[PETSC_MAX_PATH_LEN]);
PETSC_INTERN PetscErrorCode GenerateIndexedFilename(const char *, const char *, PetscInt, PetscInt, const char *, char *);
PETSC_INTERN PetscErrorCode DetermineOutputFile(RDy, PetscInt, PetscReal, const char *, char *);
PETSC_INTERN PetscErrorCode WriteXDMFOutput(TS, PetscInt, PetscReal, Vec, void *);

// checkpoint/restart functions
PETSC_INTERN PetscErrorCode InitCheckpoints(RDy);
PETSC_INTERN PetscErrorCode ReadCheckpointFile(RDy, const char *);

// time series
PETSC_INTERN PetscErrorCode InitTimeSeries(RDy);
PETSC_INTERN PetscErrorCode AccumulateBoundaryFluxes(RDy, RDyBoundary boundary, PetscInt size, PetscInt ndof, PetscReal *);
PETSC_INTERN PetscErrorCode WriteTimeSeries(TS, PetscInt, PetscReal, Vec, void *);
PETSC_INTERN PetscErrorCode DestroyTimeSeries(RDy);

// utility functions
PETSC_INTERN const char *TimeUnitAsString(RDyTimeUnit);
PETSC_INTERN PetscReal   ConvertTimeToSeconds(PetscReal, RDyTimeUnit);
PETSC_INTERN PetscReal   ConvertTimeFromSeconds(PetscReal, RDyTimeUnit);

#endif
