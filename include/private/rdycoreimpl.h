#ifndef RDYCOREIMPL_H
#define RDYCOREIMPL_H

#include <ceed/ceed.h>
#include <petsc/private/petscimpl.h>
#include <private/rdyboundaryimpl.h>
#include <private/rdyconfigimpl.h>
#include <private/rdydmimpl.h>
#include <private/rdylogimpl.h>
#include <private/rdymeshimpl.h>
#include <private/rdyoperatorimpl.h>
#include <private/rdyregionimpl.h>
#include <rdycore.h>

// CEED initialization, availability, context, useful for creating CEED
// sub-operators
PETSC_INTERN PetscErrorCode SetCeedResource(char *);
PETSC_INTERN PetscBool      CeedEnabled(void);
PETSC_INTERN Ceed           CeedContext(void);
PETSC_INTERN PetscErrorCode GetCeedVecType(VecType *);

// This type keeps track of accumulated time series data appended periodically
// to files.
typedef struct {
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

    // last step for which boundary flux time series data was written
    PetscInt last_step;
  } boundary_fluxes;

  // observations recorded at specific sites
  struct {
    struct {
      // global indices corresponding to observation site natural cell indices in input
      PetscInt *global_indices;
      // fixed x, y, z coordinates of observation sites
      PetscReal *x, *y, *z;
      // serial vector containing data for all observation sites on process 0
      Vec u;
    } sites;
    // VecScatter governing observation site vector scatter
    VecScatter scatter_u;
    // accumulation vector for averaged observations or for instantaneous values
    Vec accum_u;
    // last step for which observations data was written
    PetscInt last_step;
  } observations;
} RDyTimeSeriesData;

typedef struct {
  // base mesh before any refinements
  DM dm_base;
  DM dm_1dof_base;

  // for marking cells for refinement
  PetscBool  cells_marked_for_refinement;
  PetscBool *refine_cell;

  // for mapping data between base and current meshes
  Mat BaseToCurrentMatNDof, CurrentToBaseMatNDof;
  Mat BaseToCurrentMat1Dof, CurrentToBaseMat1Dof;

  // identifies if the mesh has been refined and the model has not taken a step
  PetscBool mesh_was_refined;

  // number of times the mesh has been refined
  int num_refinements;

  // last refined mesh that was outputted
  int last_refinement_level_outputted;

  // is refinement on or off
  PetscBool is_refinement_on;
} RDyAMR;

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

// class ID for PETSc logging events
extern PetscClassId RDY_CLASSID;

// an application context that stores data relevant to a simulation
struct _p_RDy {
  PETSCHEADER(struct _RDyOps);

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
  DM               dm;
  SectionFieldSpec soln_fields;
  PetscBool        refine;

  // auxiliary DM for diagnostics
  DM               dm_diags;
  SectionFieldSpec field_diags;
  Vec              vec_diags;

  DM               dm_1dof;
  SectionFieldSpec field_1dof;
  Vec              vec_1dof;

  // DM for sediment dynamics
  DM               sediment_dm;
  SectionFieldSpec sediment_fields;
  Vec              sediment_u_global, sediment_u_local;

  // DM for flow
  DM               flow_dm;
  SectionFieldSpec flow_fields;
  Vec              flow_u_global, flow_u_local;

  // mesh representing simulation domain
  RDyMesh mesh;

  // mesh regions
  PetscInt   num_regions;
  RDyRegion *regions;

  // mesh boundaries
  PetscInt     num_boundaries;
  RDyBoundary *boundaries;

  // sediment size classes
  PetscInt num_sediment_classes;

  // initial conditions associated with mesh regions (1 per region)
  RDyCondition *initial_conditions;

  // sources (and sinks) associated with mesh regions (1 per region)
  RDyCondition *sources;

  // boundary conditions associated with mesh boundaries (1 per boundary)
  RDyCondition *boundary_conditions;

  // log file handle
  FILE *log;

  //--------------------------
  // Refinements
  //--------------------------

  RDyAMR amr;

  //--------------------------
  // Restart
  //--------------------------
  PetscBool is_a_restart_run;  // is this a restart run?
  PetscInt  restart_step;

  //--------------------------
  // Solver and solution data
  //--------------------------

  // fixed time step size (if specified)
  PetscReal dt;

  // time₋stepping solver
  TS ts;

  // host solution vectors (global and local)
  Vec u_global, u_local;

  // host right-hand-side (residual) vector
  Vec rhs;

  // for imex methods
  Vec udot_global, udot_local;
  Mat ijacobian;

  // operator representing the system of equations
  Operator *operator;

  // time series bookkeeping
  RDyTimeSeriesData time_series;

  //-------------------
  // Simulatіon output
  //-------------------
  PetscViewer           output_viewer;
  PetscViewerAndFormat *output_vf;
};

// these are used by both the main (RDycore) driver and the MMS driver
PETSC_INTERN PetscErrorCode DetermineConfigPrefix(RDy, char *);
PETSC_INTERN PetscErrorCode ReadConfigFile(RDy);     // for RDycore driver only!
PETSC_INTERN PetscErrorCode ReadMMSConfigFile(RDy);  // for MMS driver only!
PETSC_INTERN PetscErrorCode InitBoundaries(RDy);
PETSC_INTERN PetscErrorCode InitRegions(RDy);
PETSC_INTERN PetscErrorCode OverrideParameters(RDy);
PETSC_INTERN PetscErrorCode PrintConfig(RDy);
PETSC_INTERN PetscErrorCode DestroyConfig(RDy);

PETSC_INTERN PetscErrorCode RDyDestroyVectors(RDy *);
PETSC_INTERN PetscErrorCode RDyDestroyBoundaries(RDy *);

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
PETSC_INTERN PetscErrorCode WriteTimeSeries(TS, PetscInt, PetscReal, Vec, void *);
PETSC_INTERN PetscErrorCode DestroyTimeSeries(RDy);

// utility functions
PETSC_INTERN const char *TimeUnitAsString(RDyTimeUnit);
PETSC_INTERN PetscReal   ConvertTimeToSeconds(PetscReal, RDyTimeUnit);
PETSC_INTERN PetscReal   ConvertTimeFromSeconds(PetscReal, RDyTimeUnit);

#endif
