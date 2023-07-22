#include <private/rdycoreimpl.h>
#include <private/rdymemoryimpl.h>
#include <rdycore.h>

static PetscBool initialized_ = PETSC_FALSE;

/// Initializes a process for use by RDycore. Call this at the beginning of
/// your program.
PetscErrorCode RDyInit(int argc, char *argv[], const char *help) {
  PetscFunctionBegin;
  if (!initialized_) {
    PetscBool petsc_initialized;
    PetscCall(PetscInitialized(&petsc_initialized));
    if (!petsc_initialized) {
      PetscCall(PetscInitialize(&argc, &argv, (char *)0, (char *)help));
    }
    initialized_ = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/// Initializes the RDycore library for use with Fortran. It's used by the
/// Fortran interface.
PetscErrorCode RDyInitFortran(void) {
  PetscFunctionBegin;
  if (!initialized_) {
    PetscBool petsc_initialized;
    PetscCall(PetscInitialized(&petsc_initialized));
    if (!petsc_initialized) {
      PetscCall(PetscInitializeNoArguments());
      // no need for PetscInitializeFortran because PetscInitialize is
      // called before this function in the rdycore Fortran module.
    }
    initialized_ = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

// Functions called at shutdown. This can be used by all subsystems via
// RDyOnFinalize().
typedef void (*ShutdownFunc)(void);
static ShutdownFunc *shutdown_funcs_     = NULL;
static int           num_shutdown_funcs_ = 0;
static int           shutdown_funcs_cap_ = 0;

/// Call this to register a shutdown function that is called during TDyFinalize.
PetscErrorCode RDyOnFinalize(void (*shutdown_func)(void)) {
  PetscFunctionBegin;
  if (shutdown_funcs_ == NULL) {
    shutdown_funcs_cap_ = 32;
    PetscCall(RDyAlloc(ShutdownFunc, shutdown_funcs_cap_, &shutdown_funcs_));
  } else if (num_shutdown_funcs_ == shutdown_funcs_cap_) {  // need more space!
    shutdown_funcs_cap_ *= 2;
    PetscCall(RDyRealloc(ShutdownFunc, shutdown_funcs_cap_, &shutdown_funcs_));
  }
  shutdown_funcs_[num_shutdown_funcs_] = shutdown_func;
  ++num_shutdown_funcs_;
  PetscFunctionReturn(0);
}

/// Shuts down a process in which RDyInit or RDyInitNotArguments was called.
/// (Has no effect otherwise.)
PetscErrorCode RDyFinalize(void) {
  PetscFunctionBegin;

  // Call shutdown functions in reverse order, and destroy the list.
  if (shutdown_funcs_ != NULL) {
    for (int i = num_shutdown_funcs_ - 1; i >= 0; --i) {
      shutdown_funcs_[i]();
    }
    RDyFree(shutdown_funcs_);
  }

  PetscFinalize();

  initialized_ = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/// Returns PETSC_TRUE if the RDyCore library has been initialized, PETSC_FALSE
/// otherwise.
PetscBool RDyInitialized(void) { return initialized_; }

/// Creates a new RDy object representing an RDycore simulation context.
/// @param comm        [in] the MPI communicator used by the simulation
/// @param config_file [in] a path to a configuration (.yaml) file
/// @param rdy         [out] a pointer that stores the newly created RDy.
PetscErrorCode RDyCreate(MPI_Comm comm, const char *config_file, RDy *rdy) {
  PetscFunctionBegin;

  PetscCall(PetscNew(rdy));

  // MPI comm stuff
  (*rdy)->comm = comm;
  MPI_Comm_rank(comm, &((*rdy)->rank));
  MPI_Comm_size(comm, &((*rdy)->nproc));

  // set the config file
  strncpy((*rdy)->config_file, config_file, PETSC_MAX_PATH_LEN);

  PetscFunctionReturn(0);
}

// fortran 90 version of RDyCreate
PetscErrorCode RDyCreateF90(MPI_Fint *f90_comm, const char *config_file, RDy *rdy) {
  PetscFunctionBegin;

  PetscCall(RDyCreate(MPI_Comm_f2c(*f90_comm), config_file, rdy));

  PetscFunctionReturn(0);
}

/// Destroys the given RDy object, freeing any allocated resources.
/// @param rdy [out] a pointer to the RDy object to be destroyed.
PetscErrorCode RDyDestroy(RDy *rdy) {
  PetscFunctionBegin;

  // destroy FV mesh
  if ((*rdy)->mesh.num_cells) RDyMeshDestroy((*rdy)->mesh);

  // destroy conditions
  if ((*rdy)->initial_conditions) RDyFree((*rdy)->initial_conditions);
  if ((*rdy)->sources) RDyFree((*rdy)->sources);
  if ((*rdy)->boundary_conditions) RDyFree((*rdy)->boundary_conditions);

  // destroy materials
  if ((*rdy)->materials_by_cell) RDyFree((*rdy)->materials_by_cell);
  if ((*rdy)->materials) RDyFree((*rdy)->materials);

  // destroy regions and boundaries
  for (PetscInt i = 0; i < (*rdy)->num_regions; ++i) {
    if ((*rdy)->regions[i].cell_ids) {
      RDyFree((*rdy)->regions[i].cell_ids);
    }
  }
  if ((*rdy)->region_ids) RDyFree((*rdy)->region_ids);
  if ((*rdy)->regions) RDyFree((*rdy)->regions);

  for (PetscInt i = 0; i < (*rdy)->num_boundaries; ++i) {
    if ((*rdy)->boundaries[i].edge_ids) {
      RDyFree((*rdy)->boundaries[i].edge_ids);
    }
  }
  if ((*rdy)->boundary_ids) RDyFree((*rdy)->boundary_ids);
  if ((*rdy)->boundaries) RDyFree((*rdy)->boundaries);

  // destroy solver
  if ((*rdy)->ts) TSDestroy(&((*rdy)->ts));

  // destroy vectors
  if ((*rdy)->water_src) VecDestroy(&((*rdy)->water_src));
  if ((*rdy)->R) VecDestroy(&((*rdy)->R));
  if ((*rdy)->X) VecDestroy(&((*rdy)->X));
  if ((*rdy)->X_local) VecDestroy(&((*rdy)->X_local));

  // destroy time series
  PetscCall(DestroyTimeSeries(rdy));

  // destroy DMs
  if ((*rdy)->aux_dm) DMDestroy(&((*rdy)->aux_dm));
  if ((*rdy)->dm) DMDestroy(&((*rdy)->dm));

  // destroy libCEED parts if they exist
  CeedOperatorDestroy(&(*rdy)->ceed_rhs.op);
  CeedVectorDestroy(&(*rdy)->ceed_rhs.x_ceed);
  CeedVectorDestroy(&(*rdy)->ceed_rhs.y_ceed);

  // close the log file if needed
  if (((*rdy)->log) && ((*rdy)->log != stdout)) {
    PetscCall(PetscFClose((*rdy)->comm, (*rdy)->log));
  }

  PetscCall(RDyFree(*rdy));
  *rdy = NULL;
  PetscFunctionReturn(0);
}
