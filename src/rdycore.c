#include <private/rdycoreimpl.h>
#include <private/rdymemory.h>
#include <rdycore.h>

static PetscBool initialized_ = PETSC_FALSE;

/// Initializes a process for use by RDycore. Call this at the beginning of
/// your program
PetscErrorCode RDyInit(int argc, char *argv[], const char *help) {
  PetscFunctionBegin;
  if (!initialized_) {
    PetscCall(PetscInitialize(&argc, &argv, (char *)0, (char *)help));
    initialized_ = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/// Initializes the RDycore library without arguments. It's used by the Fortran
/// interface, which calls PetscInitialize itself and then this function.
PetscErrorCode RDyInitNoArguments(void) {
  PetscFunctionBegin;
  if (!initialized_) {
    PetscCall(PetscInitializeNoArguments());
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

/// Destroys the given RDy object, freeing any allocated resources.
/// @param rdy [out] a pointer to the RDy object to be destroyed.
PetscErrorCode RDyDestroy(RDy *rdy) {
  PetscFunctionBegin;

  // Destroy tables of named flow/sediment/salinity conditions.
  // NOTE: we can make destructors for these things if they get more complex
  for (PetscInt i = 0; i < (*rdy)->num_flow_conditions; ++i) {
    RDyFree((*rdy)->flow_conditions[i].name);
  }
  for (PetscInt i = 0; i < (*rdy)->num_sediment_conditions; ++i) {
    RDyFree((*rdy)->sediment_conditions[i].name);
  }
  for (PetscInt i = 0; i < (*rdy)->num_salinity_conditions; ++i) {
    RDyFree((*rdy)->salinity_conditions[i].name);
  }

  // Destroy regions and surfaces.
  for (PetscInt i = 0; i < MAX_NUM_REGIONS; ++i) {
    RDyFree((*rdy)->regions[i].cell_ids);
  }
  for (PetscInt i = 0; i < MAX_NUM_SURFACES; ++i) {
    RDyFree((*rdy)->surfaces[i].edge_ids);
  }

  if ((*rdy)->dm) {
    DMDestroy(&((*rdy)->dm));
  }

  if (((*rdy)->log) && ((*rdy)->log != stdout)) {
    PetscCall(PetscFClose((*rdy)->comm, (*rdy)->log));
  }

  PetscCall(RDyFree(*rdy));
  *rdy = NULL;
  PetscFunctionReturn(0);
}

