#include <private/rdycoreimpl.h>
#include <rdycore.h>

static PetscBool initialized_ = PETSC_FALSE;

static void FinalizeMPI(void) {
  MPI_Finalize();
}

// Initializes MPI and adds a call to MPI_Finalize() on process exit.
// If MPI has been initialized prior to this call, it has no effect.
static PetscErrorCode InitMPI(int *argc, char ***argv) {
  PetscFunctionBegin;
  int mpi_initialized;
  PetscCall(MPI_Initialized(&mpi_initialized));
  if (!mpi_initialized) {
    PetscCall(MPI_Init(argc, argv));
    atexit(FinalizeMPI);
  }
  PetscFunctionReturn(0);
}

/// Initializes a process for use by RDycore. Call this at the beginning of
/// your program
PetscErrorCode RDyInit(int argc, char *argv[], const char *help) {
  PetscFunctionBegin;
  // We initialize MPI before PETSc to make sure that PETSc doesn't finalize
  // MPI. This allows us to run several MPI-related unit tests in the same
  // process.
  PetscCall(InitMPI(argc, argv));

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

  // We initialize MPI before PETSc to make sure that PETSc doesn't finalize
  // MPI. This allows us to run several MPI-related unit tests in the same
  // process.
  PetscCall(InitMPI(NULL, NULL));

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
/// @param comm [in] the MPI communicator used by the simulation
/// @param rdy  [out] a pointer that stores the newly created RDy.
PetscErrorCode RDyCreate(MPI_Comm comm, RDy *rdy) {
  PetscFunctionBegin;

  PetscCall(PetscNew(rdy));
  PetscFunctionReturn(0);
}

/// Configures the given RDy object with options supplied on the command line.
PetscErrorCode RDySetFromOptions(RDy rdy) {
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/// Performs any setup needed by RDy after it has been configured.
PetscErrorCode RDySetup(RDy rdy) {
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode RDyDestroy(RDy *rdy) {
  PetscFunctionBegin;
  PetscCall(RDyFree(*rdy));
  *rdy = NULL;
  PetscFunctionReturn(0);
}
