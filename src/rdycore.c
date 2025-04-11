#include <petscdmceed.h>
#include <private/rdycoreimpl.h>
#include <private/rdyoperatorimpl.h>
#include <private/rdysweimpl.h>
#include <rdycore.h>

static PetscBool initialized_ = PETSC_FALSE;
PetscClassId     RDY_CLASSID;

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

    // set up our logging class ID
    PetscCall(PetscClassIdRegister("RDycore", &RDY_CLASSID));

    // initialize our Courant number diagnostics MPI datatype / operator
    PetscCall(InitCourantNumberDiagnostics());

    initialized_ = PETSC_TRUE;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
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

    // set up our logging class ID
    PetscCall(PetscClassIdRegister("RDycore", &RDY_CLASSID));

    // initialize our Courant number diagnostics MPI datatype / operator
    PetscCall(InitCourantNumberDiagnostics());

    initialized_ = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
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
    PetscCall(PetscCalloc1(shutdown_funcs_cap_, &shutdown_funcs_));
  } else if (num_shutdown_funcs_ == shutdown_funcs_cap_) {  // need more space!
    shutdown_funcs_cap_ *= 2;
    PetscCall(PetscRealloc(sizeof(ShutdownFunc) * shutdown_funcs_cap_, &shutdown_funcs_));
  }
  shutdown_funcs_[num_shutdown_funcs_] = shutdown_func;
  ++num_shutdown_funcs_;
  PetscFunctionReturn(PETSC_SUCCESS);
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
    PetscFree(shutdown_funcs_);
  }

  PetscFinalize();

  initialized_ = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Returns PETSC_TRUE if the RDyCore library has been initialized, PETSC_FALSE
/// otherwise.
PetscBool RDyInitialized(void) { return initialized_; }

/// Creates a new RDy object representing an RDycore simulation context.
/// @param comm        [in] the MPI communicator used by the simulation or ensemble
/// @param config_file [in] a path to a configuration (.yaml) file
/// @param rdy         [out] a pointer that stores the newly created RDy.
PetscErrorCode RDyCreate(MPI_Comm comm, const char *config_file, RDy *rdy) {
  PetscFunctionBegin;

  PetscCall(PetscNew(rdy));

  // MPI comm stuff
  (*rdy)->global_comm = comm;
  MPI_Comm_rank(comm, &((*rdy)->rank));
  MPI_Comm_size(comm, &((*rdy)->nproc));
  MPI_Comm_dup((*rdy)->global_comm, &((*rdy)->comm));

  // set the config file
  strncpy((*rdy)->config_file, config_file, PETSC_MAX_PATH_LEN);

  PetscFunctionReturn(PETSC_SUCCESS);
}

// fortran 90 version of RDyCreate
PetscErrorCode RDyCreateF90(MPI_Fint *f90_comm, const char *config_file, RDy *rdy) {
  PetscFunctionBegin;

  PetscCall(RDyCreate(MPI_Comm_f2c(*f90_comm), config_file, rdy));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// returns PETSC_TRUE if the dycore was restarted from a checkpoint file,
/// PETSC_FALSE if not
PetscBool RDyRestarted(RDy rdy) { return rdy->config.restart.file[0]; }

/// @brief Destroys PETSc Vecs associated with the DM
/// @param rdy A RDy struct
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode RDyDestroyVectors(RDy *rdy) {
  PetscFunctionBegin;
  // destroy vectors
  if ((*rdy)->rhs) PetscCall(VecDestroy(&((*rdy)->rhs)));
  if ((*rdy)->u_global) PetscCall(VecDestroy(&((*rdy)->u_global)));
  if ((*rdy)->u_local) PetscCall(VecDestroy(&((*rdy)->u_local)));
  if ((*rdy)->diags_vec) PetscCall(VecDestroy(&(*rdy)->diags_vec));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Destroys a region data structure.
PetscErrorCode DestroyRegion(RDyRegion *region) {
  PetscFunctionBegin;

  if (region->owned_cell_global_ids) {
    PetscFree(region->owned_cell_global_ids);
  }
  if (region->cell_local_ids) {
    PetscFree(region->cell_local_ids);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Destroy boundary data structures
/// @param rdy A RDy struct
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode RDyDestroyBoundaries(RDy *rdy) {
  PetscFunctionBegin;

  for (PetscInt i = 0; i < (*rdy)->num_boundaries; ++i) {
    if ((*rdy)->boundaries[i].edge_ids) {
      PetscFree((*rdy)->boundaries[i].edge_ids);
    }
  }
  if ((*rdy)->boundaries) PetscFree((*rdy)->boundaries);
  (*rdy)->num_boundaries = 0;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Destroys the given RDy object, freeing any allocated resources.
/// @param rdy [out] a pointer to the RDy object to be destroyed.
PetscErrorCode RDyDestroy(RDy *rdy) {
  PetscFunctionBegin;

  // destroy FV mesh
  if ((*rdy)->mesh.num_cells) {
    RDyMeshDestroy((*rdy)->mesh);
  }

  // destroy conditions
  if ((*rdy)->initial_conditions) PetscFree((*rdy)->initial_conditions);
  if ((*rdy)->sources) PetscFree((*rdy)->sources);
  if ((*rdy)->boundary_conditions) PetscFree((*rdy)->boundary_conditions);

  // destroy regions
  for (PetscInt r = 0; r < (*rdy)->num_regions; ++r) {
    PetscCall(DestroyRegion(&((*rdy)->regions[r])));
  }
  PetscFree((*rdy)->regions);

  // destroy materials
  PetscCall(RDyDestroyBoundaries(rdy));

  // destroy solver
  if ((*rdy)->ts) TSDestroy(&((*rdy)->ts));

  PetscCall(RDyDestroyVectors(rdy));

  if ((*rdy)->operator) {
    PetscCall(DestroyOperator(&(*rdy)->operator));
  }

  // destroy time series
  PetscCall(DestroyTimeSeries(*rdy));

  // destroy DMs
  if ((*rdy)->aux_dm) DMDestroy(&((*rdy)->aux_dm));
  if ((*rdy)->dm) DMDestroy(&((*rdy)->dm));

  // destroy config data
  PetscCall(DestroyConfig((*rdy)));

  // close the log file if needed
  if (((*rdy)->log) && ((*rdy)->log != stdout)) {
    PetscCall(PetscFClose((*rdy)->comm, (*rdy)->log));
  }

  if ((*rdy)->no_overlap_dm) PetscCall(DMDestroy(&(*rdy)->no_overlap_dm));

  MPI_Comm_free(&((*rdy)->comm));
  PetscCall(PetscFree(*rdy));
  *rdy = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
