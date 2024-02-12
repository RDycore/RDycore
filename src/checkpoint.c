#include <errno.h>
#include <petscviewerhdf5.h>
#include <private/rdycoreimpl.h>
#include <sys/stat.h>
#include <sys/types.h>

// checkpoint directory name (relative to current working directory)
static const char *checkpoint_dir = "checkpoints";

// creates the checkpoint directory if it doesn't exist
static PetscErrorCode CreateCheckpointDir(RDy rdy) {
  PetscFunctionBegin;

  RDyLogDebug(rdy, "Creating output directory %s...", checkpoint_dir);

  PetscMPIInt result_and_errno[2];
  if (rdy->rank == 0) {
    result_and_errno[0] = mkdir(checkpoint_dir, 0755);
    result_and_errno[1] = errno;
  }
  MPI_Bcast(&result_and_errno, 2, MPI_INT, 0, rdy->comm);
  int result = result_and_errno[0];
  int err_no = result_and_errno[1];
  PetscCheck((result == 0) || (err_no == EEXIST), rdy->comm, PETSC_ERR_USER, "Could not create checkpoint directory: %s (errno = %d)", checkpoint_dir,
             err_no);

  PetscFunctionReturn(PETSC_SUCCESS);
}

// this struct contains metadata that is written to checkpoint files
typedef struct {
  PetscInt  nproc;  // number of MPI processes / tasks
  PetscReal t;      // simulation time
  PetscReal dt;     // timestep
} CheckpointMetadata;

// creates a new PetscBag that can store checkpoint metadata, populated with
// data from rdy
static PetscErrorCode CreateMetadata(RDy rdy, PetscBag *bag) {
  PetscFunctionBegin;

  CheckpointMetadata *metadata;
  PetscCall(PetscBagCreate(rdy->comm, sizeof(CheckpointMetadata), bag));
  PetscCall(PetscBagGetData(*bag, (void **)&metadata));
  PetscCall(PetscBagSetName(*bag, "metadata", "Checkpoint metadata"));
  PetscCall(PetscBagRegisterInt(*bag, &metadata->nproc, rdy->nproc, "nproc", "Number of MPI tasks"));
  PetscReal t;
  PetscCall(TSGetTime(rdy->ts, &t));
  PetscCall(PetscBagRegisterReal(*bag, &metadata->t, t, "t", "Simulation time"));
  PetscCall(PetscBagRegisterReal(*bag, &metadata->dt, rdy->dt, "dt", "Simulation timestep"));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// reads checkpoint metadata into rdy from the given PetscBag and destroys it
static PetscErrorCode ConsumeMetadata(RDy rdy, PetscBag bag) {
  PetscFunctionBegin;

  CheckpointMetadata *metadata;
  PetscCall(PetscBagGetData(bag, (void **)&metadata));
  rdy->nproc = metadata->nproc;
  rdy->dt    = metadata->dt;
  if (rdy->config.restart.continue_run) {
    PetscCall(TSSetTime(rdy->ts, metadata->t));
  } else {
    PetscCall(TSSetTime(rdy->ts, 0.0));
  }
  PetscCall(TSSetTimeStep(rdy->ts, rdy->dt));

  PetscCall(PetscBagDestroy(&bag));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode WriteCheckpoint(TS ts, PetscInt step, PetscReal time, Vec X, void *ctx) {
  PetscFunctionBegin;
  RDy rdy = ctx;
  if (step % rdy->config.checkpoint.interval == 0) {
    // determine an appropriate prefix for checkpoint files
    char prefix[PETSC_MAX_PATH_LEN], filename[PETSC_MAX_PATH_LEN];
    PetscCall(DetermineConfigPrefix(rdy, prefix));

    PetscViewer             viewer;
    const PetscViewerFormat format = rdy->config.checkpoint.format;
    if (format == PETSC_VIEWER_NATIVE) {  // binary
      PetscCall(GenerateIndexedFilename(checkpoint_dir, prefix, step, rdy->config.time.max_step, "bin", filename));
      PetscCall(PetscViewerBinaryOpen(rdy->comm, filename, FILE_MODE_WRITE, &viewer));
    } else {  // HDF5
      PetscCall(GenerateIndexedFilename(checkpoint_dir, prefix, step, rdy->config.time.max_step, "h5", filename));
      PetscCall(PetscViewerHDF5Open(rdy->comm, filename, FILE_MODE_WRITE, &viewer));
    }

    RDyLogInfo(rdy, "Writing checkpoint file %s...", filename);

    PetscBag bag;
    PetscCall(CreateMetadata(rdy, &bag));
    PetscCall(PetscBagView(bag, viewer));
    PetscCall(VecView(X, viewer));
    PetscCall(PetscBagDestroy(&bag));

    PetscCall(PetscViewerDestroy(&viewer));
    RDyLogInfo(rdy, "Finished writing checkpoint file.");
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Initializes the periodic writing of checkpoint files as requested in the
// YAML config file.
PetscErrorCode InitCheckpoints(RDy rdy) {
  PetscFunctionBegin;
  if (rdy->config.checkpoint.interval) {
    // make sure the checkpoint directory exists
    PetscCall(CreateCheckpointDir(rdy));

    const PetscViewerFormat format = rdy->config.checkpoint.format;
    PetscCheck((format == PETSC_VIEWER_NATIVE) || (format == PETSC_VIEWER_HDF5_PETSC), rdy->comm, PETSC_ERR_USER, "Invalid checkpoint format!");
    PetscCall(TSMonitorSet(rdy->ts, WriteCheckpoint, rdy, NULL));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// reads the checkpoint file with the given name, populating the state of the
// given RDy object
PetscErrorCode ReadCheckpointFile(RDy rdy, const char *filename) {
  PetscFunctionBegin;

  RDyLogInfo(rdy, "Reading checkpoint file %s...", filename);

  // determine the file's format from its suffix (it need not match the
  // format specified in the checkpoint section), and open an appropriate
  // viewer
  PetscViewer viewer;
  if (strstr(filename, ".bin")) {  // binary
    PetscCall(PetscViewerBinaryOpen(rdy->comm, filename, FILE_MODE_READ, &viewer));
  } else if (strstr(filename, ".h5")) {  // HDF5
    PetscCall(PetscViewerHDF5Open(rdy->comm, filename, FILE_MODE_READ, &viewer));
  } else {
    PetscCheck(PETSC_FALSE, rdy->comm, PETSC_ERR_USER, "Invalid checkpoint file: %s", filename);
  }

  PetscBag bag;
  PetscCall(CreateMetadata(rdy, &bag));
  PetscCall(PetscBagLoad(viewer, bag));
  PetscCall(ConsumeMetadata(rdy, bag));
  PetscCall(VecLoad(rdy->X, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  RDyLogInfo(rdy, "Finished reading checkpoint file.");

  // read the newly loaded solution vector into the timestepper
  PetscCall(TSSetSolution(rdy->ts, rdy->X));

  PetscFunctionReturn(PETSC_SUCCESS);
}
