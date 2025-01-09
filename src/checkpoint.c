#include <errno.h>
#include <petscviewerhdf5.h>
#include <private/rdycoreimpl.h>
#include <sys/stat.h>
#include <sys/types.h>

// we can remove this and any guarded logic if/when PetscBag supports HDF5
#define PETSCBAG_DOESNT_SUPPORT_HDF5 1

/// Returns the name of the checkpoint directory:
/// * "checkpoints"                        (single-run mode)
/// * "checkpoints/<ensemble-member-name>" (ensemble mode)
static PetscErrorCode GetCheckpointDirectory(RDy rdy, char dir[PETSC_MAX_PATH_LEN]) {
  PetscFunctionBegin;
  static char checkpoint_dir[PETSC_MAX_PATH_LEN] = {0};
  if (!checkpoint_dir[0]) {
    if (rdy->config.ensemble.size > 1) {
      snprintf(checkpoint_dir, PETSC_MAX_PATH_LEN, "checkpoints/%s", rdy->config.ensemble.members[rdy->ensemble_member_index].name);
    } else {
      strcpy(checkpoint_dir, "checkpoints");
    }
  }
  strncpy(dir, checkpoint_dir, PETSC_MAX_PATH_LEN - 1);
  PetscFunctionReturn(PETSC_SUCCESS);
}

// defined in rdyadvance.c
extern PetscErrorCode CreateDirectory(MPI_Comm comm, const char *directory);

// creates the checkpoint directory if it doesn't exist
static PetscErrorCode CreateCheckpointDirectory(RDy rdy) {
  PetscFunctionBegin;

  char checkpoint_dir[PETSC_MAX_PATH_LEN];
  PetscCall(GetCheckpointDirectory(rdy, checkpoint_dir));
  RDyLogDebug(rdy, "Creating checkpoint directory %s...", checkpoint_dir);

  // create the checkpoints/ directory on global rank 0
  if (rdy->config.ensemble.size > 1) {
    PetscCall(CreateDirectory(rdy->global_comm, "checkpoints"));
  }
  PetscCall(CreateDirectory(rdy->comm, checkpoint_dir));
  MPI_Barrier(rdy->global_comm);

  PetscFunctionReturn(PETSC_SUCCESS);
}

// this struct contains metadata that is written to checkpoint files
typedef struct {
  PetscInt  nproc;  // number of MPI processes / tasks
  PetscReal t;      // simulation time
  PetscReal dt;     // timestep
  PetscInt  step;   // timestep number
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
  PetscInt step;
  PetscCall(TSGetStepNumber(rdy->ts, &step));
  PetscCall(PetscBagRegisterInt(*bag, &metadata->step, step, "step", "Simulation step number"));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// reads checkpoint metadata into rdy from the given PetscBag and destroys it
static PetscErrorCode ConsumeMetadata(RDy rdy, PetscBag bag) {
  PetscFunctionBegin;

  CheckpointMetadata *metadata;
  PetscCall(PetscBagGetData(bag, (void **)&metadata));
  rdy->nproc = metadata->nproc;
  rdy->dt    = metadata->dt;
  if (rdy->config.restart.reinitialize) {
    PetscCall(TSSetTime(rdy->ts, 0.0));
  } else {
    PetscCall(TSSetTime(rdy->ts, metadata->t));
    PetscCall(TSSetStepNumber(rdy->ts, metadata->step));
  }
  PetscCall(TSSetTimeStep(rdy->ts, rdy->dt));

  PetscCall(PetscBagDestroy(&bag));

  PetscFunctionReturn(PETSC_SUCCESS);
}

#if PETSCBAG_DOESNT_SUPPORT_HDF5
static PetscErrorCode WriteHDF5Metadata(RDy rdy, PetscViewer viewer) {
  PetscFunctionBegin;
  PetscCall(PetscViewerHDF5WriteGroup(viewer, "/metadata"));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "/metadata"));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "nproc", PETSC_INT, &rdy->nproc));
  PetscReal t;
  PetscCall(TSGetTime(rdy->ts, &t));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "t", PETSC_DOUBLE, &t));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "dt", PETSC_DOUBLE, &rdy->dt));
  PetscInt step;
  PetscCall(TSGetStepNumber(rdy->ts, &step));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "step", PETSC_INT, &step));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ReadHDF5Metadata(RDy rdy, PetscViewer viewer) {
  PetscFunctionBegin;
  PetscCall(PetscViewerHDF5PushGroup(viewer, "/metadata"));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, NULL, "nproc", PETSC_INT, NULL, &rdy->nproc));
  if (rdy->config.restart.reinitialize) {
    PetscCall(TSSetTime(rdy->ts, 0.0));
  } else {
    PetscReal t;
    PetscCall(PetscViewerHDF5ReadAttribute(viewer, NULL, "t", PETSC_DOUBLE, NULL, &t));
    PetscCall(TSSetTime(rdy->ts, t));
    PetscInt step;
    PetscCall(PetscViewerHDF5ReadAttribute(viewer, NULL, "step", PETSC_INT, NULL, &step));
    PetscCall(TSSetStepNumber(rdy->ts, step));
  }
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, NULL, "dt", PETSC_DOUBLE, NULL, &rdy->dt));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

// this generates the ancient-looking checkpoing filename expected by E3SM
static PetscErrorCode GenerateE3SMCheckpointFilename(const char *directory, const char *prefix, PetscInt index, PetscInt max_index_val,
                                                     const char *suffix, char *filename) {
  PetscFunctionBegin;
  int  num_digits = (int)(log10((double)max_index_val)) + 1;
  char fmt[16]    = {0};
  snprintf(fmt, 15, ".%%0%dd.%%s", num_digits);
  char ending[PETSC_MAX_PATH_LEN];
  snprintf(ending, PETSC_MAX_PATH_LEN - 1, fmt, index, suffix);
  snprintf(filename, PETSC_MAX_PATH_LEN - 1, "%s/%s.rdycore.r%s", directory, prefix, ending);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode WriteCheckpoint(TS ts, PetscInt step, PetscReal time, Vec X, void *ctx) {
  PetscFunctionBegin;
  RDy rdy = ctx;
  if (step % rdy->config.checkpoint.interval == 0) {
    // determine an appropriate prefix for checkpoint files
    char prefix[PETSC_MAX_PATH_LEN], filename[PETSC_MAX_PATH_LEN];
    if (rdy->config.checkpoint.prefix[0]) {
      strncpy(prefix, rdy->config.checkpoint.prefix, PETSC_MAX_PATH_LEN);
    } else {
      PetscCall(DetermineConfigPrefix(rdy, prefix));
    }

    PetscViewer             viewer;
    const PetscViewerFormat format = rdy->config.checkpoint.format;
    char                    checkpoint_dir[PETSC_MAX_PATH_LEN];
    PetscCall(GetCheckpointDirectory(rdy, checkpoint_dir));
    if (format == PETSC_VIEWER_NATIVE) {  // binary
      PetscCall(GenerateE3SMCheckpointFilename(checkpoint_dir, prefix, step, rdy->config.time.max_step, "bin", filename));
      PetscCall(PetscViewerBinaryOpen(rdy->comm, filename, FILE_MODE_WRITE, &viewer));
    } else {  // HDF5
      PetscCall(GenerateE3SMCheckpointFilename(checkpoint_dir, prefix, step, rdy->config.time.max_step, "h5", filename));
      PetscCall(PetscViewerHDF5Open(rdy->comm, filename, FILE_MODE_WRITE, &viewer));
    }

    RDyLogInfo(rdy, "Writing checkpoint file %s...", filename);

#if PETSCBAG_DOESNT_SUPPORT_HDF5
    if (format == PETSC_VIEWER_HDF5_PETSC) {
      PetscCall(WriteHDF5Metadata(rdy, viewer));
    } else {
#endif
      PetscBag bag;
      PetscCall(CreateMetadata(rdy, &bag));
      PetscCall(PetscBagView(bag, viewer));
      PetscCall(PetscBagDestroy(&bag));
#if PETSCBAG_DOESNT_SUPPORT_HDF5
    }
#endif
    PetscCall(PetscObjectSetName((PetscObject)X, "solution"));
    PetscCall(VecView(X, viewer));

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
    PetscCall(CreateCheckpointDirectory(rdy));

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
    PetscCall(PetscViewerHDF5PushTimestepping(viewer));  // NOTE: seems to be needed
  } else {
    PetscCheck(PETSC_FALSE, rdy->comm, PETSC_ERR_USER, "Invalid checkpoint file: %s", filename);
  }

#if PETSCBAG_DOESNT_SUPPORT_HDF5
  if (strstr(filename, ".h5")) {  // binary
    PetscCall(ReadHDF5Metadata(rdy, viewer));
  } else {
#endif
    PetscBag bag;
    PetscCall(CreateMetadata(rdy, &bag));
    PetscCall(PetscBagLoad(viewer, bag));
    PetscCall(ConsumeMetadata(rdy, bag));
#if PETSCBAG_DOESNT_SUPPORT_HDF5
  }
#endif
  PetscCall(PetscObjectSetName((PetscObject)rdy->u_global, "solution"));
  PetscCall(VecLoad(rdy->u_global, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  RDyLogInfo(rdy, "Finished reading checkpoint file.");

  // read the newly loaded solution vector into the timestepper
  PetscCall(TSSetSolution(rdy->ts, rdy->u_global));

  PetscFunctionReturn(PETSC_SUCCESS);
}
