#include <errno.h>
#include <private/rdycoreimpl.h>
#include <rdycore.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

/* restart logic stubs -- not yet implemented
// writes a restart file
static PetscErrorCode WriteRestart(RDy rdy) {
  PetscFunctionBegin;
  // TODO
  PetscFunctionReturn(0);
}

// TS monitoring routine used to write restart files
static PetscErrorCode WriteRestartFiles(TS ts, PetscInt step, PetscReal time, Vec X, void *ctx) {
  RDY rdy = ctx;
  if (step % rdy->config.restart_frequency == 0) {
    WriteRestart(rdy);
  }
}
*/

// output directory name (relative to current working directory)
static const char *output_dir = "output";

static PetscErrorCode CreateOutputDir(RDy rdy) {
  PetscFunctionBegin;

  int result_and_errno[2];
  if (rdy->rank == 0) {
    result_and_errno[0] = mkdir(output_dir, 0755);
    result_and_errno[1] = errno;
  }
  MPI_Bcast(&result_and_errno, 2, MPI_INT, 0, rdy->comm);
  int result = result_and_errno[0];
  int err_no = result_and_errno[1];
  PetscCheck((result == 0) || (err_no == EEXIST), rdy->comm, PETSC_ERR_USER, "Could not create output directory: %s (errno = %d)", output_dir,
             err_no);

  PetscFunctionReturn(0);
}

static PetscErrorCode DetermineOutputFile(RDy rdy, char *filename) {
  PetscFunctionBegin;

  char *p = strstr(rdy->config_file, ".yaml");
  if (!p) {  // could be .yml, I suppose (Windows habits die hard!)
    p = strstr(rdy->config_file, ".yml");
  }
  if (p) {
    size_t prefix_len = p - rdy->config_file;
    size_t n          = strlen(output_dir) + 1 + prefix_len;
    snprintf(filename, n + 1, "%s/%s", output_dir, rdy->config_file);
  } else {
    snprintf(filename, PETSC_MAX_PATH_LEN - 1, "%s/%s", output_dir, rdy->config_file);
  }

  char suffix[4];
  if (rdy->config.output_format == PETSC_VIEWER_HDF5_XDMF) {
    strcpy(suffix, "h5");
  } else { // native binary format
    strcpy(suffix, "dat");
  }

  // concatenate some config parameters
  char addendum[PETSC_MAX_PATH_LEN];
  snprintf(addendum, PETSC_MAX_PATH_LEN - 1, "_dt_%f_%d_np%d.%s", rdy->dt, rdy->config.max_step, rdy->nproc, suffix);
  strncat(filename, addendum, PETSC_MAX_PATH_LEN - 1 - strlen(filename));

  PetscFunctionReturn(0);
}

// writes an output file
static PetscErrorCode WriteOutput(RDy rdy) {
  PetscFunctionBegin;

  // Determine the output file name.
  char fname[PETSC_MAX_PATH_LEN];
  PetscCall(DetermineOutputFile(rdy, fname));

  PetscViewer viewer;
  PetscCall(PetscViewerBinaryOpen(rdy->comm, fname, FILE_MODE_WRITE, &viewer));
  PetscCall(PetscViewerPushFormat(viewer, rdy->config.output_format));
  // TODO: break up our solution vector into (h, hu, hv) for output
  // TODO: (this constrains our file-based initial conditions)
  Vec natural;
  PetscCall(DMPlexCreateNaturalVector(rdy->dm, &natural));
  PetscCall(DMPlexGlobalToNaturalBegin(rdy->dm, rdy->X, natural));
  PetscCall(DMPlexGlobalToNaturalEnd(rdy->dm, rdy->X, natural));
  PetscCall(VecView(natural, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(VecDestroy(&natural));

  PetscFunctionReturn(0);
}

// TS monitoring routine used to write output files
static PetscErrorCode WriteOutputFiles(TS ts, PetscInt step, PetscReal time, Vec X, void *ctx) {
  RDY rdy = ctx;
  if ((rdy->config.output_frequency == -1) || // last step (interpolated)
      (time >= rdy->config.final_time) ||     // last step without interpolation
      (step % rdy->config.output_frequency == 0)) {
    WriteOutput(rdy);
  }
}

PetscErrorCode RDyRun(RDy rdy) {
  PetscFunctionBegin;

  RDyLogDebug(rdy, "Creating output directory %s...", output_dir);
  PetscCall(CreateOutputDir(rdy));

  // set up monitoring functions for handling restarts and outputs
//  if (rdy->config.restart_frequency) {
//    PetscCall(TSMonitorSet(rdy->tѕ, WriteRestartFiles, rdy, NULL));
//  }
  if (rdy->config.output_frequency) {
    PetscCall(TSMonitorSet(rdy->tѕ, WriteOutputFiles, rdy, NULL));
  }

  // do the thing!
  RDyLogDebug(rdy, "Running simulation...");
  PetscCall(TSSolve(rdy->ts, rdy->X));

  PetscFunctionReturn(0);
}
