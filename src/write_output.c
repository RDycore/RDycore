#include <errno.h>
#include <petscviewerhdf5.h>
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
  PetscFunctionBegin;
  RDY rdy = ctx;
  if (step % rdy->config.restart_frequency == 0) {
    PetscCall(WriteRestart(rdy));
  }
  PetscFunctionReturn(0);
}
*/

// output directory name (relative to current working directory)
static const char *output_dir = "output";

PetscErrorCode CreateOutputDir(RDy rdy) {
  PetscFunctionBegin;

  RDyLogDebug(rdy, "Creating output directory %s...", output_dir);

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

static PetscErrorCode DetermineOutputFile(RDy rdy, PetscInt step, PetscReal time, char *filename) {
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

  // encode specific information into the filename based on its format
  char suffix[PETSC_MAX_PATH_LEN];
  if (rdy->config.output_format == PETSC_VIEWER_HDF5_XDMF) {
    snprintf(suffix, PETSC_MAX_PATH_LEN - 1, "_dt_%f_%d_np%d.h5", rdy->dt, rdy->config.max_step, rdy->nproc);
  } else {  // native binary format
    snprintf(suffix, PETSC_MAX_PATH_LEN - 1, "_dt_%f_%d_%d_np%d.dat", rdy->dt, rdy->config.max_step, step, rdy->nproc);
  }

  // concatenate some config parameters
  strncat(filename, suffix, PETSC_MAX_PATH_LEN - 1 - strlen(filename));
  RDyLogDetail(rdy, "Step %d: writing output to %s", step, filename);

  PetscFunctionReturn(0);
}

// writes output in Petsc native binary format
static PetscErrorCode WriteBinaryOutput(RDy rdy, PetscInt step, PetscReal time) {
  PetscFunctionBegin;

  // Determine the output file name.
  char fname[PETSC_MAX_PATH_LEN];
  PetscCall(DetermineOutputFile(rdy, step, time, fname));

  PetscViewer viewer;
  PetscCall(PetscViewerBinaryOpen(rdy->comm, fname, FILE_MODE_WRITE, &viewer));
  PetscCall(PetscViewerPushFormat(viewer, rdy->config.output_format));

  // dump the solution vector in natural ordering
  Vec natural;
  PetscCall(DMPlexCreateNaturalVector(rdy->dm, &natural));
  PetscCall(DMPlexGlobalToNaturalBegin(rdy->dm, rdy->X, natural));
  PetscCall(DMPlexGlobalToNaturalEnd(rdy->dm, rdy->X, natural));
  PetscCall(VecView(natural, viewer));
  PetscCall(VecDestroy(&natural));

  PetscCall(PetscViewerPopFormat(viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscFunctionReturn(0);
}

// writes output in XDMF format
static PetscErrorCode WriteXDMFOutput(RDy rdy, PetscInt step, PetscReal time) {
  PetscFunctionBegin;

  PetscViewer viewer;

  // Determine the output file name.
  char fname[PETSC_MAX_PATH_LEN];
  PetscCall(DetermineOutputFile(rdy, step, time, fname));

  // write the grid if we're on the first step
  if (step == 0) {
    PetscCall(PetscViewerHDF5Open(rdy->comm, fname, FILE_MODE_WRITE, &viewer));
    PetscCall(PetscViewerPushFormat(viewer, rdy->config.output_format));
    PetscCall(DMView(rdy->dm, viewer));
  } else {
    PetscCall(PetscViewerHDF5Open(rdy->comm, fname, FILE_MODE_APPEND, &viewer));
    PetscCall(PetscViewerPushFormat(viewer, rdy->config.output_format));
  }

  // write solution data to a new GROUP with components in separate datasets
  PetscReal time_in_days = time / (24 * 3600);  // seconds -> days
  char      groupName[1025];
  snprintf(groupName, 1024, "%d %E d", step, time_in_days);
  PetscCall(PetscViewerHDF5PushGroup(viewer, groupName));

  // create and populate a multi-component natural vector
  Vec natural;
  PetscCall(DMPlexCreateNaturalVector(rdy->dm, &natural));
  PetscCall(DMPlexGlobalToNaturalBegin(rdy->dm, rdy->X, natural));
  PetscCall(DMPlexGlobalToNaturalEnd(rdy->dm, rdy->X, natural));

  // extract each component into a separate vector and write it to the group
  // FIXME: This setup is specific to the shallow water equations. We can
  // FIXME: generalize it later.
  const char *comp_names[3] = {
      "Water_Height",
      "X_Momentum",
      "Y_Momentum",
  };
  Vec      comp;  // single-component natural vector
  PetscInt n, N, bs;
  PetscCall(VecGetLocalSize(natural, &n));
  PetscCall(VecGetSize(natural, &N));
  PetscCall(VecGetBlockSize(natural, &bs));
  PetscCall(VecCreateMPI(rdy->comm, n / bs, N / bs, &comp));
  PetscReal *Xi;  // multi-component natural vector data
  PetscCall(VecGetArray(natural, &Xi));
  for (PetscInt c = 0; c < bs; ++c) {
    PetscObjectSetName((PetscObject)comp, comp_names[c]);
    PetscReal *Xci;  // single-component natural vector data
    PetscCall(VecGetArray(comp, &Xci));
    for (PetscInt i = 0; i < n / bs; ++i) Xci[i] = Xi[bs * i + c];
    PetscCall(VecRestoreArray(comp, &Xci));
    PetscCall(VecView(comp, viewer));
  }
  PetscCall(VecRestoreArray(natural, &Xi));

  // clean up
  PetscCall(VecDestroy(&comp));
  PetscCall(VecDestroy(&natural));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscFunctionReturn(0);
}

// TS monitoring routine used to write output files
PetscErrorCode WriteOutputFiles(TS ts, PetscInt step, PetscReal time, Vec X, void *ctx) {
  PetscFunctionBegin;
  RDy rdy = ctx;

  if ((rdy->config.output_frequency == -1) ||  // last step (interpolated)
      (time >= rdy->config.final_time) ||      // last step without interpolation
      (step % rdy->config.output_frequency == 0)) {
    if (rdy->config.output_format == PETSC_VIEWER_HDF5_XDMF) {
      PetscCall(WriteXDMFOutput(rdy, step, time));
    } else {
      PetscCall(WriteBinaryOutput(rdy, step, time));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PostprocessOutput(RDy rdy) {
  PetscFunctionBegin;

  if (rdy->config.output_format == PETSC_VIEWER_HDF5_XDMF) {
    // FIXME: write XML file here!
  }

  PetscFunctionReturn(0);
}
