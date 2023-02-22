#include <errno.h>
#include <private/rdycoreimpl.h>
#include <rdycore.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

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
    size_t n = p - rdy->config_file;
    strncpy(filename, rdy->config_file, n);
    filename[n] = 0;
  } else {
    strncpy(filename, rdy->config_file, PETSC_MAX_PATH_LEN - 1);
  }

  // concatenate some config parameters
  char suffix[PETSC_MAX_PATH_LEN];
  snprintf(suffix, PETSC_MAX_PATH_LEN - 1, "_dt_%f_%d_np%d.dat", rdy->dt, rdy->config.max_step, rdy->nproc);
  strncat(filename, suffix, PETSC_MAX_PATH_LEN - 1 - strlen(filename));

  PetscFunctionReturn(0);
}

static PetscErrorCode WriteOutput(RDy rdy) {
  PetscFunctionBegin;

  // Determine the output file name.
  char fname[PETSC_MAX_PATH_LEN];
  PetscCall(DetermineOutputFile(rdy, fname));

  PetscViewer viewer;
  PetscCall(PetscViewerBinaryOpen(rdy->comm, fname, FILE_MODE_WRITE, &viewer));
  Vec natural;
  PetscCall(DMPlexCreateNaturalVector(rdy->dm, &natural));
  PetscCall(DMPlexGlobalToNaturalBegin(rdy->dm, rdy->X, natural));
  PetscCall(DMPlexGlobalToNaturalEnd(rdy->dm, rdy->X, natural));
  PetscCall(VecView(natural, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(VecDestroy(&natural));

  PetscFunctionReturn(0);
}

PetscErrorCode RDyRun(RDy rdy) {
  PetscFunctionBegin;

  RDyLogDebug(rdy, "Creating output directory %s...", output_dir);
  PetscCall(CreateOutputDir(rdy));

  // do the thing!
  RDyLogDebug(rdy, "Running simulation...");
  PetscCall(TSSolve(rdy->ts, rdy->X));

  // write output at the end of the simulation
  RDyLogDebug(rdy, "Writing simulation output...");
  PetscCall(WriteOutput(rdy));

  PetscFunctionReturn(0);
}
