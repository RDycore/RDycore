#include <errno.h>
#include <private/rdycoreimpl.h>
#include <rdycore.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

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

/// Determines the appropriate output file name based on
/// * the desired file format as specified by rdy->config.output_format
/// * the time step index and simulation time
/// * the specified suffix
PetscErrorCode DetermineOutputFile(RDy rdy, PetscInt step, PetscReal time, const char *suffix, char *filename) {
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
  char ending[PETSC_MAX_PATH_LEN];
  if (rdy->config.output_format == OUTPUT_BINARY) {  // PETSc native binary format
    int  num_digits = (int)(log10((double)rdy->config.max_step)) + 1;
    char fmt[16]    = {0};
    snprintf(fmt, 15, "-%%0%dd.%%s", num_digits);
    snprintf(ending, PETSC_MAX_PATH_LEN - 1, fmt, step, suffix);
  } else if (rdy->config.output_format == OUTPUT_XDMF) {
    if (!strcasecmp(suffix, "h5")) {  // XDMF "heavy" data
      // for now we assume all output data goes into a single HDF5 file
      snprintf(ending, PETSC_MAX_PATH_LEN - 1, ".%s", suffix);
    } else {  // XDMF "light" data?
      PetscCheck(!strcasecmp(suffix, "xmf"), rdy->comm, PETSC_ERR_USER, "Invalid suffix for XDMF output: %s", suffix);
      // encode the step into the filename with zero-padding based on the
      // maximum step number
      int  num_digits = (int)(log10((double)rdy->config.max_step)) + 1;
      char fmt[16]    = {0};
      snprintf(fmt, 15, "-%%0%dd.%%s", num_digits);
      snprintf(ending, PETSC_MAX_PATH_LEN - 1, fmt, step, suffix);
    }
  } else {
    PetscCheck(PETSC_FALSE, rdy->comm, PETSC_ERR_USER, "Unsupported output format specified.");
  }

  // concatenate some config parameters
  strncat(filename, ending, PETSC_MAX_PATH_LEN - 1 - strlen(filename));

  PetscFunctionReturn(0);
}

// writes output in Petsc native binary format
PetscErrorCode WriteBinaryOutput(RDy rdy, PetscInt step, PetscReal time) {
  PetscFunctionBegin;

  // Determine the output file name.
  char fname[PETSC_MAX_PATH_LEN];
  PetscCall(DetermineOutputFile(rdy, step, time, "dat", fname));
  RDyLogDetail(rdy, "Step %d: writing binary output to %s", step, fname);

  PetscViewer viewer;
  PetscCall(PetscViewerBinaryOpen(rdy->comm, fname, FILE_MODE_WRITE, &viewer));
  PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_NATIVE));

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

// TS monitoring routine used to write output files
PetscErrorCode WriteOutputFiles(TS ts, PetscInt step, PetscReal time, Vec X, void *ctx) {
  PetscFunctionBegin;
  RDy rdy = ctx;

  PetscReal final_time = ConvertTimeToSeconds(rdy->config.final_time, rdy->config.time_unit);
  if ((step == -1) ||          // last step (interpolated)
      (time >= final_time) ||  // last step without interpolation
      ((step % rdy->config.output_frequency) == 0)) {
    PetscReal t = ConvertTimeFromSeconds(time, rdy->config.time_unit);
    if (rdy->config.output_format == OUTPUT_XDMF) {
      PetscCall(WriteXDMFHDF5Data(rdy, step, t));
      PetscCall(WriteXDMFXMFData(rdy, step, t));
    } else if (rdy->config.output_format == OUTPUT_CGNS) {
      PetscCall(WriteCGNSOutput(rdy, step, t));
    } else {  // binary
      PetscCall(WriteBinaryOutput(rdy, step, t));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PostprocessOutput(RDy rdy) {
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
