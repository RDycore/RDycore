#include <errno.h>
#include <private/rdycoreimpl.h>
#include <rdycore.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

// output directory name (relative to current working directory)
static const char *output_dir = "output";

/// Creates the output directory if it doesn't exist.
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
    snprintf(ending, PETSC_MAX_PATH_LEN - 1, ".%s", suffix);
  } else if (rdy->config.output_format == OUTPUT_XDMF) {
    if (!strcasecmp(suffix, "h5")) {  // XDMF "heavy" data
      if (rdy->config.output_batch_size == 1) {
        // all output data goes into a single HDF5 file
        snprintf(ending, PETSC_MAX_PATH_LEN - 1, ".%s", suffix);
      } else {
        // output data is grouped into batches of a fixed number of time steps
        PetscInt batch_size = rdy->config.output_batch_size;
        PetscInt batch      = step / batch_size;
        int      num_digits = (int)(log10((double)(rdy->config.max_step / batch_size))) + 1;
        char     fmt[16]    = {0};
        snprintf(fmt, 15, "-%%0%dd.%%s", num_digits);
        snprintf(ending, PETSC_MAX_PATH_LEN - 1, fmt, batch, suffix);
      }
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

PetscErrorCode RDyRun(RDy rdy) {
  PetscFunctionBegin;

  PetscCall(CreateOutputDir(rdy));

  // set up monitoring functions for handling restarts and outputs
  //  if (rdy->config.restart_frequency) {
  //    PetscCall(TSMonitorSet(rdy->ts, WriteRestartFiles, rdy, NULL));
  //  }

  // create a viewer with the proper format for visualization output
  PetscViewer           viz_viewer;
  PetscViewerAndFormat *viz_vf;
  PetscViewerFormat     format = PETSC_VIEWER_DEFAULT;
  char                  param[24];
  if (rdy->config.output_frequency) {
    RDyLogDebug(rdy, "Writing output every %d timestep(s)", rdy->config.output_frequency);
    PetscCall(PetscViewerCreate(rdy->comm, &viz_viewer));
    PetscBool has_param;
    switch (rdy->config.output_format) {
      case OUTPUT_CGNS:
        PetscCall(PetscViewerSetType(viz_viewer, PETSCVIEWERCGNS));
        PetscCall(PetscOptionsHasName(NULL, NULL, "-viewer_cgns_batch_size", &has_param));
        if (!has_param) {
          snprintf(param, 23, "%d", rdy->config.output_batch_size);
          PetscOptionsSetValue(NULL, "-viewer_cgns_batch_size", param);
        }
        break;
      case OUTPUT_XDMF:
        PetscCall(PetscViewerSetType(viz_viewer, PETSCVIEWERHDF5));
        format = PETSC_VIEWER_HDF5_XDMF;
        break;
      case OUTPUT_BINARY:
        PetscCall(PetscViewerSetType(viz_viewer, PETSCVIEWERBINARY));
    }

    // apply any command-line option overrides
    PetscCall(PetscViewerSetFromOptions(viz_viewer));
    PetscCall(PetscViewerAndFormatCreate(viz_viewer, format, &viz_vf));

    // set monitoring interval option if not given on the command line
    viz_vf->view_interval = rdy->config.output_frequency;

    // set up solution monitoring
    if (rdy->config.output_format == OUTPUT_XDMF) {
      // we do our own special thing for XDMF
      PetscCall(TSMonitorSet(rdy->ts, WriteXDMFOutput, rdy, NULL));
    } else {  // we let PETSc handle other formats
      PetscCall(TSMonitorSet(rdy->ts, (PetscErrorCode(*)(TS, PetscInt, PetscReal, Vec, void *))TSMonitorSolution, viz_vf, NULL));
    }
  }

  // do the thing!
  RDyLogDebug(rdy, "Running simulation...");
  PetscCall(TSSolve(rdy->ts, rdy->X));

  // clean up
  if (viz_vf) PetscCall(PetscViewerAndFormatDestroy(&viz_vf));
  if (viz_viewer) PetscCall(PetscViewerDestroy(&viz_viewer));

  PetscFunctionReturn(0);
}
