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
    if (!strcasecmp(suffix, "cgns")) {  // CGNS data
      // the CGNS viewer handles its own batching and only needs a format string
      snprintf(ending, PETSC_MAX_PATH_LEN - 1, "-%%d.%s", suffix);
    } else if (!strcasecmp(suffix, "h5")) {  // XDMF "heavy" data or CGNS
      if (rdy->config.output_batch_size == 1) {
        // all output data goes into a single file
        snprintf(ending, PETSC_MAX_PATH_LEN - 1, ".%s", suffix);
      } else {
        // output data is grouped into batches of a fixed number of time steps
        PetscInt batch_size = rdy->config.output_batch_size;
        PetscInt freq       = rdy->config.output_frequency;
        PetscInt batch      = step / freq / batch_size;
        int      num_digits = (int)(log10((double)(rdy->config.max_step / freq / batch_size))) + 1;
        char     fmt[16]    = {0};
        snprintf(fmt, 15, "-%%0%dd.%%s", num_digits);
        snprintf(ending, PETSC_MAX_PATH_LEN - 1, fmt, batch, suffix);
      }
    } else if (!strcasecmp(suffix, "xmf")) {  // XDMF "light" data
      PetscCheck(!strcasecmp(suffix, "xmf"), rdy->comm, PETSC_ERR_USER, "Invalid suffix for XDMF output: %s", suffix);
      // encode the step into the filename with zero-padding based on the
      // maximum step number
      int  num_digits = (int)(log10((double)rdy->config.max_step)) + 1;
      char fmt[16]    = {0};
      snprintf(fmt, 15, "-%%0%dd.%%s", num_digits);
      snprintf(ending, PETSC_MAX_PATH_LEN - 1, fmt, step, suffix);
    } else {
      PetscCheck(PETSC_FALSE, rdy->comm, PETSC_ERR_USER, "Unsupported file suffix: %s", suffix);
    }
  } else {
    PetscCheck(PETSC_FALSE, rdy->comm, PETSC_ERR_USER, "Unsupported output format specified.");
  }

  // concatenate some config parameters
  strncat(filename, ending, PETSC_MAX_PATH_LEN - 1 - strlen(filename));

  PetscFunctionReturn(0);
}

// This struct holds contextual data for a viewer.
typedef struct {
  PetscViewer           viewer;
  PetscViewerAndFormat *vf;
} ViewerContext;

// Destroys a ViewerContext.
static PetscErrorCode ViewerContextDestroy(ViewerContext vc) {
  PetscFunctionBegin;
  if (vc.vf) PetscCall(PetscViewerAndFormatDestroy(&vc.vf));
  if (vc.viewer) PetscCall(PetscViewerDestroy(&vc.viewer));
  PetscFunctionReturn(0);
}

// needed to construct a CGNS viewer
extern PetscErrorCode PetscViewerCreate_CGNS(PetscViewer);

// Creates a ViewerContext for visualization and sets up visualization
// monitoring. This function handles a lot of the disparities between what's
// available in PETSc's command line options and what exists in the API. We need
// this to be able to encode all of our settings in a single input file for reproducibility.
static PetscErrorCode CreateVizViewerContext(RDy rdy, ViewerContext *viz) {
  PetscFunctionBegin;

  PetscViewerFormat format = PETSC_VIEWER_DEFAULT;
  char              param[24];
  char              filename[PETSC_MAX_PATH_LEN];
  if (rdy->config.output_frequency) {
    RDyLogDebug(rdy, "Writing output every %d timestep(s)", rdy->config.output_frequency);
    PetscBool has_param;
    switch (rdy->config.output_format) {
      case OUTPUT_CGNS:
        PetscCall(PetscViewerCreate_CGNS(viz->viewer));
        PetscCall(DetermineOutputFile(rdy, 0, 0.0, "cgns", filename));
        PetscCall(PetscViewerFileSetName(viz->viewer, filename));
        PetscCall(PetscOptionsHasName(NULL, NULL, "-viewer_cgns_batch_size", &has_param));
        if (!has_param) {
          snprintf(param, 23, "%d", rdy->config.output_batch_size);
          PetscOptionsSetValue(NULL, "-viewer_cgns_batch_size", param);
        }
        break;
      case OUTPUT_XDMF:
        // we don't actually use this viewer, so maybe this doesn't matter?
        PetscCall(PetscViewerCreate(rdy->comm, &viz->viewer));
        PetscCall(PetscViewerSetType(viz->viewer, PETSCVIEWERHDF5));
        format = PETSC_VIEWER_HDF5_XDMF;
        break;
      case OUTPUT_BINARY:
        PetscCall(PetscViewerCreate(rdy->comm, &viz->viewer));
        PetscCall(PetscViewerSetType(viz->viewer, PETSCVIEWERBINARY));
    }

    // apply any command-line option overrides
    PetscCall(PetscViewerSetFromOptions(viz->viewer));
    PetscCall(PetscViewerAndFormatCreate(viz->viewer, format, &viz->vf));

    // set monitoring interval option if not given on the command line
    PetscCall(PetscOptionsHasName(NULL, NULL, "-ts_monitor_solution_interval", &has_param));
    if (!has_param) {
      viz->vf->view_interval = rdy->config.output_frequency;
    }

    // set up solution monitoring
    if (rdy->config.output_format == OUTPUT_XDMF) {
      // we do our own special thing for XDMF
      PetscCall(TSMonitorSet(rdy->ts, WriteXDMFOutput, rdy, NULL));
    } else if (rdy->config.output_format == OUTPUT_CGNS) {
    } else {  // PETSc can handle all other formats
      PetscCall(TSMonitorSet(rdy->ts, (PetscErrorCode(*)(TS, PetscInt, PetscReal, Vec, void *))TSMonitorSolution, viz->vf, NULL));
    }
  }

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
  ViewerContext viz;
  PetscCall(CreateVizViewerContext(rdy, &viz));

  // do the thing!
  RDyLogDebug(rdy, "Running simulation...");
  PetscCall(TSSolve(rdy->ts, rdy->X));

  // clean up
  PetscCall(ViewerContextDestroy(viz));

  PetscFunctionReturn(0);
}
