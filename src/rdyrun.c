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

// generates a filename in the output directory
// * with the given prefix
// * with the given zero-padded index (padded to the given max index value)
// * and the given suffix
static PetscErrorCode GenerateIndexedFilename(const char *prefix, PetscInt index, PetscInt max_index_val, const char *suffix, char *filename) {
  PetscFunctionBegin;
  int      num_digits = (int)(log10((double)max_index_val)) + 1;
  char     fmt[16]    = {0};
  snprintf(fmt, 15, "-%%0%dd.%%s", num_digits);
  char ending[PETSC_MAX_PATH_LEN];
  snprintf(ending, PETSC_MAX_PATH_LEN - 1, fmt, index, suffix);
  snprintf(filename, PETSC_MAX_PATH_LEN - 1, "%s/%s%s", output_dir, prefix, ending);
  PetscFunctionReturn(0);
}

/// Determines the appropriate output file name based on
/// * the desired file format as specified by rdy->config.output_format
/// * the time step index and simulation time
/// * the specified suffix
PetscErrorCode DetermineOutputFile(RDy rdy, PetscInt step, PetscReal time, const char *suffix, char *filename) {
  PetscFunctionBegin;

  size_t config_len = strlen(rdy->config_file);
  char prefix[config_len+1];
  memset(prefix, 0, sizeof(char)*(config_len+1));
  char *p = strstr(rdy->config_file, ".yaml");
  if (!p) {  // could be .yml, I suppose (Windows habits die hard!)
    p = strstr(rdy->config_file, ".yml");
  }
  if (p) {
    size_t prefix_len = p - rdy->config_file;
    strncpy(prefix, rdy->config_file, prefix_len);
  } else {
    strcpy(prefix, rdy->config_file);
  }

  // encode specific information into the filename based on its format
  if (rdy->config.output_format == OUTPUT_BINARY) {  // PETSc native binary format
    PetscCall(GenerateIndexedFilename(prefix, step, rdy->config.max_step, suffix, filename));
  } else if (rdy->config.output_format == OUTPUT_XDMF) {
    if (!strcasecmp(suffix, "h5")) {  // XDMF "heavy" data or CGNS
      if (rdy->config.output_batch_size == 1) {
        // all output data goes into a single file
        snprintf(filename, PETSC_MAX_PATH_LEN - 1, "%s/%s.%s", output_dir, prefix, suffix);
      } else {
        // output data is grouped into batches of a fixed number of time steps
        PetscInt batch_size = rdy->config.output_batch_size;
        PetscInt freq       = rdy->config.output_frequency;
        PetscInt batch      = step / freq / batch_size;
        PetscInt max_batch = rdy->config.max_step / freq / batch_size;
        PetscCall(GenerateIndexedFilename(prefix, batch, max_batch, suffix, filename));
      }
    } else if (!strcasecmp(suffix, "xmf")) {  // XDMF "light" data
      PetscCheck(!strcasecmp(suffix, "xmf"), rdy->comm, PETSC_ERR_USER, "Invalid suffix for XDMF output: %s", suffix);
      // encode the step into the filename with zero-padding based on the
      // maximum step number
      PetscCall(GenerateIndexedFilename(prefix, step, rdy->config.max_step, suffix, filename));
    } else {
      PetscCheck(PETSC_FALSE, rdy->comm, PETSC_ERR_USER, "Unsupported file suffix: %s", suffix);
    }
  } else if (rdy->config.output_format == OUTPUT_CGNS) {
    // the CGNS viewer handles its own batching and only needs a format string
    snprintf(filename, PETSC_MAX_PATH_LEN - 1, "%s/%s-%%d.%s", output_dir, prefix, suffix);
  } else {
    PetscCheck(PETSC_FALSE, rdy->comm, PETSC_ERR_USER, "Unsupported output format specified.");
  }

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
  if (rdy->config.output_frequency) {
    RDyLogDebug(rdy, "Writing output every %d timestep(s)", rdy->config.output_frequency);
    switch (rdy->config.output_format) {
      case OUTPUT_CGNS:
        // we've already configured this viewer in SetAdditionalOptions (see read_config_file.c)
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
    if (viz->viewer) {
      PetscCall(PetscViewerSetFromOptions(viz->viewer));
      PetscCall(PetscViewerAndFormatCreate(viz->viewer, format, &viz->vf));
    }

    /*
    // set monitoring interval option if not given on the command line
    PetscCall(PetscOptionsHasName(NULL, NULL, "-ts_monitor_solution_interval", &has_param));
    if (!has_param) {
      viz->vf->view_interval = rdy->config.output_frequency;
    }
    */

    // set up solution monitoring
    if (rdy->config.output_format == OUTPUT_XDMF) {
      // we do our own special thing for XDMF
      PetscCall(TSMonitorSet(rdy->ts, WriteXDMFOutput, rdy, NULL));
    } else if (rdy->config.output_format != OUTPUT_CGNS) { // everything else (except CGNS)
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
  ViewerContext viz = {0};
  PetscCall(CreateVizViewerContext(rdy, &viz));

  // do the thing!
  RDyLogDebug(rdy, "Running simulation...");
  PetscCall(TSSolve(rdy->ts, rdy->X));

  // clean up
  PetscCall(ViewerContextDestroy(viz));

  PetscFunctionReturn(0);
}
