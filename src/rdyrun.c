#include <private/rdycoreimpl.h>
#include <rdycore.h>

PetscErrorCode RDyRun(RDy rdy) {
  PetscFunctionBegin;

  PetscCall(CreateOutputDir(rdy));

  // set up monitoring functions for handling restarts and outputs
  //  if (rdy->config.restart_frequency) {
  //    PetscCall(TSMonitorSet(rdy->ts, WriteRestartFiles, rdy, NULL));
  //  }
  if (rdy->config.output_frequency) {
    PetscCall(TSMonitorSet(rdy->ts, WriteOutputFiles, rdy, NULL));
  }

  // do the thing!
  RDyLogDebug(rdy, "Running simulation...");
  PetscCall(TSSolve(rdy->ts, rdy->X));

  // If we need to generate any additional output files, do so here.
  PetscCall(PostprocessOutput(rdy));

  PetscFunctionReturn(0);
}
