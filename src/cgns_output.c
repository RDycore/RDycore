#include <errno.h>
#include <petscviewerhdf5.h>  // note: includes hdf5.h
#include <private/rdycoreimpl.h>
#include <rdycore.h>
#include <string.h>

// writes data to a CGNS output file
PetscErrorCode WriteCGNSOutput(RDy rdy, PetscInt step, PetscReal time) {
  PetscFunctionBegin;

  PetscViewer viewer;

  /*
  // Determine the output file name.
  char fname[PETSC_MAX_PATH_LEN];
  PetscCall(DetermineOutputFile(rdy, step, time, "h5", fname));

  // write the grid if we're on the first step
  if (step == 0) {
    PetscCall(PetscViewerHDF5Open(rdy->comm, fname, FILE_MODE_WRITE, &viewer));
    PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_HDF5_XDMF));
    PetscCall(DMView(rdy->dm, viewer));
  } else {
    PetscCall(PetscViewerHDF5Open(rdy->comm, fname, FILE_MODE_APPEND, &viewer));
    PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_HDF5_XDMF));
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
  */

  PetscFunctionReturn(0);
}
