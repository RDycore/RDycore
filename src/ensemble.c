#include <private/rdycoreimpl.h>

// in ensemble mode: configures the ensemble member residing on the local process,
// copying overridden sections from the member's configuration
PetscErrorCode ConfigureEnsembleMember(RDy rdy) {
  PetscFunctionBegin;

  // split the global communicator (evenly)
  PetscMPIInt procs_per_ensemble = rdy->nproc / rdy->config.ensemble.size;
  PetscMPIInt color              = rdy->rank / procs_per_ensemble;  // ensemble member index
  PetscMPIInt key                = 0;                               // let MPI decide how to order ranks
  MPI_Comm_free(&rdy->comm);
  MPI_Comm_split(rdy->global_comm, color, key, &rdy->comm);
  MPI_Comm_size(rdy->comm, &rdy->nproc);
  MPI_Comm_rank(rdy->comm, &rdy->rank);
  rdy->ensemble_member_index = color;

  // assign the ensemble a default name if it doesn't have one
  if (!rdy->config.ensemble.members[color].name[0]) {
    int  num_digits = (int)(log10((double)rdy->config.ensemble.size)) + 1;
    char fmt[16]    = {0};
    snprintf(fmt, 15, "%%0%dd", num_digits);
    char suffix[16];
    sprintf(suffix, fmt, color);
    sprintf(rdy->config.ensemble.members[color].name, "ensemble_%s", suffix);
  }

  // override ensemble member parameters by copying them into place
  RDyEnsembleMember member_config = rdy->config.ensemble.members[color];

  // grid
  if (member_config.grid.file[0]) {
    rdy->config.grid = member_config.grid;
  }

  // materials
  for (PetscInt m = 0; m < member_config.num_overridden_materials; ++m) {
    // find the specified material
    for (PetscInt mm = 0; mm < rdy->config.num_materials; ++mm) {
      if (!strcmp(rdy->config.materials[mm].name, member_config.materials[m].name)) {
        rdy->config.materials[mm] = member_config.materials[m];
        break;
      }
    }
  }

  // flow conditions
  for (PetscInt c = 0; c < member_config.num_overridden_flow_conditions; ++c) {
    // find the specified flow condition
    for (PetscInt cc = 0; cc < rdy->config.num_flow_conditions; ++cc) {
      if (!strcmp(rdy->config.flow_conditions[cc].name, member_config.flow_conditions[c].name)) {
        rdy->config.flow_conditions[cc] = member_config.flow_conditions[c];
        break;
      }
    }
  }

  // sediment conditions
  for (PetscInt c = 0; c < member_config.num_overridden_sediment_conditions; ++c) {
    // find the specified sediment condition
    for (PetscInt cc = 0; cc < rdy->config.num_sediment_conditions; ++cc) {
      if (!strcmp(rdy->config.sediment_conditions[cc].name, member_config.sediment_conditions[c].name)) {
        rdy->config.sediment_conditions[cc] = member_config.sediment_conditions[c];
        break;
      }
    }
  }

  // salinity conditions
  for (PetscInt c = 0; c < member_config.num_overridden_salinity_conditions; ++c) {
    // find the specified salinity condition
    for (PetscInt cc = 0; cc < rdy->config.num_salinity_conditions; ++cc) {
      if (!strcmp(rdy->config.salinity_conditions[cc].name, member_config.salinity_conditions[c].name)) {
        rdy->config.salinity_conditions[cc] = member_config.salinity_conditions[c];
        break;
      }
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}
