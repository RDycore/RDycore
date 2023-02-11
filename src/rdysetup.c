#include <petscdmplex.h>
#include <private/rdycoreimpl.h>
#include <rdycore.h>

extern PetscErrorCode ParseConfigFile(FILE *file, RDy rdy);

// sets default parameters
static PetscErrorCode SetDefaults(RDy rdy) {
  PetscFunctionBegin;

  rdy->log_level = LOG_INFO;

  PetscFunctionReturn(0);
}

// checks the validity of initial/boundary conditions and sources
static PetscErrorCode CheckConditionsAndSources(RDy rdy) {
  PetscFunctionBegin;

  if (!strlen(rdy->initial_conditions_file)) {
    // Does every region have a set of initial conditions?
    for (PetscInt r = 0; r < rdy->num_regions; ++r) {
      PetscCheck(rdy->initial_conditions[r].flow, rdy->comm, PETSC_ERR_USER,
          "Region %d has no initial flow condition!", rdy->region_ids[r]);
      if (rdy->sediment) {
        PetscCheck(rdy->initial_conditions[r].sediment, rdy->comm, PETSC_ERR_USER,
            "Region %d has no initial sediment condition!", rdy->region_ids[r]);
      }
      if (rdy->salinity) {
        PetscCheck(rdy->initial_conditions[r].salinity, rdy->comm, PETSC_ERR_USER,
            "Region %d has no initial salinity condition!", rdy->region_ids[r]);
      }
    }
  }

  // Does every surface have a set of boundary conditions?
  for (PetscInt s = 0; s < rdy->num_surfaces; ++s) {
    PetscCheck(rdy->boundary_conditions[s].flow, rdy->comm, PETSC_ERR_USER,
      "Surface %d has no flow boundary condition!", rdy->surface_ids[s]);
    if (rdy->sediment) {
      PetscCheck(rdy->boundary_conditions[s].sediment, rdy->comm, PETSC_ERR_USER,
        "Surface %d has no sediment boundary condition!", rdy->surface_ids[s]);
    }
    if (rdy->salinity) {
      PetscCheck(rdy->boundary_conditions[s].salinity, rdy->comm, PETSC_ERR_USER,
        "Surface %d has no salinity boundary condition!", rdy->surface_ids[s]);
    }
  }

  // (no source checks so far)

  PetscFunctionReturn(0);
}

/// Performs any setup needed by RDy, reading from the specified configuration
/// file.
PetscErrorCode RDySetup(RDy rdy) {
  PetscFunctionBegin;

  PetscCall(SetDefaults(rdy));

  // open the config file and set up a YAML parser for it
  FILE *file = fopen(rdy->config_file, "r");
  PetscCheck(file, rdy->comm, PETSC_ERR_USER,
    "Configuration file not found: %s", rdy->config_file);

  PetscCall(ParseConfigFile(file, rdy));
  fclose(file);

  // check initial/boundary conditions and sources
  PetscCall(CheckConditionsAndSources(rdy));

  // open the primary log file
  if (strlen(rdy->log_file)) {
    PetscCall(PetscFOpen(rdy->comm, rdy->log_file, "w", &rdy->log));
  } else {
    rdy->log = stdout;
  }

  // print configuration info
  PetscCall(RDyPrintf(rdy));

  // set up bookkeeping data structures
  // FIXME

  PetscFunctionReturn(0);
}
