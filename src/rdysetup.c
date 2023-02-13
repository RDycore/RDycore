#include <petscdmplex.h>
#include <private/rdycoreimpl.h>
#include <private/rdymemoryimpl.h>
#include <rdycore.h>

extern PetscErrorCode ReadConfigFile(RDy rdy);
extern PetscErrorCode PrintConfig(RDy rdy);

// sets default parameters
static PetscErrorCode SetDefaults(RDy rdy) {
  PetscFunctionBegin;

  rdy->log_level = LOG_INFO;

  PetscFunctionReturn(0);
}

// overrides parameters with command line arguments
static PetscErrorCode OverrideParameters(RDy rdy) {
  PetscFunctionBegin;

  // FIXME

  PetscFunctionReturn(0);
}

// checks the validity of initial/boundary conditions and sources
static PetscErrorCode CheckConditionsAndSources(RDy rdy) {
  PetscFunctionBegin;

  // Set up a reflecting flow boundary condition.
  RDyFlowCondition* reflecting_flow = NULL;
  for (PetscInt s = 0; s <= MAX_SURFACE_ID; ++s) {
    if (!rdy->flow_conditions[s].name) {
      reflecting_flow = &rdy->flow_conditions[s];
      RDyAlloc(char, strlen("reflecting")+1, &reflecting_flow->name);
      strcpy((char*)reflecting_flow->name, "reflecting");
      reflecting_flow->type = CONDITION_REFLECTING;
      break;
    }
  }
  PetscCheck(reflecting_flow, rdy->comm, PETSC_ERR_USER,
    "Could not allocate a reflecting flow condition! Please increase MAX_SURFACE_ID.");

  if (!strlen(rdy->initial_conditions_file)) {
    // Does every region have a set of initial conditions?
    for (PetscInt r = 1; r <= rdy->num_regions; ++r) {
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
  for (PetscInt s = 1; s <= rdy->num_surfaces; ++s) {
    // If no flow condition was specified for a boundary, we set it to our
    // reflecting flow condition.
    if (!rdy->boundary_conditions[s].flow) {
      RDyLogDebug(rdy, "Setting reflecting flow condition for surface %d\n", s);
      rdy->boundary_conditions[s].flow = reflecting_flow;
    }
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

// Configures bookkeeping data structures for the simulation
static PetscErrorCode InitSimulationData(RDy rdy) {
  PetscFunctionBegin;

  // FIXME

  PetscFunctionReturn(0);
}

/// Performs any setup needed by RDy, reading from the specified configuration
/// file.
PetscErrorCode RDySetup(RDy rdy) {
  PetscFunctionBegin;

  PetscCall(SetDefaults(rdy));
  PetscCall(ReadConfigFile(rdy));

  // open the primary log file
  if (strlen(rdy->log_file)) {
    PetscCall(PetscFOpen(rdy->comm, rdy->log_file, "w", &rdy->log));
  } else {
    rdy->log = stdout;
  }

  // override parameters using command line arguments
  PetscCall(OverrideParameters(rdy));

  // check initial/boundary conditions and sources
  PetscCall(CheckConditionsAndSources(rdy));

  // print configuration info
  PetscCall(PrintConfig(rdy));

  // set up bookkeeping data structures
  PetscCall(InitSimulationData(rdy));

  PetscFunctionReturn(0);
}
