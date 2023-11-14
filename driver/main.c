#include <petscsys.h>
#include <rdycore.h>

static const char *help_str =
    "rdycore - a standalone driver for RDycore\n"
    "usage: rdycore [options] <filename>\n";

static void usage(const char *exe_name) {
  fprintf(stderr, "%s: usage:\n", exe_name);
  fprintf(stderr, "%s <input.yaml>\n\n", exe_name);
}

// open a Vec that contains data in the following format:
//
// time_1 value_1
// time_2 value_2
// time_3 value_3
static PetscErrorCode OpenData(char *filename, Vec *data_vec, PetscInt *ndata) {
  PetscFunctionBegin;

  PetscViewer viewer;
  PetscCall(VecCreate(PETSC_COMM_SELF, data_vec));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF, filename, FILE_MODE_READ, &viewer));
  PetscCall(VecLoad(*data_vec, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscInt size;
  PetscCall(VecGetSize(*data_vec, &size));
  *ndata = size / 2;

  PetscFunctionReturn(PETSC_SUCCESS);
}

// For a given cur_time,
//   cur_data = value_1 if cur_time >= time_1 and cur_time < time_2
PetscErrorCode GetCurrentData(PetscScalar *data_ptr, PetscInt ndata, PetscReal cur_time, PetscBool temporally_interpolate, PetscInt *cur_data_idx, PetscReal *cur_data) {
  PetscFunctionBegin;

  PetscBool found  = PETSC_FALSE;
  PetscInt  stride = 2;
  PetscReal time_up, time_dn;
  PetscReal data_up, data_dn;

  for (PetscInt itime = 0; itime < ndata - 1; itime++) {
    time_dn = data_ptr[itime * stride];
    data_dn = data_ptr[itime * stride + 1];

    time_up = data_ptr[itime * stride + 2];
    data_up = data_ptr[itime * stride + 3];

    if (cur_time >= time_dn && cur_time < time_up) {
      found         = PETSC_TRUE;
      *cur_data_idx = itime;
      break;
    }
  }

  if (!found) {
    *cur_data_idx = ndata - 1;
    *cur_data     = data_ptr[ndata * 2 - 1];
  } else {
    if (temporally_interpolate) {
      *cur_data = (cur_time - time_dn)/(time_up - time_dn) * (data_up - data_dn) + data_dn;
    } else {
      *cur_data = data_dn;
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[]) {
  // print usage info if no arguments given
  if (argc < 2) {
    usage(argv[0]);
    exit(-1);
  }

  // initialize subsystems
  PetscCall(RDyInit(argc, argv, help_str));

  if (strcmp(argv[1], "-help")) {  // if given a config file
    // create rdycore and set it up with the given file
    MPI_Comm comm = PETSC_COMM_WORLD;
    RDy      rdy;
    PetscCall(RDyCreate(comm, argv[1], &rdy));

    // Open datasets if specified via command line arguments.
    // Currently, these datasets need to be opened before calling
    // RDySetup to avoid an error. Possible issue for the error could
    // be that RDySetup is setting a default DM for all VecLoads.
    PetscBool rain_specified, bc_specified;
    char      rainfile[PETSC_MAX_PATH_LEN], bcfile[PETSC_MAX_PATH_LEN];
    PetscCall(PetscOptionsGetString(NULL, NULL, "-rain", rainfile, sizeof(rainfile), &rain_specified));
    PetscCall(PetscOptionsGetString(NULL, NULL, "-bc", bcfile, sizeof(bcfile), &bc_specified));

    PetscBool interpolate_rain = PETSC_FALSE, interpolate_bc = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-interpolate_rain", &interpolate_rain, NULL));
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-interpolate_bc", &interpolate_bc, NULL));

    Vec          rain_vec = NULL, bc_vec = NULL;
    PetscScalar *rain_ptr = NULL, *bc_ptr = NULL;
    PetscInt     nrain, nbc;

    if (rain_specified) {
      PetscCall(OpenData(rainfile, &rain_vec, &nrain));
      PetscCall(VecGetArray(rain_vec, &rain_ptr));
    }

    if (bc_specified) {
      PetscCall(OpenData(bcfile, &bc_vec, &nbc));
      PetscCall(VecGetArray(bc_vec, &bc_ptr));
    }

    PetscCall(RDySetup(rdy));

    // allocate arrays for inspecting simulation data
    PetscInt n;
    PetscCall(RDyGetNumLocalCells(rdy, &n));
    PetscReal *h, *vx, *vy, *rain;
    PetscCalloc1(n, &h);
    PetscCalloc1(n, &vx);
    PetscCalloc1(n, &vy);
    PetscCalloc1(n, &rain);

    // get information about boundary conditions
    PetscInt nbcs, dirc_bc_idx = -1, num_edges_dirc_bc = 0;
    PetscCall(RDyGetNumBoundaryConditions(rdy, &nbcs));
    for (PetscInt ibc = 0; ibc < nbcs; ibc++) {
      PetscInt num_edges, bc_type;
      PetscCall(RDyGetNumBoundaryEdges(rdy, ibc, &num_edges));
      PetscCall(RDyGetBoundaryConditionFlowType(rdy, ibc, &bc_type));
      if (bc_type == CONDITION_DIRICHLET) {
        if (bc_specified) {
          PetscCheck(dirc_bc_idx == -1, comm, PETSC_ERR_USER,
                     "When BC file specified via -bc argument, only one CONDITION_DIRICHLET can be present in the yaml");
        }
        dirc_bc_idx       = ibc;
        num_edges_dirc_bc = num_edges;
      }
    }

    if (bc_specified) {
      PetscInt global_dirc_bc_idx = -1;
      MPI_Allreduce(&dirc_bc_idx, &global_dirc_bc_idx, 1, MPI_INT, MPI_MAX, comm);
      PetscCheck(global_dirc_bc_idx > -1, comm, PETSC_ERR_USER,
                 "The BC file specified via -bc argument, but no CONDITION_DIRICHLET found in the yaml");
    }
    PetscReal *bc_values;
    PetscCalloc1(num_edges_dirc_bc * 3, &bc_values);

    // run the simulation to completion using the time parameters in the
    // config file
    PetscReal prev_time, coupling_interval;
    RDyGetTime(rdy, &prev_time);
    RDyGetCouplingInterval(rdy, &coupling_interval);
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-coupling_interval", &coupling_interval, NULL));
    RDySetCouplingInterval(rdy, coupling_interval);

    PetscInt cur_rain_idx = -1, prev_rain_idx = -1;
    PetscInt cur_bc_idx = -1, prev_bc_idx = -1;

    while (!RDyFinished(rdy)) {  // returns true based on stopping criteria

      PetscReal time, time_step;
      PetscCall(RDyGetTime(rdy, &time));

      if (!rain_specified) {
        // apply a 1 mm/hr rain over the entire domain
        for (PetscInt icell = 0; icell < n; icell++) {
          rain[icell] = 1.0 / 3600.0 / 1000.0;  // mm/hr --> m/s
        }
        PetscCall(RDySetWaterSource(rdy, rain));
      } else {
        PetscReal cur_rain;
        PetscCall(GetCurrentData(rain_ptr, nrain, time, interpolate_rain, &cur_rain_idx, &cur_rain));

        if (interpolate_rain || cur_rain_idx != prev_rain_idx) {  // is it time to update the source term?
          prev_rain_idx = cur_rain_idx;
          for (PetscInt icell = 0; icell < n; icell++) {
            rain[icell] = cur_rain;
          }
          PetscCall(RDySetWaterSource(rdy, rain));
        }
      }

      if (bc_specified && num_edges_dirc_bc > 0) {
        PetscReal cur_bc;
        PetscCall(GetCurrentData(bc_ptr, nbc, time, interpolate_bc, &cur_bc_idx, &cur_bc));
        if (interpolate_bc || cur_bc_idx != prev_bc_idx) {  // is it time to update the bc?
          prev_bc_idx = cur_bc_idx;
          for (PetscInt iedge = 0; iedge < num_edges_dirc_bc; iedge++) {
            bc_values[iedge * 3]     = cur_bc;
            bc_values[iedge * 3 + 1] = 0.0;
            bc_values[iedge * 3 + 2] = 0.0;
          }
          PetscCall(RDySetDirichletBoundaryValues(rdy, dirc_bc_idx, num_edges_dirc_bc, 3, bc_values));
        }
      }

      // advance the solution by the coupling interval specified in the config file
      PetscCall(RDyAdvance(rdy));

      // the following just check that RDycore is doing the right thing

      PetscCall(RDyGetTime(rdy, &time));
      PetscCall(RDyGetTimeStep(rdy, &time_step));
      PetscCheck(time > prev_time, comm, PETSC_ERR_USER, "Non-increasing time!");
      PetscCheck(time_step > 0.0, comm, PETSC_ERR_USER, "Non-positive time step!");

      PetscCheck(fabs(time - prev_time - coupling_interval) < 1e-12, comm, PETSC_ERR_USER, "RDyAdvance advanced time improperly (%g, %g, %g)!",
                 prev_time, time, fabs(time - prev_time + coupling_interval));
      prev_time += coupling_interval;

      PetscInt step;
      PetscCall(RDyGetStep(rdy, &step));
      PetscCheck(step > 0, comm, PETSC_ERR_USER, "Non-positive step index!");

      PetscCall(RDyGetHeight(rdy, h));
      PetscCall(RDyGetXVelocity(rdy, vx));
      PetscCall(RDyGetYVelocity(rdy, vy));
    }

    // clean up
    if (rain_specified) {
      PetscCall(VecRestoreArray(rain_vec, &rain_ptr));
      PetscCall(VecDestroy(&rain_vec));
    }

    if (bc_specified) {
      PetscCall(VecRestoreArray(bc_vec, &bc_ptr));
      PetscCall(VecDestroy(&bc_vec));
    }

    PetscFree(h);
    PetscFree(vx);
    PetscFree(vy);
    PetscFree(rain);
    PetscFree(bc_values);
    PetscCall(RDyDestroy(&rdy));
  }

  PetscCall(RDyFinalize());
  return 0;
}
