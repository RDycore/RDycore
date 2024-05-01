// This code supports the C and Fortran MMS drivers and is not used in
// mainline RDycore (though it is built into the library).

#include <muParserDLL.h>
#include <petscconvest.h>
#include <petscdmceed.h>
#include <petscdmplex.h>
#include <petscsys.h>
#include <private/rdycoreimpl.h>
#include <private/rdydmimpl.h>
#include <private/rdymathimpl.h>

// gravitational acceleration [m/s/s]
static const PetscReal GRAVITY = 9.806;

// NOTE: our boundary conditions are expressed in terms of momenta and not flow
// velocities, so we have to chain together a few things to evaluate x and y
// momenta.

static PetscErrorCode SetSWEAnalyticBoundaryCondition(RDy rdy) {
  PetscFunctionBegin;

  // We only need a single Dirichlet boundary condition, populated with
  // manufactured solution data.
  static RDyFlowCondition analytic_flow = {
      .name = "analytic_bc",
      .type = CONDITION_DIRICHLET,
  };
  analytic_flow.height     = rdy->config.mms.swe.solutions.h;
  analytic_flow.x_momentum = rdy->config.mms.swe.solutions.u;  // NOTE: must multiply by h when enforcing!
  analytic_flow.y_momentum = rdy->config.mms.swe.solutions.v;  // NOTE: must multiply by h when enforcing!

  RDyCondition analytic_bc = {
      .flow = &analytic_flow,
  };

  // Assign the boundary condition to each boundary.
  PetscCall(PetscCalloc1(rdy->num_boundaries, &rdy->boundary_conditions));
  for (PetscInt b = 0; b < rdy->num_boundaries; ++b) {
    rdy->boundary_conditions[b] = analytic_bc;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

#define SET_SPATIAL_VARIABLES(func) \
  mupDefineBulkVar(func, "x", x);   \
  mupDefineBulkVar(func, "y", y)

// evaluates the given expression at all given x, y, placing the results into values
static PetscErrorCode EvaluateSpatialSolution(void *expr, PetscInt n, PetscReal x[n], PetscReal y[n], PetscReal values[n]) {
  PetscFunctionBegin;

  SET_SPATIAL_VARIABLES(expr);
  mupEvalBulk(expr, values, n);

  PetscFunctionReturn(PETSC_SUCCESS);
}

#define SET_SPATIOTEMPORAL_VARIABLES(func) \
  SET_SPATIAL_VARIABLES(func);             \
  mupDefineBulkVar(func, "t", t)

// evaluates the given expression at all given x, y, t, placing the results into values
static PetscErrorCode EvaluateTemporalSolution(void *expr, PetscInt n, PetscReal x[n], PetscReal y[n], PetscReal time, PetscReal values[n]) {
  PetscFunctionBegin;

  PetscReal t[n];
  for (PetscInt i = 0; i < n; ++i) t[i] = time;
  SET_SPATIOTEMPORAL_VARIABLES(expr);
  mupEvalBulk(expr, values, n);

  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef SET_SPATIAL_VARIABLES
#undef SET_SPATIOTEMPORAL_VARIABLES

// this function gets called at the beginning of each time step, updating
// source terms and boundary conditions at a properly centered time
static PetscErrorCode MMSPreStep(TS ts) {
  PetscFunctionBegin;

  RDy rdy;
  PetscCall(TSGetApplicationContext(ts, (void *)&rdy));

  PetscReal t, dt;
  PetscCall(TSGetTime(ts, &t));
  PetscCall(TSGetTimeStep(ts, &dt));

  PetscCall(RDyMMSEnforceBoundaryConditions(rdy, t + 0.5 * dt));
  PetscCall(RDyMMSComputeSourceTerms(rdy, t + 0.5 * dt));

  PetscFunctionReturn(PETSC_SUCCESS);
}

extern PetscErrorCode PauseIfRequested(RDy rdy);  // for -pause support

// this can be used in place of RDySetup for the MMS driver, which uses a
// modified YAML input schema (see ReadMMSConfigFile in yaml_input.c)
PetscErrorCode RDyMMSSetup(RDy rdy) {
  PetscFunctionBegin;

  PetscCall(PauseIfRequested(rdy));

  PetscCall(ReadMMSConfigFile(rdy));

  // open the primary log file
  if (strlen(rdy->config.logging.file)) {
    PetscCall(PetscFOpen(rdy->comm, rdy->config.logging.file, "w", &rdy->log));
  } else {
    rdy->log = stdout;
  }

  // override parameters using command line arguments
  PetscCall(OverrideParameters(rdy));

  // initialize CEED if needed
  if (rdy->ceed_resource[0]) {
    PetscCallCEED(CeedInit(rdy->ceed_resource, &rdy->ceed));
  }

  RDyLogDebug(rdy, "Creating DMs...");
  PetscCall(CreateDM(rdy));           // for mesh and solution vector
  PetscCall(CreateAuxiliaryDM(rdy));  // for diagnostics
  PetscCall(CreateVectors(rdy));      // global and local vectors, residuals

  RDyLogDebug(rdy, "Initializing regions...");
  PetscCall(InitRegions(rdy));

  // note: this must be done after global vectors are created so a global
  // note: section exists for the DM
  RDyLogDebug(rdy, "Creating FV mesh...");
  PetscCall(RDyMeshCreateFromDM(rdy->dm, &rdy->mesh));

  RDyLogDebug(rdy, "Initializing boundaries and boundary conditions...");
  PetscCall(InitBoundaries(rdy));
  PetscCall(SetSWEAnalyticBoundaryCondition(rdy));

  RDyLogDebug(rdy, "Initializing shallow water equations solver...");
  PetscCall(InitSWE(rdy));
  PetscCall(TSSetPreStep(rdy->ts, MMSPreStep));

  RDyLogDebug(rdy, "Initializing solution and source data...");
  PetscCall(RDyMMSComputeSolution(rdy, 0.0, rdy->X));
  PetscCall(PetscCalloc1(rdy->mesh.num_cells, &rdy->materials_by_cell));
  PetscCall(RDyMMSUpdateMaterialProperties(rdy));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// evaluates the relevant manufactured solution at the given time, placing the
// solution into the given vector
PetscErrorCode RDyMMSComputeSolution(RDy rdy, PetscReal time, Vec solution) {
  PetscFunctionBegin;

  PetscCall(VecZeroEntries(solution));

  // initialize the manufactured solution on each region
  PetscInt n_local, ndof;
  PetscCall(VecGetLocalSize(solution, &n_local));
  PetscCall(VecGetBlockSize(solution, &ndof));
  PetscScalar *x_ptr;
  PetscCall(VecGetArray(solution, &x_ptr));

  for (PetscInt r = 0; r < rdy->num_regions; ++r) {
    RDyRegion region = rdy->regions[r];

    // Create vectorized (x, y, t) triples for bulk expression evaluation
    PetscReal cell_x[region.num_cells], cell_y[region.num_cells];
    PetscInt  N = 0;  // number of bulk evaluations
    for (PetscInt c = 0; c < region.num_cells; ++c) {
      PetscInt cell_id = region.cell_ids[c];
      if (3 * cell_id < n_local) {
        cell_x[N] = rdy->mesh.cells.centroids[cell_id].X[0];
        cell_y[N] = rdy->mesh.cells.centroids[cell_id].X[1];
        ++N;
      }
    }

    if (rdy->config.physics.flow.mode == FLOW_SWE) {
      PetscCheck(ndof == 3, rdy->comm, PETSC_ERR_USER, "SWE solution vector has %" PetscInt_FMT " DOF (should have 3)", ndof);

      // evaluate the manufactured ѕolutions at all (x, y, t)
      PetscReal h[N], u[N], v[N];
      PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.h, N, cell_x, cell_y, time, h));
      PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.u, N, cell_x, cell_y, time, u));
      PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.v, N, cell_x, cell_y, time, v));

      // TODO: salinity and sediment initial conditions go here.

      PetscInt l = 0;
      for (PetscInt c = 0; c < region.num_cells; ++c) {
        PetscInt cell_id = region.cell_ids[c];
        if (3 * cell_id < n_local) {  // skip ghost cells
          x_ptr[3 * cell_id]     = h[l];
          x_ptr[3 * cell_id + 1] = h[l] * u[l];
          x_ptr[3 * cell_id + 2] = h[l] * v[l];
          ++l;
        }
      }
    }
  }

  PetscCall(VecRestoreArray(solution, &x_ptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// evaluates the source terms associated with the manufactured solutions
PetscErrorCode RDyMMSComputeSourceTerms(RDy rdy, PetscReal time) {
  PetscFunctionBegin;

  RDyMesh  *mesh  = &rdy->mesh;
  RDyCells *cells = &mesh->cells;

  PetscInt N;
  PetscCall(RDyGetNumLocalCells(rdy, &N));
  PetscReal cell_x[N], cell_y[N];

  PetscInt l = 0;
  for (PetscInt icell = 0; icell < mesh->num_cells; icell++) {
    if (cells->is_local[icell]) {
      cell_x[l] = rdy->mesh.cells.centroids[icell].X[0];
      cell_y[l] = rdy->mesh.cells.centroids[icell].X[1];
      ++l;
    }
  }

  if (rdy->config.physics.flow.mode == FLOW_SWE) {
    // evaluate the manufactured ѕolutions at all (x, y, t)
    PetscReal h[N], u[N], v[N];
    PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.h, N, cell_x, cell_y, time, h));
    PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.u, N, cell_x, cell_y, time, u));
    PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.v, N, cell_x, cell_y, time, v));

    PetscReal dhdx[N], dhdy[N], dhdt[N];
    PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.dhdx, N, cell_x, cell_y, time, dhdx));
    PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.dhdy, N, cell_x, cell_y, time, dhdy));
    PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.dhdt, N, cell_x, cell_y, time, dhdt));

    PetscReal dudx[N], dudy[N], dudt[N];
    PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.dudx, N, cell_x, cell_y, time, dudx));
    PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.dudy, N, cell_x, cell_y, time, dudy));
    PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.dudt, N, cell_x, cell_y, time, dudt));

    PetscReal dvdx[N], dvdy[N], dvdt[N];
    PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.dvdx, N, cell_x, cell_y, time, dvdx));
    PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.dvdy, N, cell_x, cell_y, time, dvdy));
    PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.dvdt, N, cell_x, cell_y, time, dvdt));

    PetscReal n[N];
    PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.n, N, cell_x, cell_y, time, n));

    PetscReal dzdx[N], dzdy[N];
    PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.dzdx, N, cell_x, cell_y, time, dzdx));
    PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.dzdy, N, cell_x, cell_y, time, dzdy));

    PetscReal h_source[N], hu_source[N], hv_source[N];

    l = 0;
    for (PetscInt icell = 0; icell < mesh->num_cells; icell++) {
      if (cells->is_local[icell]) {
        PetscReal Cd = GRAVITY * Square(n[l]) * PetscPowReal(h[l], -1.0 / 3.0);

        h_source[l] = dhdt[l] + u[l] * dhdx[l] + h[l] * dudx[l] + v[l] * dhdy[l] + h[l] * dvdy[l];

        hu_source[l] = u[l] * dhdt[l] + h[l] * dudt[l];
        hu_source[l] += 2.0 * u[l] * h[l] * dudx[l] + u[l] * u[l] * dhdx[l] + GRAVITY * h[l] * dhdx[l];
        hu_source[l] += u[l] * h[l] * dvdy[l] + v[l] * h[l] * dudy[l] + u[l] * v[l] * dhdy[l];
        hu_source[l] += dzdx[l] * GRAVITY * h[l];
        hu_source[l] += Cd * u[l] * PetscSqrtReal(u[l] * u[l] + v[l] * v[l]);

        hv_source[l] = v[l] * dhdt[l] + h[l] * dvdt[l];
        hv_source[l] += u[l] * h[l] * dvdx[l] + v[l] * h[l] * dudx[l] + u[l] * v[l] * dhdx[l];
        hv_source[l] += v[l] * v[l] * dhdy[l] + 2.0 * v[l] * h[l] * dvdy[l] + GRAVITY * h[l] * dhdy[l];
        hv_source[l] += dzdy[l] * GRAVITY * h[l];
        hv_source[l] += Cd * v[l] * PetscSqrtReal(u[l] * u[l] + v[l] * v[l]);
        ++l;
      }
    }

    PetscCall(RDySetWaterSourceForLocalCells(rdy, N, h_source));
    PetscCall(RDySetXMomentumSourceForLocalCells(rdy, N, hu_source));
    PetscCall(RDySetYMomentumSourceForLocalCells(rdy, N, hv_source));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// call this to enforce analytical boundary conditions in the MMS driver
PetscErrorCode RDyMMSEnforceBoundaryConditions(RDy rdy, PetscReal time) {
  PetscFunctionBegin;

  RDyLogDebug(rdy, "Enforcing MMS boundary conditions...");

  for (PetscInt b = 0; b < rdy->num_boundaries; ++b) {
    // fetch x, y for each edge (and set t = time)
    RDyBoundary boundary  = rdy->boundaries[b];
    PetscInt    num_edges = boundary.num_edges;
    PetscReal   x[num_edges], y[num_edges];
    for (PetscInt e = 0; e < num_edges; ++e) {
      PetscInt edge_id       = boundary.edge_ids[e];
      RDyPoint edge_centroid = rdy->mesh.edges.centroids[edge_id];
      x[e]                   = edge_centroid.X[0];
      y[e]                   = edge_centroid.X[1];
    }

    // compute h, hu, hv on each edge (SWE-specific)
    RDyFlowCondition *flow_bc = rdy->boundary_conditions[b].flow;
    PetscReal         h[num_edges], u[num_edges], v[num_edges];
    PetscCall(EvaluateTemporalSolution(flow_bc->height, num_edges, x, y, time, h));
    PetscCall(EvaluateTemporalSolution(flow_bc->x_momentum, num_edges, x, y, time, u));
    PetscCall(EvaluateTemporalSolution(flow_bc->y_momentum, num_edges, x, y, time, v));

    // set the boundary values (SWE-specific)
    // NOTE: ndof == 3 for SWE
    PetscReal boundary_values[3 * num_edges];
    for (PetscInt e = 0; e < num_edges; ++e) {
      boundary_values[3 * e]     = h[e];
      boundary_values[3 * e + 1] = h[e] * u[e];
      boundary_values[3 * e + 2] = h[e] * v[e];
    }
    PetscCall(RDySetDirichletBoundaryValues(rdy, b, num_edges, 3, boundary_values));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// updates relevant material properties for the method of manufactured solutions
// at the given time
PetscErrorCode RDyMMSUpdateMaterialProperties(RDy rdy) {
  PetscFunctionBegin;

  // initialize the material properties on each region
  PetscInt n_local;
  PetscCall(VecGetLocalSize(rdy->X, &n_local));

  for (PetscInt r = 0; r < rdy->num_regions; ++r) {
    RDyRegion region = rdy->regions[r];

    // create vectorized (x, y) pairs for bulk expression evaluation
    PetscReal cell_x[region.num_cells], cell_y[region.num_cells];
    PetscInt  N = 0;  // number of bulk evaluations
    for (PetscInt c = 0; c < region.num_cells; ++c) {
      PetscInt cell_id = region.cell_ids[c];
      if (3 * cell_id < n_local) {
        cell_x[N] = rdy->mesh.cells.centroids[cell_id].X[0];
        cell_y[N] = rdy->mesh.cells.centroids[cell_id].X[1];
        ++N;
      }
    }

    // evaluate and set material properties
    if (rdy->config.physics.flow.mode == FLOW_SWE) {
      PetscReal manning[N];
      PetscCall(EvaluateSpatialSolution(rdy->config.mms.swe.solutions.n, N, cell_x, cell_y, manning));
      PetscCall(RDySetManningsNForLocalCells(rdy, N, manning));
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Computes the componentwise L1, L2, and Linf error norms for the relevant
// manufactured solution at the given time. L1_norms, L2_norms, and Linf_norms
// are all arrays large enough to store the number of dof. If non-NULL,
// num_global_cells stores the number of distinct global cells and global_area
// stores the total area covered by distinct global cells.
PetscErrorCode RDyMMSComputeErrorNorms(RDy rdy, PetscReal time, PetscReal *L1_norms, PetscReal *L2_norms, PetscReal *Linf_norms,
                                       PetscInt *num_global_cells, PetscReal *global_area) {
  PetscFunctionBegin;
  // compute the error vector
  Vec error;
  PetscCall(RDyCreatePrognosticVec(rdy, &error));
  PetscCall(RDyMMSComputeSolution(rdy, time, error));
  PetscCall(VecAYPX(error, -1.0, rdy->X));

  PetscInt ndof;
  PetscCall(VecGetBlockSize(error, &ndof));

  // compute the componentwise error norms on local cells
  PetscReal *e;
  PetscCall(VecGetArray(error, &e));
  PetscReal area_sum = 0.0;
  memset(L1_norms, 0, ndof * sizeof(PetscReal));
  memset(L2_norms, 0, ndof * sizeof(PetscReal));
  memset(Linf_norms, 0, ndof * sizeof(PetscReal));
  for (PetscInt i = 0; i < rdy->mesh.num_cells_local; ++i) {
    PetscReal area = rdy->mesh.cells.areas[i];

    for (PetscInt dof = 0; dof < ndof; ++dof) {
      PetscReal e_dof = e[ndof * i + dof];
      L1_norms[dof] += PetscAbsReal(e_dof) * area;
      L2_norms[dof] += e_dof * e_dof * area;
      Linf_norms[dof] = PetscMax(e_dof, Linf_norms[dof]);
    }
    area_sum += area;
  }
  PetscCall(VecRestoreArray(error, &e));
  PetscCall(VecDestroy(&error));

  // obtain global error norms
  PetscCall(MPI_Allreduce(MPI_IN_PLACE, L1_norms, ndof, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD));
  PetscCall(MPI_Allreduce(MPI_IN_PLACE, L2_norms, ndof, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD));
  PetscCall(MPI_Allreduce(MPI_IN_PLACE, Linf_norms, ndof, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD));

  for (PetscInt dof = 0; dof < ndof; ++dof) {
    L2_norms[dof] = PetscSqrtReal(L2_norms[dof]);
  }

  // obtain optional diagnostics
  if (num_global_cells) {
    PetscMPIInt ncells;
    PetscCall(MPI_Reduce(&rdy->mesh.num_cells_local, &ncells, 1, MPI_INT, MPI_SUM, 0, PETSC_COMM_WORLD));
    *num_global_cells = (PetscInt)ncells;
  }
  if (global_area) {
    PetscCall(MPI_Reduce(&area_sum, global_area, 1, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// performs a temporo-spatial convergence study using the given instance of RDy
// as a coarse grid, uniformly refining it the specified number of times,
// evolving the solution to the given time, computing error norms for each
// component, and calculating rates of convergence (and variances) with linear
// regression
PetscErrorCode RDyMMSEstimateConvergenceRates(RDy rdy, PetscInt num_refinements, PetscReal *L1_conv_rates, PetscReal *L2_conv_rates,
                                              PetscReal *Linf_conv_rates) {
  PetscFunctionBegin;

  PetscReal final_time = rdy->config.time.final_time;

  PetscInt dim;
  PetscCall(DMGetDimension(rdy->dm, &dim));

  // error norm storage
  PetscInt  num_comps = 3;  // SWE only!
  PetscReal L1_norms[num_refinements + 1][num_comps], L2_norms[num_refinements + 1][num_comps], Linf_norms[num_refinements + 1][num_comps];

  // create refined RDy objects and set them up (dumb, but easy)
  RDy rdys[num_refinements + 1];
  rdys[0] = rdy;
  for (PetscInt r = 1; r <= num_refinements; ++r) {
    PetscCall(RDyCreate(rdy->comm, rdy->config_file, &rdys[r]));
    char num_refinements[5];
    snprintf(num_refinements, 4, "%" PetscInt_FMT, r);
    PetscOptionsSetValue(NULL, "-dm_refine", num_refinements);
    PetscCall(RDyMMSSetup(rdys[r]));

    // override timestepping info (no good way to do this currently)
    rdys[r]->config.time.time_step = rdys[r - 1]->config.time.time_step / 2.0;
    rdys[r]->config.time.max_step  = rdys[r - 1]->config.time.max_step * 2;
    TSSetTimeStep(rdys[r]->ts, rdys[r]->config.time.time_step);
    TSSetMaxSteps(rdys[r]->ts, rdys[r]->config.time.max_step);
  }

  for (PetscInt r = 0; r <= num_refinements; ++r) {
    PetscPrintf(rdys[r]->comm, "Refinement level %" PetscInt_FMT ":\n", r);

    // run the problem to completion
    while (!RDyFinished(rdys[r])) {
      PetscCall(RDyAdvance(rdys[r]));
    }

    // compute error norms for this refinement level
    PetscCall(RDyMMSComputeErrorNorms(rdys[r], final_time, L1_norms[r], L2_norms[r], Linf_norms[r], NULL, NULL));
    PetscPrintf(rdys[r]->comm, "  Error norms at t = %g:\n", final_time);
    const char *comp_names[3] = {" h", "hu", "hv"};
    for (PetscInt c = 0; c < num_comps; ++c) {
      PetscPrintf(rdys[r]->comm, "    %s: L1 = %g, L2 = %g, Linf = %g\n", comp_names[c], L1_norms[r][c], L2_norms[r][c], Linf_norms[r][c]);
    }
    PetscPrintf(rdys[r]->comm, "\n");
  }

  // calculate the spatial discretization parameter N, where h^{-dim} = N.
  PetscReal x[num_refinements + 1];
  for (PetscInt r = 0; r <= num_refinements; ++r) {
    PetscInt N = rdys[r]->mesh.num_cells_global;
    x[r]       = PetscLog10Real(N);
  }

  // fit convergence rates
  PetscReal y1[num_refinements + 1], y2[num_refinements + 1], yinf[num_refinements + 1];
  for (PetscInt c = 0; c < num_comps; ++c) {
    for (PetscInt r = 0; r <= num_refinements; ++r) {
      y1[r]   = PetscLog10Real(L1_norms[r][c]);
      y2[r]   = PetscLog10Real(L2_norms[r][c]);
      yinf[r] = PetscLog10Real(Linf_norms[r][c]);
    }

    // since h^{-dim} = N, log err = s log N + b = -s dim log h + b
    PetscReal slope, intercept;
    PetscCall(PetscLinearRegression(num_refinements + 1, x, y1, &slope, &intercept));
    L1_conv_rates[c] = -slope * dim;
    PetscCall(PetscLinearRegression(num_refinements + 1, x, y2, &slope, &intercept));
    L2_conv_rates[c] = -slope * dim;
    PetscCall(PetscLinearRegression(num_refinements + 1, x, yinf, &slope, &intercept));
    Linf_conv_rates[c] = -slope * dim;
  }

  // clean up
  for (PetscInt r = 1; r <= num_refinements; ++r) {
    PetscCall(RDyDestroy(&rdys[r]));
  }

  // PetscCall(PetscConvEstDestroy(&convEst));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define CheckConvergence(comp, comp_index, norm)                                                                                            \
  if (norm##_conv_rates[comp_index] <= rdy->config.mms.swe.convergence.expected_rates.comp.norm) {                                          \
    SETERRQ(rdy->comm, PETSC_ERR_USER, "FAIL: %s convergence rate for %s is %g (expected %g)", #norm, #comp, norm##_conv_rates[comp_index], \
            rdy->config.mms.swe.convergence.expected_rates.comp.norm);                                                                      \
  }

PetscErrorCode RDyMMSRun(RDy rdy) {
  PetscFunctionBegin;

  // FIXME: SWE only at the moment
  if (rdy->config.mms.swe.convergence.num_refinements) {
    // run a convergence study
    PetscInt  num_comps       = 3;
    PetscInt  num_refinements = 3;
    PetscReal L1_conv_rates[num_comps], L2_conv_rates[num_comps], Linf_conv_rates[num_comps];
    PetscCall(RDyMMSEstimateConvergenceRates(rdy, num_refinements, L1_conv_rates, L2_conv_rates, Linf_conv_rates));

    const char *comp_names[3] = {" h", "hu", "hv"};
    PetscPrintf(rdy->comm, "Convergence rates:\n");
    for (PetscInt idof = 0; idof < 3; idof++) {
      PetscPrintf(rdy->comm, "  %s: L1 = %g, L2 = %g, Linf = %g\n", comp_names[idof], L1_conv_rates[idof], L2_conv_rates[idof],
                  Linf_conv_rates[idof]);
    }

    // check the convergence rates and print PASS or FAIL
    CheckConvergence(h, 0, L1);
    CheckConvergence(h, 0, L2);
    CheckConvergence(h, 0, Linf);
    CheckConvergence(hu, 1, L1);
    CheckConvergence(hu, 1, L2);
    CheckConvergence(hu, 1, Linf);
    CheckConvergence(hv, 2, L1);
    CheckConvergence(hv, 2, L2);
    CheckConvergence(hv, 2, Linf);
    PetscPrintf(rdy->comm, "PASS: all convergence rates satisfy thresholds.\n");
  } else {
    // run the problem to completion and print error norms
    while (!RDyFinished(rdy)) {
      PetscCall(RDyAdvance(rdy));
    }

    // compute error norms for the final solution
    RDyTimeUnit time_unit;
    PetscCall(RDyGetTimeUnit(rdy, &time_unit));
    PetscReal cur_time;
    PetscCall(RDyGetTime(rdy, time_unit, &cur_time));
    PetscReal L1_norms[3], L2_norms[3], Linf_norms[3], global_area;
    PetscInt  num_global_cells;
    PetscCall(RDyMMSComputeErrorNorms(rdy, cur_time, L1_norms, L2_norms, Linf_norms, &num_global_cells, &global_area));

    PetscPrintf(rdy->comm, "Avg-cell-area    : %18.16f\n", global_area / num_global_cells);
    PetscPrintf(rdy->comm, "Avg-length-scale : %18.16f\n", PetscSqrtReal(global_area / num_global_cells));

    PetscPrintf(rdy->comm, "Error-Norm-1     : ");
    for (PetscInt idof = 0; idof < 3; idof++) printf("%18.16f ", L1_norms[idof]);
    PetscPrintf(rdy->comm, "\n");

    PetscPrintf(rdy->comm, "Error-Norm-2     : ");
    for (PetscInt idof = 0; idof < 3; idof++) printf("%18.16f ", L2_norms[idof]);
    PetscPrintf(rdy->comm, "\n");

    PetscPrintf(rdy->comm, "Error-Norm-Max   : ");
    for (PetscInt idof = 0; idof < 3; idof++) printf("%18.16f ", Linf_norms[idof]);
    PetscPrintf(rdy->comm, "\n");
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}
