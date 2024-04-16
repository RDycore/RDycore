// This code supports the C and Fortran MMS drivers and is not used in
// mainline RDycore (though it is built into the library).

#include <muParserDLL.h>
#include <petscdmceed.h>
#include <petscdmplex.h>
#include <petscsys.h>
#include <private/rdycoreimpl.h>
#include <private/rdydmimpl.h>

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
  analytic_flow.height = rdy->config.mms.swe.solutions.h;

  analytic_flow.x_momentum = mupCreate(muBASETYPE_FLOAT);
  MathExpression x_momentum_expr;
  snprintf(x_momentum_expr, MAX_EXPRESSION_LEN, "%s * %s", rdy->config.mms.swe.expressions.h, rdy->config.mms.swe.expressions.u);
  mupSetExpr(analytic_flow.x_momentum, x_momentum_expr);

  analytic_flow.y_momentum = mupCreate(muBASETYPE_FLOAT);
  MathExpression y_momentum_expr;
  snprintf(y_momentum_expr, MAX_EXPRESSION_LEN, "%s * %s", rdy->config.mms.swe.expressions.h, rdy->config.mms.swe.expressions.v);
  mupSetExpr(analytic_flow.y_momentum, y_momentum_expr);

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
static PetscErrorCode EvaluateTemporalSolution(void *expr, PetscInt n, PetscReal x[n], PetscReal y[n], PetscReal t[n], PetscReal values[n]) {
  PetscFunctionBegin;

  SET_SPATIOTEMPORAL_VARIABLES(expr);
  mupEvalBulk(expr, values, n);

  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef SET_SPATIAL_VARIABLES
#undef SET_SPATIOTEMPORAL_VARIABLES

static PetscErrorCode SetSWEAnalyticSolution(RDy rdy) {
  PetscFunctionBegin;

  PetscCall(VecZeroEntries(rdy->X));

  // initialize the manufactured solution on each region
  PetscInt n_local, ndof;
  PetscCall(VecGetLocalSize(rdy->X, &n_local));
  PetscCall(VecGetBlockSize(rdy->X, &ndof));
  PetscScalar *x_ptr;
  PetscCall(VecGetArray(rdy->X, &x_ptr));

  for (PetscInt r = 0; r < rdy->num_regions; ++r) {
    RDyRegion region = rdy->regions[r];

    // Create vectorized (x, y, t) triples for bulk expression evaluation
    PetscReal cell_x[region.num_cells], cell_y[region.num_cells], t[region.num_cells];
    PetscInt  N = 0;  // number of bulk evaluations
    for (PetscInt c = 0; c < region.num_cells; ++c) {
      PetscInt cell_id = region.cell_ids[c];
      if (3 * cell_id < n_local) {
        cell_x[N] = rdy->mesh.cells.centroids[cell_id].X[0];
        cell_y[N] = rdy->mesh.cells.centroids[cell_id].X[1];
        t[N]      = 0.0;  // initial time
        ++N;
      }
    }

    if (rdy->config.physics.flow.mode == FLOW_SWE) {
      PetscCheck(ndof == 3, rdy->comm, PETSC_ERR_USER, "SWE solution vector has %" PetscInt_FMT " DOF (should have 3)", ndof);

      // evaluate the manufactured ѕolutions at all (x, y, t)
      PetscReal h[N], u[N], v[N];
      PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.h, N, cell_x, cell_y, t, h));
      PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.u, N, cell_x, cell_y, t, u));
      PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.v, N, cell_x, cell_y, t, v));

      // TODO: salinity and sediment initial conditions go here.

      // evaluate and set material properties
      PetscReal manning[N];
      PetscCall(EvaluateSpatialSolution(rdy->config.mms.swe.solutions.n, N, cell_x, cell_y, manning));
      PetscCall(RDySetManningsNForLocalCells(rdy, N, manning));

      PetscInt l = 0;
      for (PetscInt c = 0; c < region.num_cells; ++c) {
        PetscInt cell_id = region.cell_ids[c];
        if (3 * cell_id < n_local) {  // skip ghost cells
          x_ptr[3 * cell_id]     = h[l];
          x_ptr[3 * cell_id + 1] = h[l] * u[l];
          x_ptr[3 * cell_id + 2] = h[l] * v[l];
        }
      }
    }
  }

  PetscCall(VecRestoreArray(rdy->X, &x_ptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// this can be used in place of RDySetup for the MMS driver, which uses a
// modified YAML input schema (see ReadMMSConfigFile in yaml_input.c)
PetscErrorCode RDySetupMMS(RDy rdy) {
  PetscFunctionBegin;

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

  RDyLogDebug(rdy, "Initializing solution and source data...");
  PetscCall(SetSWEAnalyticSolution(rdy));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// call this to enforce analytical boundary conditions in the MMS driver
PetscErrorCode RDyEnforceMMSBoundaryConditions(RDy rdy, PetscReal time) {
  PetscFunctionBegin;

  RDyLogDebug(rdy, "Enforcing MMS boundary conditions...");

  for (PetscInt b = 0; b < rdy->num_boundaries; ++b) {
    // fetch x, y for each edge (and set t = time)
    RDyBoundary boundary  = rdy->boundaries[b];
    PetscInt    num_edges = boundary.num_edges;
    PetscReal   x[num_edges], y[num_edges], t[num_edges];
    for (PetscInt e = 0; e < num_edges; ++e) {
      PetscInt edge_id       = boundary.edge_ids[e];
      RDyPoint edge_centroid = rdy->mesh.edges.centroids[edge_id];
      x[e]                   = edge_centroid.X[0];
      y[e]                   = edge_centroid.X[1];
      t[e]                   = time;
    }

    // compute h, hu, hv on each edge (SWE-specific)
    RDyFlowCondition *flow_bc = rdy->boundary_conditions[b].flow;
    PetscReal         h[num_edges], hu[num_edges], hv[num_edges];
    PetscCall(EvaluateTemporalSolution(flow_bc->height, num_edges, x, y, t, h));
    PetscCall(EvaluateTemporalSolution(flow_bc->x_momentum, num_edges, x, y, t, hu));
    PetscCall(EvaluateTemporalSolution(flow_bc->y_momentum, num_edges, x, y, t, hv));

    // set the boundary values (SWE-specific)
    // NOTE: ndof == 3 for SWE
    PetscReal boundary_values[3 * num_edges];
    for (PetscInt e = 0; e < num_edges; ++e) {
      boundary_values[3 * e]     = h[e];
      boundary_values[3 * e + 1] = hu[e];
      boundary_values[3 * e + 2] = hv[e];
    }
    PetscCall(RDySetDirichletBoundaryValues(rdy, b, num_edges, 3, boundary_values));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}
