// This code supports the C and Fortran MMS drivers and is not used in
// mainline RDycore (though it is built into the library).

#include <muParserDLL.h>
#include <petscdmceed.h>
#include <petscdmplex.h>
#include <petscsys.h>
#include <private/rdycoreimpl.h>
#include <private/rdydmimpl.h>
#include <private/rdyheatimpl.h>
#include <private/rdymathimpl.h>
#include <private/rdyoperatorimpl.h>

#include "petscstring.h"
#include "private/config.h"

static const PetscReal GRAVITY                = 9.806;   // gravitational acceleration [m/s^2]
static const PetscReal DENSITY_OF_WATER       = 1000.0;  // [kg/m^3]
static const PetscReal SPECIFIC_HEAT_OF_WATER = 4186.0;  // [J/(kg·K)]

// NOTE: our boundary conditions are expressed in terms of momenta and not flow
// velocities, so we have to chain together a few things to evaluate x and y
// momenta.

static PetscErrorCode SetAnalyticBoundaryCondition(RDy rdy) {
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

  static RDySedimentCondition analytic_sediment = {
      .name = "analytic_sediment_bc",
      .type = CONDITION_DIRICHLET,
  };
  for (PetscInt i = 0; i < rdy->config.physics.sediment.num_classes; ++i) {
    strncpy(analytic_sediment.classes[i].expression, rdy->config.mms.sediment.expressions.c[i], MAX_EXPRESSION_LEN);
    analytic_sediment.classes[i].value = (void*)rdy->config.mms.sediment.solutions.c[i];
  };
  static RDySalinityCondition analytic_salinity = {
      .name = "analytic_bc",
      .type = CONDITION_DIRICHLET,
  };
  analytic_salinity.concentration = rdy->config.mms.salinity.solutions.S;

  static RDyHeatCondition analytic_heat = {
      .name = "analytic_bc",
      .type = CONDITION_DIRICHLET,
  };
  analytic_heat.water_temperature = rdy->config.mms.temperature.solutions.T;
  RDyCondition analytic_bc        = {
             .flow     = &analytic_flow,
             .sediment = &analytic_sediment,
             .salinity = (rdy->config.physics.salinity ? &analytic_salinity : NULL),
             .heat     = (rdy->config.physics.heat ? &analytic_heat : NULL),
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
static PetscErrorCode EvaluateSpatialSolution(void* expr, PetscInt n, PetscReal* x, PetscReal* y, PetscReal* values) {
  PetscFunctionBegin;

  SET_SPATIAL_VARIABLES(expr);
  mupEvalBulk(expr, values, n);

  PetscFunctionReturn(PETSC_SUCCESS);
}

#define SET_SPATIOTEMPORAL_VARIABLES(func) \
  SET_SPATIAL_VARIABLES(func);             \
  mupDefineBulkVar(func, "t", t)

// evaluates the given expression at all given x, y, t, placing the results into values
static PetscErrorCode EvaluateTemporalSolution(void* expr, PetscInt n, PetscReal* x, PetscReal* y, PetscReal time, PetscReal* values) {
  PetscFunctionBegin;

  PetscReal* t;
  PetscCalloc1(n, &t);
  for (PetscInt i = 0; i < n; ++i) t[i] = time;
  SET_SPATIOTEMPORAL_VARIABLES(expr);
  mupEvalBulk(expr, values, n);
  PetscCall(PetscFree(t));

  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef SET_SPATIAL_VARIABLES
#undef SET_SPATIOTEMPORAL_VARIABLES

// Collects the centroids of this rank's owned cells, in owned-cell order (the
// same order used by the operator's per-cell forcing arrays and by the error
// norm loop). The caller frees cell_x and cell_y.
static PetscErrorCode GetOwnedCellCentroids(RDy rdy, PetscInt* num_owned_cells, PetscReal** cell_x, PetscReal** cell_y) {
  PetscFunctionBegin;

  PetscInt N;
  PetscCall(RDyGetNumOwnedCells(rdy, &N));
  PetscCall(PetscCalloc1(N, cell_x));
  PetscCall(PetscCalloc1(N, cell_y));

  PetscInt l = 0;
  for (PetscInt icell = 0; icell < rdy->mesh.num_cells; ++icell) {
    if (rdy->mesh.cells.is_owned[icell]) {
      (*cell_x)[l] = rdy->mesh.cells.centroids[icell].X[0];
      (*cell_y)[l] = rdy->mesh.cells.centroids[icell].X[1];
      ++l;
    }
  }
  *num_owned_cells = N;

  PetscFunctionReturn(PETSC_SUCCESS);
}

// index of the prognostic heat DOF (hT) within the solution vector
static PetscInt MMSHeatComponentIndex(RDy rdy) { return 3 + rdy->config.physics.sediment.num_classes + (rdy->config.physics.salinity ? 1 : 0); }

// Number of components for which error norms are reported. Beyond the
// 3 + num_tracers prognostic DOF, heat-enabled runs report the derived
// temperature T = hT/h as one additional trailing component, since hT is what
// the solver advances but T is the physically meaningful quantity.
static PetscInt MMSNumErrorComponents(RDy rdy) { return 3 + rdy->num_tracers + (rdy->config.physics.heat ? 1 : 0); }

// running max over the simulation of |Q_mms|_inf, the manufactured heat source
// installed by MMSPostStep. Case 1 (source-free moving transport) is defined by
// this being zero to roundoff, which is otherwise unobservable from the output.
static PetscReal mms_max_heat_source = 0.0;

// Computes the prescribed MMS heat flux at one time without changing any
// production forcing state. The MMS driver selects the temporal quadrature
// for this source before invoking the heat TS.
static PetscErrorCode ComputeMMSHeatSource(RDy rdy, PetscReal time, PetscReal source[]) {
  PetscFunctionBegin;

  PetscInt   N;
  PetscReal *cell_x, *cell_y;
  PetscCall(GetOwnedCellCentroids(rdy, &N, &cell_x, &cell_y));

  PetscReal *h, *u, *v, *T;
  PetscReal *dhdx, *dhdy, *dhdt, *dudx, *dvdy;
  PetscReal *dTdx, *dTdy, *dTdt;
  PetscCall(PetscCalloc1(N, &h));
  PetscCall(PetscCalloc1(N, &u));
  PetscCall(PetscCalloc1(N, &v));
  PetscCall(PetscCalloc1(N, &T));
  PetscCall(PetscCalloc1(N, &dhdx));
  PetscCall(PetscCalloc1(N, &dhdy));
  PetscCall(PetscCalloc1(N, &dhdt));
  PetscCall(PetscCalloc1(N, &dudx));
  PetscCall(PetscCalloc1(N, &dvdy));
  PetscCall(PetscCalloc1(N, &dTdx));
  PetscCall(PetscCalloc1(N, &dTdy));
  PetscCall(PetscCalloc1(N, &dTdt));

  PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.h, N, cell_x, cell_y, time, h));
  PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.u, N, cell_x, cell_y, time, u));
  PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.v, N, cell_x, cell_y, time, v));
  PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.dhdx, N, cell_x, cell_y, time, dhdx));
  PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.dhdy, N, cell_x, cell_y, time, dhdy));
  PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.dhdt, N, cell_x, cell_y, time, dhdt));
  PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.dudx, N, cell_x, cell_y, time, dudx));
  PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.dvdy, N, cell_x, cell_y, time, dvdy));
  PetscCall(EvaluateTemporalSolution(rdy->config.mms.temperature.solutions.T, N, cell_x, cell_y, time, T));
  PetscCall(EvaluateTemporalSolution(rdy->config.mms.temperature.solutions.dTdx, N, cell_x, cell_y, time, dTdx));
  PetscCall(EvaluateTemporalSolution(rdy->config.mms.temperature.solutions.dTdy, N, cell_x, cell_y, time, dTdy));
  PetscCall(EvaluateTemporalSolution(rdy->config.mms.temperature.solutions.dTdt, N, cell_x, cell_y, time, dTdt));

  for (PetscInt i = 0; i < N; ++i) {
    source[i] = DENSITY_OF_WATER * SPECIFIC_HEAT_OF_WATER *
                (h[i] * dTdt[i] + T[i] * dhdt[i] + h[i] * u[i] * dTdx[i] + T[i] * h[i] * dudx[i] + T[i] * u[i] * dhdx[i] + h[i] * v[i] * dTdy[i] +
                 T[i] * h[i] * dvdy[i] + T[i] * v[i] * dhdy[i]);
  }

  PetscCall(PetscFree(cell_x));
  PetscCall(PetscFree(cell_y));
  PetscCall(PetscFree(h));
  PetscCall(PetscFree(u));
  PetscCall(PetscFree(v));
  PetscCall(PetscFree(T));
  PetscCall(PetscFree(dhdx));
  PetscCall(PetscFree(dhdy));
  PetscCall(PetscFree(dhdt));
  PetscCall(PetscFree(dudx));
  PetscCall(PetscFree(dvdy));
  PetscCall(PetscFree(dTdx));
  PetscCall(PetscFree(dTdy));
  PetscCall(PetscFree(dTdt));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// sets the z coordinate of refined mesh vertices to match the analytic value
// z(x, y)
static PetscErrorCode SnapVerticesToBathymetry(RDy rdy) {
  PetscFunctionBegin;

  Vec          coordinates;
  PetscSection coordSection;
  PetscScalar* coords;
  PetscInt     v, vStart, vEnd, offset;
  PetscReal    x, y, z;

  PetscCall(DMGetCoordinateSection(rdy->dm, &coordSection));
  PetscCall(DMGetCoordinatesLocal(rdy->dm, &coordinates));
  PetscCall(DMPlexGetDepthStratum(rdy->dm, 0, &vStart, &vEnd));

  PetscCall(VecGetArray(coordinates, &coords));
  for (v = vStart; v < vEnd; v++) {
    PetscCall(PetscSectionGetOffset(coordSection, v, &offset));
    x = coords[offset];
    y = coords[offset + 1];
    mupDefineVar(rdy->config.mms.swe.solutions.z, "x", &x);
    mupDefineVar(rdy->config.mms.swe.solutions.z, "y", &y);
    z                  = mupEval(rdy->config.mms.swe.solutions.z);
    coords[offset + 2] = z;
  }
  PetscCall(VecRestoreArray(coordinates, &coords));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// this function gets called at the beginning of each time step, updating
// source terms and  boundary conditions at a properly centered time
static PetscErrorCode MMSPreStep(TS ts) {
  PetscFunctionBegin;

  RDy rdy;
  PetscCall(TSGetApplicationContext(ts, (void*)&rdy));

  PetscReal t, dt;
  PetscCall(TSGetTime(ts, &t));
  PetscCall(TSGetTimeStep(ts, &dt));

  PetscCall(RDyMMSEnforceBoundaryConditions(rdy, t + 0.5 * dt));
  PetscCall(RDyMMSComputeSourceTerms(rdy, t + 0.5 * dt));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Guards the heat component of the transport operator's external source vector.
//
// Today this is vacuous: nothing can write that slot, because rdycore.h exposes
// RDySetRegionalWaterSource / XMomentum / YMomentum / Sediment but no heat
// equivalent. The check exists so that if such an API is ever added and wired
// into MMSPreStep by mistake, the manufactured d(hT)/dt would be applied in both
// split solves and this fails loudly instead of silently halving the measured
// order.
static PetscErrorCode AssertZeroTransportHeatSource(RDy rdy) {
  PetscFunctionBegin;

  PetscInt heat_comp = MMSHeatComponentIndex(rdy);
  for (PetscInt r = 0; r < rdy->num_regions; ++r) {
    RDyRegion region = rdy->regions[r];
    if (!region.num_owned_cells) continue;

    OperatorData source_data;
    PetscCall(GetOperatorRegionalExternalSource(rdy->operator, region, &source_data));
    PetscReal max_heat_source = 0.0;
    for (PetscInt c = 0; c < region.num_owned_cells; ++c) {
      max_heat_source = PetscMax(max_heat_source, PetscAbsReal(source_data.values[heat_comp][c]));
    }
    PetscCall(RestoreOperatorRegionalExternalSource(rdy->operator, region, &source_data));
    PetscCheck(max_heat_source == 0.0, rdy->comm, PETSC_ERR_PLIB,
               "The transport external source vector has a nonzero heat component (%g) in region %" PetscInt_FMT
               ". The manufactured heat residual belongs to the heat solve alone; installing it in both solves would "
               "apply d(hT)/dt twice.",
               (double)max_heat_source, r);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// TS post-step callback for MMS heat: applies the source quadrature associated
// with the configured one-step implicit heat method after each transport step.
// Manufactured expressions are only available in the MMS driver, so this
// sampling intentionally remains outside HeatIFunction.
//
// How the manufactured correction is split between the two solves
// ---------------------------------------------------------------
// RDycore advances a coupling interval with a Lie split: the transport TSSolve
// advances SWE and carries hT as a passive tracer, then the heat TSSolve holds
// the flow fixed and changes only hT. Writing D(C) = div(huT, hvT), the unsplit
// manufactured correction is S_C = C_t + D(C) - R_atm, and consistency requires
// the two split corrections to sum to it -- so C_t must appear once in that sum,
// not once per solve. RDycore allocates it as
//
//   transport solve: the full manufactured h, hu, hv sources, and *nothing* for
//                    heat, so the numerical tracer flux performs the complete hT
//                    transport rather than having it cancelled analytically;
//   heat solve:      the complete conservative residual C_t + D(C), which the
//                    direct-source branch consumes in place of the atmospheric
//                    parameterization.
//
// To leading order the composite update is then
//
//   C^{n+1} = C^n + dt*C_t + dt*[D(C) - D_h(C)] + O(dt^2),
//
// with D_h the discrete flux divergence, so the residual measures exactly the
// spatial truncation error of the tracer flux plus the temporal and splitting
// error. MMSPreStep installing a heat source too would double-count C_t.
//
// The zero heat component of the transport source is structural, not a
// convention: no public API writes that slot (see RDySetRegional*Source in
// rdycore.h -- there is no heat equivalent), and the operator's external source
// vector is zero-initialized. AssertZeroTransportHeatSource below guards against
// a future API that changes this.
//
// Scope limit: because this branch *replaces* HeatQNet(T) rather than correcting
// it, the direct_source MMS path does not verify the nonlinear atmospheric
// parameterization or its analytic Jacobian. In this path the heat TS type also
// does not determine the order of the source step -- the residual has no state
// dependence, so every consistent one-step method gives the same update and the
// TS type selects only which manufactured quadrature is sampled below.
static PetscErrorCode MMSPostStep(TS ts) {
  PetscFunctionBegin;
  RDy rdy;
  PetscCall(TSGetApplicationContext(ts, (void*)&rdy));

  PetscCall(AssertZeroTransportHeatSource(rdy));

  // The heat source is applied over the transport interval that just completed,
  // as a single implicit step of exactly that length.
  PetscReal t0, t1;
  PetscCall(TSGetPrevTime(ts, &t0));
  PetscCall(TSGetTime(ts, &t1));

  RDyHeat   heat = rdy->heat_context;
  TSType    heat_ts_type;
  PetscBool is_beuler, is_cn;
  PetscCall(TSGetType(rdy->heat_ts, &heat_ts_type));
  PetscCall(PetscStrcmp(heat_ts_type, TSBEULER, &is_beuler));
  PetscCall(PetscStrcmp(heat_ts_type, TSCN, &is_cn));
  PetscCheck(is_beuler || is_cn, rdy->comm, PETSC_ERR_SUP, "MMS heat source sampling supports only TSBEULER and TSCN, not '%s'", heat_ts_type);

  if (is_beuler) {
    PetscCall(ComputeMMSHeatSource(rdy, t1, heat->forcing.direct_source));
  } else {
    PetscInt   num_owned_cells;
    PetscReal* left_source;
    PetscCall(RDyGetNumOwnedCells(rdy, &num_owned_cells));
    PetscCall(PetscMalloc1(num_owned_cells, &left_source));
    PetscCall(ComputeMMSHeatSource(rdy, t0, left_source));
    PetscCall(ComputeMMSHeatSource(rdy, t1, heat->forcing.direct_source));
    for (PetscInt c = 0; c < num_owned_cells; ++c) {
      heat->forcing.direct_source[c] = 0.5 * (left_source[c] + heat->forcing.direct_source[c]);
    }
    PetscCall(PetscFree(left_source));
  }

  // Track |Q_mms|_inf so the manufactured construction can be checked directly
  // rather than inferred from the final hT error. A source-free moving-transport
  // case (C_t + div(huT, hvT) == 0) must keep this at roundoff scale.
  {
    PetscInt num_owned_cells;
    PetscCall(RDyGetNumOwnedCells(rdy, &num_owned_cells));
    for (PetscInt c = 0; c < num_owned_cells; ++c) {
      mms_max_heat_source = PetscMax(mms_max_heat_source, PetscAbsReal(heat->forcing.direct_source[c]));
    }
  }

  heat->use_direct_source = PETSC_TRUE;
  // NOTE: on the failure path below the flag stays raised. That is harmless
  // NOTE: because the MMS driver aborts, but it is the reason production code
  // NOTE: must never rely on this reset.
  PetscCall(RDyHeatAdvance(rdy, t0, t1));
  // Reset flag so stale MMS forcing cannot leak into subsequent non-MMS calls
  heat->use_direct_source = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

extern PetscErrorCode PauseIfRequested(RDy rdy);  // for -pause support
extern PetscErrorCode InitOperator(RDy rdy);
extern PetscErrorCode InitSolver(RDy rdy);

// prognostic DOF plus one slot for the derived temperature T = hT/h
#define MAX_NUM_COMPONENTS (3 + MAX_NUM_TRACERS + 1)
static char mms_comp_names[MAX_NUM_COMPONENTS][MAX_NAME_LEN + 1] = {0};

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

  // set names of solution components
  PetscStrncpy(mms_comp_names[0], " h ", MAX_NAME_LEN);
  PetscStrncpy(mms_comp_names[1], "hu ", MAX_NAME_LEN);
  PetscStrncpy(mms_comp_names[2], "hv ", MAX_NAME_LEN);
  PetscInt index = 3, num_classes = rdy->config.physics.sediment.num_classes;
  for (PetscInt i = 0; i < num_classes; ++i) {
    snprintf(mms_comp_names[index + i], MAX_NAME_LEN, "c%" PetscInt_FMT " ", i);
  }
  index += num_classes;
  if (rdy->config.physics.salinity) {
    PetscStrncpy(mms_comp_names[index], "salinity", MAX_NAME_LEN);
    ++index;
  }
  if (rdy->config.physics.heat) {
    // NOTE: the prognostic heat DOF is the conservative variable hT; the
    // NOTE: derived temperature T = hT/h is reported as a trailing component.
    PetscStrncpy(mms_comp_names[index], "hT ", MAX_NAME_LEN);
    ++index;
    PetscStrncpy(mms_comp_names[3 + rdy->num_tracers], " T ", MAX_NAME_LEN);
  }

  // reset the manufactured heat source diagnostic for this run
  mms_max_heat_source = 0.0;

  // if a refinement level is not specified, set the base refinement level
  PetscInt refine_level = 0;
  PetscOptionsGetInt(NULL, NULL, "-dm_refine", &refine_level, NULL);
  if (!refine_level) {
    PetscInt base_refinement = rdy->config.mms.convergence.base_refinement;
    char     refinement[5];
    snprintf(refinement, 4, "%" PetscInt_FMT, base_refinement);
    PetscOptionsSetValue(NULL, "-dm_refine", refinement);
    // the following line is apparently needed when we give -dm_refine above
    PetscOptionsSetValue(NULL, "-dm_plex_transform_label_match_strata", "1");
  }

  RDyLogDebug(rdy, "Creating DMs...");

  rdy->soln_fields = (SectionFieldSpec){
      .num_fields            = 1,
      .num_field_components  = {3 + rdy->num_tracers},
      .field_names           = {"Solution"},
      .field_component_names = {{
          "Height",
          "MomentumX",
          "MomentumY",
      }},
  };
  for (PetscInt i = 0; i < rdy->num_tracers; ++i) {
    snprintf(rdy->soln_fields.field_component_names[0][3 + i], MAX_NAME_LEN, "SedimentMassPerUnitArea%" PetscInt_FMT, i);
  }

  // set up solution time-averaged field spec
  rdy->soln_output.avg_fields.num_fields              = 1;
  rdy->soln_output.avg_fields.num_field_components[0] = rdy->soln_fields.num_field_components[0];
  strcpy(rdy->soln_output.avg_fields.field_names[0], "SolutionMean");
  strcpy(rdy->soln_output.avg_fields.field_component_names[0][0], "Height_Mean");
  strcpy(rdy->soln_output.avg_fields.field_component_names[0][1], "MomentumX_Mean");
  strcpy(rdy->soln_output.avg_fields.field_component_names[0][2], "MomentumY_Mean");
  for (PetscInt i = 0; i < rdy->num_tracers; ++i) {
    snprintf(rdy->soln_output.avg_fields.field_component_names[0][3 + i], MAX_NAME_LEN, "SedimentMassPerUnitArea%" PetscInt_FMT "_Mean", i);
  }
  rdy->soln_output.skip_first_component = PETSC_FALSE;

  // set up primitive variables field spec for time-averaged (mean) output
  rdy->prim_vars_output.avg_fields.num_fields              = 1;
  rdy->prim_vars_output.avg_fields.num_field_components[0] = rdy->soln_fields.num_field_components[0];
  strcpy(rdy->prim_vars_output.avg_fields.field_names[0], "PrimitiveVariables");
  strcpy(rdy->prim_vars_output.avg_fields.field_component_names[0][0], "Height_Mean");
  strcpy(rdy->prim_vars_output.avg_fields.field_component_names[0][1], "VelocityX_Mean");
  strcpy(rdy->prim_vars_output.avg_fields.field_component_names[0][2], "VelocityY_Mean");
  for (PetscInt i = 0; i < rdy->num_tracers; ++i) {
    snprintf(rdy->prim_vars_output.avg_fields.field_component_names[0][3 + i], MAX_NAME_LEN, "SedimentConcentration%" PetscInt_FMT "_Mean", i);
  }

  // set up primitive variables field spec for instantaneous output
  rdy->prim_vars_output.inst_fields.num_fields              = 1;
  rdy->prim_vars_output.inst_fields.num_field_components[0] = rdy->soln_fields.num_field_components[0];
  strcpy(rdy->prim_vars_output.inst_fields.field_names[0], "PrimitiveVariablesInstantaneous");
  strcpy(rdy->prim_vars_output.inst_fields.field_component_names[0][0], "Height");  // skipped at write time
  strcpy(rdy->prim_vars_output.inst_fields.field_component_names[0][1], "VelocityX");
  strcpy(rdy->prim_vars_output.inst_fields.field_component_names[0][2], "VelocityY");
  for (PetscInt i = 0; i < rdy->num_tracers; ++i) {
    snprintf(rdy->prim_vars_output.inst_fields.field_component_names[0][3 + i], MAX_NAME_LEN, "SedimentConcentration%" PetscInt_FMT, i);
  }
  rdy->prim_vars_output.skip_first_component = PETSC_TRUE;

  // set up source output field specs (instantaneous and time-averaged)
  {
    PetscInt num_src_comp                               = 3 + rdy->num_tracers;
    rdy->src_output.inst_fields.num_fields              = 1;
    rdy->src_output.inst_fields.num_field_components[0] = num_src_comp;
    strcpy(rdy->src_output.inst_fields.field_names[0], "Sources");
    strcpy(rdy->src_output.inst_fields.field_component_names[0][0], "WaterSource");
    strcpy(rdy->src_output.inst_fields.field_component_names[0][1], "MomentumXSource");
    strcpy(rdy->src_output.inst_fields.field_component_names[0][2], "MomentumYSource");
    for (PetscInt i = 0; i < rdy->num_tracers; ++i) {
      snprintf(rdy->src_output.inst_fields.field_component_names[0][3 + i], MAX_NAME_LEN, "SedimentMassPerUnitArea%" PetscInt_FMT "Source", i);
    }

    rdy->src_output.avg_fields.num_fields              = 1;
    rdy->src_output.avg_fields.num_field_components[0] = num_src_comp;
    strcpy(rdy->src_output.avg_fields.field_names[0], "SourcesMean");
    strcpy(rdy->src_output.avg_fields.field_component_names[0][0], "WaterSource_Mean");
    strcpy(rdy->src_output.avg_fields.field_component_names[0][1], "MomentumXSource_Mean");
    strcpy(rdy->src_output.avg_fields.field_component_names[0][2], "MomentumYSource_Mean");
    for (PetscInt i = 0; i < rdy->num_tracers; ++i) {
      snprintf(rdy->src_output.avg_fields.field_component_names[0][3 + i], MAX_NAME_LEN, "SedimentMassPerUnitArea%" PetscInt_FMT "Source_Mean", i);
    }
  }
  rdy->src_output.skip_first_component = PETSC_FALSE;

  PetscCall(CreateDM(rdy));

  PetscCall(CreateAuxiliaryDMs(rdy));

  if (rdy->num_tracers) {
    PetscCall(CreateFlowDM(rdy));
    PetscCall(CreateTracerDM(rdy));
  } else {
    rdy->flow_fields = rdy->soln_fields;
    rdy->flow_dm     = rdy->dm;
  }

  // create global and local vectors
  PetscCall(CreateVectors(rdy));

  // adjust the vertices of a refined mesh to conform to our analytical z(x, y)
  PetscCall(SnapVerticesToBathymetry(rdy));

  // note: this must be done after global vectors are created so a global
  // note: section exists for the DM
  RDyLogDebug(rdy, "Creating FV mesh...");
  PetscCall(RDyMeshCreateFromDM(rdy->dm, 0, &rdy->mesh));
  if (rdy->config.physics.flow.well_balancing == WELL_BALANCING_HR) {
    PetscCall(RDyMeshOverride2DProjection(&rdy->mesh));
  }
  if (rdy->config.grid.cell_elevation.file[0]) {
    PetscCall(OverrideCellElevation(rdy));
  }

  RDyLogDebug(rdy, "Initializing regions...");
  PetscCall(InitRegions(rdy));

  RDyLogDebug(rdy, "Initializing boundaries and boundary conditions...");
  PetscCall(InitBoundaries(rdy));
  PetscCall(SetAnalyticBoundaryCondition(rdy));

  RDyLogDebug(rdy, "Initializing operator...");
  PetscCall(InitOperator(rdy));

  // Wire OutputVar references to Operator-owned vectors (must happen after InitOperator).
  {
    Operator* op                      = rdy->operator;
    rdy->prim_vars_output.petsc_inst  = op->primitive_variables;
    rdy->prim_vars_output.petsc_accum = op->primitive_variables_accum;
    rdy->prim_vars_output.ceed_inst   = op->ceed.primitive_variables;
    rdy->prim_vars_output.ceed_accum  = op->ceed.primitive_variables_accum;
    rdy->src_output.petsc_inst        = op->src_inst;
    rdy->src_output.petsc_accum       = op->src_accum;
    rdy->src_output.ceed_inst         = op->ceed.ceed_src_inst;
    rdy->src_output.ceed_accum        = op->ceed.ceed_src_accum;
  }

  RDyLogDebug(rdy, "Initializing solver...");
  PetscCall(InitSolver(rdy));

  PetscCall(TSSetPreStep(rdy->ts, MMSPreStep));

  if (rdy->config.physics.heat) {
    RDyLogDebug(rdy, "Initializing MMS heat TS...");
    PetscCall(RDyHeatCreate(rdy));
    PetscCall(TSSetPostStep(rdy->ts, MMSPostStep));
  }

  RDyLogDebug(rdy, "Initializing solution and source data...");
  PetscCall(RDyMMSComputeSolution(rdy, 0.0, rdy->u_global));
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
  PetscScalar* x_ptr;
  PetscCall(VecGetArray(solution, &x_ptr));

  PetscInt flow_ndof;
  switch (rdy->config.physics.flow.mode) {
    case FLOW_SWE:
      flow_ndof = 3;
      break;
    default:
      PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "Extend code to support flow mode other than SWE");
      break;
  }

  for (PetscInt r = 0; r < rdy->num_regions; ++r) {
    RDyRegion region = rdy->regions[r];

    // Create vectorized (x, y, t) triples for bulk expression evaluation
    PetscReal *cell_x, *cell_y;
    PetscCall(PetscCalloc1(region.num_local_cells, &cell_x));
    PetscCall(PetscCalloc1(region.num_local_cells, &cell_y));

    PetscInt N = 0;  // number of bulk evaluations
    for (PetscInt c = 0; c < region.num_local_cells; ++c) {
      PetscInt cell_id = region.cell_local_ids[c];
      if (3 * cell_id < n_local) {
        cell_x[N] = rdy->mesh.cells.centroids[cell_id].X[0];
        cell_y[N] = rdy->mesh.cells.centroids[cell_id].X[1];
        ++N;
      }
    }

    if (rdy->config.physics.flow.mode == FLOW_SWE) {
      PetscCheck(ndof == flow_ndof + rdy->num_tracers, rdy->comm, PETSC_ERR_USER,
                 "SWE solution vector has %" PetscInt_FMT " DOF that does not match the sum of flow_dof (%" PetscInt_FMT
                 ") and number of sediment classes (%" PetscInt_FMT ")",
                 ndof, flow_ndof, rdy->num_tracers);

      // evaluate the manufactured ѕolutions at all (x, y, t)

      // flow equations
      PetscReal *h, *u, *v;
      PetscCall(PetscCalloc1(region.num_local_cells, &h));
      PetscCall(PetscCalloc1(region.num_local_cells, &u));
      PetscCall(PetscCalloc1(region.num_local_cells, &v));
      PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.h, N, cell_x, cell_y, time, h));
      PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.u, N, cell_x, cell_y, time, u));
      PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.v, N, cell_x, cell_y, time, v));

      {
        PetscInt l = 0;
        for (PetscInt c = 0; c < region.num_local_cells; ++c) {
          PetscInt cell_id = region.cell_local_ids[c];
          if (ndof * cell_id < n_local) {  // skip ghost cells
            x_ptr[ndof * cell_id]     = h[l];
            x_ptr[ndof * cell_id + 1] = h[l] * u[l];
            x_ptr[ndof * cell_id + 2] = h[l] * v[l];
            ++l;
          }
        }
      }

      // sediment class concentrations
      PetscInt num_sediment_classes = rdy->config.physics.sediment.num_classes;
      if (num_sediment_classes > 0) {
        PetscInt   offset = 3;
        PetscReal* ci;
        PetscCall(PetscCalloc1(region.num_local_cells, &ci));
        for (PetscInt i = 0; i < num_sediment_classes; ++i) {
          PetscInt l = 0;
          PetscCall(EvaluateTemporalSolution((void*)rdy->config.mms.sediment.solutions.c[i], N, cell_x, cell_y, time, ci));
          for (PetscInt c = 0; c < region.num_local_cells; ++c) {
            PetscInt cell_id = region.cell_local_ids[c];
            if (ndof * cell_id < n_local) {  // skip ghost cells
              x_ptr[ndof * cell_id + offset + i] = h[l] * ci[l];
              ++l;
            }
          }
        }
        PetscCall(PetscFree(ci));
      }

      // salinity concentration
      if (rdy->config.physics.salinity) {
        PetscInt   offset = 3 + num_sediment_classes;
        PetscReal* s;
        PetscInt   l = 0;
        PetscCall(PetscCalloc1(region.num_local_cells, &s));
        PetscCall(EvaluateTemporalSolution((void*)rdy->config.mms.salinity.solutions.S, N, cell_x, cell_y, time, s));
        for (PetscInt c = 0; c < region.num_local_cells; ++c) {
          PetscInt cell_id = region.cell_local_ids[c];
          if (ndof * cell_id < n_local) {  // skip ghost cells
            x_ptr[ndof * cell_id + offset] = h[l] * s[l];
            ++l;
          }
        }
        PetscCall(PetscFree(s));
      }

      // temperature profile
      if (rdy->config.physics.heat) {
        PetscInt   offset = 3 + num_sediment_classes + (rdy->config.physics.salinity ? 1 : 0);
        PetscReal* T;
        PetscInt   l = 0;
        PetscCall(PetscCalloc1(region.num_local_cells, &T));
        PetscCall(EvaluateTemporalSolution((void*)rdy->config.mms.temperature.solutions.T, N, cell_x, cell_y, time, T));
        for (PetscInt c = 0; c < region.num_local_cells; ++c) {
          PetscInt cell_id = region.cell_local_ids[c];
          if (ndof * cell_id < n_local) {  // skip ghost cells
            x_ptr[ndof * cell_id + offset] = h[l] * T[l];
            ++l;
          }
        }
        PetscCall(PetscFree(T));
      }

      PetscCall(PetscFree(h));
      PetscCall(PetscFree(u));
      PetscCall(PetscFree(v));
    }
    PetscCall(PetscFree(cell_x));
    PetscCall(PetscFree(cell_y));
  }

  PetscCall(VecRestoreArray(solution, &x_ptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// evaluates the source terms associated with the manufactured solutions
PetscErrorCode RDyMMSComputeSourceTerms(RDy rdy, PetscReal time) {
  PetscFunctionBegin;

  RDyMesh*  mesh  = &rdy->mesh;
  RDyCells* cells = &mesh->cells;

  PetscInt N;
  PetscCall(RDyGetNumOwnedCells(rdy, &N));
  PetscReal *cell_x, *cell_y;
  PetscCall(PetscCalloc1(N, &cell_x));
  PetscCall(PetscCalloc1(N, &cell_y));

  PetscInt l = 0;
  for (PetscInt icell = 0; icell < mesh->num_cells; icell++) {
    if (cells->is_owned[icell]) {
      cell_x[l] = rdy->mesh.cells.centroids[icell].X[0];
      cell_y[l] = rdy->mesh.cells.centroids[icell].X[1];
      ++l;
    }
  }

  if (rdy->config.physics.flow.mode == FLOW_SWE) {
    // evaluate the manufactured ѕolutions at all (x, y, t)

    PetscReal *h, *u, *v;
    PetscCall(PetscCalloc1(N, &h));
    PetscCall(PetscCalloc1(N, &u));
    PetscCall(PetscCalloc1(N, &v));
    PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.h, N, cell_x, cell_y, time, h));
    PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.u, N, cell_x, cell_y, time, u));
    PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.v, N, cell_x, cell_y, time, v));

    PetscReal *dhdx, *dhdy, *dhdt;
    PetscCall(PetscCalloc1(N, &dhdx));
    PetscCall(PetscCalloc1(N, &dhdy));
    PetscCall(PetscCalloc1(N, &dhdt));
    PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.dhdx, N, cell_x, cell_y, time, dhdx));
    PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.dhdy, N, cell_x, cell_y, time, dhdy));
    PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.dhdt, N, cell_x, cell_y, time, dhdt));

    PetscReal *dudx, *dudy, *dudt;
    PetscCall(PetscCalloc1(N, &dudx));
    PetscCall(PetscCalloc1(N, &dudy));
    PetscCall(PetscCalloc1(N, &dudt));
    PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.dudx, N, cell_x, cell_y, time, dudx));
    PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.dudy, N, cell_x, cell_y, time, dudy));
    PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.dudt, N, cell_x, cell_y, time, dudt));

    PetscReal *dvdx, *dvdy, *dvdt;
    PetscCall(PetscCalloc1(N, &dvdx));
    PetscCall(PetscCalloc1(N, &dvdy));
    PetscCall(PetscCalloc1(N, &dvdt));
    PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.dvdx, N, cell_x, cell_y, time, dvdx));
    PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.dvdy, N, cell_x, cell_y, time, dvdy));
    PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.dvdt, N, cell_x, cell_y, time, dvdt));

    PetscReal* n;
    PetscCall(PetscCalloc1(N, &n));
    PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.n, N, cell_x, cell_y, time, n));

    PetscReal *dzdx, *dzdy;
    PetscCall(PetscCalloc1(N, &dzdx));
    PetscCall(PetscCalloc1(N, &dzdy));
    PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.dzdx, N, cell_x, cell_y, time, dzdx));
    PetscCall(EvaluateTemporalSolution(rdy->config.mms.swe.solutions.dzdy, N, cell_x, cell_y, time, dzdy));

    PetscReal *h_source, *hu_source, *hv_source;
    PetscCall(PetscCalloc1(N, &h_source));
    PetscCall(PetscCalloc1(N, &hu_source));
    PetscCall(PetscCalloc1(N, &hv_source));

    l = 0;
    for (PetscInt icell = 0; icell < mesh->num_cells; icell++) {
      if (cells->is_owned[icell]) {
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

    PetscCall(RDySetRegionalWaterSource(rdy, 1, N, h_source));
    PetscCall(RDySetRegionalXMomentumSource(rdy, 1, N, hu_source));
    PetscCall(RDySetRegionalYMomentumSource(rdy, 1, N, hv_source));

    PetscInt num_sediment_classes = rdy->config.physics.sediment.num_classes;
    if (num_sediment_classes) {
      PetscReal *ci[MAX_NUM_SEDIMENT_CLASSES], *dcidx[MAX_NUM_SEDIMENT_CLASSES], *dcidy[MAX_NUM_SEDIMENT_CLASSES], *dcidt[MAX_NUM_SEDIMENT_CLASSES];
      PetscReal* hci_source;

      PetscCall(PetscCalloc1(N, &hci_source));
      for (PetscInt i = 0; i < num_sediment_classes; ++i) {
        PetscCall(PetscCalloc1(N, &ci[i]));
        PetscCall(PetscCalloc1(N, &dcidx[i]));
        PetscCall(PetscCalloc1(N, &dcidy[i]));
        PetscCall(PetscCalloc1(N, &dcidt[i]));

        // NOTE: we cast to void * here because sediment solutions are stored as
        // NOTE: pointer-sensible integers so they fit into an array
        PetscCall(EvaluateTemporalSolution((void*)rdy->config.mms.sediment.solutions.c[i], N, cell_x, cell_y, time, ci[i]));
        PetscCall(EvaluateTemporalSolution((void*)rdy->config.mms.sediment.solutions.dcdx[i], N, cell_x, cell_y, time, dcidx[i]));
        PetscCall(EvaluateTemporalSolution((void*)rdy->config.mms.sediment.solutions.dcdy[i], N, cell_x, cell_y, time, dcidy[i]));
        PetscCall(EvaluateTemporalSolution((void*)rdy->config.mms.sediment.solutions.dcdt[i], N, cell_x, cell_y, time, dcidt[i]));
      }

      // FIXME: Need to move these constants into a struct that is specific to the erosion/deposition
      // FIXME: parameterization
      const PetscReal kp_constant             = 0.001;
      const PetscReal settling_velocity       = 0.01;
      const PetscReal tau_critical_erosion    = 0.1;
      const PetscReal tau_critical_deposition = 1000.0;
      const PetscReal rhow                    = DENSITY_OF_WATER;

      for (PetscInt i = 0; i < num_sediment_classes; ++i) {
        l = 0;
        for (PetscInt icell = 0; icell < mesh->num_cells; icell++) {
          if (cells->is_owned[icell]) {
            hci_source[l] = ci[i][l] * dhdt[l] + h[l] * dcidt[i][l];
            hci_source[l] += u[l] * ci[i][l] * dhdx[l] + h[l] * ci[i][l] * dudx[l] + u[l] * h[l] * dcidx[i][l];
            hci_source[l] += v[l] * ci[i][l] * dhdy[l] + h[l] * ci[i][l] * dvdy[l] + v[l] * h[l] * dcidy[i][l];

            PetscReal Cd    = GRAVITY * Square(n[l]) * PetscPowReal(h[l], -1.0 / 3.0);
            PetscReal tau_b = 0.5 * rhow * Cd * (Square(u[l]) + Square(v[l]));
            PetscReal ei    = kp_constant * (tau_b - tau_critical_erosion) / tau_critical_erosion;
            PetscReal di    = settling_velocity * ci[i][l] * (1.0 - tau_b / tau_critical_deposition);
            hci_source[l] += -(ei - di);
            ++l;
          }
        }
        PetscCall(RDySetRegionalSedimentSource(rdy, 1, i, N, hci_source));
      }

      for (PetscInt i = 0; i < num_sediment_classes; ++i) {
        PetscCall(PetscFree(ci[i]));
        PetscCall(PetscFree(dcidx[i]));
        PetscCall(PetscFree(dcidy[i]));
        PetscCall(PetscFree(dcidt[i]));
      }
      PetscCall(PetscFree(hci_source));
    }

    PetscReal *s, *dsdx, *dsdy, *dsdt;
    if (rdy->config.physics.salinity) {
      PetscCall(PetscCalloc1(N, &s));
      PetscCall(PetscCalloc1(N, &dsdx));
      PetscCall(PetscCalloc1(N, &dsdy));
      PetscCall(PetscCalloc1(N, &dsdt));

      PetscCall(EvaluateTemporalSolution(rdy->config.mms.salinity.solutions.S, N, cell_x, cell_y, time, s));
      PetscCall(EvaluateTemporalSolution(rdy->config.mms.salinity.solutions.dSdx, N, cell_x, cell_y, time, dsdx));
      PetscCall(EvaluateTemporalSolution(rdy->config.mms.salinity.solutions.dSdy, N, cell_x, cell_y, time, dsdy));
      PetscCall(EvaluateTemporalSolution(rdy->config.mms.salinity.solutions.dSdt, N, cell_x, cell_y, time, dsdt));

      // TODO: salinity logic goes here!

      PetscCall(PetscFree(s));
      PetscCall(PetscFree(dsdx));
      PetscCall(PetscFree(dsdy));
      PetscCall(PetscFree(dsdt));
    }

    PetscCall(PetscFree(h));
    PetscCall(PetscFree(u));
    PetscCall(PetscFree(v));
    PetscCall(PetscFree(dhdx));
    PetscCall(PetscFree(dhdy));
    PetscCall(PetscFree(dhdt));
    PetscCall(PetscFree(dudx));
    PetscCall(PetscFree(dudy));
    PetscCall(PetscFree(dudt));
    PetscCall(PetscFree(dvdx));
    PetscCall(PetscFree(dvdy));
    PetscCall(PetscFree(dvdt));
    PetscCall(PetscFree(n));
    PetscCall(PetscFree(dzdx));
    PetscCall(PetscFree(dzdy));
    PetscCall(PetscFree(h_source));
    PetscCall(PetscFree(hu_source));
    PetscCall(PetscFree(hv_source));
  }
  PetscCall(PetscFree(cell_x));
  PetscCall(PetscFree(cell_y));

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
    PetscReal * x, *y;
    PetscCall(PetscCalloc1(num_edges, &x));
    PetscCall(PetscCalloc1(num_edges, &y));
    for (PetscInt e = 0; e < num_edges; ++e) {
      PetscInt edge_id       = boundary.edge_ids[e];
      RDyPoint edge_centroid = rdy->mesh.edges.centroids[edge_id];
      x[e]                   = edge_centroid.X[0];
      y[e]                   = edge_centroid.X[1];
    }

    // compute h, hu, hv on each edge (SWE-specific)
    RDyFlowCondition* flow_bc = rdy->boundary_conditions[b].flow;
    PetscReal *       h, *u, *v;
    PetscCall(PetscCalloc1(num_edges, &h));
    PetscCall(PetscCalloc1(num_edges, &u));
    PetscCall(PetscCalloc1(num_edges, &v));
    PetscCall(EvaluateTemporalSolution(flow_bc->height, num_edges, x, y, time, h));
    PetscCall(EvaluateTemporalSolution(flow_bc->x_momentum, num_edges, x, y, time, u));
    PetscCall(EvaluateTemporalSolution(flow_bc->y_momentum, num_edges, x, y, time, v));

    // set flow boundary values (SWE-specific, ndof == 3)
    PetscReal* boundary_values;
    PetscCall(PetscCalloc1(3 * num_edges, &boundary_values));
    for (PetscInt e = 0; e < num_edges; ++e) {
      boundary_values[3 * e]     = h[e];
      boundary_values[3 * e + 1] = h[e] * u[e];
      boundary_values[3 * e + 2] = h[e] * v[e];
    }
    PetscCall(RDySetFlowDirichletBoundaryValues(rdy, b, num_edges, 3, boundary_values));

    // set tracer boundary values
    PetscInt num_sediment_classes = rdy->config.physics.sediment.num_classes;
    if (num_sediment_classes > 0) {
      PetscReal *sediment_boundary_values, *ci;
      PetscCall(PetscCalloc1(num_sediment_classes * num_edges, &sediment_boundary_values));
      PetscCall(PetscCalloc1(num_edges, &ci));
      RDySedimentCondition* sediment_bc = rdy->boundary_conditions[b].sediment;
      for (PetscInt i = 0; i < num_sediment_classes; ++i) {
        PetscCall(EvaluateTemporalSolution(sediment_bc->classes[i].value, num_edges, x, y, time, ci));
        for (PetscInt e = 0; e < num_edges; ++e) {
          sediment_boundary_values[num_sediment_classes * e + i] = h[e] * ci[e];
        }
      }
      PetscCall(RDySetSedimentDirichletBoundaryValues(rdy, b, num_edges, num_sediment_classes, sediment_boundary_values));
      PetscCall(PetscFree(sediment_boundary_values));
      PetscCall(PetscFree(ci));
    }
    if (rdy->config.physics.salinity) {
      PetscReal *salinity_boundary_values, *s;
      PetscCall(PetscCalloc1(num_edges, &salinity_boundary_values));
      PetscCall(PetscCalloc1(num_edges, &s));
      RDySalinityCondition* salinity_bc = rdy->boundary_conditions[b].salinity;
      PetscCall(EvaluateTemporalSolution(salinity_bc->concentration, num_edges, x, y, time, s));
      for (PetscInt e = 0; e < num_edges; ++e) {
        salinity_boundary_values[e] = h[e] * s[e];
      }
      PetscCall(RDySetSalinityDirichletBoundaryValues(rdy, b, num_edges, salinity_boundary_values));
      PetscCall(PetscFree(salinity_boundary_values));
      PetscCall(PetscFree(s));
    }
    if (rdy->config.physics.heat) {
      PetscReal *temperature_boundary_values, *T;
      PetscCall(PetscCalloc1(num_edges, &temperature_boundary_values));
      PetscCall(PetscCalloc1(num_edges, &T));
      RDyHeatCondition* heat_bc = rdy->boundary_conditions[b].heat;
      PetscCall(EvaluateTemporalSolution(heat_bc->water_temperature, num_edges, x, y, time, T));
      for (PetscInt e = 0; e < num_edges; ++e) {
        temperature_boundary_values[e] = T[e];
      }
      PetscCall(RDySetHeatDirichletBoundaryValues(rdy, b, num_edges, temperature_boundary_values));
      PetscCall(PetscFree(temperature_boundary_values));
      PetscCall(PetscFree(T));
    }

    PetscCall(PetscFree(x));
    PetscCall(PetscFree(y));
    PetscCall(PetscFree(h));
    PetscCall(PetscFree(u));
    PetscCall(PetscFree(v));
    PetscCall(PetscFree(boundary_values));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// updates relevant material properties for the method of manufactured solutions
// at the given time
PetscErrorCode RDyMMSUpdateMaterialProperties(RDy rdy) {
  PetscFunctionBegin;

  // initialize the material properties on each region
  PetscInt n_local;
  PetscCall(VecGetLocalSize(rdy->u_global, &n_local));
  PetscInt ndof;
  PetscCall(VecGetBlockSize(rdy->u_global, &ndof));

  for (PetscInt r = 0; r < rdy->num_regions; ++r) {
    RDyRegion region = rdy->regions[r];

    // create vectorized (x, y) pairs for bulk expression evaluation
    PetscReal *cell_x, *cell_y;
    PetscCall(PetscCalloc1(region.num_local_cells, &cell_x));
    PetscCall(PetscCalloc1(region.num_local_cells, &cell_y));

    PetscInt N = 0;  // number of bulk evaluations
    for (PetscInt c = 0; c < region.num_local_cells; ++c) {
      PetscInt cell_id = region.cell_local_ids[c];
      if (ndof * cell_id < n_local) {
        cell_x[N] = rdy->mesh.cells.centroids[cell_id].X[0];
        cell_y[N] = rdy->mesh.cells.centroids[cell_id].X[1];
        ++N;
      }
    }

    // evaluate and set material properties
    if (rdy->config.physics.flow.mode == FLOW_SWE) {
      OperatorData material_properties;
      PetscCall(GetOperatorRegionalMaterialProperties(rdy->operator, region, &material_properties));
      PetscCall(EvaluateSpatialSolution(rdy->config.mms.swe.solutions.n, N, cell_x, cell_y, material_properties.values[MATERIAL_PROPERTY_MANNINGS]));
      PetscCall(RestoreOperatorRegionalMaterialProperties(rdy->operator, region, &material_properties));
    }
    PetscCall(PetscFree(cell_x));
    PetscCall(PetscFree(cell_y));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Computes componentwise L1, L2, and Linf norms of (u_global - reference) over
// this rank's owned cells and reduces them across ranks. When heat is enabled a
// trailing derived-temperature component is appended: T is recovered as hT/h
// with the same tiny_h guard the operators use, and the reference temperature
// comes from T_reference when given, or from the reference vector's own hT/h
// otherwise.
static PetscErrorCode ComputeComponentwiseNorms(RDy rdy, Vec reference, const PetscReal* T_reference, PetscReal* L1_norms, PetscReal* L2_norms,
                                                PetscReal* Linf_norms, PetscInt* num_global_cells, PetscReal* global_area) {
  PetscFunctionBegin;

  PetscInt ndof;
  PetscCall(VecGetBlockSize(reference, &ndof));
  PetscInt num_comps        = MMSNumErrorComponents(rdy);
  PetscInt heat_comp        = rdy->config.physics.heat ? MMSHeatComponentIndex(rdy) : -1;
  PetscInt temperature_comp = rdy->config.physics.heat ? 3 + rdy->num_tracers : -1;

  const PetscReal tiny_h = rdy->config.physics.flow.tiny_h;

  const PetscScalar *u, *r;
  PetscCall(VecGetArrayRead(rdy->u_global, &u));
  PetscCall(VecGetArrayRead(reference, &r));

  PetscReal area_sum = 0.0;
  memset(L1_norms, 0, num_comps * sizeof(PetscReal));
  memset(L2_norms, 0, num_comps * sizeof(PetscReal));
  memset(Linf_norms, 0, num_comps * sizeof(PetscReal));
  for (PetscInt i = 0; i < rdy->mesh.num_owned_cells; ++i) {
    PetscInt  cell_id = rdy->mesh.cells.owned_to_local[i];
    PetscReal area    = rdy->mesh.cells.areas[cell_id];

    for (PetscInt dof = 0; dof < ndof; ++dof) {
      PetscReal e_dof = PetscRealPart(u[ndof * i + dof] - r[ndof * i + dof]);
      L1_norms[dof] += PetscAbsReal(e_dof) * area;
      L2_norms[dof] += e_dof * e_dof * area;
      Linf_norms[dof] = PetscMax(PetscAbsReal(e_dof), Linf_norms[dof]);
    }

    if (temperature_comp >= 0) {
      PetscReal h_num = PetscRealPart(u[ndof * i]);
      PetscReal T_num = (h_num >= tiny_h) ? PetscRealPart(u[ndof * i + heat_comp]) / h_num : 0.0;
      PetscReal T_ref;
      if (T_reference) {
        T_ref = T_reference[i];
      } else {
        PetscReal h_ref = PetscRealPart(r[ndof * i]);
        T_ref           = (h_ref >= tiny_h) ? PetscRealPart(r[ndof * i + heat_comp]) / h_ref : 0.0;
      }
      PetscReal e_T = T_num - T_ref;
      L1_norms[temperature_comp] += PetscAbsReal(e_T) * area;
      L2_norms[temperature_comp] += e_T * e_T * area;
      Linf_norms[temperature_comp] = PetscMax(PetscAbsReal(e_T), Linf_norms[temperature_comp]);
    }
    area_sum += area;
  }
  PetscCall(VecRestoreArrayRead(reference, &r));
  PetscCall(VecRestoreArrayRead(rdy->u_global, &u));

  // obtain global norms
  PetscCall(MPI_Allreduce(MPI_IN_PLACE, L1_norms, num_comps, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD));
  PetscCall(MPI_Allreduce(MPI_IN_PLACE, L2_norms, num_comps, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD));
  PetscCall(MPI_Allreduce(MPI_IN_PLACE, Linf_norms, num_comps, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD));

  for (PetscInt dof = 0; dof < num_comps; ++dof) {
    L2_norms[dof] = PetscSqrtReal(L2_norms[dof]);
  }

  // obtain optional diagnostics
  if (num_global_cells) {
    PetscMPIInt ncells;
    PetscCall(MPI_Reduce(&rdy->mesh.num_owned_cells, &ncells, 1, MPI_INT, MPI_SUM, 0, PETSC_COMM_WORLD));
    *num_global_cells = (PetscInt)ncells;
  }
  if (global_area) {
    PetscCall(MPI_Reduce(&area_sum, global_area, 1, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Computes the componentwise L1, L2, and Linf error norms for the relevant
// manufactured solution at the given time. L1_norms, L2_norms, and Linf_norms
// are all arrays large enough to store MMSNumErrorComponents(rdy) values --
// the prognostic dof, plus the derived temperature when heat is enabled. If
// non-NULL, num_global_cells stores the number of distinct global cells and
// global_area stores the total area covered by distinct global cells.
PetscErrorCode RDyMMSComputeErrorNorms(RDy rdy, PetscReal time, PetscReal* L1_norms, PetscReal* L2_norms, PetscReal* Linf_norms,
                                       PetscInt* num_global_cells, PetscReal* global_area) {
  PetscFunctionBegin;

  Vec exact;
  PetscCall(RDyCreatePrognosticVec(rdy, &exact));
  PetscCall(RDyMMSComputeSolution(rdy, time, exact));

  // Evaluate the exact temperature from the manufactured expression at the owned
  // cell centroids, so the reported T error is a true pointwise difference
  // rather than a ratio of two error norms.
  PetscReal* T_exact = NULL;
  if (rdy->config.physics.heat) {
    PetscInt   N;
    PetscReal *cell_x, *cell_y;
    PetscCall(GetOwnedCellCentroids(rdy, &N, &cell_x, &cell_y));
    PetscCall(PetscCalloc1(N, &T_exact));
    PetscCall(EvaluateTemporalSolution(rdy->config.mms.temperature.solutions.T, N, cell_x, cell_y, time, T_exact));
    PetscCall(PetscFree(cell_x));
    PetscCall(PetscFree(cell_y));
  }

  PetscCall(ComputeComponentwiseNorms(rdy, exact, T_exact, L1_norms, L2_norms, Linf_norms, num_global_cells, global_area));

  PetscCall(PetscFree(T_exact));
  PetscCall(VecDestroy(&exact));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PrintErrorNorms(MPI_Comm comm, PetscReal time, int num_comps, PetscReal* L1_norms, PetscReal* L2_norms, PetscReal* Linf_norms) {
  PetscFunctionBegin;
  PetscPrintf(comm, "  Error norms at t = %g:\n", time);
  for (PetscInt c = 0; c < num_comps; ++c) {
    PetscPrintf(comm, "    %s: L1 = %g, L2 = %g, Linf = %g\n", mms_comp_names[c], L1_norms[c], L2_norms[c], Linf_norms[c]);
  }
  PetscPrintf(comm, "\n");
  PetscFunctionReturn(PETSC_SUCCESS);
}

// performs a temporo-spatial convergence study using the given instance of RDy
// as a coarse grid, uniformly refining it the number of times specific in the
// mms section of the configuration and evolving the solution to the given time,
// computing error norms for each component, and calculating rates of
// convergence (and variances) with linear regression
PetscErrorCode RDyMMSEstimateConvergenceRates(RDy rdy, PetscReal* L1_conv_rates, PetscReal* L2_conv_rates, PetscReal* Linf_conv_rates) {
  PetscFunctionBegin;

  PetscReal final_time = rdy->config.time.stop;

  PetscInt dim;
  PetscCall(DMGetDimension(rdy->dm, &dim));

  int num_refinements = rdy->config.mms.convergence.num_refinements;
  int base_refinement = rdy->config.mms.convergence.base_refinement;

#define MAX_NUM_REFINEMENTS 8
  PetscCheck(num_refinements <= MAX_NUM_REFINEMENTS, rdy->comm, PETSC_ERR_USER, "Number of refinements (%d) exceeds maximum (%d)", num_refinements,
             MAX_NUM_REFINEMENTS);

  // error norm storage
  PetscReal L1_norms[MAX_NUM_REFINEMENTS + 1][MAX_NUM_COMPONENTS], L2_norms[MAX_NUM_REFINEMENTS + 1][MAX_NUM_COMPONENTS],
      Linf_norms[MAX_NUM_REFINEMENTS + 1][MAX_NUM_COMPONENTS];

  int num_comps = (int)MMSNumErrorComponents(rdy);

  // timestep refinement schedule, taken from the unrefined (level 0) instance so
  // every level scales relative to it rather than chaining off its predecessor
  PetscInt  dt_exponent    = rdy->config.mms.convergence.timestep_refinement_exponent;
  PetscReal base_dt        = rdy->dt;  // seconds; honors any -dt override
  PetscReal base_time_step = rdy->config.time.time_step;
  PetscInt  base_stop_n    = rdy->config.time.stop_n;
  PetscCheck(dt_exponent >= 0, rdy->comm, PETSC_ERR_USER, "mms.convergence.timestep_refinement_exponent (%" PetscInt_FMT ") must be non-negative",
             dt_exponent);
  if (dt_exponent) {
    PetscPrintf(rdy->comm, "Refining the timestep with the mesh: dt ~ dx^%" PetscInt_FMT " (dt_0 = %g s, stop_n_0 = %" PetscInt_FMT ")\n",
                dt_exponent, (double)base_dt, base_stop_n);
  }

  // create refined RDy objects and set them up (dumb, but easy)
  RDy rdys[MAX_NUM_REFINEMENTS + 1];
  rdys[0] = rdy;
  for (PetscInt r = 1; r <= num_refinements; ++r) {
    PetscCall(RDyCreate(rdy->comm, rdy->config_file, &rdys[r]));
    char num_refinements[5];
    snprintf(num_refinements, 4, "%" PetscInt_FMT, r + base_refinement);
    PetscCall(PetscOptionsSetValue(NULL, "-dm_refine", num_refinements));
    PetscCall(RDyMMSSetup(rdys[r]));

    // Override timestepping info (no good way to do this currently). Each level
    // halves dx, so the timestep is scaled by 2^(-p) per level and stop_n by the
    // reciprocal; without the stop_n change the finer levels would stop early at
    // a different final time and the error comparison would be meaningless.
    PetscReal level_factor         = PetscPowReal(2.0, (PetscReal)(dt_exponent * r));
    rdys[r]->config.time.time_step = base_time_step / level_factor;
    rdys[r]->config.time.stop_n    = (PetscInt)PetscCeilReal(base_stop_n * level_factor);
    rdys[r]->dt                    = base_dt / level_factor;
    TSSetTimeStep(rdys[r]->ts, rdys[r]->dt);
    TSSetMaxSteps(rdys[r]->ts, rdys[r]->config.time.stop_n);
  }

  for (PetscInt r = 0; r <= num_refinements; ++r) {
    PetscPrintf(rdys[r]->comm, "Refinement level %" PetscInt_FMT ":\n", r + base_refinement);

    // run the problem to completion
    PetscCall(TSSolve(rdys[r]->ts, rdys[r]->u_global));

    // compute error norms for this refinement level
    PetscCall(RDyMMSComputeErrorNorms(rdys[r], final_time, L1_norms[r], L2_norms[r], Linf_norms[r], NULL, NULL));
    PrintErrorNorms(rdys[r]->comm, final_time, num_comps, L1_norms[r], L2_norms[r], Linf_norms[r]);
  }

  // calculate the spatial discretization parameter N, where h^{-dim} = N.
  PetscReal x[MAX_NUM_REFINEMENTS + 1];
  for (PetscInt r = 0; r <= num_refinements; ++r) {
    PetscInt N = rdys[r]->mesh.num_cells_global;
    x[r]       = PetscLog10Real(N);
  }

  // fit convergence rates
  PetscReal y1[MAX_NUM_REFINEMENTS + 1], y2[MAX_NUM_REFINEMENTS + 1], yinf[MAX_NUM_REFINEMENTS + 1];
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

// Writes the final prognostic state to a PETSc binary file so a later run can
// difference against it. Refining dt on a fixed mesh does not drive the error
// against the exact solution to zero -- it converges to the semi-discrete
// solution, whose distance from the exact solution is the O(dx) transport error.
// Differencing two runs on the *same* mesh removes that floor and leaves the
// temporal-plus-splitting error, which is what a temporal order claim needs.
//
// NOTE: the file stores the global vector in DMPlex's distributed ordering, so a
// NOTE: reference is only comparable to a run on the same mesh with the same
// NOTE: number of ranks.
static PetscErrorCode SaveFinalState(RDy rdy, const char* filename) {
  PetscFunctionBegin;
  PetscViewer viewer;
  PetscCall(PetscViewerBinaryOpen(rdy->comm, filename, FILE_MODE_WRITE, &viewer));
  PetscCall(VecView(rdy->u_global, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscPrintf(rdy->comm, "  Wrote final state to %s\n", filename);
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Loads a reference state written by SaveFinalState and reports componentwise
// difference norms against the current solution.
static PetscErrorCode ReportReferenceDifference(RDy rdy, const char* filename, PetscInt num_comps) {
  PetscFunctionBegin;
  Vec reference;
  PetscCall(RDyCreatePrognosticVec(rdy, &reference));

  PetscViewer viewer;
  PetscCall(PetscViewerBinaryOpen(rdy->comm, filename, FILE_MODE_READ, &viewer));
  PetscCall(VecLoad(reference, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscReal L1_norms[MAX_NUM_COMPONENTS], L2_norms[MAX_NUM_COMPONENTS], Linf_norms[MAX_NUM_COMPONENTS];
  PetscCall(ComputeComponentwiseNorms(rdy, reference, NULL, L1_norms, L2_norms, Linf_norms, NULL, NULL));

  PetscPrintf(rdy->comm, "  Reference differences (%s):\n", filename);
  for (PetscInt c = 0; c < num_comps; ++c) {
    PetscPrintf(rdy->comm, "    ref %s: L1 = %g, L2 = %g, Linf = %g\n", mms_comp_names[c], L1_norms[c], L2_norms[c], Linf_norms[c]);
  }
  PetscPrintf(rdy->comm, "\n");

  PetscCall(VecDestroy(&reference));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define CheckConvergence(comp, comp_index, norm)                                                                                         \
  if (isnan(norm##_conv_rates[comp_index]) || (norm##_conv_rates[comp_index] <= rdy->config.mms.convergence.expected_rates.comp.norm)) { \
    SETERRQ(rdy->comm, PETSC_ERR_USER, "FAIL: %s convergence rate for %s is %g (expected %g)", #norm, mms_comp_names[comp_index],        \
            norm##_conv_rates[comp_index], rdy->config.mms.convergence.expected_rates.comp.norm);                                        \
  }

PetscErrorCode RDyMMSRun(RDy rdy) {
  PetscFunctionBegin;

  PetscInt num_comps = MMSNumErrorComponents(rdy);  // NOTE: SWE assumed!
  if (rdy->config.mms.convergence.num_refinements) {
    PetscReal L1_conv_rates[MAX_NUM_COMPONENTS], L2_conv_rates[MAX_NUM_COMPONENTS], Linf_conv_rates[MAX_NUM_COMPONENTS];
    // run a convergence study
    PetscCall(RDyMMSEstimateConvergenceRates(rdy, L1_conv_rates, L2_conv_rates, Linf_conv_rates));

    PetscPrintf(rdy->comm, "Convergence rates:\n");
    for (PetscInt idof = 0; idof < num_comps; idof++) {
      PetscPrintf(rdy->comm, "  %s: L1 = %g, L2 = %g, Linf = %g\n", mms_comp_names[idof], L1_conv_rates[idof], L2_conv_rates[idof],
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

    // NOTE: rdy->num_tracers counts salinity and heat as well as sediment, so
    // NOTE: this loop is restricted to the sediment classes that c[] describes;
    // NOTE: salinity and heat are checked against their own thresholds below.
    for (PetscInt i = 0; i < rdy->config.physics.sediment.num_classes; ++i) {
      CheckConvergence(c[i], 3 + i, L1);
      CheckConvergence(c[i], 3 + i, L2);
      CheckConvergence(c[i], 3 + i, Linf);
    }
    if (rdy->config.physics.salinity) {
      PetscInt salinity_index = 3 + rdy->config.physics.sediment.num_classes;
      CheckConvergence(S, salinity_index, L1);
      CheckConvergence(S, salinity_index, L2);
      CheckConvergence(S, salinity_index, Linf);
    }
    // The conservative variable hT is what the solver advances; T = hT/h is the
    // physically meaningful quantity and can combine or cancel h and hT errors,
    // so the two carry independent thresholds.
    if (rdy->config.physics.heat) {
      PetscInt heat_index        = MMSHeatComponentIndex(rdy);
      PetscInt temperature_index = 3 + rdy->num_tracers;
      CheckConvergence(hT, heat_index, L1);
      CheckConvergence(hT, heat_index, L2);
      CheckConvergence(hT, heat_index, Linf);
      CheckConvergence(T, temperature_index, L1);
      CheckConvergence(T, temperature_index, L2);
      CheckConvergence(T, temperature_index, Linf);
    }
    PetscPrintf(rdy->comm, "PASS: all convergence rates satisfy thresholds.\n");
  } else {
    PetscReal L1_norms[MAX_NUM_COMPONENTS], L2_norms[MAX_NUM_COMPONENTS], Linf_norms[MAX_NUM_COMPONENTS];

    // run the problem to completion and print error norms
    if (rdy->config.physics.heat) {
      // For MMS heat runs, use the transport TSSolve directly so MMSPostStep
      // handles the second heat TSSolve after each transport step. Using
      // RDyAdvance would trigger a duplicate heat solve.
      PetscReal final_time = ConvertTimeToSeconds(rdy->config.time.stop, rdy->config.time.unit);
      PetscCall(TSSetMaxTime(rdy->ts, final_time));
      PetscCall(TSSetExactFinalTime(rdy->ts, TS_EXACTFINALTIME_MATCHSTEP));
      PetscCall(TSSolve(rdy->ts, rdy->u_global));
    } else {
      while (!RDyFinished(rdy)) {
        PetscCall(RDyAdvance(rdy));
      }
    }

    // compute error norms for the final solution
    RDyTimeUnit time_unit;
    PetscCall(RDyGetTimeUnit(rdy, &time_unit));
    PetscReal cur_time;
    PetscCall(RDyGetTime(rdy, time_unit, &cur_time));
    PetscReal global_area;
    PetscInt  num_global_cells;
    PetscCall(RDyMMSComputeErrorNorms(rdy, cur_time, L1_norms, L2_norms, Linf_norms, &num_global_cells, &global_area));

    PrintErrorNorms(rdy->comm, cur_time, num_comps, L1_norms, L2_norms, Linf_norms);

    if (rdy->config.physics.heat) {
      PetscReal max_heat_source;
      PetscCall(MPI_Allreduce(&mms_max_heat_source, &max_heat_source, 1, MPI_DOUBLE, MPI_MAX, rdy->comm));
      PetscPrintf(rdy->comm, "  Max-|Q_mms|-inf  : %18.12e\n", max_heat_source);
    }
    PetscPrintf(rdy->comm, "  Avg-cell-area    : %18.16f\n", global_area / num_global_cells);
    PetscPrintf(rdy->comm, "  Avg-length-scale : %18.16f\n", PetscSqrtReal(global_area / num_global_cells));

    // same-mesh reference-solution support for temporal/splitting order studies
    char      reference_file[PETSC_MAX_PATH_LEN] = {0};
    PetscBool have_reference = PETSC_FALSE, save_reference = PETSC_FALSE;
    PetscCall(PetscOptionsGetString(NULL, NULL, "-mms_save_final_state", reference_file, sizeof(reference_file), &save_reference));
    if (save_reference) PetscCall(SaveFinalState(rdy, reference_file));
    PetscCall(PetscOptionsGetString(NULL, NULL, "-mms_reference_solution", reference_file, sizeof(reference_file), &have_reference));
    if (have_reference) PetscCall(ReportReferenceDifference(rdy, reference_file, num_comps));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}
