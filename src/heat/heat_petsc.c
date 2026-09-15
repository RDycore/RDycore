#include <math.h>
#include <muParserDLL.h>
#include <petscdm.h>
#include <private/rdycoreimpl.h>
#include <private/rdyheatimpl.h>
#include <private/rdymathimpl.h>

static const PetscReal WATER_ALBEDO             = 0.08;
static const PetscReal WATER_EMISSIVITY         = 0.97;
static const PetscReal STEFAN_BOLTZMANN         = 5.670374419e-8;
static const PetscReal DENSITY_OF_AIR           = 1.225;
static const PetscReal SPECIFIC_HEAT_OF_AIR     = 1005.0;
static const PetscReal LATENT_HEAT_VAPORIZATION = 2.5e6;
static const PetscReal DENSITY_OF_WATER         = 1000.0;
static const PetscReal SPECIFIC_HEAT_OF_WATER   = 4186.0;
static const PetscReal STANDARD_AIR_PRESSURE    = 101325.0;
static const PetscReal WATER_VAPOR_EPSILON      = 0.622;
static const PetscReal CELSIUS_TO_KELVIN        = 273.15;

static PetscReal SaturationSpecificHumidity(PetscReal temp_c) {
  PetscReal e_sat = 611.2 * PetscExpReal(17.67 * temp_c / (temp_c + 243.5));
  PetscReal denom = STANDARD_AIR_PRESSURE - (1.0 - WATER_VAPOR_EPSILON) * e_sat;
  return WATER_VAPOR_EPSILON * e_sat / denom;
}

static PetscReal DSaturationSpecificHumidityDTemperature(PetscReal temp_c) {
  PetscReal e_sat = 611.2 * PetscExpReal(17.67 * temp_c / (temp_c + 243.5));
  PetscReal de_dT = e_sat * 17.67 * 243.5 / Square(temp_c + 243.5);
  PetscReal denom = STANDARD_AIR_PRESSURE - (1.0 - WATER_VAPOR_EPSILON) * e_sat;
  PetscReal dq_de = WATER_VAPOR_EPSILON * STANDARD_AIR_PRESSURE / Square(denom);
  return dq_de * de_dT;
}

static PetscReal HeatQNet(RDyHeat heat, PetscInt owned_cell, PetscReal temp_c) {
  RDyHeatForcing* forcing = &heat->forcing;
  PetscReal       temp_k  = temp_c + CELSIUS_TO_KELVIN;
  PetscReal       r_inv   = 0.2 + 0.1 * forcing->wind_speed[owned_cell];

  PetscReal q_sw = (1.0 - WATER_ALBEDO) * forcing->downwelling_shortwave[owned_cell];
  PetscReal q_lw = forcing->downwelling_longwave[owned_cell] - WATER_EMISSIVITY * STEFAN_BOLTZMANN * PetscPowReal(temp_k, 4.0);
  PetscReal q_sh = DENSITY_OF_AIR * SPECIFIC_HEAT_OF_AIR * (forcing->air_temperature[owned_cell] - temp_c) * r_inv;
  PetscReal q_e  = DENSITY_OF_AIR * LATENT_HEAT_VAPORIZATION * (forcing->specific_humidity[owned_cell] - SaturationSpecificHumidity(temp_c)) * r_inv;

  return q_sw + q_lw + q_sh + q_e;
}

static PetscReal DHeatQNetDTemperature(RDyHeat heat, PetscInt owned_cell, PetscReal temp_c) {
  RDyHeatForcing* forcing = &heat->forcing;
  PetscReal       temp_k  = temp_c + CELSIUS_TO_KELVIN;
  PetscReal       r_inv   = 0.2 + 0.1 * forcing->wind_speed[owned_cell];

  PetscReal d_q_lw = -4.0 * WATER_EMISSIVITY * STEFAN_BOLTZMANN * Cube(temp_k);
  PetscReal d_q_sh = -DENSITY_OF_AIR * SPECIFIC_HEAT_OF_AIR * r_inv;
  PetscReal d_q_e  = -DENSITY_OF_AIR * LATENT_HEAT_VAPORIZATION * DSaturationSpecificHumidityDTemperature(temp_c) * r_inv;

  return d_q_lw + d_q_sh + d_q_e;
}

// The implicit atmospheric source step comes in two flavors that differ only in
// how the net surface heat flux Q_net is obtained, so each gets its own
// IFunction/IJacobian pair rather than branching per DOF inside the loop:
//
//   * "prescribed source" - Q_net is read straight from RDyHeatForcing::direct_source
//     (a YAML heat_flux override, or the manufactured source used by the MMS driver)
//   * "atmospheric source" - Q_net is computed from the local atmospheric forcing
//     via HeatQNet()
//
// SetHeatTSCallbacks() selects the pair; see its comment for when that happens.

static PetscErrorCode HeatIFunctionPrescribedSource(TS ts, PetscReal t, Vec U, Vec Udot, Vec F, void* ctx) {
  (void)ts;
  (void)t;
  PetscFunctionBegin;
  RDy     rdy  = ctx;
  RDyHeat heat = rdy->heat_context;

  PetscInt n_dof;
  PetscCall(VecGetBlockSize(U, &n_dof));
  PetscInt start, end;
  PetscCall(VecGetOwnershipRange(U, &start, &end));

  const PetscScalar *u, *udot;
  PetscScalar*       f;
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArrayRead(Udot, &udot));
  PetscCall(VecGetArray(F, &f));

  PetscInt n_local;
  PetscCall(VecGetLocalSize(U, &n_local));

  const PetscInt   heat_comp     = heat->heat_comp;
  const PetscReal  tiny_h        = heat->config->physics.flow.tiny_h;
  const PetscReal* direct_source = heat->forcing.direct_source;

  for (PetscInt j = 0; j < n_local; ++j) {
    PetscInt comp = (start + j) % n_dof;
    f[j]          = udot[j];
    if (comp == heat_comp) {
      PetscInt  owned_cell = j / n_dof;
      PetscReal h          = u[n_dof * owned_cell];
      if (h >= tiny_h) {
        f[j] = udot[j] - direct_source[owned_cell] / (DENSITY_OF_WATER * SPECIFIC_HEAT_OF_WATER);
      }
    }
  }

  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArrayRead(Udot, &udot));
  PetscCall(VecRestoreArray(F, &f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode HeatIFunctionAtmosphericSource(TS ts, PetscReal t, Vec U, Vec Udot, Vec F, void* ctx) {
  (void)ts;
  (void)t;
  PetscFunctionBegin;
  RDy     rdy  = ctx;
  RDyHeat heat = rdy->heat_context;

  PetscInt n_dof;
  PetscCall(VecGetBlockSize(U, &n_dof));
  PetscInt start, end;
  PetscCall(VecGetOwnershipRange(U, &start, &end));

  const PetscScalar *u, *udot;
  PetscScalar*       f;
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArrayRead(Udot, &udot));
  PetscCall(VecGetArray(F, &f));

  PetscInt n_local;
  PetscCall(VecGetLocalSize(U, &n_local));

  const PetscInt  heat_comp = heat->heat_comp;
  const PetscReal tiny_h    = heat->config->physics.flow.tiny_h;

  for (PetscInt j = 0; j < n_local; ++j) {
    PetscInt comp = (start + j) % n_dof;
    f[j]          = udot[j];
    if (comp == heat_comp) {
      PetscInt  owned_cell = j / n_dof;
      PetscReal h          = u[n_dof * owned_cell];
      if (h >= tiny_h) {
        PetscReal T = u[j] / h;
        f[j]        = udot[j] - HeatQNet(heat, owned_cell, T) / (DENSITY_OF_WATER * SPECIFIC_HEAT_OF_WATER);
      }
    }
  }

  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArrayRead(Udot, &udot));
  PetscCall(VecRestoreArray(F, &f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// A prescribed Q_net does not depend on temperature, so the residual reduces to
// Udot and its Jacobian is exactly shift*I for every DOF. That needs no state and
// no per-cell work, which is why this callback is shared by both backends.
static PetscErrorCode HeatIJacobianPrescribedSource(TS ts, PetscReal t, Vec U, Vec Udot, PetscReal shift, Mat J, Mat P, void* ctx) {
  (void)ts;
  (void)t;
  (void)U;
  (void)Udot;
  (void)ctx;
  PetscFunctionBegin;

  PetscCall(MatZeroEntries(P));
  PetscCall(MatShift(P, shift));
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  if (J != P) {
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode HeatIJacobianAtmosphericSource(TS ts, PetscReal t, Vec U, Vec Udot, PetscReal shift, Mat J, Mat P, void* ctx) {
  (void)ts;
  (void)t;
  (void)Udot;
  PetscFunctionBegin;
  RDy     rdy  = ctx;
  RDyHeat heat = rdy->heat_context;

  PetscInt n_dof, start, end;
  PetscCall(VecGetBlockSize(U, &n_dof));
  PetscCall(VecGetOwnershipRange(U, &start, &end));

  const PetscScalar* u;
  PetscCall(VecGetArrayRead(U, &u));

  const PetscInt  heat_comp = heat->heat_comp;
  const PetscReal tiny_h    = heat->config->physics.flow.tiny_h;

  PetscCall(MatZeroEntries(P));
  for (PetscInt j = 0; j < end - start; ++j) {
    PetscInt  global = start + j;
    PetscInt  comp   = global % n_dof;
    PetscReal diag   = shift;
    if (comp == heat_comp) {
      PetscInt  owned_cell = j / n_dof;
      PetscReal h          = u[n_dof * owned_cell];
      if (h >= tiny_h) {
        PetscReal T  = u[j] / h;
        PetscReal dQ = DHeatQNetDTemperature(heat, owned_cell, T);
        diag         = shift - dQ / (DENSITY_OF_WATER * SPECIFIC_HEAT_OF_WATER * h);
      }
    }
    PetscCall(MatSetValue(P, global, global, diag, INSERT_VALUES));
  }
  PetscCall(VecRestoreArrayRead(U, &u));

  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  if (J != P) {
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Installs the IFunction/IJacobian pair matching the active source treatment and
// backend.
//
// NOTE: use_direct_source is not fixed for the lifetime of the TS - the MMS driver
// NOTE: raises it around every RDyHeatAdvance() and lowers it afterwards - so this
// NOTE: runs immediately before each solve rather than only at setup.
static PetscErrorCode SetHeatTSCallbacks(RDy rdy) {
  PetscFunctionBegin;
  RDyHeat heat = rdy->heat_context;

  TSIFunctionFn* ifunction;
  TSIJacobianFn* ijacobian;
  if (heat->use_direct_source) {
    ifunction = CeedEnabled() ? HeatIFunctionCeedPrescribedSource : HeatIFunctionPrescribedSource;
    ijacobian = HeatIJacobianPrescribedSource;  // shift*I, so no backend-specific variant
  } else {
    ifunction = CeedEnabled() ? HeatIFunctionCeedAtmosphericSource : HeatIFunctionAtmosphericSource;
    ijacobian = CeedEnabled() ? HeatIJacobianCeedAtmosphericSource : HeatIJacobianAtmosphericSource;
  }

  PetscCall(TSSetIFunction(rdy->heat_ts, NULL, ifunction, rdy));
  PetscCall(TSSetIJacobian(rdy->heat_ts, rdy->heat_jac, rdy->heat_jac, ijacobian, rdy));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FillForcingFromSources(RDy rdy) {
  PetscFunctionBegin;
  RDyHeat heat = rdy->heat_context;

  // In MMS mode, sources are not set up; nothing to do
  if (!rdy->sources) PetscFunctionReturn(PETSC_SUCCESS);

  for (PetscInt r = 0; r < rdy->num_regions; ++r) {
    RDyRegion    region = rdy->regions[r];
    RDyCondition src    = rdy->sources[r];
    PetscCheck(src.heat, rdy->comm, PETSC_ERR_USER, "Region '%s' has no heat source condition!", region.name);

    // If heat_flux is specified, use direct_source instead of the atmospheric parameterization
    if (src.heat->heat_flux) {
      PetscReal qnet = mupEval(src.heat->heat_flux);
      for (PetscInt c = 0; c < region.num_owned_cells; ++c) {
        PetscInt owned_cell                     = region.owned_cell_global_ids[c];
        heat->forcing.direct_source[owned_cell] = qnet;
      }
      heat->use_direct_source = PETSC_TRUE;
      continue;
    }

    PetscReal downwelling_shortwave = mupEval(src.heat->downwelling_shortwave);
    PetscReal downwelling_longwave  = mupEval(src.heat->downwelling_longwave);
    PetscReal wind_speed            = mupEval(src.heat->wind_speed);
    PetscReal air_temperature       = mupEval(src.heat->air_temperature);
    PetscReal specific_humidity     = mupEval(src.heat->specific_humidity);

    for (PetscInt c = 0; c < region.num_owned_cells; ++c) {
      PetscInt owned_cell                             = region.owned_cell_global_ids[c];
      heat->forcing.downwelling_shortwave[owned_cell] = downwelling_shortwave;
      heat->forcing.downwelling_longwave[owned_cell]  = downwelling_longwave;
      heat->forcing.wind_speed[owned_cell]            = wind_speed;
      heat->forcing.air_temperature[owned_cell]       = air_temperature;
      heat->forcing.specific_humidity[owned_cell]     = specific_humidity;
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyHeatCreate(RDy rdy) {
  PetscFunctionBegin;
  PetscCall(PetscCalloc1(1, &rdy->heat_context));
  RDyHeat heat    = rdy->heat_context;
  heat->mesh      = &rdy->mesh;
  heat->config    = &rdy->config;
  heat->heat_comp = 3 + rdy->config.physics.sediment.num_classes + (rdy->config.physics.salinity ? 1 : 0);
  heat->dt        = rdy->dt;

  PetscCall(DMCreateMatrix(rdy->dm, &rdy->heat_jac));

  PetscInt num_owned_cells = rdy->mesh.num_owned_cells;
  PetscCall(PetscCalloc1(num_owned_cells, &heat->forcing.downwelling_shortwave));
  PetscCall(PetscCalloc1(num_owned_cells, &heat->forcing.downwelling_longwave));
  PetscCall(PetscCalloc1(num_owned_cells, &heat->forcing.wind_speed));
  PetscCall(PetscCalloc1(num_owned_cells, &heat->forcing.air_temperature));
  PetscCall(PetscCalloc1(num_owned_cells, &heat->forcing.specific_humidity));
  PetscCall(PetscCalloc1(num_owned_cells, &heat->forcing.direct_source));
  PetscCall(FillForcingFromSources(rdy));

  PetscCall(TSCreate(rdy->comm, &rdy->heat_ts));
  PetscCall(TSSetType(rdy->heat_ts, TSBEULER));
  if (CeedEnabled()) PetscCall(CreateCeedHeatOperators(rdy));
  PetscCall(SetHeatTSCallbacks(rdy));
  PetscCall(TSSetOptionsPrefix(rdy->heat_ts, "heat_"));
  PetscCall(TSSetFromOptions(rdy->heat_ts));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyHeatDestroy(RDy rdy) {
  PetscFunctionBegin;
  if (rdy->heat_ts) PetscCall(TSDestroy(&rdy->heat_ts));
  if (rdy->heat_jac) PetscCall(MatDestroy(&rdy->heat_jac));
  if (rdy->heat_context) {
    RDyHeat heat = rdy->heat_context;
    if (CeedEnabled()) PetscCall(DestroyCeedHeatOperators(rdy));
    PetscCall(PetscFree(heat->forcing.downwelling_shortwave));
    PetscCall(PetscFree(heat->forcing.downwelling_longwave));
    PetscCall(PetscFree(heat->forcing.wind_speed));
    PetscCall(PetscFree(heat->forcing.air_temperature));
    PetscCall(PetscFree(heat->forcing.specific_humidity));
    PetscCall(PetscFree(heat->forcing.direct_source));
    PetscCall(PetscFree(rdy->heat_context));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyHeatUpdateForcing(RDy rdy, PetscReal time) {
  PetscFunctionBegin;
  (void)time;
  rdy->heat_context->dt = rdy->dt;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyHeatAdvance(RDy rdy, PetscReal start_time, PetscReal end_time) {
  PetscFunctionBegin;
  PetscCheck(end_time > start_time, rdy->comm, PETSC_ERR_ARG_OUTOFRANGE, "Heat end time %g must be greater than start time %g", (double)end_time,
             (double)start_time);

  // the source treatment may have changed since the last solve, so pick the
  // matching callbacks and push any forcing updates through to the CEED operators
  PetscCall(SetHeatTSCallbacks(rdy));
  if (CeedEnabled()) PetscCall(UpdateCeedHeatForcing(rdy));

  PetscReal interval = end_time - start_time;
  PetscCall(TSSetTime(rdy->heat_ts, start_time));
  PetscCall(TSSetMaxTime(rdy->heat_ts, end_time));
  PetscCall(TSSetExactFinalTime(rdy->heat_ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetTimeStep(rdy->heat_ts, interval));
  PetscCall(TSSolve(rdy->heat_ts, rdy->u_global));

  PetscFunctionReturn(PETSC_SUCCESS);
}
