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

static PetscErrorCode HeatResidual(SNES snes, Vec X, Vec F, void* ctx) {
  PetscFunctionBegin;
  RDy     rdy  = ctx;
  RDyHeat heat = rdy->heat_context;

  PetscInt n_dof;
  PetscCall(VecGetBlockSize(X, &n_dof));
  PetscInt start, end;
  PetscCall(VecGetOwnershipRange(X, &start, &end));

  const PetscScalar *x, *star;
  PetscScalar*       f;
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(VecGetArrayRead(heat->star_state, &star));
  PetscCall(VecGetArray(F, &f));

  PetscInt n_local;
  PetscCall(VecGetLocalSize(X, &n_local));
  for (PetscInt j = 0; j < n_local; ++j) {
    PetscInt comp = (start + j) % n_dof;
    f[j]          = x[j] - star[j];
    if (comp == heat->heat_comp) {
      PetscInt  owned_cell = j / n_dof;
      PetscReal h          = x[n_dof * owned_cell];
      if (h >= heat->config->physics.flow.tiny_h) {
        PetscReal hT = x[j];
        PetscReal T  = hT / h;
        f[j]         = hT - star[j] - heat->dt * HeatQNet(heat, owned_cell, T) / (DENSITY_OF_WATER * SPECIFIC_HEAT_OF_WATER);
      }
    }
  }

  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecRestoreArrayRead(heat->star_state, &star));
  PetscCall(VecRestoreArray(F, &f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode HeatJacobian(SNES snes, Vec X, Mat J, Mat P, void* ctx) {
  PetscFunctionBegin;
  RDy     rdy  = ctx;
  RDyHeat heat = rdy->heat_context;

  PetscInt n_dof, start, end;
  PetscCall(VecGetBlockSize(X, &n_dof));
  PetscCall(VecGetOwnershipRange(X, &start, &end));

  const PetscScalar* x;
  PetscCall(VecGetArrayRead(X, &x));

  PetscCall(MatZeroEntries(P));
  for (PetscInt j = 0; j < end - start; ++j) {
    PetscInt  global = start + j;
    PetscInt  comp   = global % n_dof;
    PetscReal diag   = 1.0;
    if (comp == heat->heat_comp) {
      PetscInt  owned_cell = j / n_dof;
      PetscReal h          = x[n_dof * owned_cell];
      if (h >= heat->config->physics.flow.tiny_h) {
        PetscReal T  = x[j] / h;
        PetscReal dQ = DHeatQNetDTemperature(heat, owned_cell, T);
        diag         = 1.0 - heat->dt * dQ / (DENSITY_OF_WATER * SPECIFIC_HEAT_OF_WATER * h);
      }
    }
    PetscCall(MatSetValue(P, global, global, diag, INSERT_VALUES));
  }
  PetscCall(VecRestoreArrayRead(X, &x));

  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  if (J != P) {
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FillForcingFromSources(RDy rdy) {
  PetscFunctionBegin;
  RDyHeat heat = rdy->heat_context;

  for (PetscInt r = 0; r < rdy->num_regions; ++r) {
    RDyRegion    region = rdy->regions[r];
    RDyCondition src    = rdy->sources[r];
    PetscCheck(src.heat, rdy->comm, PETSC_ERR_USER, "Region '%s' has no heat source condition!", region.name);

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
  PetscCheck(!CeedEnabled(), rdy->comm, PETSC_ERR_USER, "heat equation support is currently implemented only for the PETSc backend");

  PetscCall(PetscCalloc1(1, &rdy->heat_context));
  RDyHeat heat    = rdy->heat_context;
  heat->mesh      = &rdy->mesh;
  heat->config    = &rdy->config;
  heat->heat_comp = 3 + rdy->config.physics.sediment.num_classes + (rdy->config.physics.salinity ? 1 : 0);
  heat->dt        = rdy->dt;

  PetscCall(VecDuplicate(rdy->u_global, &heat->star_state));
  PetscCall(VecDuplicate(rdy->u_global, &rdy->heat_residual));
  PetscCall(DMCreateMatrix(rdy->dm, &rdy->heat_jac));

  PetscInt num_owned_cells = rdy->mesh.num_owned_cells;
  PetscCall(PetscCalloc1(num_owned_cells, &heat->forcing.downwelling_shortwave));
  PetscCall(PetscCalloc1(num_owned_cells, &heat->forcing.downwelling_longwave));
  PetscCall(PetscCalloc1(num_owned_cells, &heat->forcing.wind_speed));
  PetscCall(PetscCalloc1(num_owned_cells, &heat->forcing.air_temperature));
  PetscCall(PetscCalloc1(num_owned_cells, &heat->forcing.specific_humidity));
  PetscCall(FillForcingFromSources(rdy));

  PetscCall(SNESCreate(rdy->comm, &rdy->heat_snes));
  PetscCall(SNESSetFunction(rdy->heat_snes, rdy->heat_residual, HeatResidual, rdy));
  PetscCall(SNESSetJacobian(rdy->heat_snes, rdy->heat_jac, rdy->heat_jac, HeatJacobian, rdy));
  PetscCall(SNESSetTolerances(rdy->heat_snes, 1e-10, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(SNESSetOptionsPrefix(rdy->heat_snes, "heat_"));
  PetscCall(SNESSetFromOptions(rdy->heat_snes));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyHeatDestroy(RDy rdy) {
  PetscFunctionBegin;
  if (rdy->heat_snes) PetscCall(SNESDestroy(&rdy->heat_snes));
  if (rdy->heat_jac) PetscCall(MatDestroy(&rdy->heat_jac));
  if (rdy->heat_residual) PetscCall(VecDestroy(&rdy->heat_residual));
  if (rdy->heat_context) {
    RDyHeat heat = rdy->heat_context;
    if (heat->star_state) PetscCall(VecDestroy(&heat->star_state));
    PetscCall(PetscFree(heat->forcing.downwelling_shortwave));
    PetscCall(PetscFree(heat->forcing.downwelling_longwave));
    PetscCall(PetscFree(heat->forcing.wind_speed));
    PetscCall(PetscFree(heat->forcing.air_temperature));
    PetscCall(PetscFree(heat->forcing.specific_humidity));
    PetscCall(PetscFree(rdy->heat_context));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyHeatCaptureStarState(RDy rdy) {
  PetscFunctionBegin;
  PetscCall(VecCopy(rdy->u_global, rdy->heat_context->star_state));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyHeatUpdateForcing(RDy rdy, PetscReal time) {
  PetscFunctionBegin;
  (void)time;
  rdy->heat_context->dt = rdy->dt;
  PetscFunctionReturn(PETSC_SUCCESS);
}
