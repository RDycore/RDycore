#ifndef HEAT_SOURCES_CEED_H
#define HEAT_SOURCES_CEED_H

#include "heat_types_ceed.h"

// we disable compiler warnings for implicitly-declared math functions known to
// the JIT compiler
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wimplicit-function-declaration"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wimplicit-function-declaration"

// The following Q functions use C99 VLA features for shaping multidimensional
// arrays, which don't have the same drawbacks as VLA allocations.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvla"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla"

/// Saturation specific humidity over water at the given temperature [deg C],
/// using the Magnus-Tetens approximation for the saturation vapor pressure.
CEED_QFUNCTION_HELPER CeedScalar HeatSaturationSpecificHumidity(const HeatContext context, CeedScalar temp_c) {
  const CeedScalar e_sat = 611.2 * exp(17.67 * temp_c / (temp_c + 243.5));
  const CeedScalar denom = context->standard_air_pressure - (1.0 - context->water_vapor_epsilon) * e_sat;
  return context->water_vapor_epsilon * e_sat / denom;
}

/// Derivative of HeatSaturationSpecificHumidity() with respect to temperature.
CEED_QFUNCTION_HELPER CeedScalar HeatDSaturationSpecificHumidityDTemperature(const HeatContext context, CeedScalar temp_c) {
  const CeedScalar e_sat = 611.2 * exp(17.67 * temp_c / (temp_c + 243.5));
  const CeedScalar de_dT = e_sat * 17.67 * 243.5 / Square(temp_c + 243.5);
  const CeedScalar denom = context->standard_air_pressure - (1.0 - context->water_vapor_epsilon) * e_sat;
  const CeedScalar dq_de = context->water_vapor_epsilon * context->standard_air_pressure / Square(denom);
  return dq_de * de_dT;
}

/// Bulk transfer velocity for the turbulent fluxes [m/s].
CEED_QFUNCTION_HELPER CeedScalar HeatTransferVelocityCeed(const CeedScalar forcing[NUM_HEAT_FORCINGS]) {
  return 0.2 + 0.1 * forcing[HEAT_FORCING_WIND_SPEED];
}

// The net surface heat flux is carried as two pieces rather than one, mirroring
// HeatQNonLatent()/HeatQLatent() in heat_petsc.c. The latent flux Q_e is the only
// component that moves mass, the only one subject to the evaporation cap, and the
// only one whose temperature derivative drops out of the capped Jacobian.

/// Q_sw + Q_lw + Q_sh [W/m^2]: the radiative and sensible components.
CEED_QFUNCTION_HELPER CeedScalar HeatQNonLatentCeed(const HeatContext context, const CeedScalar forcing[NUM_HEAT_FORCINGS], CeedScalar temp_c) {
  const CeedScalar temp_k = temp_c + context->celsius_to_kelvin;
  const CeedScalar r_inv  = HeatTransferVelocityCeed(forcing);

  const CeedScalar q_sw = (1.0 - context->water_albedo) * forcing[HEAT_FORCING_DOWNWELLING_SHORTWAVE];
  const CeedScalar q_lw = forcing[HEAT_FORCING_DOWNWELLING_LONGWAVE] - context->water_emissivity * context->stefan_boltzmann * pow(temp_k, 4.0);
  const CeedScalar q_sh = context->density_of_air * context->specific_heat_of_air * (forcing[HEAT_FORCING_AIR_TEMPERATURE] - temp_c) * r_inv;

  return q_sw + q_lw + q_sh;
}

/// Derivative of HeatQNonLatentCeed() with respect to temperature.
CEED_QFUNCTION_HELPER CeedScalar HeatDQNonLatentDTemperatureCeed(const HeatContext context, const CeedScalar forcing[NUM_HEAT_FORCINGS],
                                                                 CeedScalar temp_c) {
  const CeedScalar temp_k = temp_c + context->celsius_to_kelvin;
  const CeedScalar r_inv  = HeatTransferVelocityCeed(forcing);

  const CeedScalar d_q_lw = -4.0 * context->water_emissivity * context->stefan_boltzmann * temp_k * temp_k * temp_k;
  const CeedScalar d_q_sh = -context->density_of_air * context->specific_heat_of_air * r_inv;

  return d_q_lw + d_q_sh;
}

/// Q_e [W/m^2], the latent heat flux: negative while the cell evaporates (and so
/// loses water), positive under condensation (and so gains it).
CEED_QFUNCTION_HELPER CeedScalar HeatQLatentCeed(const HeatContext context, const CeedScalar forcing[NUM_HEAT_FORCINGS], CeedScalar temp_c) {
  const CeedScalar r_inv = HeatTransferVelocityCeed(forcing);

  return context->density_of_air * context->latent_heat_vaporization *
         (forcing[HEAT_FORCING_SPECIFIC_HUMIDITY] - HeatSaturationSpecificHumidity(context, temp_c)) * r_inv;
}

/// Derivative of HeatQLatentCeed() with respect to temperature.
CEED_QFUNCTION_HELPER CeedScalar HeatDQLatentDTemperatureCeed(const HeatContext context, const CeedScalar forcing[NUM_HEAT_FORCINGS],
                                                              CeedScalar temp_c) {
  const CeedScalar r_inv = HeatTransferVelocityCeed(forcing);

  return -context->density_of_air * context->latent_heat_vaporization * HeatDSaturationSpecificHumidityDTemperature(context, temp_c) * r_inv;
}

/// Lower bound on Q_e [W/m^2]: a cell may only evaporate the water it has. Over one
/// implicit step of length dt the depth changes by q_e*dt/(rho_w*L_v), so holding the
/// cell at or above tiny_h requires q_e >= -(h - tiny_h)*rho_w*L_v/dt, which is what
/// this returns. Condensation is never limited. CEED analogue of MinLatentHeatFlux()
/// in heat_petsc.c, where the reasoning is spelled out in full.
CEED_QFUNCTION_HELPER CeedScalar HeatMinLatentHeatFluxCeed(const HeatContext context, CeedScalar h) {
  return -(h - context->tiny_h) * context->density_of_water * context->latent_heat_vaporization / context->dt;
}

// The implicit source step comes in two flavors (prescribed vs. atmospheric
// Q_net); each gets its own Q-function so the choice costs nothing per
// quadrature point. See SetHeatTSCallbacks() in heat_petsc.c for the selection.

/// Q-function evaluating the residual of the implicit heat source step when the
/// net surface heat flux is prescribed per cell. CEED analogue of
/// HeatIFunctionPrescribedSource() in heat_petsc.c; purely pointwise.
///
/// Input fields:
///   in[0]: q[num_owned_cells][num_comp]                — state (active)
///   in[1]: q_dot[num_owned_cells][num_comp]            — state time derivative (passive)
///   in[2]: forcing[num_owned_cells][NUM_HEAT_FORCINGS] — atmospheric forcing (passive)
///
/// Output fields:
///   out[0]: residual[num_owned_cells][num_comp]        — implicit residual (active)
CEED_QFUNCTION(HeatIFunctionPrescribedSourceQF)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  const CeedScalar(*q)[CEED_Q_VLA]       = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_dot)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  const CeedScalar(*forcing)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];

  CeedScalar(*residual)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const HeatContext context   = (HeatContext)ctx;
  const CeedInt     num_comp  = context->num_comp;
  const CeedInt     heat_comp = context->heat_comp;
  const CeedScalar  tiny_h    = context->tiny_h;
  const CeedScalar  rho_cp    = context->density_of_water * context->specific_heat_of_water;

  for (CeedInt i = 0; i < Q; i++) {
    // every component (and every dry cell) carries the trivial residual Udot
    for (CeedInt c = 0; c < num_comp; ++c) residual[c][i] = q_dot[c][i];

    const CeedScalar h = q[0][i];
    if (h >= tiny_h) {
      residual[heat_comp][i] = q_dot[heat_comp][i] - forcing[HEAT_FORCING_DIRECT_SOURCE][i] / rho_cp;
    }
  }
  return 0;
}

/// Q-function evaluating the residual of the implicit heat source step when the
/// net surface heat flux is computed from atmospheric forcing. CEED analogue of
/// HeatIFunctionAtmosphericSource() in heat_petsc.c, which documents the physics;
/// purely pointwise.
///
/// Unlike the prescribed-source flavor, this one writes the h, hu, and hv rows as
/// well: its latent component moves water, and the momentum rows carry that water
/// away at the local flow velocity so that u = hu/h is unaffected. The hT row gains
/// no -T*hdot term, so heat concentrates into the water that remains.
///
/// Field layout matches HeatIFunctionPrescribedSourceQF().
CEED_QFUNCTION(HeatIFunctionAtmosphericSourceQF)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  const CeedScalar(*q)[CEED_Q_VLA]       = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_dot)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  const CeedScalar(*forcing)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];

  CeedScalar(*residual)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const HeatContext context   = (HeatContext)ctx;
  const CeedInt     num_comp  = context->num_comp;
  const CeedInt     heat_comp = context->heat_comp;
  const CeedScalar  tiny_h    = context->tiny_h;
  const CeedScalar  rho_lv    = context->density_of_water * context->latent_heat_vaporization;
  const CeedScalar  rho_cp    = context->density_of_water * context->specific_heat_of_water;

  for (CeedInt i = 0; i < Q; i++) {
    // every component (and every dry cell) carries the trivial residual Udot
    for (CeedInt c = 0; c < num_comp; ++c) residual[c][i] = q_dot[c][i];

    const CeedScalar h = q[0][i];
    if (h >= tiny_h) {
      CeedScalar cell_forcing[NUM_HEAT_FORCINGS];
      for (CeedInt c = 0; c < NUM_HEAT_FORCINGS; ++c) cell_forcing[c] = forcing[c][i];

      const CeedScalar temp_c  = q[heat_comp][i] / h;
      const CeedScalar q_e_raw = HeatQLatentCeed(context, cell_forcing, temp_c);
      const CeedScalar q_e_min = HeatMinLatentHeatFluxCeed(context, h);
      const CeedScalar q_e     = q_e_raw < q_e_min ? q_e_min : q_e_raw;
      const CeedScalar hdot    = q_e / rho_lv;

      residual[0][i]         = q_dot[0][i] - hdot;
      residual[1][i]         = q_dot[1][i] - (q[1][i] / h) * hdot;
      residual[2][i]         = q_dot[2][i] - (q[2][i] / h) * hdot;
      residual[heat_comp][i] = q_dot[heat_comp][i] - (HeatQNonLatentCeed(context, cell_forcing, temp_c) + q_e) / rho_cp;
    }
  }
  return 0;
}

/// Q-function evaluating the IJacobian for the atmospheric source step. The residual
/// is pointwise, so the Jacobian is block diagonal with one dense num_comp x num_comp
/// block per cell; the block is emitted in row-major order, matching the COO pattern
/// PreallocateHeatJacobianBlocks() (heat_petsc.c) installs on heat_jac, so the output
/// can be handed straight to MatSetValuesCOO(). CEED analogue of
/// HeatIJacobianAtmosphericSource() in heat_petsc.c, which derives the entries.
///
/// There is deliberately no prescribed-source counterpart: a prescribed Q_net is
/// temperature-independent and moves no water, so that Jacobian is shift*I and both
/// backends share HeatIJacobianPrescribedSource().
///
/// Input fields:
///   in[0]: q[num_owned_cells][num_comp]                — state (active)
///   in[1]: forcing[num_owned_cells][NUM_HEAT_FORCINGS] — atmospheric forcing (passive)
///
/// Output fields:
///   out[0]: jacobian[num_owned_cells][num_comp*num_comp] — row-major blocks (active)
CEED_QFUNCTION(HeatIJacobianAtmosphericSourceQF)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  const CeedScalar(*q)[CEED_Q_VLA]       = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*forcing)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];

  CeedScalar(*jacobian)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const HeatContext context   = (HeatContext)ctx;
  const CeedInt     num_comp  = context->num_comp;
  const CeedInt     heat_comp = context->heat_comp;
  const CeedScalar  tiny_h    = context->tiny_h;
  const CeedScalar  shift     = context->shift;
  const CeedScalar  rho_lv    = context->density_of_water * context->latent_heat_vaporization;
  const CeedScalar  rho_cp    = context->density_of_water * context->specific_heat_of_water;

  for (CeedInt i = 0; i < Q; i++) {
    // d(Udot)/dU is just the shift for every component (and every dry cell)
    for (CeedInt c = 0; c < num_comp * num_comp; ++c) jacobian[c][i] = 0.0;
    for (CeedInt c = 0; c < num_comp; ++c) jacobian[c * num_comp + c][i] = shift;

    const CeedScalar h = q[0][i];
    if (h >= tiny_h) {
      CeedScalar cell_forcing[NUM_HEAT_FORCINGS];
      for (CeedInt c = 0; c < NUM_HEAT_FORCINGS; ++c) cell_forcing[c] = forcing[c][i];

      const CeedScalar temp_c  = q[heat_comp][i] / h;
      const CeedScalar q_e_raw = HeatQLatentCeed(context, cell_forcing, temp_c);
      const CeedScalar q_e_min = HeatMinLatentHeatFluxCeed(context, h);
      const CeedInt    capped  = q_e_raw < q_e_min;
      const CeedScalar q_e     = capped ? q_e_min : q_e_raw;

      // the capped branch is linear in h alone, so the temperature derivative of the
      // latent flux drops out of the Jacobian entirely
      CeedScalar dqe_dh, dqe_dhT;
      if (capped) {
        dqe_dh  = -rho_lv / context->dt;
        dqe_dhT = 0.0;
      } else {
        const CeedScalar dqe_dT = HeatDQLatentDTemperatureCeed(context, cell_forcing, temp_c);
        dqe_dh                  = -dqe_dT * temp_c / h;
        dqe_dhT                 = dqe_dT / h;
      }

      const CeedScalar dnl_dT    = HeatDQNonLatentDTemperatureCeed(context, cell_forcing, temp_c);
      const CeedScalar dqnet_dh  = -dnl_dT * temp_c / h + dqe_dh;
      const CeedScalar dqnet_dhT = dnl_dT / h + dqe_dhT;

      const CeedScalar vel[2] = {q[1][i] / h, q[2][i] / h};

      jacobian[0 * num_comp + 0][i]         = shift - dqe_dh / rho_lv;
      jacobian[0 * num_comp + heat_comp][i] = -dqe_dhT / rho_lv;

      for (CeedInt m = 1; m <= 2; ++m) {
        jacobian[m * num_comp + 0][i]         = vel[m - 1] * (q_e / h - dqe_dh) / rho_lv;
        jacobian[m * num_comp + m][i]         = shift - q_e / (h * rho_lv);
        jacobian[m * num_comp + heat_comp][i] = -vel[m - 1] * dqe_dhT / rho_lv;
      }

      jacobian[heat_comp * num_comp + 0][i]         = -dqnet_dh / rho_cp;
      jacobian[heat_comp * num_comp + heat_comp][i] = shift - dqnet_dhT / rho_cp;
    }
  }
  return 0;
}

#pragma GCC diagnostic   pop
#pragma GCC diagnostic   pop
#pragma clang diagnostic pop
#pragma clang diagnostic pop

#endif
