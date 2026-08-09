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

/// Net surface heat flux Q_net [W/m^2] for water at temperature @p temp_c [deg C],
/// given the atmospheric forcing of a single cell.
CEED_QFUNCTION_HELPER CeedScalar HeatQNetCeed(const HeatContext context, const CeedScalar forcing[NUM_HEAT_FORCINGS], CeedScalar temp_c) {
  const CeedScalar temp_k = temp_c + context->celsius_to_kelvin;
  const CeedScalar r_inv  = 0.2 + 0.1 * forcing[HEAT_FORCING_WIND_SPEED];

  const CeedScalar q_sw = (1.0 - context->water_albedo) * forcing[HEAT_FORCING_DOWNWELLING_SHORTWAVE];
  const CeedScalar q_lw = forcing[HEAT_FORCING_DOWNWELLING_LONGWAVE] - context->water_emissivity * context->stefan_boltzmann * pow(temp_k, 4.0);
  const CeedScalar q_sh = context->density_of_air * context->specific_heat_of_air * (forcing[HEAT_FORCING_AIR_TEMPERATURE] - temp_c) * r_inv;
  const CeedScalar q_e  = context->density_of_air * context->latent_heat_vaporization *
                         (forcing[HEAT_FORCING_SPECIFIC_HUMIDITY] - HeatSaturationSpecificHumidity(context, temp_c)) * r_inv;

  return q_sw + q_lw + q_sh + q_e;
}

/// Derivative of HeatQNetCeed() with respect to temperature.
CEED_QFUNCTION_HELPER CeedScalar HeatDQNetDTemperatureCeed(const HeatContext context, const CeedScalar forcing[NUM_HEAT_FORCINGS],
                                                           CeedScalar temp_c) {
  const CeedScalar temp_k = temp_c + context->celsius_to_kelvin;
  const CeedScalar r_inv  = 0.2 + 0.1 * forcing[HEAT_FORCING_WIND_SPEED];

  const CeedScalar d_q_lw = -4.0 * context->water_emissivity * context->stefan_boltzmann * temp_k * temp_k * temp_k;
  const CeedScalar d_q_sh = -context->density_of_air * context->specific_heat_of_air * r_inv;
  const CeedScalar d_q_e =
      -context->density_of_air * context->latent_heat_vaporization * HeatDSaturationSpecificHumidityDTemperature(context, temp_c) * r_inv;

  return d_q_lw + d_q_sh + d_q_e;
}

/// Q-function evaluating the residual of the implicit atmospheric heat source
/// step. This is the CEED analogue of HeatIFunction() in heat_petsc.c; it is
/// purely pointwise, with no spatial coupling.
///
/// Input fields:
///   in[0]: q[num_owned_cells][num_comp]                — state (active)
///   in[1]: q_dot[num_owned_cells][num_comp]            — state time derivative (passive)
///   in[2]: forcing[num_owned_cells][NUM_HEAT_FORCINGS] — atmospheric forcing (passive)
///
/// Output fields:
///   out[0]: residual[num_owned_cells][num_comp]        — implicit residual (active)
CEED_QFUNCTION(HeatIFunctionQF)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  const CeedScalar(*q)[CEED_Q_VLA]       = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_dot)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  const CeedScalar(*forcing)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];

  CeedScalar(*residual)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const HeatContext context   = (HeatContext)ctx;
  const CeedInt     num_comp  = context->num_comp;
  const CeedInt     heat_comp = context->heat_comp;
  const CeedScalar  rho_cp    = context->density_of_water * context->specific_heat_of_water;

  for (CeedInt i = 0; i < Q; i++) {
    // every component (and every dry cell) carries the trivial residual Udot
    for (CeedInt c = 0; c < num_comp; ++c) residual[c][i] = q_dot[c][i];

    const CeedScalar h = q[0][i];
    if (h >= context->tiny_h) {
      CeedScalar cell_forcing[NUM_HEAT_FORCINGS];
      for (CeedInt c = 0; c < NUM_HEAT_FORCINGS; ++c) cell_forcing[c] = forcing[c][i];

      CeedScalar q_net;
      if (context->use_direct_source) {
        q_net = cell_forcing[HEAT_FORCING_DIRECT_SOURCE];
      } else {
        q_net = HeatQNetCeed(context, cell_forcing, q[heat_comp][i] / h);
      }
      residual[heat_comp][i] = q_dot[heat_comp][i] - q_net / rho_cp;
    }
  }
  return 0;
}

/// Q-function evaluating the diagonal of the IJacobian for the implicit
/// atmospheric heat source step. The residual is pointwise, so the Jacobian is
/// exactly diagonal and can be written straight into a PETSc Mat with
/// MatDiagonalSet(). This is the CEED analogue of HeatIJacobian() in heat_petsc.c.
///
/// Input fields:
///   in[0]: q[num_owned_cells][num_comp]                — state (active)
///   in[1]: forcing[num_owned_cells][NUM_HEAT_FORCINGS] — atmospheric forcing (passive)
///
/// Output fields:
///   out[0]: diagonal[num_owned_cells][num_comp]        — Jacobian diagonal (active)
CEED_QFUNCTION(HeatIJacobianDiagonalQF)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  const CeedScalar(*q)[CEED_Q_VLA]       = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*forcing)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];

  CeedScalar(*diagonal)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const HeatContext context   = (HeatContext)ctx;
  const CeedInt     num_comp  = context->num_comp;
  const CeedInt     heat_comp = context->heat_comp;
  const CeedScalar  shift     = context->shift;
  const CeedScalar  rho_cp    = context->density_of_water * context->specific_heat_of_water;

  for (CeedInt i = 0; i < Q; i++) {
    // d(Udot)/dU is just the shift for every component (and every dry cell)
    for (CeedInt c = 0; c < num_comp; ++c) diagonal[c][i] = shift;

    const CeedScalar h = q[0][i];
    if (h >= context->tiny_h && !context->use_direct_source) {
      CeedScalar cell_forcing[NUM_HEAT_FORCINGS];
      for (CeedInt c = 0; c < NUM_HEAT_FORCINGS; ++c) cell_forcing[c] = forcing[c][i];

      const CeedScalar dQ_dT = HeatDQNetDTemperatureCeed(context, cell_forcing, q[heat_comp][i] / h);
      diagonal[heat_comp][i] = shift - dQ_dT / (rho_cp * h);
    }
  }
  return 0;
}

#pragma GCC diagnostic   pop
#pragma GCC diagnostic   pop
#pragma clang diagnostic pop
#pragma clang diagnostic pop

#endif
