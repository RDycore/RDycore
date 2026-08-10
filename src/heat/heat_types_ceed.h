#ifndef HEAT_TYPES_CEED_H
#define HEAT_TYPES_CEED_H

#include "heat_ceed.h"

// per-cell atmospheric forcing components, laid out as a single
// forcing[num_owned_cells][NUM_HEAT_FORCINGS] passive input field
#define HEAT_FORCING_DOWNWELLING_SHORTWAVE 0
#define HEAT_FORCING_DOWNWELLING_LONGWAVE 1
#define HEAT_FORCING_WIND_SPEED 2
#define HEAT_FORCING_AIR_TEMPERATURE 3
#define HEAT_FORCING_SPECIFIC_HUMIDITY 4
#define HEAT_FORCING_DIRECT_SOURCE 5
#define NUM_HEAT_FORCINGS 6

// Q-function context for the implicit atmospheric heat source step. These
// values mirror the constants used by the PETSc backend in heat_petsc.c.
typedef struct HeatContext_ *HeatContext;
struct HeatContext_ {
  CeedScalar tiny_h;                    // water height below which dry conditions are assumed
  CeedScalar shift;                     // TS shift (dU/dUdot) supplied to the IJacobian callback
  CeedScalar water_albedo;              //
  CeedScalar water_emissivity;          //
  CeedScalar stefan_boltzmann;          // [W/m^2/K^4]
  CeedScalar density_of_air;            // [kg/m^3]
  CeedScalar specific_heat_of_air;      // [J/kg/K]
  CeedScalar latent_heat_vaporization;  // [J/kg]
  CeedScalar density_of_water;          // [kg/m^3]
  CeedScalar specific_heat_of_water;    // [J/kg/K]
  CeedScalar standard_air_pressure;     // [Pa]
  CeedScalar water_vapor_epsilon;       //
  CeedScalar celsius_to_kelvin;         //
  CeedInt    heat_comp;                 // index of the h*T component within the solution state
  CeedInt    num_comp;                  // number of solution components
};

#endif
