#include <ceed/ceed.h>
#include <petscdmceed.h>
#include <private/rdycoreimpl.h>
#include <private/rdyheatimpl.h>

// CEED uses C99 VLA features for shaping multidimensional arrays, which don't
// have the same drawbacks as VLA allocations.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvla"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla"

#include "heat_sources_ceed.h"

// physical constants, mirroring those used by the PETSc backend in heat_petsc.c
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

static inline CeedMemType MemTypeP2C(PetscMemType mem_type) { return PetscMemTypeDevice(mem_type) ? CEED_MEM_DEVICE : CEED_MEM_HOST; }

// frees a data context allocated using PETSc, returning a libCEED error code
static int FreeContextPetsc(void *data) {
  if (PetscFree(data)) return CeedError(NULL, CEED_ERROR_ACCESS, "PetscFree failed");
  return CEED_ERROR_SUCCESS;
}

// Creates the Q-function context shared by the heat IFunction and IJacobian
// Q-functions. The "time shift" and "use direct source" fields are registered so
// that they can be updated between solves without rebuilding the operators.
static PetscErrorCode CreateHeatQFunctionContext(Ceed ceed, RDy rdy, CeedQFunctionContext *qf_context) {
  PetscFunctionBeginUser;

  HeatContext heat_ctx;
  PetscCall(PetscCalloc1(1, &heat_ctx));

  heat_ctx->tiny_h                   = rdy->config.physics.flow.tiny_h;
  heat_ctx->shift                    = 0.0;
  heat_ctx->water_albedo             = WATER_ALBEDO;
  heat_ctx->water_emissivity         = WATER_EMISSIVITY;
  heat_ctx->stefan_boltzmann         = STEFAN_BOLTZMANN;
  heat_ctx->density_of_air           = DENSITY_OF_AIR;
  heat_ctx->specific_heat_of_air     = SPECIFIC_HEAT_OF_AIR;
  heat_ctx->latent_heat_vaporization = LATENT_HEAT_VAPORIZATION;
  heat_ctx->density_of_water         = DENSITY_OF_WATER;
  heat_ctx->specific_heat_of_water   = SPECIFIC_HEAT_OF_WATER;
  heat_ctx->standard_air_pressure    = STANDARD_AIR_PRESSURE;
  heat_ctx->water_vapor_epsilon      = WATER_VAPOR_EPSILON;
  heat_ctx->celsius_to_kelvin        = CELSIUS_TO_KELVIN;
  heat_ctx->heat_comp                = (CeedInt)rdy->heat_context->heat_comp;
  heat_ctx->num_comp                 = (CeedInt)(3 + rdy->num_tracers);  // NOTE: SWE assumed!
  heat_ctx->use_direct_source        = 0;

  PetscCallCEED(CeedQFunctionContextCreate(ceed, qf_context));
  PetscCallCEED(CeedQFunctionContextSetData(*qf_context, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*heat_ctx), heat_ctx));
  PetscCallCEED(CeedQFunctionContextSetDataDestroy(*qf_context, CEED_MEM_HOST, FreeContextPetsc));

  PetscCallCEED(CeedQFunctionContextRegisterDouble(*qf_context, "time shift", offsetof(struct HeatContext_, shift), 1,
                                                   "Shift dU/dUdot supplied by the TS to its IJacobian callback"));
  PetscCallCEED(CeedQFunctionContextRegisterInt32(*qf_context, "use direct source", offsetof(struct HeatContext_, use_direct_source), 1,
                                                  "Nonzero if a prescribed net heat flux replaces the atmospheric parameterization"));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Creates the CEED operators backing the implicit atmospheric heat source step,
/// along with the CeedVectors used to wrap PETSc data during their application.
///
/// Both operators are pointwise over owned cells, so they use strided element
/// restrictions and require no basis evaluation or neighbor coupling:
///
///  * the IFunction operator maps the active state `q` to the active residual,
///    reading `q_dot` and the per-cell atmospheric `forcing` passively
///  * the IJacobian operator maps the active state `q` to the active Jacobian
///    diagonal, reading `forcing` passively
///
/// @param [inout] rdy the RDycore simulation context (must have heat enabled)
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode CreateCeedHeatOperators(RDy rdy) {
  PetscFunctionBegin;

  Ceed    ceed            = CeedContext();
  RDyHeat heat            = rdy->heat_context;
  CeedInt num_comp        = (CeedInt)(3 + rdy->num_tracers);  // NOTE: SWE assumed!
  CeedInt num_owned_cells = (CeedInt)rdy->mesh.num_owned_cells;

  // the Q-function context is shared by both Q-functions so that a single
  // update of "time shift"/"use direct source" is seen by both operators
  CeedQFunctionContext qf_context;
  PetscCall(CreateHeatQFunctionContext(ceed, rdy, &qf_context));

  // NOTE: the order in which inputs and outputs are added below determines their
  // NOTE: indexing within the Q-function implementations
  CeedQFunction qf_ifunction, qf_ijacobian;
  PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, HeatIFunctionQF, HeatIFunctionQF_loc, &qf_ifunction));
  PetscCallCEED(CeedQFunctionAddInput(qf_ifunction, "q", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf_ifunction, "q_dot", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf_ifunction, "forcing", NUM_HEAT_FORCINGS, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf_ifunction, "residual", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionSetContext(qf_ifunction, qf_context));

  PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, HeatIJacobianDiagonalQF, HeatIJacobianDiagonalQF_loc, &qf_ijacobian));
  PetscCallCEED(CeedQFunctionAddInput(qf_ijacobian, "q", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf_ijacobian, "forcing", NUM_HEAT_FORCINGS, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf_ijacobian, "diagonal", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionSetContext(qf_ijacobian, qf_context));

  PetscCallCEED(CeedQFunctionContextDestroy(&qf_context));

  // the heat TS operates on global vectors, whose local portions hold exactly
  // the owned cells in block-interleaved order, so strided restrictions suffice
  CeedElemRestriction restrict_state, restrict_forcing;
  CeedInt             strides_state[]   = {num_comp, 1, num_comp};
  CeedInt             strides_forcing[] = {NUM_HEAT_FORCINGS, 1, NUM_HEAT_FORCINGS};
  PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_owned_cells, 1, num_comp, num_owned_cells * num_comp, strides_state, &restrict_state));
  PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_owned_cells, 1, NUM_HEAT_FORCINGS, num_owned_cells * NUM_HEAT_FORCINGS, strides_forcing,
                                                 &restrict_forcing));

  // CeedVectors used to wrap PETSc arrays during operator application; their
  // arrays are attached and detached on every callback
  PetscCallCEED(CeedElemRestrictionCreateVector(restrict_state, &heat->ceed.u, NULL));
  PetscCallCEED(CeedElemRestrictionCreateVector(restrict_state, &heat->ceed.u_dot, NULL));
  PetscCallCEED(CeedElemRestrictionCreateVector(restrict_state, &heat->ceed.residual, NULL));
  PetscCallCEED(CeedElemRestrictionCreateVector(restrict_state, &heat->ceed.diagonal, NULL));

  // the forcing vector is owned by us and refreshed by UpdateCeedHeatForcing()
  PetscCallCEED(CeedElemRestrictionCreateVector(restrict_forcing, &heat->ceed.forcing, NULL));
  PetscCallCEED(CeedVectorSetValue(heat->ceed.forcing, 0.0));

  PetscCallCEED(CeedOperatorCreate(ceed, qf_ifunction, NULL, NULL, &heat->ceed.ifunction_op));
  PetscCallCEED(CeedOperatorSetField(heat->ceed.ifunction_op, "q", restrict_state, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(heat->ceed.ifunction_op, "q_dot", restrict_state, CEED_BASIS_NONE, heat->ceed.u_dot));
  PetscCallCEED(CeedOperatorSetField(heat->ceed.ifunction_op, "forcing", restrict_forcing, CEED_BASIS_NONE, heat->ceed.forcing));
  PetscCallCEED(CeedOperatorSetField(heat->ceed.ifunction_op, "residual", restrict_state, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

  PetscCallCEED(CeedOperatorCreate(ceed, qf_ijacobian, NULL, NULL, &heat->ceed.ijacobian_op));
  PetscCallCEED(CeedOperatorSetField(heat->ceed.ijacobian_op, "q", restrict_state, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(heat->ceed.ijacobian_op, "forcing", restrict_forcing, CEED_BASIS_NONE, heat->ceed.forcing));
  PetscCallCEED(CeedOperatorSetField(heat->ceed.ijacobian_op, "diagonal", restrict_state, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

  // cache the context field labels used to update the operators between solves
  PetscCallCEED(CeedOperatorGetContextFieldLabel(heat->ceed.ijacobian_op, "time shift", &heat->ceed.shift_label));
  PetscCallCEED(CeedOperatorGetContextFieldLabel(heat->ceed.ifunction_op, "use direct source", &heat->ceed.ifunction_direct_source_label));
  PetscCallCEED(CeedOperatorGetContextFieldLabel(heat->ceed.ijacobian_op, "use direct source", &heat->ceed.ijacobian_direct_source_label));

  // a PETSc Vec to receive the Jacobian diagonal before it is handed to MatDiagonalSet()
  PetscCall(VecDuplicate(rdy->u_global, &heat->ceed.diagonal_vec));

  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_state));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_forcing));
  PetscCallCEED(CeedQFunctionDestroy(&qf_ifunction));
  PetscCallCEED(CeedQFunctionDestroy(&qf_ijacobian));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Destroys the CEED operators and vectors created by CreateCeedHeatOperators().
/// @param [inout] rdy the RDycore simulation context
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode DestroyCeedHeatOperators(RDy rdy) {
  PetscFunctionBegin;

  RDyHeat heat = rdy->heat_context;
  if (!heat) PetscFunctionReturn(PETSC_SUCCESS);

  if (heat->ceed.diagonal_vec) PetscCall(VecDestroy(&heat->ceed.diagonal_vec));
  if (heat->ceed.ifunction_op) PetscCallCEED(CeedOperatorDestroy(&heat->ceed.ifunction_op));
  if (heat->ceed.ijacobian_op) PetscCallCEED(CeedOperatorDestroy(&heat->ceed.ijacobian_op));
  if (heat->ceed.u) PetscCallCEED(CeedVectorDestroy(&heat->ceed.u));
  if (heat->ceed.u_dot) PetscCallCEED(CeedVectorDestroy(&heat->ceed.u_dot));
  if (heat->ceed.residual) PetscCallCEED(CeedVectorDestroy(&heat->ceed.residual));
  if (heat->ceed.diagonal) PetscCallCEED(CeedVectorDestroy(&heat->ceed.diagonal));
  if (heat->ceed.forcing) PetscCallCEED(CeedVectorDestroy(&heat->ceed.forcing));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Copies the host-side atmospheric forcing arrays into the interleaved CeedVector
/// read by the heat Q-functions, and propagates the current use_direct_source flag
/// to both operators. This must be called after any change to RDyHeatForcing and
/// before the heat TS is advanced.
/// @param [inout] rdy the RDycore simulation context
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode UpdateCeedHeatForcing(RDy rdy) {
  PetscFunctionBegin;

  RDyHeat  heat            = rdy->heat_context;
  PetscInt num_owned_cells = rdy->mesh.num_owned_cells;

  CeedScalar(*f)[NUM_HEAT_FORCINGS];
  PetscCallCEED(CeedVectorGetArray(heat->ceed.forcing, CEED_MEM_HOST, (CeedScalar **)&f));
  for (PetscInt c = 0; c < num_owned_cells; ++c) {
    f[c][HEAT_FORCING_DOWNWELLING_SHORTWAVE] = heat->forcing.downwelling_shortwave[c];
    f[c][HEAT_FORCING_DOWNWELLING_LONGWAVE]  = heat->forcing.downwelling_longwave[c];
    f[c][HEAT_FORCING_WIND_SPEED]            = heat->forcing.wind_speed[c];
    f[c][HEAT_FORCING_AIR_TEMPERATURE]       = heat->forcing.air_temperature[c];
    f[c][HEAT_FORCING_SPECIFIC_HUMIDITY]     = heat->forcing.specific_humidity[c];
    f[c][HEAT_FORCING_DIRECT_SOURCE]         = heat->forcing.direct_source[c];
  }
  PetscCallCEED(CeedVectorRestoreArray(heat->ceed.forcing, (CeedScalar **)&f));

  int32_t use_direct_source = heat->use_direct_source ? 1 : 0;
  PetscCallCEED(CeedOperatorSetContextInt32(heat->ceed.ifunction_op, heat->ceed.ifunction_direct_source_label, &use_direct_source));
  PetscCallCEED(CeedOperatorSetContextInt32(heat->ceed.ijacobian_op, heat->ceed.ijacobian_direct_source_label, &use_direct_source));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// TS IFunction callback for the implicit atmospheric heat source step, evaluated
/// with the CEED backend. Equivalent to HeatIFunction() in heat_petsc.c.
PetscErrorCode HeatIFunctionCeed(TS ts, PetscReal t, Vec U, Vec Udot, Vec F, void *ctx) {
  (void)ts;
  (void)t;
  PetscFunctionBegin;

  RDy     rdy  = ctx;
  RDyHeat heat = rdy->heat_context;

  const PetscScalar *u_ptr, *udot_ptr;
  PetscScalar       *f_ptr;
  PetscMemType       u_mem_type, udot_mem_type, f_mem_type;
  PetscCall(VecGetArrayReadAndMemType(U, &u_ptr, &u_mem_type));
  PetscCall(VecGetArrayReadAndMemType(Udot, &udot_ptr, &udot_mem_type));
  PetscCall(VecGetArrayAndMemType(F, &f_ptr, &f_mem_type));

  PetscCallCEED(CeedVectorSetArray(heat->ceed.u, MemTypeP2C(u_mem_type), CEED_USE_POINTER, (CeedScalar *)u_ptr));
  PetscCallCEED(CeedVectorSetArray(heat->ceed.u_dot, MemTypeP2C(udot_mem_type), CEED_USE_POINTER, (CeedScalar *)udot_ptr));
  PetscCallCEED(CeedVectorSetArray(heat->ceed.residual, MemTypeP2C(f_mem_type), CEED_USE_POINTER, f_ptr));

  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCEED(CeedOperatorApply(heat->ceed.ifunction_op, heat->ceed.u, heat->ceed.residual, CEED_REQUEST_IMMEDIATE));
  PetscCall(PetscLogGpuTimeEnd());

  PetscCallCEED(CeedVectorTakeArray(heat->ceed.residual, MemTypeP2C(f_mem_type), &f_ptr));
  PetscCallCEED(CeedVectorTakeArray(heat->ceed.u_dot, MemTypeP2C(udot_mem_type), (CeedScalar **)&udot_ptr));
  PetscCallCEED(CeedVectorTakeArray(heat->ceed.u, MemTypeP2C(u_mem_type), (CeedScalar **)&u_ptr));

  PetscCall(VecRestoreArrayAndMemType(F, &f_ptr));
  PetscCall(VecRestoreArrayReadAndMemType(Udot, &udot_ptr));
  PetscCall(VecRestoreArrayReadAndMemType(U, &u_ptr));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// TS IJacobian callback for the implicit atmospheric heat source step, evaluated
/// with the CEED backend. The residual is pointwise, so the Jacobian is exactly
/// diagonal: the CEED operator produces the diagonal values and MatDiagonalSet()
/// installs them. Equivalent to HeatIJacobian() in heat_petsc.c.
PetscErrorCode HeatIJacobianCeed(TS ts, PetscReal t, Vec U, Vec Udot, PetscReal shift, Mat J, Mat P, void *ctx) {
  (void)ts;
  (void)t;
  (void)Udot;
  PetscFunctionBegin;

  RDy     rdy  = ctx;
  RDyHeat heat = rdy->heat_context;

  double ceed_shift = (double)shift;
  PetscCallCEED(CeedOperatorSetContextDouble(heat->ceed.ijacobian_op, heat->ceed.shift_label, &ceed_shift));

  const PetscScalar *u_ptr;
  PetscScalar       *diag_ptr;
  PetscMemType       u_mem_type, diag_mem_type;
  PetscCall(VecGetArrayReadAndMemType(U, &u_ptr, &u_mem_type));
  PetscCall(VecGetArrayAndMemType(heat->ceed.diagonal_vec, &diag_ptr, &diag_mem_type));

  PetscCallCEED(CeedVectorSetArray(heat->ceed.u, MemTypeP2C(u_mem_type), CEED_USE_POINTER, (CeedScalar *)u_ptr));
  PetscCallCEED(CeedVectorSetArray(heat->ceed.diagonal, MemTypeP2C(diag_mem_type), CEED_USE_POINTER, diag_ptr));

  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCEED(CeedOperatorApply(heat->ceed.ijacobian_op, heat->ceed.u, heat->ceed.diagonal, CEED_REQUEST_IMMEDIATE));
  PetscCall(PetscLogGpuTimeEnd());

  PetscCallCEED(CeedVectorTakeArray(heat->ceed.diagonal, MemTypeP2C(diag_mem_type), &diag_ptr));
  PetscCallCEED(CeedVectorTakeArray(heat->ceed.u, MemTypeP2C(u_mem_type), (CeedScalar **)&u_ptr));

  PetscCall(VecRestoreArrayAndMemType(heat->ceed.diagonal_vec, &diag_ptr));
  PetscCall(VecRestoreArrayReadAndMemType(U, &u_ptr));

  PetscCall(MatZeroEntries(P));
  PetscCall(MatDiagonalSet(P, heat->ceed.diagonal_vec, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  if (J != P) {
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop
