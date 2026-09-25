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

// bulk transfer velocity for the turbulent fluxes [m/s]
static PetscReal HeatTransferVelocity(RDyHeat heat, PetscInt owned_cell) { return 0.2 + 0.1 * heat->forcing.wind_speed[owned_cell]; }

// The net surface heat flux is carried as two pieces rather than one. The split is
// not cosmetic: the latent flux Q_e is the only component that moves mass, it is
// the only one subject to the evaporation cap (see MinLatentHeatFlux()), and in the
// capped branch its temperature derivative drops out of the Jacobian entirely.

// Q_sw + Q_lw + Q_sh: the radiative and sensible components [W/m^2].
static PetscReal HeatQNonLatent(RDyHeat heat, PetscInt owned_cell, PetscReal temp_c) {
  RDyHeatForcing* forcing = &heat->forcing;
  PetscReal       temp_k  = temp_c + CELSIUS_TO_KELVIN;
  PetscReal       r_inv   = HeatTransferVelocity(heat, owned_cell);

  PetscReal q_sw = (1.0 - WATER_ALBEDO) * forcing->downwelling_shortwave[owned_cell];
  PetscReal q_lw = forcing->downwelling_longwave[owned_cell] - WATER_EMISSIVITY * STEFAN_BOLTZMANN * PetscPowReal(temp_k, 4.0);
  PetscReal q_sh = DENSITY_OF_AIR * SPECIFIC_HEAT_OF_AIR * (forcing->air_temperature[owned_cell] - temp_c) * r_inv;

  return q_sw + q_lw + q_sh;
}

static PetscReal DHeatQNonLatentDTemperature(RDyHeat heat, PetscInt owned_cell, PetscReal temp_c) {
  PetscReal temp_k = temp_c + CELSIUS_TO_KELVIN;
  PetscReal r_inv  = HeatTransferVelocity(heat, owned_cell);

  PetscReal d_q_lw = -4.0 * WATER_EMISSIVITY * STEFAN_BOLTZMANN * Cube(temp_k);
  PetscReal d_q_sh = -DENSITY_OF_AIR * SPECIFIC_HEAT_OF_AIR * r_inv;

  return d_q_lw + d_q_sh;
}

// Q_e, the latent heat flux [W/m^2]. Negative when the cell is evaporating, which
// is also when it removes water; positive under condensation, which adds water.
static PetscReal HeatQLatent(RDyHeat heat, PetscInt owned_cell, PetscReal temp_c) {
  RDyHeatForcing* forcing = &heat->forcing;
  PetscReal       r_inv   = HeatTransferVelocity(heat, owned_cell);

  return DENSITY_OF_AIR * LATENT_HEAT_VAPORIZATION * (forcing->specific_humidity[owned_cell] - SaturationSpecificHumidity(temp_c)) * r_inv;
}

static PetscReal DHeatQLatentDTemperature(RDyHeat heat, PetscInt owned_cell, PetscReal temp_c) {
  PetscReal r_inv = HeatTransferVelocity(heat, owned_cell);

  return -DENSITY_OF_AIR * LATENT_HEAT_VAPORIZATION * DSaturationSpecificHumidityDTemperature(temp_c) * r_inv;
}

// Lower bound on Q_e: a cell can only evaporate the water it has. Over one implicit
// step of length dt the depth changes by q_e*dt/(rho_w*L_v), so holding the cell at
// or above tiny_h requires
//
//   q_e >= q_e_min = -(h - tiny_h)*rho_w*L_v/dt
//
// which is what this returns (a non-positive number, zero at h == tiny_h). Without
// it an aggressive evaporative demand over a long coupling interval drives h
// negative, and the negative depth then propagates into T = hT/h and into the next
// flow solve.
//
// Condensation (q_e > 0) adds water and is never limited. The bound is applied to
// the energy budget as well as the mass budget -- latent heat may only be removed
// for water that actually leaves the cell -- so a capped cell also cools more
// slowly than the uncapped parameterization would have it.
//
// NOTE: dt here is the step the heat TS is attempting, which for the backward Euler
// NOTE: default is exactly the interval over which the depth update applies.
static PetscReal MinLatentHeatFlux(PetscReal h, PetscReal tiny_h, PetscReal dt) {
  return -(h - tiny_h) * DENSITY_OF_WATER * LATENT_HEAT_VAPORIZATION / dt;
}

// The implicit atmospheric source step comes in two flavors that differ only in
// how the net surface heat flux Q_net is obtained, so each gets its own
// IFunction/IJacobian pair rather than branching per DOF inside the loop:
//
//   * "prescribed source" - Q_net is read straight from RDyHeatForcing::direct_source
//     (a YAML heat_flux override, or the manufactured source used by the MMS driver)
//   * "atmospheric source" - Q_net is computed from the local atmospheric forcing
//     via HeatQNonLatent() + HeatQLatent()
//
// SetHeatTSCallbacks() selects the pair; see its comment for when that happens.
//
// The two flavors differ in one more way, which is why they cannot share a loop.
// The atmospheric flavor's latent component Q_e moves water as well as energy, so
// that flavor also writes the h, hu, and hv rows of the residual; the prescribed
// flavor receives a single net flux with no latent component to extract, so it
// leaves the flow untouched. See HeatIFunctionAtmosphericSource().

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

// Residual of the implicit atmospheric source step.
//
// Evaporation takes water with it, so this step is not confined to the heat DOF.
// Writing the capped latent flux as Q_e and the evaporative depth rate as
//
//   hdot = Q_e/(rho_w*L_v)          [m/s, negative while evaporating]
//
// the cell-local system solved here is
//
//   dh/dt    = hdot
//   d(hu)/dt = u*hdot,   d(hv)/dt = v*hdot
//   d(hT)/dt = Q_net/(rho_w*c_w)
//
// The momentum rows carry the mass away at the local flow velocity, which leaves
// u = hu/h unchanged; dropping them would instead accelerate the flow as the depth
// fell. The same convention is used for condensation, where vapor is taken to join
// the flow at its own velocity rather than arriving at rest.
//
// The hT row is deliberately *not* given a -T*hdot term: hT is conserved under the
// mass loss, so the remaining water warms as it concentrates. Sediment and salinity
// rows are likewise left alone, so those tracers concentrate too.
//
// Dry cells (h < tiny_h) exchange nothing at all, and every remaining component
// carries the trivial residual Udot so the solve leaves it unchanged.
static PetscErrorCode HeatIFunctionAtmosphericSource(TS ts, PetscReal t, Vec U, Vec Udot, Vec F, void* ctx) {
  (void)t;
  PetscFunctionBegin;
  RDy     rdy  = ctx;
  RDyHeat heat = rdy->heat_context;

  // the evaporation cap is a limit on how much water one step may remove, so it
  // needs the length of the step being attempted
  PetscReal dt;
  PetscCall(TSGetTimeStep(ts, &dt));

  PetscInt n_dof, n_local;
  PetscCall(VecGetBlockSize(U, &n_dof));
  PetscCall(VecGetLocalSize(U, &n_local));

  const PetscScalar *u, *udot;
  PetscScalar*       f;
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArrayRead(Udot, &udot));
  PetscCall(VecGetArray(F, &f));

  const PetscInt  heat_comp = heat->heat_comp;
  const PetscReal tiny_h    = heat->config->physics.flow.tiny_h;
  const PetscReal rho_lv    = DENSITY_OF_WATER * LATENT_HEAT_VAPORIZATION;
  const PetscReal rho_cp    = DENSITY_OF_WATER * SPECIFIC_HEAT_OF_WATER;

  for (PetscInt owned_cell = 0; owned_cell < n_local / n_dof; ++owned_cell) {
    const PetscInt cell = n_dof * owned_cell;

    // every component (and every dry cell) carries the trivial residual Udot
    for (PetscInt c = 0; c < n_dof; ++c) f[cell + c] = udot[cell + c];

    PetscReal h = u[cell];
    if (h < tiny_h) continue;

    PetscReal T    = u[cell + heat_comp] / h;
    PetscReal q_e  = PetscMax(HeatQLatent(heat, owned_cell, T), MinLatentHeatFlux(h, tiny_h, dt));
    PetscReal hdot = q_e / rho_lv;

    f[cell + 0]         = udot[cell + 0] - hdot;
    f[cell + 1]         = udot[cell + 1] - (u[cell + 1] / h) * hdot;
    f[cell + 2]         = udot[cell + 2] - (u[cell + 2] / h) * hdot;
    f[cell + heat_comp] = udot[cell + heat_comp] - (HeatQNonLatent(heat, owned_cell, T) + q_e) / rho_cp;
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

// Jacobian of HeatIFunctionAtmosphericSource(). The residual is still pointwise, so
// the Jacobian is block diagonal with one dense n_dof x n_dof block per cell -- but
// it is no longer a scalar diagonal, because evaporation couples the h, hu, hv, and
// hT rows through T = hT/h and through the capped Q_e.
//
// With rho_lv = rho_w*L_v, rho_cp = rho_w*c_w, u = hu/h, and v = hv/h, differentiating
// the four active rows gives
//
//   dF_h/dh     = shift - (dQe/dh)/rho_lv          dF_h/d(hT)  = -(dQe/dhT)/rho_lv
//   dF_hu/dh    = u*(Qe/h - dQe/dh)/rho_lv         dF_hu/d(hu) = shift - Qe/(h*rho_lv)
//   dF_hu/d(hT) = -u*(dQe/dhT)/rho_lv              (and likewise for hv, with v)
//   dF_hT/dh    = -(dQnet/dh)/rho_cp               dF_hT/d(hT) = shift - (dQnet/dhT)/rho_cp
//
// The two branches of the cap differ only in dQe/dh and dQe/dhT, which is the reason
// the flux was split into latent and non-latent pieces:
//
//   uncapped: Qe depends on the state only through T, so with dT/dh = -T/h and
//             dT/d(hT) = 1/h, dQe/dh = -(dQe/dT)*T/h and dQe/dhT = (dQe/dT)/h
//   capped:   Qe = -(h - tiny_h)*rho_lv/dt depends on h alone, so dQe/dh = -rho_lv/dt
//             and dQe/dhT = 0 -- the temperature derivative drops out entirely
//
// The cap makes the residual non-smooth where it activates, so Newton can need an
// extra iteration or two on a cell that crosses it; it stays convergent because
// every uncapped derivative dQnet/dT is negative and the capped branch is linear.
static PetscErrorCode HeatIJacobianAtmosphericSource(TS ts, PetscReal t, Vec U, Vec Udot, PetscReal shift, Mat J, Mat P, void* ctx) {
  (void)t;
  (void)Udot;
  PetscFunctionBegin;
  RDy     rdy  = ctx;
  RDyHeat heat = rdy->heat_context;

  PetscReal dt;
  PetscCall(TSGetTimeStep(ts, &dt));

  PetscInt n_dof, start, end;
  PetscCall(VecGetBlockSize(U, &n_dof));
  PetscCall(VecGetOwnershipRange(U, &start, &end));

  const PetscScalar* u;
  PetscCall(VecGetArrayRead(U, &u));

  const PetscInt  heat_comp = heat->heat_comp;
  const PetscReal tiny_h    = heat->config->physics.flow.tiny_h;
  const PetscReal rho_lv    = DENSITY_OF_WATER * LATENT_HEAT_VAPORIZATION;
  const PetscReal rho_cp    = DENSITY_OF_WATER * SPECIFIC_HEAT_OF_WATER;

  PetscCall(MatZeroEntries(P));

  PetscCheck(n_dof <= MAX_NUM_FIELD_COMPONENTS, rdy->comm, PETSC_ERR_SUP,
             "The heat Jacobian block is sized for at most %d components, but the solution has %" PetscInt_FMT, MAX_NUM_FIELD_COMPONENTS, n_dof);
  PetscInt  rows[MAX_NUM_FIELD_COMPONENTS], cols[MAX_NUM_FIELD_COMPONENTS];
  PetscReal block[MAX_NUM_FIELD_COMPONENTS * MAX_NUM_FIELD_COMPONENTS];

  for (PetscInt owned_cell = 0; owned_cell < (end - start) / n_dof; ++owned_cell) {
    const PetscInt cell = n_dof * owned_cell;
    for (PetscInt c = 0; c < n_dof; ++c) rows[c] = cols[c] = start + cell + c;

    // d(Udot)/dU is just the shift for every component (and every dry cell)
    for (PetscInt c = 0; c < n_dof * n_dof; ++c) block[c] = 0.0;
    for (PetscInt c = 0; c < n_dof; ++c) block[c * n_dof + c] = shift;

    PetscReal h = u[cell];
    if (h >= tiny_h) {
      PetscReal T       = u[cell + heat_comp] / h;
      PetscReal q_e_raw = HeatQLatent(heat, owned_cell, T);
      PetscReal q_e_min = MinLatentHeatFlux(h, tiny_h, dt);
      PetscBool capped  = (PetscBool)(q_e_raw < q_e_min);
      PetscReal q_e     = capped ? q_e_min : q_e_raw;

      PetscReal dqe_dh, dqe_dhT;
      if (capped) {
        dqe_dh  = -rho_lv / dt;
        dqe_dhT = 0.0;
      } else {
        PetscReal dqe_dT = DHeatQLatentDTemperature(heat, owned_cell, T);
        dqe_dh           = -dqe_dT * T / h;
        dqe_dhT          = dqe_dT / h;
      }

      PetscReal dnl_dT    = DHeatQNonLatentDTemperature(heat, owned_cell, T);
      PetscReal dqnet_dh  = -dnl_dT * T / h + dqe_dh;
      PetscReal dqnet_dhT = dnl_dT / h + dqe_dhT;

      PetscReal vel[2] = {u[cell + 1] / h, u[cell + 2] / h};

      block[0 * n_dof + 0]         = shift - dqe_dh / rho_lv;
      block[0 * n_dof + heat_comp] = -dqe_dhT / rho_lv;

      for (PetscInt m = 1; m <= 2; ++m) {
        block[m * n_dof + 0]         = vel[m - 1] * (q_e / h - dqe_dh) / rho_lv;
        block[m * n_dof + m]         = shift - q_e / (h * rho_lv);
        block[m * n_dof + heat_comp] = -vel[m - 1] * dqe_dhT / rho_lv;
      }

      block[heat_comp * n_dof + 0]         = -dqnet_dh / rho_cp;
      block[heat_comp * n_dof + heat_comp] = shift - dqnet_dhT / rho_cp;
    }

    PetscCall(MatSetValues(P, n_dof, rows, n_dof, cols, block, INSERT_VALUES));
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

// The implicit heat Jacobian is block diagonal in the global DOF numbering: each cell
// couples only to itself, through T = hT/h and the capped evaporative flux (see
// HeatIJacobianAtmosphericSource(), HeatIJacobianPrescribedSource(), and
// HeatIJacobianCeedAtmosphericSource()). heat_jac is therefore preallocated with a
// COO pattern of one dense n_dof x n_dof block per owned cell, rather than the wider
// FV-stencil sparsity DMCreateMatrix() would otherwise give it -- heat_jac is
// dedicated to the heat TS, so narrowing its sparsity doesn't affect anything else
// built from rdy->dm.
//
// A COO preallocation also lets the CEED backend write Jacobian values with
// MatSetValuesCOO(), which (unlike MatSetValues()/MatShift()/MatDiagonalSet()) has a
// device-native implementation for GPU Mat types, so the block values it computes on
// the GPU never have to round-trip through the host.
//
// NOTE: the entry ordering below -- cell-major, then row, then column within the
// NOTE: block -- is a contract with the CEED Jacobian Q-function, whose output field
// NOTE: carries n_dof*n_dof components per cell laid out in exactly this order. The
// NOTE: two must be changed together.
static PetscErrorCode PreallocateHeatJacobianBlocks(Mat heat_jac, PetscInt n_dof) {
  PetscFunctionBegin;

  PetscInt start, end;
  PetscCall(MatGetOwnershipRange(heat_jac, &start, &end));
  PetscCheck((end - start) % n_dof == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB,
             "Heat Jacobian row range [%" PetscInt_FMT ", %" PetscInt_FMT ") is not a whole number of %" PetscInt_FMT "-component cells", start, end,
             n_dof);

  PetscCount n_coo = (PetscCount)(end - start) * n_dof;

  PetscInt *rows, *cols;
  PetscCall(PetscMalloc2(n_coo, &rows, n_coo, &cols));
  PetscCount k = 0;
  for (PetscInt cell = start; cell < end; cell += n_dof) {
    for (PetscInt r = 0; r < n_dof; ++r) {
      for (PetscInt c = 0; c < n_dof; ++c, ++k) {
        rows[k] = cell + r;
        cols[k] = cell + c;
      }
    }
  }
  PetscCall(MatSetPreallocationCOO(heat_jac, n_coo, rows, cols));
  PetscCall(PetscFree2(rows, cols));

  // MatSetPreallocationCOO() alone leaves the matrix unassembled (mat->assembled ==
  // PETSC_FALSE) until values are written with MatSetValuesCOO(), which the plain
  // PETSc Jacobian callbacks (HeatIJacobianPrescribedSource(), MatShift() in
  // particular) don't do -- they require an already-assembled matrix. DMCreateMatrix()
  // used to hand back an assembled (all-zero) matrix; restore that guarantee here so
  // heat_jac is usable by either the COO or the MatShift()/MatSetValues() write path,
  // whichever runs first.
  PetscCall(MatAssemblyBegin(heat_jac, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(heat_jac, MAT_FINAL_ASSEMBLY));

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
  PetscCall(PreallocateHeatJacobianBlocks(rdy->heat_jac, 3 + rdy->num_tracers));

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
