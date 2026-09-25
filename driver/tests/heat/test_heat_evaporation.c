#include <private/rdycoreimpl.h>
#include <private/rdyheatimpl.h>
#include <rdycore.h>

static const char* help_str =
    "Verifies the evaporative mass and momentum sinks of the atmospheric heat source,\n"
    "and the cap that keeps evaporation from drawing a cell below tiny_h.\n";

// Both checks below drive RDyHeatAdvance() directly instead of RDyAdvance(), so the
// transport solve never runs and every change in the state is attributable to the
// surface exchange. That isolation is what lets them assert exact identities rather
// than tolerances chosen to accommodate the flow.
static PetscErrorCode SetUniformState(RDy rdy, PetscReal h, PetscReal u, PetscReal v, PetscReal temperature) {
  PetscFunctionBegin;

  PetscInt n_dof, n_local;
  PetscCall(VecGetBlockSize(rdy->u_global, &n_dof));
  PetscCall(VecGetLocalSize(rdy->u_global, &n_local));

  PetscScalar* x;
  PetscCall(VecGetArray(rdy->u_global, &x));
  for (PetscInt c = 0; c < n_local / n_dof; ++c) {
    PetscInt cell = n_dof * c;
    for (PetscInt comp = 0; comp < n_dof; ++comp) x[cell + comp] = 0.0;
    x[cell + 0]                            = h;
    x[cell + 1]                            = h * u;
    x[cell + 2]                            = h * v;
    x[cell + rdy->heat_context->heat_comp] = h * temperature;
  }
  PetscCall(VecRestoreArray(rdy->u_global, &x));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Reduces the per-cell maxima the assertions below need. Every cell carries the same
// state, so a single set of extrema characterizes the whole domain.
typedef struct {
  PetscReal min_h;          // smallest depth anywhere, for the tiny_h floor
  PetscReal max_h;          // largest depth anywhere
  PetscReal max_vel_error;  // largest |u - u0| or |v - v0|
} StateSummary;

static PetscErrorCode SummarizeState(RDy rdy, PetscReal u0, PetscReal v0, StateSummary* summary) {
  PetscFunctionBegin;

  PetscInt n_dof, n_local;
  PetscCall(VecGetBlockSize(rdy->u_global, &n_dof));
  PetscCall(VecGetLocalSize(rdy->u_global, &n_local));

  PetscReal local_min_h = PETSC_MAX_REAL, local_max_h = PETSC_MIN_REAL, local_vel_error = 0.0;

  const PetscScalar* x;
  PetscCall(VecGetArrayRead(rdy->u_global, &x));
  for (PetscInt c = 0; c < n_local / n_dof; ++c) {
    PetscInt  cell = n_dof * c;
    PetscReal h    = PetscRealPart(x[cell + 0]);
    local_min_h    = PetscMin(local_min_h, h);
    local_max_h    = PetscMax(local_max_h, h);

    PetscReal u     = PetscRealPart(x[cell + 1]) / h;
    PetscReal v     = PetscRealPart(x[cell + 2]) / h;
    local_vel_error = PetscMax(local_vel_error, PetscAbsReal(u - u0));
    local_vel_error = PetscMax(local_vel_error, PetscAbsReal(v - v0));
  }
  PetscCall(VecRestoreArrayRead(rdy->u_global, &x));

  PetscCallMPI(MPI_Allreduce(&local_min_h, &summary->min_h, 1, MPIU_REAL, MPI_MIN, rdy->comm));
  PetscCallMPI(MPI_Allreduce(&local_max_h, &summary->max_h, 1, MPIU_REAL, MPI_MAX, rdy->comm));
  PetscCallMPI(MPI_Allreduce(&local_vel_error, &summary->max_vel_error, 1, MPIU_REAL, MPI_MAX, rdy->comm));

  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char* argv[]) {
  PetscCall(RDyInit(argc, argv, help_str));
  PetscCheck(argc >= 2, PETSC_COMM_WORLD, PETSC_ERR_USER, "usage: %s <input.yaml>", argv[0]);

  RDy rdy;
  PetscCall(RDyCreate(PETSC_COMM_WORLD, argv[1], &rdy));
  PetscCall(RDySetup(rdy));

  const PetscReal tiny_h = rdy->config.physics.flow.tiny_h;
  const PetscReal dt     = 1.0;
  const PetscReal u0 = 0.5, v0 = 0.2, temperature = 30.0;

  PetscCheck(!rdy->heat_context->use_direct_source, rdy->comm, PETSC_ERR_USER,
             "This test requires the atmospheric heat source; the input file prescribes a heat_flux instead.");

  // ---------------------------------------------------------------------------
  // Case 1: deep water, so the evaporative demand is nowhere near the available
  // water and the cap stays inactive.
  //
  // The assertion is an exact identity of the discretization rather than a
  // regression value. Backward Euler on
  //
  //   h1 = h0 + dt*m,   (hu)1 = (hu)0 + dt*u1*m,   u1 = (hu)1/h1
  //
  // gives (hu)1*(1 - dt*m/h1) = (hu)0, and since dt*m = h1 - h0 the bracket is
  // h0/h1, leaving u1 = u0 exactly -- for any m, and so independently of the
  // parameterization that produced it. Evaporation must not accelerate the flow.
  // Dropping the momentum rows, or scaling them by anything other than the local
  // velocity, breaks this identity immediately.
  // ---------------------------------------------------------------------------
  const PetscReal h_deep = 0.5;
  PetscCall(SetUniformState(rdy, h_deep, u0, v0, temperature));
  PetscCall(RDyHeatAdvance(rdy, 0.0, dt));

  StateSummary deep;
  PetscCall(SummarizeState(rdy, u0, v0, &deep));

  PetscCheck(deep.max_vel_error < 1.0e-12, rdy->comm, PETSC_ERR_PLIB,
             "Evaporation changed the flow velocity by %g; the momentum sink must remove mass at the local velocity, leaving u and v unchanged.",
             (double)deep.max_vel_error);
  PetscCheck(deep.max_h < h_deep, rdy->comm, PETSC_ERR_PLIB,
             "Evaporation into dry air did not reduce the water depth (h = %g, was %g); the mass sink is missing.", (double)deep.max_h,
             (double)h_deep);
  PetscCheck(deep.min_h > tiny_h, rdy->comm, PETSC_ERR_PLIB, "Deep water was drawn down to %g, at or below tiny_h = %g, in a single step.",
             (double)deep.min_h, (double)tiny_h);

  // ---------------------------------------------------------------------------
  // Case 2: a film holding only 1e-6 m of water above tiny_h, against an
  // evaporative demand three orders of magnitude larger. The cap must bind.
  //
  // Once capped, q_e = -(h1 - tiny_h)*rho_w*L_v/dt by construction, so the depth
  // update closes analytically:
  //
  //   h1 = h0 + dt*q_e/(rho_w*L_v) = h0 - (h1 - tiny_h)  =>  h1 = (h0 + tiny_h)/2
  //
  // The excess over tiny_h halves each capped step and never reaches zero, which is
  // the property that keeps a drying cell from going negative. This pins the mass
  // equation and the cap together, with no dependence on the flux parameterization.
  // ---------------------------------------------------------------------------
  const PetscReal h_film     = tiny_h + 1.0e-6;
  const PetscReal h_expected = 0.5 * (h_film + tiny_h);
  PetscCall(SetUniformState(rdy, h_film, u0, v0, temperature));
  PetscCall(RDyHeatAdvance(rdy, dt, 2.0 * dt));

  StateSummary film;
  PetscCall(SummarizeState(rdy, u0, v0, &film));

  PetscCheck(film.min_h > tiny_h, rdy->comm, PETSC_ERR_PLIB,
             "The evaporation cap failed: depth reached %g, at or below tiny_h = %g. A capped step must leave the cell strictly wet.",
             (double)film.min_h, (double)tiny_h);
  PetscReal cap_error = PetscMax(PetscAbsReal(film.min_h - h_expected), PetscAbsReal(film.max_h - h_expected));
  PetscCheck(cap_error < 1.0e-14 * tiny_h, rdy->comm, PETSC_ERR_PLIB,
             "A capped step left h = [%g, %g] instead of the analytic (h0 + tiny_h)/2 = %g (error %g).", (double)film.min_h, (double)film.max_h,
             (double)h_expected, (double)cap_error);
  PetscCheck(film.max_vel_error < 1.0e-12, rdy->comm, PETSC_ERR_PLIB,
             "A capped evaporation step changed the flow velocity by %g; the momentum sink must use the same capped flux as the mass sink.",
             (double)film.max_vel_error);

  PetscCall(RDyDestroy(&rdy));
  PetscCall(RDyFinalize());
  return 0;
}
