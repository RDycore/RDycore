#include <private/rdycoreimpl.h>
#include <private/rdyheatimpl.h>
#include <rdycore.h>

static const char* help_str = "Verifies that the heat source advances over the complete RDyAdvance coupling interval.\n";

int main(int argc, char* argv[]) {
  PetscCall(RDyInit(argc, argv, help_str));
  PetscCheck(argc >= 2, PETSC_COMM_WORLD, PETSC_ERR_USER, "usage: %s <input.yaml>", argv[0]);

  RDy rdy;
  PetscCall(RDyCreate(PETSC_COMM_WORLD, argv[1], &rdy));
  PetscCall(RDySetup(rdy));
  PetscCall(RDyAdvance(rdy));

  PetscInt ndof, n_local;
  PetscCall(VecGetBlockSize(rdy->u_global, &ndof));
  PetscCall(VecGetLocalSize(rdy->u_global, &n_local));

  const PetscScalar* u;
  PetscCall(VecGetArrayRead(rdy->u_global, &u));
  PetscReal local_max_error = 0.0;
  for (PetscInt c = 0; c < n_local / ndof; ++c) {
    PetscReal h           = PetscRealPart(u[ndof * c]);
    PetscReal temperature = PetscRealPart(u[ndof * c + rdy->heat_context->heat_comp]) / h;
    local_max_error       = PetscMax(local_max_error, PetscAbsReal(temperature - 21.0));
  }
  PetscCall(VecRestoreArrayRead(rdy->u_global, &u));

  PetscReal global_max_error;
  PetscCallMPI(MPI_Allreduce(&local_max_error, &global_max_error, 1, MPIU_REAL, MPI_MAX, rdy->comm));
  PetscCheck(global_max_error < 1.0e-10, rdy->comm, PETSC_ERR_PLIB, "Heat coupling-interval regression failed: maximum temperature error is %g",
             (double)global_max_error);

  PetscCall(RDyDestroy(&rdy));
  PetscCall(RDyFinalize());
  return 0;
}
