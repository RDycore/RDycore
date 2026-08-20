static char help[] = "Minimal TAOBLMVM test: min 1/2|x-a|^2 with bounds.\n";
#include <petsctao.h>
static PetscErrorCode ObjGrad(Tao tao, Vec x, PetscReal *f, Vec g, void *ctx)
{
  const PetscScalar *xa;
  PetscScalar       *ga;
  PetscInt           lo, hi;
  PetscFunctionBeginUser;
  /* ill-conditioned diagonal quadratic: f = 1/2 sum w_i (x_i - 0.5)^2, w_i spans 1..1e4 */
  PetscCall(VecGetOwnershipRange(x, &lo, &hi));
  PetscCall(VecGetArrayRead(x, &xa));
  PetscCall(VecGetArray(g, &ga));
  *f = 0.0;
  for (PetscInt i = lo; i < hi; i++) {
    PetscReal w = 1.0 + 1e4 * (PetscReal)i;
    PetscReal r = PetscRealPart(xa[i - lo]) - 0.5;
    ga[i - lo]  = w * r;
    *f += 0.5 * w * r * r;
  }
  PetscCall(VecRestoreArrayRead(x, &xa));
  PetscCall(VecRestoreArray(g, &ga));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, f, 1, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)tao)));
  PetscFunctionReturn(PETSC_SUCCESS);
}
int main(int argc, char **argv)
{
  Tao tao; Vec x, lb, ub;
  PetscInt n = 2;
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, n, &x));
  PetscCall(VecSet(x, 0.1));
  PetscCall(VecDuplicate(x, &lb)); PetscCall(VecSet(lb, 0.0));
  PetscCall(VecDuplicate(x, &ub)); PetscCall(VecSet(ub, 1.0));
  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao));
  PetscCall(TaoSetType(tao, TAOBLMVM));
  PetscCall(TaoSetSolution(tao, x));
  PetscCall(TaoSetVariableBounds(tao, lb, ub));
  PetscCall(TaoSetObjectiveAndGradient(tao, NULL, ObjGrad, NULL));
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSolve(tao));
  PetscInt its; PetscCall(TaoGetIterationNumber(tao, &its));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "done: %d iterations\n", (int)its));
  PetscCall(TaoDestroy(&tao)); PetscCall(VecDestroy(&x)); PetscCall(VecDestroy(&lb)); PetscCall(VecDestroy(&ub));
  PetscCall(PetscFinalize());
  return 0;
}
