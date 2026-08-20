## Summary

On current `main` (v3.25.4-518-g40779859693 and confirmed again after
pulling to dd2b456823a, 2026-08-18), `TAOBLMVM` (and `TAOBQNLS`)
segfault during the quasi-Newton matrix update, on the second or third
iteration of a plain bound-constrained minimization. The same programs
run correctly on v3.25.3-361-g8caedce40c7.

Backtrace (macOS arm64, opt build; identical from two independent
applications):

```
MatDenseCreateColumnVec_Private +464
MatDenseGetColumnVecWrite_SeqDense +76
MatDenseGetColumnVecWrite +108
LMBasisGetVec_Internal +16
LMBasisGetNextVec +36
MatUpdateKernel_LMVM +44
MatUpdate_LMVMSymBrdn +1244
MatLMVMUpdate +412
TaoSolve_BLMVM +508
TaoSolve +408
```

## Reproducer

40-line standalone program (attached, `ex_tao_blmvm_segv.c`): minimize
an ill-conditioned diagonal quadratic
`f = 1/2 sum_i w_i (x_i - 0.5)^2`, `w_i` in `[1, 5e5]`, with bounds
`[0, 1]`, `TAOBLMVM`, defaults otherwise.

```
mpicc -o ex_tao_blmvm_segv ex_tao_blmvm_segv.c $(pkg-config --cflags --libs $PETSC_DIR/$PETSC_ARCH/lib/pkgconfig/PETSc.pc)
./ex_tao_blmvm_segv -n 50     # SEGV on main; "done: N iterations" on v3.25.3
```

(A well-conditioned problem that converges in one iteration does NOT
crash -- the failure needs the history update on iteration >= 2, which
matches the LMBasisGetNextVec frame.)

## Context

Found while running RDycore's adjoint-based Manning calibration (TAO
driving TSAdjoint gradients) against a freshly pulled main: the
calibration CTests that pass on v3.25.3-361 segfault identically
through `TaoSolve_BLMVM -> MatLMVMUpdate`. The reproducer above removes
RDycore and TSAdjoint from the picture entirely.

## Environment

macOS arm64 (Apple clang 21), OpenMPI 5.0.9, opt build
(`--with-debugging=no --with-strict-petscerrorcode`),
`--download-f2cblaslapack`.

## Attribution

Reported by Mark Adams (LBNL). Diagnosed and reproducer written with
Claude (Fable 5).

## Reproducer source

<details><summary>ex_tao_blmvm_segv.c</summary>

```c
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
```

</details>
