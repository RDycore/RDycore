## Summary

When computing parameter sensitivities with `TSARKIMEX`, the adjoint
step unconditionally applies **both** the implicit-part and
explicit-part parameter Jacobians once `ts->vecs_sensip` is set. If the
user's parameter appears only in the IFunction and they therefore
register only `TSSetIJacobianP` (no `TSSetRHSJacobianP`), `ts->Jacprhs`
is NULL and the adjoint step segfaults.

In `TSAdjointStep_ARKIMEX` (src/ts/impls/arkimex/arkimex.c, lines
~1591–1633 at v3.25.3-361-g8caedce40c7):

```c
if (ts->vecs_sensip) {
  PetscCall(TSComputeIJacobianP(ts, ..., ts->Jacp, PETSC_TRUE));   // dFdP
  PetscCall(TSComputeRHSJacobianP(ts, ..., ts->Jacprhs));          // dGdP
}
...
PetscCall(MatMultTranspose(ts->Jacprhs, VecsSensiTemp[nadj], VecsSensiPTemp[nadj]));  // SEGV: Jacprhs == NULL
```

`MatMultTranspose` on the NULL `Jacprhs` fails: signal 11 (SEGV) in an
optimized build, "Null argument, when expecting valid pointer" in a
debug build. Verified on both v3.25.3-361-g8caedce40c7 and current main
(v3.25.4-518-g40779859693, 2026-08-16) -- the unguarded calls are at
arkimex.c:1592 and :1633 in both.

## Reproduction context

RDycore (E3SM river dynamical core) IMEX shallow-water setup: stiff
Manning friction in the IFunction (`F = u_t - S_fric(u; n)`), fluxes in
the RHS function. The Manning coefficient n appears only in the
implicit part, so the natural registration is `TSSetIJacobianP` alone.
Sequence: `TSSetRHSFunction` + `TSSetIFunction`/`TSSetIJacobian` +
`TSSetIJacobianP` + `TSSetCostGradients(ts, 1, &lambda, &mu)` +
`TSSolve` + `TSAdjointSolve` -> SEGV in the first adjoint step.

**Standalone reproducer below** (`ex_arkimex_adjoint_jacprhs.c`,
~150 lines, no external deps): u_t + p u = -u with the parameter p only
in the IFunction, cost J = u(T).

```
./ex_arkimex_adjoint_jacprhs                            # fails (SEGV opt / Null argument debug)
./ex_arkimex_adjoint_jacprhs -workaround -ts_adapt_type none
# dJ/dp = -0.0740818608 (exact -T e^{-(1+p)T} = -0.0740818221)
```

A pitfall the reproducer also documents: the RHS and I Jacobians must
be registered with SEPARATE Mat objects -- with a shared Mat the
adjoint step silently overwrites one with the other and produces
garbage gradients (arguably worth a runtime check as well).

## Expected behavior

A missing RHSJacobianP should be treated as dG/dp = 0 (skip the
`MatMultTranspose`/`MatMultTransposeAdd` on the explicit side), matching
the natural user expectation that only the parts where the parameter
appears need Jacobians. Alternatively, `TSAdjointSetUp` could error
with a clear message requiring both to be registered for ARKIMEX.

## Workaround (works)

Register a preallocated all-zero matrix with a trivial callback:

```c
PetscCall(MatCreateAIJ(comm, nloc, nploc, PETSC_DETERMINE, PETSC_DETERMINE, 0, NULL, 0, NULL, &Jacprhs_zero));
/* assemble empty */
PetscCall(TSSetRHSJacobianP(ts, Jacprhs_zero, RHSJacobianPZero, ctx));  /* callback: MatZeroEntries + assemble */
```

With this in place the ARKIMEX discrete-adjoint gradients check out
against central finite differences of the full nonlinear solve to
2e-8 (dJ/du0) and 2.5e-6 (dJ/dn) on our SWE test problem — the adjoint
itself is in fine shape.

## Two small related notes (docs-level, can split out if preferred)

1. **Adaptive controller vs. FD validation of adjoint gradients**: with
   the default ARKIMEX error controller active, finite-difference
   checks of adjoint gradients appear to fail at ~1e-4 because each
   perturbed forward solve selects a different step sequence; with
   `-ts_adapt_type none` the same check passes at 2e-8. Entirely
   expected on reflection, but a sentence in the TSAdjoint manual page
   ("validate gradients with a fixed step sequence") would save users a
   debugging session.

2. For implicit integrators (tested with TSTHETA/BEULER), the
   discrete-adjoint gradient accuracy tracks the SNES/KSP tolerances of
   the forward and adjoint solves (defaults gave ~2e-4 on dJ/dp;
   `-snes_rtol 1e-12 -ksp_rtol 1e-12` recovered ~5e-6). Also probably
   worth a manual-page sentence.

## Environment

Verified on PETSc v3.25.3-361-g8caedce40c7 (opt) and main
v3.25.4-518-g40779859693 (debug), macOS arm64, OpenMPI 5.0.9.

## Attribution

Reported by Mark Adams (LBNL). Diagnosed and reproducer written with
Claude (Fable 5) during RDycore adjoint development.


## Reproducer

<details><summary>ex_arkimex_adjoint_jacprhs.c</summary>

```c
static char help[] = "Reproducer: TSAdjoint/ARKIMEX SEGV when the parameter\n"
                     "appears only in the IFunction and only TSSetIJacobianP is registered.\n"
                     "Problem: u_t + p u = -u  (implicit stiff decay p u, explicit -u).\n"
                     "Cost: J = u(T). Run:\n"
                     "  ./ex_arkimex_adjoint_jacprhs              -> SEGV in TSAdjointSolve\n"
                     "  ./ex_arkimex_adjoint_jacprhs -workaround  -> runs; prints dJ/dp\n\n";

#include <petscts.h>

typedef struct {
  PetscReal p;
} AppCtx;

/* explicit part G(u) = -u */
static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec U, Vec G, void *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(VecCopy(U, G));
  PetscCall(VecScale(G, -1.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec U, Mat J, Mat P, void *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(MatZeroEntries(P));
  PetscCall(MatSetValue(P, 0, 0, -1.0, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* implicit part F(u, u_t) = u_t + p u */
static PetscErrorCode IFunction(TS ts, PetscReal t, Vec U, Vec Udot, Vec F, void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;
  PetscFunctionBeginUser;
  PetscCall(VecCopy(U, F));
  PetscCall(VecScale(F, user->p));
  PetscCall(VecAXPY(F, 1.0, Udot));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode IJacobian(TS ts, PetscReal t, Vec U, Vec Udot, PetscReal shift, Mat J, Mat P, void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;
  PetscFunctionBeginUser;
  PetscCall(MatZeroEntries(P));
  PetscCall(MatSetValue(P, 0, 0, shift + user->p, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* dF/dp = u */
static PetscErrorCode IJacobianP(TS ts, PetscReal t, Vec U, Vec Udot, PetscReal shift, Mat Jacp, void *ctx)
{
  const PetscScalar *u;
  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(MatSetValue(Jacp, 0, 0, u[0], INSERT_VALUES));
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(MatAssemblyBegin(Jacp, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Jacp, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* workaround: the explicit part has no p-dependence -> dG/dp = 0 */
static PetscErrorCode RHSJacobianPZero(TS ts, PetscReal t, Vec U, Mat Jacp, void *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(MatZeroEntries(Jacp));
  PetscCall(MatAssemblyBegin(Jacp, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Jacp, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  TS        ts;
  Vec       U, lambda, mu;
  Mat       Jrhs, Ji, Jacp, Jacprhs;
  AppCtx    user = {.p = 2.0};
  PetscBool workaround = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-workaround", &workaround, NULL));

  PetscCall(VecCreateSeq(PETSC_COMM_SELF, 1, &U));
  PetscCall(VecSet(U, 1.0));
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, 1, 1, 1, NULL, &Jrhs));
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, 1, 1, 1, NULL, &Ji));
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, 1, 1, 1, NULL, &Jacp));

  PetscCall(TSCreate(PETSC_COMM_SELF, &ts));
  PetscCall(TSSetType(ts, TSARKIMEX));
  PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction, &user));
  PetscCall(TSSetRHSJacobian(ts, Jrhs, Jrhs, RHSJacobian, &user));
  PetscCall(TSSetIFunction(ts, NULL, IFunction, &user));
  PetscCall(TSSetIJacobian(ts, Ji, Ji, IJacobian, &user));
  PetscCall(TSSetIJacobianP(ts, Jacp, IJacobianP, &user));

  if (workaround) {
    PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, 1, 1, 0, NULL, &Jacprhs));
    PetscCall(MatAssemblyBegin(Jacprhs, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Jacprhs, MAT_FINAL_ASSEMBLY));
    PetscCall(TSSetRHSJacobianP(ts, Jacprhs, RHSJacobianPZero, &user));
  }

  PetscCall(TSSetSaveTrajectory(ts));
  PetscCall(TSSetTime(ts, 0.0));
  PetscCall(TSSetTimeStep(ts, 0.01));
  PetscCall(TSSetMaxTime(ts, 0.1));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSolve(ts, U));

  /* J(p) = u(T): lambda(T) = dJ/du(T) = 1, mu(T) = 0 */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, 1, &lambda));
  PetscCall(VecSet(lambda, 1.0));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, 1, &mu));
  PetscCall(VecSet(mu, 0.0));
  PetscCall(TSSetCostGradients(ts, 1, &lambda, &mu));

  PetscCall(TSAdjointSolve(ts)); /* <- SEGV here without -workaround */

  {
    const PetscScalar *m;
    PetscCall(VecGetArrayRead(mu, &m));
    /* exact: u(T) = exp(-(1+p) T), dJ/dp = -T exp(-(1+p) T) = -0.0740818... */
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "dJ/dp = %.10f (exact -T e^{-(1+p)T} = %.10f)\n", (double)PetscRealPart(m[0]),
                          (double)(-0.1 * PetscExpReal(-3.0 * 0.1))));
    PetscCall(VecRestoreArrayRead(mu, &m));
  }

  PetscCall(VecDestroy(&U));
  PetscCall(VecDestroy(&lambda));
  PetscCall(VecDestroy(&mu));
  PetscCall(MatDestroy(&Jrhs));
  PetscCall(MatDestroy(&Ji));
  PetscCall(MatDestroy(&Jacp));
  if (workaround) PetscCall(MatDestroy(&Jacprhs));
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFinalize());
  return 0;
}
```

</details>

cc @caidao22
