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
