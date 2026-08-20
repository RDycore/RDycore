// Does MatInvertBlockDiagonal() return a STALE cached inverse after a matrix is
// re-assembled with new values but an unchanged nonzero structure?
//
// Tests BOTH MATAIJ (with a block size set) and MATBAIJ, because the two have
// separate caches: Mat_SeqAIJ::ibdiagvalid and Mat_SeqBAIJ::idiagvalid. Only
// MatAssemblyEnd_SeqBAIJ clears its flag at all, and it does so AFTER an early
// return taken whenever
//     A->was_assembled && A->ass_nonzerostate == A->nonzerostate
// i.e. exactly the "same sparsity, new values" case.
//
// This matters because PCPBJACOBI obtains its blocks via MatInvertBlockDiagonal
// (PCSetUp_PBJacobi_Host), so a stale cache silently preconditions with the
// wrong blocks -- and with -ksp_type preonly, silently solves the wrong system.
// AIJ is the case that matters in practice; BAIJ is less well maintained.
//
// Build: mpicc -o ex_invertblockdiag_stale ex_invertblockdiag_stale.c \
//          $(pkg-config --cflags --libs $PETSC_DIR/$PETSC_ARCH/lib/pkgconfig/PETSc.pc)
// Run:   ./ex_invertblockdiag_stale
// Pass:  both report 0.5 after the second assembly.

#include <petscksp.h>

static PetscErrorCode CheckType(MatType type, PetscBool *ok) {
  Mat                A;
  const PetscScalar *inv;
  const PetscInt     bs = 2, nb = 4;

  PetscFunctionBeginUser;
  PetscCall(MatCreate(PETSC_COMM_SELF, &A));
  PetscCall(MatSetSizes(A, bs * nb, bs * nb, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetType(A, type));
  PetscCall(MatSetBlockSize(A, bs));
  PetscCall(MatSeqAIJSetPreallocation(A, bs, NULL));   // no-op for BAIJ
  PetscCall(MatSeqBAIJSetPreallocation(A, bs, 1, NULL));  // no-op for AIJ

  for (PetscInt pass = 0; pass < 2; ++pass) {
    PetscScalar d = (pass == 0) ? 1.0 : 2.0;  // diagonal blocks = I, then 2I
    for (PetscInt b = 0; b < nb; ++b) {
      PetscScalar blk[4] = {d, 0.0, 0.0, d};
      PetscCall(MatSetValuesBlocked(A, 1, &b, 1, &b, blk, INSERT_VALUES));
    }
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatInvertBlockDiagonal(A, &inv));
    PetscReal got = PetscRealPart(inv[0]), want = 1.0 / PetscRealPart(d);
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "  %-8s pass %" PetscInt_FMT ": inverse[0] = %-6g (expected %g) %s\n", type, pass, (double)got,
                          (double)want, (PetscAbsReal(got - want) < 1e-12) ? "ok" : "<-- STALE"));
    if (pass == 1 && PetscAbsReal(got - want) > 1e-12) *ok = PETSC_FALSE;
  }
  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// The user-visible consequence: KSP(preonly) + PCPBJACOBI on a block-diagonal
// matrix is an exact solve, so x must equal A^-1 b. Re-assemble A with new
// values, solve again, and the answer is silently wrong if the PC reused the
// previous blocks. No error is raised anywhere.
static PetscErrorCode CheckSolve(MatType type, PetscBool *ok) {
  Mat            A;
  Vec            b, x;
  KSP            ksp;
  PC             pc;
  const PetscInt bs = 2, nb = 4;

  PetscFunctionBeginUser;
  PetscCall(MatCreate(PETSC_COMM_SELF, &A));
  PetscCall(MatSetSizes(A, bs * nb, bs * nb, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetType(A, type));
  PetscCall(MatSetBlockSize(A, bs));
  PetscCall(MatSeqAIJSetPreallocation(A, bs, NULL));
  PetscCall(MatSeqBAIJSetPreallocation(A, bs, 1, NULL));
  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(VecSet(b, 1.0));

  PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));
  PetscCall(KSPSetType(ksp, KSPPREONLY));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCPBJACOBI));

  for (PetscInt pass = 0; pass < 2; ++pass) {
    PetscScalar d = (pass == 0) ? 1.0 : 2.0;
    for (PetscInt bl = 0; bl < nb; ++bl) {
      PetscScalar blk[4] = {d, 0.0, 0.0, d};
      PetscCall(MatSetValuesBlocked(A, 1, &bl, 1, &bl, blk, INSERT_VALUES));
    }
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSolve(ksp, b, x));
    PetscReal got;
    PetscCall(VecMax(x, NULL, &got));
    PetscReal want = 1.0 / PetscRealPart(d);
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "  %-8s pass %" PetscInt_FMT ": KSPSolve x[0] = %-6g (exact answer %g) %s\n", type, pass,
                          (double)got, (double)want, (PetscAbsReal(got - want) < 1e-12) ? "ok" : "<-- WRONG ANSWER"));
    if (pass == 1 && PetscAbsReal(got - want) > 1e-12) *ok = PETSC_FALSE;
  }
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv) {
  PetscBool ok = PETSC_TRUE;
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "MatInvertBlockDiagonal after re-assembly with unchanged sparsity:\n"));
  PetscCall(CheckType(MATAIJ, &ok));
  PetscCall(CheckType(MATBAIJ, &ok));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "\nKSP(preonly) + PCPBJACOBI on the same block-diagonal matrix:\n"));
  PetscCall(CheckSolve(MATAIJ, &ok));
  PetscCall(CheckSolve(MATBAIJ, &ok));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n%s\n", ok ? "PASS: cache invalidated on re-assembly" : "FAIL: stale block-diagonal inverse reused"));
  PetscCall(PetscFinalize());
  return 0;
}
