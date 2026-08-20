# PETSc issue draft: MatInvertBlockDiagonal() returns a stale cache after re-assembly; PCPBJACOBI silently solves the wrong system

## Summary

`MatInvertBlockDiagonal()` caches the inverted diagonal blocks in the matrix.
That cache is not invalidated when a matrix is re-assembled with new values but
an unchanged nonzero structure, so the routine returns the *previous* values'
inverse. Because `PCSetUp_PBJacobi_Host()` obtains its blocks this way,
`PCPBJACOBI` preconditions with stale blocks after any such re-assembly -- and
with `-ksp_type preonly` on a block-diagonal operator, where the preconditioner
is the exact solve, `KSPSolve()` returns a **silently wrong answer**. No error
or warning is raised.

Affects both `MATAIJ` (with a block size set) and `MATBAIJ`.

## Reproducer

`ex_invertblockdiag_stale.c` (attached). Block-diagonal matrix, 2x2 blocks.
Assemble with diagonal blocks `= I`, then re-assemble the same sparsity with
`= 2I`, checking `MatInvertBlockDiagonal()` directly and via
`KSP(preonly) + PCPBJACOBI`.

```
cc -o ex ex_invertblockdiag_stale.c -I$PETSC_DIR/include \
    -I$PETSC_DIR/$PETSC_ARCH/include -L$PETSC_DIR/$PETSC_ARCH/lib -lpetsc
./ex
```

Output on current `main`:

```
MatInvertBlockDiagonal after re-assembly with unchanged sparsity:
  aij      pass 0: inverse[0] = 1      (expected 1.) ok
  aij      pass 1: inverse[0] = 1      (expected 0.5) <-- STALE
  baij     pass 0: inverse[0] = 1      (expected 1.) ok
  baij     pass 1: inverse[0] = 1      (expected 0.5) <-- STALE

KSP(preonly) + PCPBJACOBI on the same block-diagonal matrix:
  aij      pass 0: KSPSolve x[0] = 1      (exact answer 1.) ok
  aij      pass 1: KSPSolve x[0] = 1      (exact answer 0.5) <-- WRONG ANSWER
  baij     pass 0: KSPSolve x[0] = 1      (exact answer 1.) ok
  baij     pass 1: KSPSolve x[0] = 1      (exact answer 0.5) <-- WRONG ANSWER
```

Expected: `0.5` in every `pass 1` line.

## Build

Pristine `origin/main` worktree at `dd2b456823a`, freshly configured, minimal:

```
./configure PETSC_ARCH=arch-ibd-min --with-debugging=0 --with-fc=0 \
            --with-mpi=0 --download-f2cblaslapack
```

macOS arm64. No external packages beyond f2cblaslapack, to rule out
environment effects.

## Analysis

`MatInvertBlockDiagonal_SeqBAIJ()` (`src/mat/impls/baij/seq/baij.c`) and
`MatInvertBlockDiagonal_SeqAIJ()` (`src/mat/impls/aij/seq/aij.c`) both return
the cached blocks when `idiagvalid` / `ibdiagvalid` is set.

`MatAssemblyEnd_SeqBAIJ()` does contain the invalidation, under a comment
showing it is intended:

```c
/* diagonals may have moved, so kill the diagonal pointers */
a->idiagvalid = PETSC_FALSE;
```

but that line sits at the end of the routine, while the routine returns early
at the top whenever

```c
if (mode == MAT_FLUSH_ASSEMBLY || (A->was_assembled && A->ass_nonzerostate == A->nonzerostate)) PetscFunctionReturn(PETSC_SUCCESS);
```

which is exactly the "same sparsity, new values" case. So the invalidation is
unreachable in the common path.

`MatAssemblyEnd_SeqAIJ()` has the same early return and never clears
`ibdiagvalid` at all -- the only assignment of `PETSC_FALSE` in `aij.c` is in
matrix creation.

Suggested fix: clear the flag before the early return in both routines (or on
`MatSetValues`).

## Why this may have gone unnoticed

- `PCJACOBI` uses `MatGetDiagonal()` and is unaffected; `PCPBJACOBI` is less
  widely used.
- With a Krylov method rather than `preonly`, a stale preconditioner degrades
  convergence rather than producing a wrong answer, so it looks like a
  conditioning problem rather than a bug.
- If the nonzero *structure* changes between assemblies, the early return does
  not fire and the cache is correctly invalidated.

## How it was found

RDycore's shallow-water ARK-IMEX solver has a per-cell block-diagonal implicit
Jacobian (friction only), re-assembled every Newton step with a fixed sparsity.
`preonly + pbjacobi` should be an exact solve; instead the first linear solve
converged in 1 iteration and later ones took 78, 44, 24..., and with `preonly`
the Newton iteration diverged. We now default that solver to
`preonly + bjacobi`, which is unaffected.
