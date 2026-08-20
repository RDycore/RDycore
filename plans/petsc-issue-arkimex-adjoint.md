# Draft PETSc issue (for Hong) — paste into gitlab.com/petsc/petsc/-/issues

**Title:** TSAdjoint/ARKIMEX: SEGV when parameter dependence is supplied
only via TSSetIJacobianP (NULL Jacprhs dereferenced)

---

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

**Standalone reproducer attached** (`ex_arkimex_adjoint_jacprhs.c`,
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
