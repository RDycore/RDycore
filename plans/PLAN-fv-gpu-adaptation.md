# Adapting the petsc-claude GPU pipeline to RDycore's FV implicit solve

Mark's directive (2026-08-21): the petsc-claude fork is a mature GPU DMPlex
pipeline — look at adapting it to FV and using it.

## What the fork has (from plans/notes/ROADMAP-fem-gamg-bjkokkos.md)

Built and gated for FE elasticity (ex56k, bs=3 — same block size as SWE):

- **BAIJKOKKOS / BAIJCUDA**: native blocked device matrices — block SpMV
  (flat/multi-row/team kernels), **blocked COO assembly** (Phase 4b),
  device products AB/AtB/**PtAP** (the 180x CUDA-vs-Kokkos PtAPNumeric gap
  is the paper's headline), MIS coarsener on device.
- **Device GAMG-SA**: setup fully device-resident (ProlFilt/RowCorr/
  FormProl0/Coarsen/CreateG), prolongator filtering, 25x setup win;
  **device PBJacobi apply** (GpuToCpu 182 -> 0); cudss coarse solve.
- **Stale block-diagonal cache fix** now landed on main (state-keyed),
  so cached pbjacobi inverses are correct under per-Newton reassembly.
- Canonical config pattern:
  `-dm_mat_type baijkokkos -dm_vec_type kokkos -pc_type gamg
   -mg_levels_pc_type pbjacobi -mg_coarse_pc_type bjacobi
   -mg_coarse_sub_pc_factor_mat_solver_type cudss`
- All measured on m1516_g A100s — the allocation we're already using.

## Why FV is the EASY case

The FE pipeline's hard part is device closure maps for element assembly
(PetscFE-Kokkos). RDycore's FV Jacobian has trivially simpler structure:
per interior edge, four 3x3 blocks at (L,L),(L,R),(R,R),(R,L); per cell,
one diagonal source/BC block. The COO index pattern is a flat function of
the edge list — no closures, no quadrature, no section permutations.

## Motivating measurement (job 57325605, dt=1 BEULER, 64 ranks, 2.93M cells)

SNESJacobianEval = 67% of SNESSolve (1.07 s per assembly) — the current
assembly is ~17M one-block MatSetValuesBlockedLocal calls per Jacobian
(4 per edge + diagonals). The linear solve under EW is only 17%.

## Staged plan

- **P1 (CPU, immediate, no GPU dependency): COO-ize the assembly.**
  At RDySetup: build coo_i/coo_j for the fixed FV pattern (4 blocks/edge
  + cell diagonals), MatSetPreallocationCOO once. Each assembly: fill a
  flat values array in the existing edge loop (same tangent-flux code),
  one MatSetValuesCOO(ADD). Kills the MatSetValues overhead AND replaces
  the DMCreateMatrix closure-superset preallocation (open question #4 to
  Jeff) with the exact FV pattern. Expected ~2x on implicit stepping at
  scale. Works on plain AIJ; type-agnostic afterwards.
- **P2: device solve, host physics.** `-dm_mat_type baijkokkos` (or
  aijkokkos first): values computed on host in the same loop, uploaded by
  COO (one transfer per assembly); SpMV/pbjacobi/GAMG run on device via
  the fork. BEULER outer fgmres. This is config + P1, near-zero new code.
- **P3: device assembly.** Port the edge-loop tangent-flux kernel to
  Kokkos (mirror of the CEED operator split: flux + source kernels);
  fill the COO values array device-side. FV analog of the fork's
  PetscFE-Kokkos arc, but without closure machinery.
- **P4: device GAMG study for SWE.** Re-run the dt=15/30 parameter study
  with the fork's device GAMG. SWE caveats vs the elasticity canon:
  nonsymmetric (fgmres outer, gmres+pbjacobi smoothers, NO cg),
  advective near-null space is not RBM (no SetNearNullSpace analog yet —
  open research: what is the right near-null space for upwinded SWE at
  large CFL? constants-per-field is the naive start), and the Newton
  metric includes PtAPNumeric per reassembly — the fork's 180x device
  PtAP is directly relevant because GAMG setup reruns every Newton.

## Prerequisites / blockers

- Nonlinear robustness at dt>=5 first (line-search-off job 57363544):
  no point making a fast solve for steps Newton can't take.
- P2+ needs a kokkos/cuda PETSc arch on PM from petsc-claude main
  (arch-perlmutter-opt-gcc-kokkos-cuda exists in the sandbox but must be
  rebuilt after the stale-fix commits; check its base).
- The adjoint path stays CPU for now (TSAdjoint transposes on host);
  device forward + host adjoint is fine — the gradient bottleneck is the
  forward sweeps anyway.
