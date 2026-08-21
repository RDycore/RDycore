# Kickoff: device RHS + GPU adjoint for the SWE implicit path (session 2)

Charter for the next GPU session, continuing plans/gpu-session-kickoff.md
(P2+P3 DONE, P4 shelved). Read plans/RESULTS-gpu-implicit.md first — it has
every number, environment gotcha, and decision below. Branches
adams/gpu-implicit == adams/manning-draft at the P3 merge (8e826ec7).

## Goal

Remove the two remaining host components of the implicit forward+gradient
path at the Turning 30 m scale:

- **B1 — device RHS (PETSc operator backend).** TSFunctionEval is now 74%
  of n4 SNESSolve (21.6 of 29.1 s per 20 steps). Port the PETSc-backend
  RHS (interior flux + boundary flux + source) to Kokkos kernels the same
  way P3 ported the Jacobian: reuse the SAME edge/cell geometry tables and
  offset machinery in swe_jacobian_kokkos.kokkos.cxx (generalize the
  context; do not duplicate it), and convert swe_roe_flux_petsc.h with the
  RDY_MATH_FN pattern used on swe_roe_flux_jacobian_petsc.h (see commit
  071257a4 — mechanical: strip PetscErrorCode plumbing, call sites drop
  PetscCall). Do NOT touch the CEED path. Expected: n4 SNESSolve ~7.5 s
  (~15x vs CPU n64).
  GATE: explicit-path ctest subset unchanged (the PETSc RHS also serves
  explicit runs — do not regress them); jacobian/adjoint/calibrate 14/14;
  dt=1 x20 PM run twins the P3 trajectory (norms to 16 digits) with
  TSFunctionEval GPU %F = 100 and no per-eval GpuToCpu of the state.

- **B2 — GPU adjoint.** After B1 the HOST adjoint is ~70-80% of gradient
  time (estimate in the review section of RESULTS-gpu-implicit.md). This
  was charter-deferred, not technically blocked. Work items in order:
  1. TSTrajectory with kokkos vec types HANGS (reproduce on the laptop:
     rdycore_adjoint adjoint_beuler.yaml -dm_vec_type kokkos
     -snes_rtol 1e-12 -snes_atol 1e-50 -ksp_rtol 1e-12; killed after
     10 min; plain run takes seconds). Diagnose: likely D2H sync or I/O
     interaction in trajectory save/restore. Fix may belong in the fork.
  2. Transpose solve path on device: MatMultTranspose_SeqBAIJKokkos
     exists; VERIFY transpose PBJacobi apply (KSPSolveTranspose +
     PCApplyTranspose on pbjacobi with baijkokkos) — add a fork gate test
     if missing (pattern: src/mat/tests/ex337.c, the lifecycle test this
     effort added).
  3. Then the adjoint's per-step Jacobian assemblies get the P3 device
     path for free (rhs_jac type-driven).
  GATE: adjoint FD gates (rel < 1e-5) with device forward+backward at
  np 1 and 2; calibrate_manning_* ctest unchanged.

## Standing decisions (do not relitigate)

- pbjacobi is the production PC for the working dt range (dt<=1). GAMG is
  shelved with P4 until large-dt implicit research lands (see the
  cad91dce verdict in RESULTS-manning-draft.md: nonsmooth Newton + error
  control both block dt>=5; unlock candidates are SNES VI, smoothed
  wet/dry, loosened ts tolerances — research, not tuning).
- SWE GAMG notes for whenever P4 revives: -pc_gamg_threshold -1
  (value-filtering breaks graph symmetry; -pc_gamg_sym_graph not honored
  on the fork's device CreateGraph — fork follow-up), prolongator_filter
  0.03, aggressive coarsening; PCSetUp moves ~66-82 MB/setup (apply and
  solve are GPU %F 100 — that transfer is the only lead).
- Blocked COO (MatCOOUseBlockIndices) for BAIJ types; scalar COO
  otherwise; -rhs_jac_mat_type overrides the Jacobian's type alone.

## Environments (see RESULTS-gpu-implicit.md for full gotchas)

- Laptop: petsc-claude arch-macosx-gnu-rdycore-kokkos-O (host kokkos +
  RDycore deps); RDycore worktree ~/Codes/RDycore-gpu, build-claude.
  Verify FIRST on the laptop — the whole P3 debugging loop ran there.
- PM: petsc-claude arch-perlmutter-claude-kokkos-cuda-O (cuda 13.2, no
  cudss); rdycore-manning build-claude (CPU) + build-claude-gpu; run dir
  $SCRATCH/gpu-implicit (yaml, compare_traj.py, all p2i/p3i/p4prep logs).
  MPICH_GPU_SUPPORT_ENABLED=1 in jobs. PM /tmp is per-login-node.
- The SHARED arch arch-perlmutter-opt-gcc-kokkos-cuda is CUDA-chimeric
  (12.9 configure / 13.2 objects): runs only with cudatoolkit/12.9 loaded
  AND 13.2 cuda+math lib64 appended to LD_LIBRARY_PATH; its configured
  python is gone (PYTHON=python3). Clean fix = full reconfigure under one
  toolkit (coordinate — nothing currently depends on it beyond fork tests;
  QA passed 66/66 with the workaround).
- petsc-claude is reconciled at fdaf8ca2c1a everywhere local; Mark pushes
  to GitLab (origin = markadams4/petsc-claude, never upstream petsc).

## Correctness invariants (unchanged from session 1)

- numerics.jacobian: fd stays the permanent verification baseline.
- CPU builds (stock PETSc or fork-CPU) must build and pass: the device
  paths are all compile- and runtime-guarded (RDY_HAVE_KOKKOS_JACOBIAN,
  RDY_HAVE_MAT_COO_BLOCK_INDICES, type checks).
- The 12/17-gauge Harvey machinery and paper are NOT in scope.

## Open delegated items (not this session's critical path)

- cudss-on-cuda13 wiring in the fork (P4 coarse solve).
- Fork: honor -pc_gamg_sym_graph on the device CreateGraph path.
- Shared-arch clean reconfigure under one CUDA toolkit.
