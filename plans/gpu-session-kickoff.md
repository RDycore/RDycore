# Kickoff: full-GPU RDycore implicit path on the petsc-claude fork

Charter for a fresh session (Mark's second agent). Recommended model:
**claude-fable-5** (do NOT use Haiku for this — long multi-step port with
subtle correctness gates; see also the project memory note on subagents).

## Goal

A device-resident implicit (BEULER) solve path for RDycore's SWE at the
Turning 30 m scale, built on Mark's PETSc fork `petsc-claude` (main), which
provides GPU FEM assembly patterns (PetscFE-Kokkos), BAIJKOKKOS/BAIJCUDA
blocked matrices with blocked COO assembly, device GAMG-SA with device
PBJacobi smoothers, and device PtAP. Phases P2-P4 of
`plans/PLAN-fv-gpu-adaptation.md` (P1 is DONE — read that file first).

## State you inherit (all pushed to RDycore branch `adams/manning-draft`)

- **P1 done** (commit `3c1847cd`): the analytic SWE RHS Jacobian is
  assembled via MatSetPreallocationCOO / MatSetValuesCOO with the exact FV
  pattern (4 blocks per interior edge for owned sides + boundary (l,l) +
  cell diagonal). Measured A/B on a PM GPU node (dt=1 BEULER, 64 ranks,
  20 steps, fixed ksp_rtol 1e-3, pbjacobi):
  pre-COO SNESSolve 203.8 s (JacEval 118.8 s, 1.15 s/assembly, KSP 53.0 s)
  ->  COO SNESSolve 134.6 s (JacEval 79.4 s, 0.77 s/assembly, KSP 25.6 s).
  The tighter pattern halves stored nonzeros (closure superset gone), which
  is what halved KSPSolve. Remaining assembly cost = flux-tangent loop
  (host) + DMGlobalToLocal + COO scatter (7.7 s, load-imbalanced 23x).
- **Solver status at 30 m** (plans/RESULTS-manning-draft.md, bottom
  sections): dt=1 converges (4-5 Newton its, bt LS); dt>=5 fails in the
  LINE SEARCH (wet/dry nonsmooth merit function), not the linear solve —
  basic (no) line search runs were in flight at handoff (logs
  b_dt{5,15,30}_pbjac.log, b_dt15_gamg.log in $SCRATCH/manning-beuler).
  Mark's solver preferences: NO -snes_ksp_ew for now, fixed ksp_rtol
  1e-2..1e-3; pbjacobi (never plain bjacobi) as simple PC and as GAMG
  smoother (-mg_levels_pc_type pbjacobi), gmres smoothers with fgmres
  outside if needed; gamg knobs to explore: -pc_gamg_agg_nsmooths 0|1,
  -pc_gamg_aggressive_coarsening 0|1 (nsmooths 0 relevant for this
  NONSYMMETRIC operator — no CG anywhere).

## Perlmutter environment (account m1516_g, GPU nodes; CPU allocation m4267)

- `~/Codes/petsc-claude` = Mark's sandbox fork, branch main at
  `5d7d9997275` (includes the stale block-diagonal-inverse cache fix,
  state-keyed — REQUIRED for pbjacobi under per-Newton reassembly).
  ANOTHER AGENT works in this repo too — coordinate before rebuilding
  arches or switching branches; do not rewrite history.
- CPU arch `arch-perlmutter-claude-O` is built (gotcha: `module unload
  cudatoolkit gpu` before configure, else the sandbox tries to
  CUDAC-compile agg_device.cuda.cu even with --with-cuda=0).
- For the GPU path you need a kokkos-cuda arch of petsc-claude for
  RDycore to link: `arch-perlmutter-opt-gcc-kokkos-cuda` exists in the
  sandbox (configs in ~/Codes/petsc-configs/) but predates the 3 new
  commits — rebuild it (coordinate with the other agent).
- RDycore worktree: `~/Codes/rdycore-manning` (branch adams/manning-draft;
  `git pull` first). Builds: `build-claude` (CPU petsc-claude arch;
  export PETSC_DIR/PETSC_ARCH, `module load cmake cray-hdf5-parallel`,
  `cmake -DENABLE_FORTRAN=OFF`). Add a `build-claude-gpu` for the kokkos
  arch. NB `build-claude/driver/rdycore.preCOO` is a preserved old binary.
- Run dir: `$SCRATCH/manning-beuler` (Turning 30 m mesh + IC symlinked,
  beuler_dt*.yaml configs, job scripts). Mesh/IC source of truth:
  /global/cfs/cdirs/m4267/shared/data/harvey/Turning_30m/.
- PM /tmp is per-login-node: keep scratch files in $HOME or $SCRATCH.
- Laptop stack for CPU verification: build_rev + petsc_gem
  arch-macosx-gnu-rdycore-O-rev (export PETSC_DIR/ARCH before cmake).

## The work (in order; STOP at each gate)

- **P2 — device matrices, host physics.** Run the BEULER path with
  `-dm_mat_type aijkokkos` first, then baijkokkos (`-dm_vec_type kokkos`).
  RDycore creates rhs_jac itself in CreateAnalyticJacobianCOO
  (src/swe/swe_jacobian_petsc.c) with MatSetType(MATAIJ) — make the type
  option-overridable (MatSetFromOptions or honor dm_mat_type) rather than
  hardwired. Values stay host-computed; MatSetValuesCOO uploads.
  GATE: jacobian/adjoint ctest subset passes on CPU unchanged, and a PM
  dt=1 run reproduces the CPU trajectory (state norms to ~1e-12) with
  solver work on device (check -log_view GPU columns + no per-iteration
  GpuToCpu transfers in the solve).
- **P3 — device assembly.** Port the FV Jacobian fill (interior-edge loop
  + source diagonal; boundary edges can stay host initially, they are
  O(surface)) to a Kokkos kernel filling the COO values array on device;
  MatSetValuesCOO with a device pointer. Follow the fork's kernel style
  (see baijkok.kokkos.cxx and the PetscFE-Kokkos arc). The RHS itself has
  a CEED path already — do NOT entangle with CEED; this is the PETSc
  operator backend. GATE: FD gates (test_swe_jacobian, adjoint suite)
  pass with device assembly at np 1 and 2; A/B the 0.77 s/assembly number.
- **P4 — device GAMG for SWE at large dt.** Once the nonlinear robustness
  question (line search / dt continuation) is settled on CPU, run the
  GAMG parameter menu on device at dt=15/30 and compare against pbjacobi.
  Respect Mark's solver preferences above. Metric: KSPSolve + PCSetUp
  (PtAPNumeric reruns every Newton — the fork's device PtAP is the
  headline asset here).

## Correctness invariants (do not regress)

- numerics.jacobian: fd stays the permanent verification baseline
  (DMCreateMatrix closure pattern — do not COO-ize or device-ify it).
- The adjoint path (TSAdjoint, host transposes) must keep working with
  the CPU build; device work is forward-solve-first.
- ctest subset: `ctest -R "adjoint|calibrate|jacobian"` = 14/14 on the
  laptop stack; full suite has exactly 7 known env failures (CGNS/AMR).
- The 12-gauge/17-gauge Harvey machinery and paper are NOT in scope.

## Open questions to resolve early

- baijkokkos vs aijkokkos for bs=3 FV: the fork's BAIJ path gives block
  SpMV + device PBJacobi; check MatSetPreallocationCOO support for
  MATBAIJKOKKOS (the fork added "blocked COO" — BAIJCUDA arc Phase 4b) —
  if scalar-COO-on-BAIJ is unsupported, use the blocked COO API.
- Where the DMPlex l2g map lives on device for P3 (precompute the global
  index arrays once on host and copy — they are static).
