# GPU-implicit session log (P2-P4 of PLAN-fv-gpu-adaptation)

Branch `adams/gpu-implicit` (off `adams/manning-draft` @ 6765cd86). Second
agent, per plans/gpu-session-kickoff.md.

## P2 — device matrices, host physics

### Code (laptop-verified, commits c1487f6d + 054717cb)

- `CreateAnalyticJacobianCOO` no longer hardwires MATAIJ: type comes from
  `DMGetMatType` (`-dm_mat_type`; default unchanged), plus a prefixed
  `MatSetFromOptions` so `-rhs_jac_mat_type` overrides THIS matrix alone.
  (A bare `-mat_type` is also consumed inside DMCreateMatrix_Plex and flips
  the FD twin in the jacobian tests — segfaults MATCOLORINGSL on an empty
  BAIJKokkos, whose host CSR is only built at assembly.)
- Blocked COO for BAIJ types via the fork's `MatCOOUseBlockIndices`:
  block-index pattern (9x fewer index entries), values = one dense row-major
  3x3 block per entry == exactly the P1 values layout, so the assembly loop
  is untouched. Detected at configure time by grepping petscmat.h ->
  `RDY_HAVE_MAT_COO_BLOCK_INDICES` (config.h); builds against stock PETSc
  unchanged.
- Only types that IMPLEMENT blocked COO opt in: `*baijkokkos`, `*baijcuda`,
  `mpibaij` (host twin). Plain `seqbaij` ignores the flag by design and
  would misread block indices as scalar (verified: rel err ~1) — it keeps
  scalar COO. Serial host blocked-COO testing: force `mpibaij`.

### Laptop verification (fork stack)

New PETSc arch `arch-macosx-gnu-rdycore-kokkos-O` of ~/Codes/petsc-claude
(RDycore dep set + kokkos host); RDycore worktree build `build-claude`.
- `ctest -R "adjoint|calibrate|jacobian"`: 14/14 (both this stack and the
  petsc_gem `build_rev` stack).
- `test_swe_jacobian_global` FD-vs-analytic rel err 1.5-1.9e-8 (gate 1e-6),
  IDENTICAL across `-rhs_jac_mat_type` in {aij, baij, mpibaij, aijkokkos,
  baijkokkos}, np 1 and 2.
- Standalone fork check: blocked COO on seqbaijkokkos is exact (0.0 diff vs
  MatSetValuesBlocked reference, repeated blocks summed). NB
  MatSetPreallocationCOO may permute its index arrays in place.

### Perlmutter status / environment findings (IMPORTANT)

- PM moved to CPE 26.03: default `cudatoolkit/13.2`, `cray-mpich/9.1.0`
  whose `libmpi_gtl_cuda.so` REQUIRES `libcudart.so.13` — CUDA 12 builds of
  new arches are no longer linkable with the default MPICH. The
  `--download-cudss` archive is cuda12-only => cudss is OFF in the new arch
  until a cuda13 archive is wired (P4 coarse-solve fallback: host lu).
- `arch-perlmutter-opt-gcc-kokkos-cuda` (shared with the other agent) was
  STALE w.r.t. the 3 stale-cache-fix commits; incrementally rebuilt Aug 21
  (touched the 14 device TUs including changed headers first — header-dep
  gotcha). CAVEAT: that arch records `CUDAC = nvcc` (PATH-resolved), so the
  rebuilt objects compiled under nvcc 13.2 into a 12.9-configured arch. It
  links; a from-scratch reconfigure under the new CPE is the clean fix if
  anything smells.
- New arch for RDycore: `arch-perlmutter-claude-kokkos-cuda-O` = device
  pipeline (kokkos/kokkos-kernels/metis/parmetis/triangle, cuda 13.2,
  minus cudss) + RDycore deps (exodus/hdf5/libceed/muparser/netcdf/pnetcdf/
  zlib/revolve), COPTFLAGS=-O2. Config script:
  ~/Codes/petsc-configs/arch-perlmutter-claude-kokkos-cuda-O.py (+ build
  script build-claude-kokkos-cuda.sh). Building at session time.
- RDycore worktree ~/Codes/rdycore-manning now on `adams/gpu-implicit`;
  `cmake-claude-gpu.sh` stages `build-claude-gpu` against the new arch.
- Run dir: `$SCRATCH/gpu-implicit` (mesh+IC linked, beuler_dt1.yaml,
  `p2_ab.sbatch` staged: CPU baseline, GPU-binary/host-mats sanity, then
  aijkokkos and baijkokkos at n4 and n64, all dt=1 20 steps, fgmres+pbjacobi
  rtol 1e-3, solution monitors for trajectory comparison).
- The OTHER agent was actively iterating the dt>=5 robustness question this
  morning in $SCRATCH/manning-beuler (d_dt5_*, b_dt5_* logs; interactive job
  on nid001000): at dt=5 with basic LS the FIRST Newton linear solve
  diverges (300 its, pbjacobi) — a linear-solver hardness problem, not just
  the line search. P4 waits on that thread; staying out of their dir.

## P3 — device assembly (design sketch, not started)

- swe_roe_flux_jacobian_petsc.h is pure self-contained scalar math — ports
  to KOKKOS_INLINE_FUNCTION nearly mechanically (drop PetscFunctionBegin/
  PetscCall error plumbing, keep branches).
- Setup-time: per-edge and per-cell OFFSET tables into the COO values array
  (host cursor replay), device views of static geometry (l, r, wl, wr, sn,
  cn per edge; dz_dx, dz_dy per cell; owned flags folded in).
- Assembly: parallel_for internal edges -> interior region; parallel_for
  owned cells -> diagonal region; boundary edges stay HOST (O(surface)) in
  a contiguous slice of the values array, deep_copied in.
- MatSetValuesCOO with a device pointer (kokkos impls detect memtype).
- State: -dm_vec_type kokkos + VecKokkosGetDeviceView of u_local after
  DMGlobalToLocal; manning via device view of material_properties.

## Session updates (2026-08-21, later)

- **petsc-claude reconciliation (now owned by this session)**: three
  lineages — laptop main a5fbbf51c88 (33 ahead of GitLab 18bdd7fc26b, an
  ancestor), PM main 5d7d9997275 (11ef47e1f04 + 3 recovered commits).
  Verified: the two PM cache-fix commits are content-identical cherry-picks
  of laptop's 9c4fd83f3fb/d10463c1824, and PM-unique 5d7d9997275 is a
  content twin of laptop's 1f8484fdbe7 (MatFlatABSchedBuildDevice + ex322
  gates present on both). Merge commit 6e627442cdd joins the histories
  (tree == laptop main); laptop AND PM main fast-forwarded to it, so all
  three PM commits are ancestors — nothing lost by construction. GitLab
  push is fast-forward and PENDING (permission-blocked for the agent):
  `cd ~/Codes/petsc-claude && git push origin main`.
  Only source delta 5d7d9997275 -> merge: src/ksp/pc/impls/gamg/
  agg_device.kokkos.cxx (the shared opt-gcc-kokkos-cuda arch needs an
  incremental make for it; delegated as a QA job).
- **arch-perlmutter-claude-kokkos-cuda-O BUILT** on the merged tree
  (cuda 13.2, no cudss). RDycore `build-claude-gpu` built against it
  (RDY_HAVE_MAT_COO_BLOCK_INDICES=1); CPU `build-claude` rebuilt at
  054717cb on arch-perlmutter-claude-O.
- **P2 gate job 57369566** submitted ($SCRATCH/gpu-implicit/p2_ab.sbatch):
  dt=1 x20 steps, fgmres+pbjacobi rtol 1e-3, MPICH_GPU_SUPPORT_ENABLED=1;
  runs: CPU baseline n64, GPU-binary/host-mats n64 (build sanity),
  aijkokkos and baijkokkos each at n4 and n64 (-dm_vec_type kokkos);
  -ts_monitor_solution binary dumps -> compare_traj.py (n64-vs-n64 is the
  apples-to-apples trajectory pair; n4 runs differ by partition/reductions
  at Newton-tolerance level, expected).

### Fork bug found & fixed: MatShift/MatNorm on BAIJKokkos (fdaf8ca2c1a)

PM baijkokkos smoke SEGV'd at startup; reproduced on the laptop (np>=2
driver run, then a 60-line unit test). Root cause: the device-native
BAIJKokkos classes INHERITED MatShift_SeqBAIJ(), which reads the base
Mat_SeqBAIJ->nz (0 for device-built matrices), concludes the matrix is
empty, and re-preallocates it 1 block/row -- destroying the device
structure/values (serial: Jacobian degenerates to shift*I; parallel:
dangling COO struct -> SEGV). Trigger: TSComputeIJacobianDefault() =
MatScale+MatShift every Newton step of the BEULER path -- the fork's
FE/SNES pipelines never called MatShift, so it went unseen. MatNorm was
also inherited and silently returned 0 (reads empty base arrays).

Fix (petsc-claude fdaf8ca2c1a, laptop+PM): device MatShift_SeqBAIJKokkos
(in-place add on each block-row's diagonal block; SUP if a diagonal block
is missing from the pattern), MatShift_MPIBAIJKokkos (delegates to the
diag sub-block), MatNorm_SeqBAIJKokkos (frobenius from synced values;
1/inf via temporary SEQAIJ), + lifecycle gate ex337 (blocked COO cycled
with Norm/Scale/Shift/Mult vs an AIJ reference; seq, mpi, and the MPIBAIJ
host twin). Verified: ex337 3/3, ex311/322/334/335 all ok, cycle repro
np1-4 exact, RDycore ctest 14/14, adjoint_beuler full solve 40/40 at
np1/2/4 baijkokkos, PM 4-GPU baijkokkos smoke converges. NB for the QA
agent: the SHARED arch needs an incremental make to pick this up, and the
cuda-gated blocked-COO tests (ex313/314/316/319/320) + ex337 should run
on a GPU node.

P2 job 57369566 held during the PM relink and released; queued with the
fixed lib.

## P2 GATE: PASSED (2026-08-21, interactive runs; batch 57369566 redundant/queued for n4 data)

dt=1 x20 steps, n64 (4 A100s shared for device runs), fgmres+pbjacobi
rtol 1e-3, no EW. All three runs take IDENTICAL Newton paths (20 snes
conv, 717 total linear its) and identical state norms to 16 digits at
every dumped step; aijkokkos vs baijkokkos trajectories are bit-identical
(rel diff <= 2e-16). CPU-vs-device entry ORDER differs only because
arch-perlmutter-claude-O has no parmetis (different partition) — compare
norms, not raw dumps. Logs/dumps: $SCRATCH/gpu-implicit/p2i_*.log,
usol.*.bin, compare_traj.py.

| metric (s)          | CPU aij | aijkokkos | baijkokkos |
|---------------------|---------|-----------|------------|
| SNESSolve           | 116.7   | 60.6      | 29.5       |
| SNESJacobianEval    | 80.1    | 24.2      | 8.9        |
| KSPSolve            | 13.0    | 22.7      | 15.6       |
| MatMult (717)       | 10.8    | 9.1 GPU100| 8.6 GPU100 |
| PCApply (614)       | 1.31    | 0.039     | 0.019      |
| MatSetValuesCOO(103)| 3.5     | 13.4      | 4.5        |
| TSFunctionEval(125) | 20.5    | 10.1      | 3.7        |

- Device residency: MatMult/PCApply GPU %F = 100; KSPSolve GpuToCpu =
  511 transfers x 0.017 MB TOTAL (convergence scalars). Gate condition
  "no per-iteration GpuToCpu in the solve" met.
- Fork's device PBJacobi: 1.31 -> 0.019 s (~68x on the apply).
- baijkokkos beats aijkokkos 2x overall: scalar COO uploads 23 MB/assembly
  through a 9x larger index/perm scatter (logged 2.37 GB H2D total);
  blocked COO's leaner scatter cuts JacEval 24.2 -> 8.9 s.
- KSPSolve is SLOWER on device than CPU at n64 (context thrash: 16
  ranks/GPU, no MPS) — the n4 batch runs + P3/P4 are where device KSP
  perf gets its fair shot. CPU MatAssemblyEnd showed a 62 s / 23000x
  imbalance spike in this interactive session (worse than the 7.7 s
  charter number) — device runs sidestep it entirely.
- CPU JacEval 80.1 s is the P3 target: values are still host-computed
  (flux-tangent loop + DMGlobalToLocal + upload). P3 moves the fill to a
  device kernel over the precomputed COO offsets.
