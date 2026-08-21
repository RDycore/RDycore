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

## P3 GATE: PASSED (2026-08-21) -- device assembly

Implementation (commit 375b9067 + this log): swe_jacobian_kokkos.kokkos.cxx
fills the COO values buffer on device (interior-edge, boundary-edge, and
cell kernels writing at setup-precomputed offsets; math shared verbatim
with the host loop via RDY_MATH_FN); MatSetValuesCOO consumes the device
pointer. TU compiled by PETSc's .kokkos.cxx rule (nvcc_wrapper) driven
from CMake (src/swe/kokkos_compile.mk); host loop kept for non-Kokkos
types. Laptop FD gates: jacobian_global FD-limited & digit-identical to
host assembly, np 1/2, device path PetscInfo-confirmed; full BEULER
solves 40/40 np 1/2/4; ctest 14/14; host adjoint gates unchanged.

PM A/B (dt=1 x20, protocol of the P2 table; trajectory BIT-IDENTICAL to
the P2 device runs, rel diff 2.3e-16; same 20 snes / 717 lin its):

| SNESJacobianEval (103) | total  | per-assembly |
|------------------------|--------|--------------|
| CPU (P1 COO)           | 80.1 s | 0.78 s       |
| P2 baijkokkos n64      | 8.9 s  | 86 ms        |
| P3 baijkokkos n64      | 1.86 s | 18 ms        |
| P3 aijkokkos n64       | 1.53 s | 15 ms        |
| P3 baijkokkos n4       | 0.35 s | 3.4 ms       |

- The charter's A/B target (0.77 s/assembly P1 baseline): 226x faster
  per assembly at n4, 43x at n64. JacEval GPU %F = 100, ZERO CpuToGpu in
  the event (only Dirichlet ghosts + material props are staged, ~KBs).
- MatSetValuesCOO: 4.5 s (P2 host upload) -> 14 ms (device pointer).
- SNESSolve n64: 116.7 (CPU) -> 29.5 (P2) -> 22.0 s (P3). At n4: 29.1 s,
  now dominated by TSFunctionEval 21.6 s = the HOST RHS physics on 4
  ranks -- the remaining host component (stays host per charter: no CEED
  entanglement; at n64 it spreads to 3.6 s).
- Known limitation (recorded in the P3 commit): TSAdjoint with kokkos VEC
  types hangs in trajectory machinery -- adjoint stays on host types per
  the charter invariant (forward-solve-first).
- Batch job 57369566 cancelled as redundant (all gate data collected via
  interactive runs).

## P4 status: BLOCKED on the dt>=5 nonlinear robustness question

Charter prerequisite not met: the other agent's morning runs show the
FIRST Newton linear solve diverging at dt=5 (300 its, pbjacobi, basic LS)
-- a linear-solver hardness problem upstream of the line-search question.
P4's device-GAMG parameter menu waits for that CPU-side resolution.
Useful P4 prep that can proceed: device GAMG machinery validation at dt=1
(PtAP/smoother path), and the delegated cudss-on-CUDA-13 job for the
coarse solve.

### P4 prep: device GAMG machinery validation at dt=1 (n4, baijkokkos)

Config: fgmres + gamg, gmres(4)+pbjacobi smoothers, nsmooths {0,1},
aggressive_coarsening 0, coarse solve host LU (cudss pending), NO other
gamg tuning. Logs: $SCRATCH/gpu-implicit/p4prep_gamg{0,1}_n4.log.

- WORKS end-to-end: 20/20 Newton both variants, 206 total linear its
  (pbjacobi: 717 -- GAMG already cuts iterations 3.5x at dt=1).
- Device PtAP validated under per-Newton reassembly: PtAPSymbolic once
  (14 products, 0.57 s), PtAPNumeric re-run 1442x in 1.63 s; the
  Galerkin products show ~zero transfers.
- SNESSolve 41.4/38.9 s vs pbjacobi's 29.1 at n4 -- GAMG loses at dt=1
  as expected (easy problem); the machinery, not the crown, was the point.
- FINDING for P4: PCSetUp moves ~82 MB/setup EACH WAY (8.5 GB per run,
  1484 transfers) and PCApply logs GPU %F ~10 -- some hierarchy stage
  still host-resident with the BARE option set. The elasticity canon adds
  -pc_gamg_threshold 0.05 -pc_gamg_threshold_scale 0.5
  -pc_gamg_prolongator_filter 0.03 -pc_gamg_aggressive_coarsening 1
  -pc_gamg_process_eq_limit 1000; first move when P4 unblocks is to rerun
  with the canon set + chase the remaining transfers (and wire cudss for
  the coarse solve -- delegated).

### P4 prep #2 + shared-arch QA (post-cert-renewal session)

- **Shared-arch GPU QA: 66 ok / 0 fail** — the full cuda-gated fork suite
  (ex311/313/314/316/319/320/322/334/335/337) on an A100 against
  arch-perlmutter-opt-gcc-kokkos-cuda rebuilt at the reconciled main +
  fdaf8ca fix. CAVEATS for anyone running it: (a) that arch is
  CUDA-chimeric until reconfigured — needs `module load cudatoolkit/12.9`
  AND 13.2's cuda/math lib64 appended to LD_LIBRARY_PATH; (b) its
  configured python (~/cutile-venv) is gone — pass PYTHON=python3; (c) the
  fork's query_tests chokes on multi-pattern search= — one pattern per
  invocation, and use gmakefile.test directly (the top-level wrapper
  dropped my search var); (d) generated test scripts honor $PETSCMPIEXEC
  for redirecting mpiexec into an existing allocation.
- **GAMG earlier "GPU %F ~10" was a TRUNCATION artifact in my log
  extraction — KSPSolve/PCSetUp/PCApply all report GPU %F = 100.** The
  genuine P4 lead is narrower: PCSetUp moves ~66-82 MB per Newton setup
  (6.8-8.5 GB per 103-setup run) — chase in the aggregation/graph or
  coarse-op construction path.
- **SWE-vs-elasticity GAMG delta found**: `-pc_gamg_threshold 0.05` makes
  the VALUE-filtered graph asymmetric on the upwinded SWE operator and
  GAMG errors ("un-symmetric graph"); `-pc_gamg_sym_graph true` is NOT
  honored on this path (likely the device CreateGraph lacks symmetrize —
  fork follow-up). Working SWE canon: `-pc_gamg_threshold -1` (structure
  IS symmetric) + `-pc_gamg_prolongator_filter 0.03` + aggressive
  coarsening: 20/20 Newton, 206 its, PCSetUp 4.24 s/103 (vs 5.4 bare).
