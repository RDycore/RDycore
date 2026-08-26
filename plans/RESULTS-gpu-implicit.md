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

## REVIEW: the full-GPU SWE Manning conversion (with Mark, 2026-08-21)

Protocol everywhere: Turning 30 m (2.93M cells), dt=1 BEULER x20 steps,
fgmres+pbjacobi, fixed ksp_rtol 1e-3, one GPU node.

Correctness: identical Newton path (20 snes / 717 lin its) across all
seven configurations; device-format and device-assembly trajectories
bit-identical (2e-16); norms vs CPU identical to 16 digits every step
(entry order differs only by partitioner availability); device assembly
FD-limited (1.5-1.9e-8, digit-identical to host); ctest 14/14; host
adjoint gates untouched.

| SNESSolve (s)     | CPU n64 | P2 baij | P3 baij | P3 baij n4 |
|-------------------|---------|---------|---------|------------|
| total             | 116.7   | 29.5    | 22.0    | 29.1       |
| JacobianEval      | 80.1    | 8.9     | 1.86    | 0.35       |
| KSPSolve          | 13.0    | 15.6    | 13.1    | 5.6        |
| TSFunctionEval    | 20.5    | 3.7     | 3.6     | 21.6       |

- Assembly (the campaign target) is dead: 1.15 (pre-COO) -> 0.77 (P1) ->
  0.086 (P2) -> 0.018/0.0034 s per assembly (P3, n64/n4). 5.3x end-to-end.
- NOT yet on GPU, honestly: (1) the RHS -- now 74% of n4 SNESSolve; the
  next increment (device RHS in the PETSc backend, reusing the P3 tables
  + shared-math-header pattern on swe_roe_flux_petsc.h) takes n4
  SNESSolve to ~7.5 s (~15x vs CPU n64). (2) adjoint (host by charter).
  (3) GAMG coarse solve (host LU until cudss/cuda13).

Decisions from the review:
- **pbjacobi is the production PC for the working dt range.** At dt=1 the
  shifted system is strongly diagonally dominant: pbjacobi 717 its /
  KSP 5.6 s at n4, zero setup; GAMG 206 its but 4.2-5.4 s PCSetUp per run
  (PtAP every Newton) -- net ~3x slower. GAMG stays shelved with P4 until
  large-dt implicit research (SNES VI / smoothed wet-dry / ts tolerances)
  lands.
- **GPU adjoint is NOT technically out of scope -- only charter-deferred**
  (forward-solve-first). After the device RHS lands, the host adjoint is
  an estimated 70-80% of calibration gradient time (GPU forward ~7.5 s vs
  host backward ~18-20 s per 20 steps: one assembly + one transpose solve
  per backward step + trajectory I/O). Needed: TSTrajectory with device
  vecs (the observed hang -- the real blocker), transpose PBJacobi
  verification (MatMultTranspose on baijkokkos exists in the fork). Once
  trajectory works, the adjoint's assemblies get the P3 device path for
  free.

## B1 -- device RHS (session 2, 2026-08-21): laptop gates PASSED

Implementation (this commit): the PETSc-backend SWE RHS (interior flux +
boundary flux + source) runs in Kokkos kernels that REUSE the P3 Jacobian
context's geometry views (swe_jacobian_kokkos.kokkos.cxx gains SetupRHS/
ApplyFlux/ApplySource; no duplicated tables). Key pieces:

- swe_roe_flux_petsc.h converted with the RDY_MATH_FN pattern (071257a4
  style): ComputeSWERoeEigenspectrum -> void, new per-edge
  ComputeSWERoeFluxEdge + ComputeSWERiemannVelocity; the host array wrapper
  keeps its signature, so every host caller (explicit, MUSCL, HR, tracer)
  is source- and bitwise-unchanged.
- Determinism: per-edge flux kernels write an edge-indexed buffer; a
  per-owned-cell gather kernel accumulates in EXACTLY the host loops'
  order (interior edges by ascending index, then boundary edges), built as
  a CSR at setup. Device sums are bitwise identical to the host RHS.
- Boundary fluxes: raw per-edge fluxes are D2H-copied (O(surface)) into
  the per-boundary vecs each eval and accumulated as before (time-series
  semantics preserved; non-owned entries -- never read -- stay zero).
- Courant diagnostics via a MaxLoc reduction over amax*len/min(A) with
  host-side id resolution (same update rule; per-eval reset unchanged).
- Sources/material props: host seq vecs staged to device with
  PetscObjectState change tracking (upload only when forcing updates, not
  per eval). Dirichlet ghosts reuse the Jacobian's staging buffer.
- Wiring: CreateAnalyticJacobianCOO attaches SWERHSKokkosData to
  Operator.petsc when config is eligible (SWE, no sediment, non-HR, Roe,
  first-order, source explicit|ark_imex); ApplyPetscOperator dispatches to
  ApplySWEPetscOperatorsKokkos when the state Vec is a Kokkos type,
  mirroring the flux -> flux_divergence copy -> source sequence. CEED path
  untouched. `-swe_rhs_kokkos false` keeps the host RHS with device
  matrices/vecs (the A/B baseline that isolates the RHS delta).

Laptop gates (arch-macosx-gnu-rdycore-kokkos-O, host Kokkos):
- ctest adjoint|calibrate|jacobian: 14/14; swe tests minus cgns: 97/97
  (the 6 swe_roe cgns tests fail PRE-EXISTINGLY: this arch has no CGNS --
  "Unknown PetscViewer type: cgns").
- Bitwise A/B (same baijkokkos+kokkos-vec linear algebra, host RHS vs
  device RHS): adjoint_beuler (reflecting BCs), a dirichlet +
  critical-outflow variant, and adjoint_arkimex -- trajectories
  BIT-IDENTICAL (cmp) at np 1 and 2; np 4 beuler solve 40/40.
- NB an earlier aij-vs-baijkokkos comparison showed 2e-6 rel diffs -- that
  is the LINEAR ALGEBRA delta (different MatMult/dot rounding), present
  with the host RHS too; the -swe_rhs_kokkos A/B isolates the RHS proper.

PM gate (pending): dt=1 x20 Turning protocol, trajectory twin vs P3 dumps,
TSFunctionEval GPU %F = 100, no per-eval GpuToCpu of the state, n4
SNESSolve ~7.5 s expected.

## B1 PM GATE: PASSED (2026-08-21) -- device RHS at Turning 30 m

Protocol: dt=1 x20, baijkokkos + kokkos vecs, fgmres+pbjacobi rtol 1e-3,
one GPU node. Newton path IDENTICAL across all runs (20 snes conv, 614
converged-linear-iteration sum). Trajectories: n4 device-RHS vs n4
host-RHS (same partition, only the RHS source differs) max rel diff
2.5e-16; n64 device-RHS vs the P3 dump 2.4e-16 -- same ulp-level noise as
the P3-vs-P2 "bit-identical" comparison (gcc host vs nvcc device FMA
contraction; on the laptop, where one compiler builds both, the pair is
exactly bitwise). Norms twin to 16 digits: gate met.

| n4 (s)              | host RHS (P3) | device RHS (B1) |
|---------------------|---------------|-----------------|
| SNESSolve           | 28.6          | 6.94            |
| TSFunctionEval (125)| 21.6          | 0.226  (96x)    |
| KSPSolve            | 5.13          | 5.00            |
| SNESJacobianEval    | 0.37          | 0.33            |

- TSFunctionEval GPU %F = 100; CpuToGpu = 2 transfers / 23 MB TOTAL (the
  one-time material-props + external-sources staging -- the state-change
  caching works); GpuToCpu = 125 x 37 KB (boundary fluxes only). The
  host-RHS baseline moved 2.2 GB up + 4.4 GB down per run in the same
  event. "No per-eval GpuToCpu of the state": met.
- n4 SNESSolve 6.94 s beats the charter estimate (~7.5 s): 16.8x vs the
  CPU n64 116.7 s baseline. n64: 20.2 s (KSP-bound at 16 ranks/GPU as
  before; n4 is the GPU-friendly rank count).
- Runs/logs: $SCRATCH/gpu-implicit/b1i_{hostrhs_n4,devrhs_n4,devrhs_n64}.log,
  usol.b1*.bin, b1_interactive.sh.

## B2 progress (2026-08-21, same session)

1. **TSTrajectory + kokkos vec hang: GONE at the current stacks.** The
   charter repro (rdycore_adjoint adjoint_beuler.yaml -dm_vec_type kokkos,
   tight tolerances) completes in ~0.4 s on the laptop -- as do
   -rhs_jac_mat_type baijkokkos and full -dm_mat_type baijkokkos configs,
   np 1-4. Most plausibly fixed by fdaf8ca (MatShift/MatNorm on device
   BAIJKokkos: the backward sweep runs TSComputeIJacobianDefault's
   MatScale+MatShift every step, and pre-fix MatShift destroyed the device
   matrix). Not re-bisected; the symptom is unreproducible.
2. **Adjoint FD gates with device forward+backward PASS at np 1 and 2**
   (full-kokkos: dJ/du0 rel 1.25e-8 / 1.35e-8, Manning aggregate within
   gate). Fixed a DRIVER FD-check bug exposed at np>1: every rank called
   VecSetValue(ADD_VALUES) for the sampled-dof perturbation, scaling the
   FD slope by nproc (fd was exactly 2x at np2 / 4x at np4 while the
   ADJOINT gradients were identical to np1 all along). Perturbation now
   applied from rank 0 only. ctest 14/14 after the fix.
3. **Transpose PBJacobi verification**: end-to-end covered by the FD gates
   above (KSPSolveTranspose + PCApplyTranspose on baijkokkos in the
   backward sweep); fork gate ex337 EXTENDED with per-cycle
   MatMultTranspose and KSPSolveTranspose(preonly+pbjacobi) checks vs the
   AIJ reference (petsc-claude d1f202c9e6b laptop / e58c2dabe05 PM, 3/3 on
   the laptop; GPU-node run pending with the next fork QA batch).
4. The adjoint's per-step Jacobian assemblies take the P3 device path
   type-driven, as predicted (device assembly PetscInfo confirmed in the
   full-kokkos adjoint runs).

Remaining for B2: PM-scale adjoint timing/residency (Turning-protocol
backward sweep with device types vs host types).

## B2 PM GATE: PASSED (2026-08-21) -- GPU adjoint at Turning 30 m

Full gradient runs (rdycore_adjoint beuler_dt1.yaml: truth forward +
perturbed forward + one TSAdjoint backward sweep), dt=1 x20,
fgmres+pbjacobi, one GPU node, -adjoint_fd_samples 0 (FD correctness ran
on the laptop at tight tolerances; see B2 progress above).

Correctness at scale: with the SAME binary and partition (GPU binary,
n64, parmetis) at ksp_rtol 1e-8 / snes_rtol 1e-10, host-types vs
device-types (kokkos vec + baijkokkos) gradients are IDENTICAL to all
printed digits: J = 143.434, |dJ/du0| = 1574->1564.37 both, sum(dJ/dn) =
-446848 both. (An apparent 0.15%/2.8% host-vs-device discrepancy was the
CPU binary's DIFFERENT PARTITION -- arch-perlmutter-claude-O has no
parmetis -- combined with the driver's partition-dependent observation
set; it persists at tight tolerance for that reason and vanishes under a
matched partition. Each configuration's gradient is tolerance-stable from
rtol 1e-3 to 1e-8 at <2e-4 relative.)

Performance (rtol 1e-3 production protocol, logs b2i_*.log):

| event (s)            | host n64 | device n4 | device n64 |
|----------------------|----------|-----------|------------|
| total wall           | 271.9    | 107.1     | 184.2      |
| TSStep (2 forwards)  | 217.8    | 14.4      | 40.3       |
| TSAdjointStep (20)   | 18.1     | 3.0       | 3.8        |
| TSTrajectorySet (42) | 8.9      | 8.2       | 9.9        |
| SNESJacobianEval(246)| 160.6    | 0.78      | --         |
| MatMultTranspose(151)| 2.15     | 0.29      | --         |

- Gradient loop (forwards + trajectory + backward) 245.9 -> 26.5 s at n4
  (~9.3x); the backward sweep runs at GPU %F = 99 with the P3 device
  assembly (246 assemblies in 0.78 s) and device transpose solves.
- Remaining device-side hotspot: TSTrajectorySet ~8 s/window (a third of
  the n4 gradient loop) -- the memory-trajectory staging (78 GpuToCpu,
  1.37 GB, plus its MPI traffic); ~constant across host/device and rank
  counts. Follow-up lead: device-resident or async trajectory staging.
- The wall-total gap beyond the loop (~80 s) is one-time setup (mesh
  distribution etc.), amortized across calibration iterations.
- Fork gate ex337 (now with MatMultTranspose + KSPSolveTranspose/
  PCApplyTranspose-pbjacobi cycles): All cycles OK on A100, seq and mpi.

Session-2 charter status: B1 PASSED, B2 PASSED (all charter work items:
hang gone, transpose verified + fork-gated, device assemblies engaged;
gates: FD np1/2, calibrate ctest, scale twin). Full laptop ctest: green
except PRE-EXISTING failures (6x swe_roe cgns: arch lacks CGNS;
amr_c_np_3_basic: SEGV reproduced at parent commit 670062ee -- AMR is
out of charter scope).

### B2 follow-up: the TSTrajectorySet lead is CLOSED by -ts_trajectory_type memory

The 8 s/window trajectory cost was the DEFAULT disk ("basic") trajectory:
rank-0-gathered binary writes of every checkpoint (4.5 GB MPI + 1.37 GB
D2H per gradient at n4) and reads on the backward sweep. With
`-ts_trajectory_type memory` the checkpoints are Vec copies -- which for
kokkos vecs live ON DEVICE -- and the driver already supports it
(ResetTrajectory's stack handling exists for exactly this):

- TSTrajectorySet 8.21 s -> 0.084 s (98x); TSTrajectoryGet 0.92 s ->
  0.5 ms; zero MPI, zero D2H in either event.
- Gradient identical to the disk-trajectory run to every printed digit.
- Laptop FD gates with memory trajectory, full-kokkos, np 1/2: 1.25e-8 /
  1.35e-8 (same values as disk).
- Device n4 gradient loop: 26.5 -> 17.2 s (TSStep 14.1 + TSAdjointStep
  3.0 + trajectory 0.09) = ~14.3x vs the 245.9 s CPU-n64 loop.
  Log: $SCRATCH/gpu-implicit/b2m_dev_n4.log.

NOT made the default: the memory trajectory holds every step of the
window (no disk spill), which is fine for 20-step gate windows but not
for long Harvey windows on hosts; use the option on the GPU path (or
revolve-style checkpointing when windows outgrow device memory).

### KSP round: VecMDot was 96% of KSPSolve -- a KokkosBlas::gemv fallback, fixed in the fork

Rank sweep first (gradient, memory trajectory): KSPSolve 37.3 s (n1) ->
20.3 (n2) -> 10.3 (n4): clean throughput scaling, so per-iteration cost
was bandwidth-shaped, not sync latency. pipefgmres at n4: 12.2 s (no
win). Inside KSPSolve at n4, VecMDot was 9.9 of 10.3 s at 0.5 GF/s --
~1000x off roofline for the FGMRES orthogonalization.

Micro-benchmark (bench_mdot.c, $SCRATCH/gpu-implicit, single A100,
n = 2.2M, nv = 16): the DEFAULT VecMDot path (-vec_mdot_use_gemv true,
pooled work vectors -> KokkosBlas::gemv) runs at 36 GB/s; the
parallel_reduce path (-vec_mdot_use_gemv 0) at 1050 GB/s; per-vector
fallbacks at ~1000 GB/s. Root cause: PETSc's --download-kokkos-kernels
builds with vendor TPLs OFF (KokkosKernels_ENABLE_TPL_CUBLAS=OFF), and
KokkosBlas's NATIVE tall-skinny transpose gemv is ~30x off roofline.

Fork fix (petsc-claude 90d0d174a34 laptop / ddac7cfaecd PM): default
-vec_mdot_use_gemv to FALSE when the device exec space has no vendor
BLAS TPL behind gemv (CUDA without cuBLAS TPL, HIP without rocBLAS);
option still overrides. Host builds unchanged. Alternative long-term
fix for the next full reconfigure: enable the cuBLAS TPL in the
kokkos-kernels download.

Effect at Turning n4 (dt=1 x20, fgmres+pbjacobi rtol 1e-3):
- forward SNESSolve 6.94 -> 2.99 s (VecMDot 9.9 -> 0.40 s over the
  gradient's 206 solves; KSPSolve 10.3 -> 1.08 s per-forward-pair
  share). Now 39x vs the CPU n64 forward baseline.
- gradient loop (2 forwards + trajectory + backward): 17.2 -> 8.1 s
  (TSStep 5.9-6.3 + TSAdjointStep 2.1 + trajectory 0.11) = ~30x vs the
  245.9 s CPU-n64 gradient. Gradient values identical throughout.
- Verified the flipped DEFAULT (no flag): b2v_grad_default.log.

Solver variants tested at n4 (same protocol, gemv fix on):
- lgmres+pbjacobi: 625 its, SNESSolve 3.89 s -- no win over fgmres.
- lgmres + gamg (-pc_gamg_agg_nsmooths 0 -mg_levels_pc_type pbjacobi
  -pc_gamg_threshold -1): 1084 its, SNESSolve 32.4 s (PCSetUp 10.2,
  PCApply 19.3) -- unsmoothed aggregation with pbjacobi level smoothing
  is far too weak here; pbjacobi stays the production PC (per the
  standing decision). Average its/solve is ~6, so nested KSP
  preconditioning (-pc_type ksp) has no role at dt=1 either.

Current n4 SNESSolve profile after all fixes: KSP 1.1 s + PCSetUp
(pbjacobi block inversion, 103x) 1.2 s + JacEval 0.33 + RHS 0.23 --
roughly balanced; further gains are diminishing-returns territory.

## Calibration validation (B1+B2 integration; 2026-08-21 evening)

The full TAO calibration loop (gauge-twin mode) runs on device types --
the first exercise of the Manning-update path (RDySetDomainManningsN ->
material-properties state bump -> device re-upload in RHS and Jacobian)
and of repeated memory-trajectory reset/reuse across TAO iterations.

Laptop (adjoint_beuler.yaml + -adjoint_calibrate_gauges
-adjoint_gauges_twin, 60 TAO its, np 1 and 2): host and device runs
converge IDENTICALLY -- J reduction 9.02x and two-zone recovery rel err
3.071e-01 to all printed digits; J endpoints agree to ~5 digits (the
benign aij-vs-baijkokkos linear-algebra rounding).

Config foot-gun found and hardened: the gauges CTEST config
(adjoint_dam_break.yaml) uses numerics.jacobian: fd, and FD COLORING on a
device-native matrix type SEGVs inside MatGetRowIJ_SeqBAIJ (coloring
reads host CSR arrays that device matrices only build at assembly -- the
known P2-era landmine, reached via -dm_mat_type this time).
RegisterSWERHSJacobian now rejects jacobian: fd with device matrix types
with an explanatory error. Device calibration uses the implicit config
(beuler + analytic), which is the production path anyway.

### Turning-scale calibration (gauge twin) + an honest-baseline correction

PM runs (beuler_dt1.yaml + gauges twin: 39-step windows, 13 obs times,
418,076 twin gauges, 2,926,532 per-cell Manning parameters, BLMVM,
fgmres+pbjacobi rtol 1e-3, memory trajectory):

- Device n4 (kokkos types): 20 TAO its, J 1.283e6 -> 2.315e4 (55.4x),
  TaoSolve 106.2 s = 5.3 s per TAO iteration. Total wall 414.7 s
  (~60 s of that is one-time DMPlexDistribute).
- Host types n64 (SAME binary and parmetis partition): 5 TAO its,
  TaoSolve 165.5 s = 33.1 s per iteration. J-trace IDENTICAL to the
  device run to every printed digit through the common iterations
  (gauge sets are natural-ordered, hence partition-independent).
- Single-node apples-to-apples (4 A100s vs 64 cores of the same node):
  ~6.2x per TAO iteration; recovery converging identically.

CORRECTION to the morning's baselines: the "CPU n64" figures used the
CPU-arch binary, which has NO PARMETIS -- its native partition inflates
host times ~4x (e.g. TSAdjointStep 0.90 s/step there vs 0.247 s/step for
host types on the parmetis partition). The honest single-node
host-vs-device ratios at Turning are ~6-8x (gradient ~50 s -> 8.1 s per
20-step window), not the ~30x implied by the no-parmetis baseline. The
device-side numbers themselves are unaffected.

Device per-iteration profile (TaoSolve 5.3 s/it): forward SNESSolve
~3.3 s + adjoint sweep ~2.3 s per window-pair share; inside them PCSetUp
(pbjacobi block inversion, 2473 calls) 41 s total now rivals KSPSolve
29.7 s -- a possible next squeeze, along with host-H observation
applications. Diminishing returns.

### 1-simulated-hour gradient (real-window check, device n4)

beuler_dt1_1hr.yaml (Turning, spun-up Harvey IC draining through the
critical-outflow outlet -- NO rain forcing in this config; storm-forced
fidelity needs the calibration project's forcing files), 3600 steps at
dt=1, FD-twin driver mode, -ts_trajectory_type memory
-ts_trajectory_max_cps_ram 400 (revolve checkpointing), fgmres+pbjacobi
rtol 1e-3. Log: $SCRATCH/gpu-implicit/b4_1hr_dev_n4.log.

- COMPLETED in 27.3 min wall (includes the driver's extra truth
  forward): forward 0.114 s/step -- Newton holds ~4 its/step across the
  full hour -- and the checkpointed backward works: 3199 revolve
  recompute steps (~0.89 extra forwards, near optimal for 400 cps),
  TSAdjointStep 0.101 s/step, gradient finite and sane.
- Per-gradient (1 forward + checkpointed backward) at a 1-hr window:
  ~19 min at n4. Measured-basis extrapolation per TAO iteration:
  3 hr ~= 57 min, 6 hr ~= 1.9 hr (n4). A 20-iteration calibration on a
  6-hr window is ~1.6 GPU-node-days -- the economic case for multi-node
  scaling, window design, and the dt>=5 research (5x on everything).
- Follow-up lead: the revolve checkpoints stage through HOST memory
  (hundreds of GB of PCIe across the sweep -- visible as GpuToCpu in
  TSStep/TSAdjointStep/TSTrajectoryGet). Device-resident checkpoint
  storage is the next trajectory squeeze if long-window gradients become
  the daily workload.

## Rainfall wiring for the adjoint driver (campaign prep, 2026-08-21 night)

The forcing subsystem (-raster_rain_dir + -raster_rain_start_date, hourly
PETSc-binary rasters, mm/hr -> m/s, raster->mesh map) is only invoked by
RDyAdvance, which the adjoint driver bypasses -- and the datasets stream
strictly forward in time while checkpoint recomputes REWIND time. Driver
fix (adjoint_test.c): preload the window's hourly per-cell rain once at
setup (monotone RDyApplyForcing calls), then a TSPreStage hook -- which
fires inside every TSStep, INCLUDING trajectory ReCompute replays --
swaps the active hour in as a pure function of stage time (hour h covers
(h*3600,(h+1)*3600], matching RDyAdvance's coupling semantics; region
id 1 as in RDyApplyForcing; source is state-independent so no adjoint
terms arise).

Laptop validation (single-region BEULER dam-break variant
rain_beuler_test.yaml, synthetic 2-hour raster series, 7200 steps at
dt=1, revolve max_cps_ram 50 => recomputes cross the hour switch):
- adjoint-vs-FD dJ/du0 gates: 2.4e-7..8.9e-7 (gate 1e-5), host and
  device types, np 1 and 2 -- the forced trajectory is reproduced
  exactly through hour switches and replays.
- device-RHS vs host-RHS with hourly source re-uploads: J, |dJ/du0|,
  sum(dJ/dn) identical to every printed digit (the source state-cache
  invalidation handles time-varying rain).
- dJ/dn on this test is genuinely ~0 (motionless by hour 2): added a
  degenerate-gradient guard to the driver's Manning FD gate (absolute
  floor fd_tol*max(J,1) when the relative test is noise-vs-noise).
- Known limitation (pre-existing): RDyApplyForcing hardcodes region
  id 1 = whole domain; multi-region configs (e.g. the planar dam tests)
  error cleanly. Turning's single "domain" region is fine.

Data: full Harvey hourly MRMS (+ mswep/nldas/daymet) rasters at
/global/cfs/cdirs/m4267/shared/data/harvey/spatially-distributed-rainfall/
mm-per-hr/*/bin (Aug 24-30); repo tree carries a 2-hour sample.

### Turning 1-hr FORCED gradient: the campaign pipeline is complete

b5_1hr_rain_n4.log: beuler_dt1_1hr.yaml + real MRMS hourly rain
(-raster_rain_dir .../mm-per-hr/mrms/bin -raster_rain_start_date
2017,8,26,0,0), device n4, memory trajectory with revolve (400 cps).
Completed in 27.7 min wall -- same as the unforced run (rain adds no
measurable cost). J(forced) = 3802.79 vs J(unforced) = 5330.22 (the rain
demonstrably reshapes the trajectory); gradient finite
(|dJ/du0| = 3.14e8, sum(dJ/dn) = -1.07e10).

End-to-end status for the Wednesday campaign: real forcing + Turning
30 m + device forward + checkpointed device adjoint all compose, at
~19 min per gradient per simulated hour of window at n4. Next campaign
steps: (1) dt-verification (BEULER dt=1 vs ARK dt=0.25 gauge
hydrographs on a forced window -- the main rdycore driver applies rain
natively via RDyAdvance, so the ARK reference needs no new wiring);
(2) window placement on the rising limb per the observation-time
sensitivity study (confirm at 30 m); (3) the scientists' answers on
trusted gauges/window (dam-release influence) and WSE accuracy target.

## Session 3 (2026-08-22): optimization pass -- pbjacobi setup, device Jacp/H, staging caches

Scope: the measured leads from the session-2 profiles, in value order.
Every step gated on ctest adjoint|calibrate|jacobian 14/14 + laptop
bitwise A/B (device config with vs without each optimization) + PM
A/B under the standing Turning protocols. RDycore commits 827ec4fb..
4e854188; fork commit 5b4f3d82d1e (patch applied on PM via git am).

### Lead 1 (fork 5b4f3d82d1e): pbjacobi PCSetUp device-pointer handoff

Root cause of the 41.1 s / 2473 setups: PCSetUp_PBJacobi_Kokkos went
through the host MatInvertBlockDiagonal() contract -- the BAIJKokkos
implementation inverted ON DEVICE but copied the finished inverses D2H
into a->idiag, and the PC pushed the same bytes H2D again: 2 x 52.7
MB/rank/setup of PCIe plus a fresh 53 MB device allocation per call.
Fix: Mat_SeqBAIJKokkos caches the inverse in a device view (idiag_d,
state-keyed, allocation reused); new composed
"MatInvertBlockDiagonalDevice_C" (mpi delegates to the diag block)
returns the device pointer; PCSetUp_PBJacobi_Kokkos aliases it with
zero host traffic. `-pc_pbjacobi_invert_device false` = the old path
(A/B baseline). Bit-identical by construction (same inversion kernel).

### Leads 3+4+6 (RDycore): device Jacp, device H, staging caches

- **The "revolve checkpoints stage through host" lead was a
  MISATTRIBUTION**: the b4_1hr log's per-event transfers show
  TSTrajectorySet/Get move ~ZERO bytes (checkpoint vecs are device-
  resident kokkos duplicates). The TBs of PCIe inside
  TSStep/TSAdjointStep/TSTrajectoryGet were the NESTED pbjacobi
  PCSetUps (45,445 calls, 2.39 TB EACH WAY per rank, 542 s = 32% of
  the 1-hr wall) + the host Jacp path + unlogged mp staging. Nothing
  to fix in the trajectory itself.
- **Device parameter Jacobian dF/dn (ced890f2)**: SWERHSJacobianP ran
  on host every backward step (full-state D2H, MatSetValue loop into
  host AIJ, and Jacp^T applies pulling lambda D2H -- 17.6 MB x 3600
  steps = 63 GB per 1-hr gradient). Device-assembly configs now get a
  COO-preallocated MATAIJKOKKOS Jacp filled by a device kernel
  (SWEKokkosJacobianP); assembly and MatMultTranspose(Jacp) stay on
  device (0 transfers in the o3 profile).
- **Observation operator H follows the state vec type (ebca3206)**:
  aijkokkos on device runs; MatCreateVecs hands back kokkos obs-space
  vecs, so per-observation MatMult/MatMultTranspose stay on device.
- **Material-props staging unified + state-keyed (827ec4fb)**: the
  Jacobian assembly deep_copied mp EVERY assembly (unlogged; ~530 GB
  over the 1-hr gradient). Now one shared state-keyed cache
  (StageTracked) serves Jacobian, RHS source, and Jacp. Read-only
  VecGetArray uses of material_properties converted to VecGetArrayRead
  (write-mode gets bumped the state and would have retired the cache).
- **src_inst snapshot state-keyed + Dirichlet staging skip (2fe34b5e)**:
  the per-RHS-eval VecCopy(external_sources -> src_inst) (~920 GB of
  host memcpy per 1-hr gradient) now runs only when the forcing vec's
  state changes; the 3*n_bedges Dirichlet upload is skipped when the
  config has no Dirichlet boundary (has_dirichlet, computed at setup).

### PM A/B (Turning 30 m, n4, standing protocols; logs o1..o6, opt_interactive.sh)

Forward dt=1 x20 (vs `-pc_pbjacobi_invert_device false` -- bitwise
pair, 20/20 Newton, 614 lin its both):

| n4 forward (s)  | invert off | optimized |
|-----------------|-----------|-----------|
| SNESSolve       | 2.59      | 1.76      |
| PCSetUp (103)   | 0.896     | 0.086     |
| PCSetUp PCIe    | 5.4 GB x2 | 0         |

Trajectories o1-vs-o2 cmp-BITWISE IDENTICAL. (NB comparisons against
the STORED usol.b1.n4.bin now differ in late digits: that dump predates
the fork's -vec_mdot_use_gemv default flip -- different FGMRES
orthogonalization rounding. In-session A/B pairs are the valid check.)

Gradient (20-step window, memory trajectory; J = 143.152, |dJ/du0| =
1555.3, sum(dJ/dn) = -445566 -- IDENTICAL to b2v_grad_default to every
printed digit, and identical between invert on/off):

| n4 gradient (s)   | b2v (recorded) | optimized |
|-------------------|----------------|-----------|
| TSStep (2 fwd)    | 6.26           | 3.37      |
| TSAdjointStep (20)| 2.08           | 0.31      |
| PCSetUp (226)     | 2.99           | 0.19      |
| KSPSolve (206)    | 2.39           | 2.01      |
| gradient loop     | ~8.4           | ~3.8 (2.2x)|

Calibration (gauges twin, 20 TAO its; J-trace identical: 1.283391e6 ->
2.315231e4, 55.43x):

| n4 calibration (s)  | b3cal (recorded) | optimized |
|---------------------|------------------|-----------|
| TaoSolve (20 its)   | 106.2 (5.3/it)   | 34.6 (1.73/it) |
| PCSetUp (2473)      | 41.1             | 2.06      |
| TSAdjointStep (440) | 45.8             | 5.81      |
| TSStep (460)        | 66.2             | 30.7      |
| KSPSolve (2033)     | 29.7             | 18.3      |

Single-node honest device-vs-host is now ~19x per TAO iteration
(33.1 s host-types n64 vs 1.73 s device n4).

### Foot-gun found on PM: TAO/LMVM aborts on CUDA device solution vecs

With the device Jacp, MatCreateVecs(rhs_jac_p) returns kokkos vecs;
duplicating the TAO solution vec from mu put BLMVM's LMVM dense
internals on device, where MatDenseGetColumnVecWrite hits a Kokkos
DualView "concurrent modification" abort inside MatLMVMUpdate (o5;
invisible on the laptop -- host-space Kokkos has no separate mirror).
Fix 4e854188: TAO-side vecs (solution/bounds/prior/gradient) are
explicit host-layout twins; the adjoint keeps its device mu. J-trace
unchanged. FORK FOLLOW-UP LEAD: fix MatLMVM/densekokkos on CUDA.

### Style pass (Mark review): PetscObjectTypeCompare everywhere

Type-name strstr/strcmp sniffing replaced by PetscObjectTypeCompareAny
against canonical type constants at: the device-assembly gate, the
device-RHS dispatch, the FD-coloring guard, the blocked-COO opt-in,
the driver's observation-matrix choice and trajectory-type check
(de180e3c, 8cba778c). Simplify pass extracted the shared one-pass
Dirichlet ghost gather (ddc9b618). MatSeqAIJGetCSRAndMemType noted as
the sharper memtype query where a SEQ AIJ-family matrix is in hand
(not applicable at these MPI/BAIJ/pre-assembly sites).

### 1-hr revolve gradient re-verified (o6_1hr_opt_n4.log)

b4 protocol exactly (3600 steps, revolve max_cps_ram 400, unforced):
gradient IDENTICAL to b4 to every printed digit (J = 5330.22,
|dJ/du0| = 9.32912e10, sum(dJ/dn) = -3.25331e12); wall 1638 -> 808 s
(2.03x). TSStep (10399 incl. recomputes) 654 s (62.9 ms/step),
TSAdjointStep 53.4 s (14.8 ms/step, was 101), PCSetUp 542 -> 37.7 s,
KSPSolve 437 -> 401 s (now the dominant term -- the next lead if one
is ever needed). Per-gradient at a 1-hr window: ~19 -> ~9.4 min at n4;
6-hr window ~= 56 min/TAO-it; the campaign's 6-hr 20-iteration
calibration drops from ~1.6 to ~0.8 GPU-node-days.

### Deferred

- DMPlexDistribute caching (lead 5): the mesh pipeline (options-driven
  distribute -> refine -> overlap -> natural SF) would need
  topology+distribution+natural-order save/load; the natural SF is
  load-bearing for the gauge observation set (the known partition-
  dependence trap). 60-90 s is one-time per job, amortized over a
  calibration -- poor risk/benefit days before the campaign.
- Fork GitLab push (Mark): main now also carries 5b4f3d82d1e.

### KSP solver sweep (2026-08-22, with Mark; Mark's interactive node, forward dt=1 x20 n4)

The solve under the microscope (logs/ksp_look_n4.log in the repo, full
-ksp_view + -ksp_monitor): fgmres(30)+pbjacobi bs=3, right PC,
unpreconditioned norm, rtol 1e-3 -- 81 of 103 solves take exactly 6
iterations at a flat ~3.2x/iteration; VecMDot is ~40% of KSPSolve even
post-gemv-fix (inherent to GMRES at 6 its). Variants (SNESSolve /
KSPSolve seconds; all 20/20 Newton unless noted):

| solver                | rtol | lin its | KSP    | SNES  |
|-----------------------|------|---------|--------|-------|
| fgmres (baseline)     | 1e-3 | 614     | 1.02   | 1.76  |
| gmres right           | 1e-3 | 614     | 1.00   | 1.66  |
| gmres right           | 1e-2 | 457     | 0.72   | 1.43  |
| gmres right           | 1e-1 | DIVERGED step 1 (Newton 31 its -> NaN) |
| gmres left            | 1e-3 | 625     | 1.02   | 1.68  |
| gmres left            | 1e-2 | 476     | 0.75   | 1.44  |
| bicg                  | 1e-3 | 721     | 2.13   | 2.76  |
| bcgs                  | 1e-3 | 365     | 1.13   | 1.77  |
| bcgs                  | 1e-2 | 290     | 0.92   | 1.60  |

- fgmres's flexibility is unused (pbjacobi is stationary): gmres+right
  is bit-equivalent at 1e-3 (identical 614-it path, traj 2.3e-16) and
  ~5% faster -- a free swap.
- rtol 1e-2: 19% faster forward SNESSolve, trajectory 1.2e-9 from
  baseline, gradient moves 0.1-0.55% (|dJ/du0| 1555.3->1556.8,
  sum(dJ/dn) -445566->-448015): likely fine for BLMVM but ~25x the
  1e-3->1e-8 stability band -- run the laptop FD gates at 1e-2 before
  making it the campaign default. rtol 1e-1 is unusable.
- Left PC: within noise of right, slightly more its (predicted:
  P^-1 ~ well-scaled here, only the test norm changes). bicg 2x slower
  (2 matvecs/it incl. transpose, MORE its); bcgs no win over gmres.
- RECOMMENDATION: -ksp_type gmres -ksp_pc_side right (rtol 1e-3 now;
  1e-2 pending gradient FD gates). PM logs: ksp_*_n4.log,
  grad_gmres_rtol*_n4.log in $SCRATCH/gpu-implicit.

### rtol 1e-2 validation + rain-forced rerun (2026-08-22, later; Mark's node)

- FD gates at ksp rtol 1e-2 (laptop, tight snes): adjoint-vs-FD
  4.5e-3 / 6.2e-4 at np1/2 -- formally FAILS the 1e-5 gate, as expected
  (gradient error is inner-solve-tolerance-limited; the gate is a
  tight-tolerance instrument). The decision-level tests all pass:
  laptop gauge-twin calibration at 1e-2 indistinguishable from 1e-3
  (9.02x, J to 5-6 digits, same 30 its); Turning 20-it calibration at
  gmres+right rtol 1e-2: J 1.283391e6 -> 2.313797e4 (55.47x vs 55.43x
  at 1e-3), TaoSolve 34.6 -> 31.5 s (KSPSolve 18.3 -> 14.5).
  o7_cal_gmres_rtol2_n4.log. Campaign recommendation: gmres+right
  rtol 1e-2, one 1e-3 spot-check for the record.
- Rain-forced 1-hr gradient RERUN on the optimized stack
  (o8_1hr_rain_opt_n4.log, b5 protocol): J = 3802.79, |dJ/du0| =
  3.14266e8, sum(dJ/dn) = -1.07497e10 -- the recorded b5 values
  exactly; wall 27.7 -> 13.7 min (2.02x). Rain x optimizations
  validated; the campaign pipeline is confirmed end-to-end on the
  final stack.
- Status vs campaign: real-gauge observations still UNUSED (all
  Turning calibrations are twins) -- that is the campaign itself.
  Plan for the Wednesday meeting: plans/campaign-wednesday.md
  (supersedes campaign-tonight.md, whose no-revolve premise is dead).

### FINDING: drag discontinuity at tiny_h blocks NLCD-prior implicit runs (2026-08-22 night)

The Turning class-mode twin (NLCD prior applied as the Manning field,
mean n = 0.1005, max 0.16) fails DIVERGED_NONLINEAR_SOLVE within the
first few dt=1 BEULER steps (logs o9*-o13* in $SCRATCH/gpu-implicit).
Diagnosis (o9c_diag.log, snes+linesearch monitors): Newton descends
cleanly 903 -> 1.26e-3 in ~13 its, then the residual FREEZES; at the
stall a lambda = 1e-13 step jumps the residual 8x (1.26e-3 -> 1.01e-2)
-- the residual is DISCONTINUOUS at the iterate. Cause: the drag term
g n^2 h^(-7/3) q|q| is gated by h >= tiny_h with tiny_h = 1e-7 and a
PLAIN q/h velocity (h_anuga only regularizes the flux/primitive
velocities, NOT the source kernel), so a cell crossing the cutoff with
retained momentum switches an enormous drag contribution on/off
discontinuously. NLCD-scale n^2 raised the jump amplitude ~100x above
where the mild-field runs left it (invisible below solver tolerance).
Confirmed NOT fixable by: rtol (1e-2/1e-3), snes_rtol 1e-5, bt/basic
damped 0.7/newtontr, h_anuga 0.001/0.01 alone (moves the failing step
only), tiny_h 1e-4 (WORSE - more cells at a fatter threshold), dt=0.5
(fails step 1; the floor scales oddly with dt). The gauge-twin
calibrations (n0 0.03, two-zone 0.03/0.06) never see it -- amplitude
below tolerance. Related to but distinct from the other agent's dt>=5
linear-solve hardness.

DECISION (Mark): implement fix (2) -- regularize the drag itself with
the ANUGA velocity (drag -> 0 smoothly as h -> 0), pending scientist
OK Wednesday (the paper's open-questions section already asks exactly
this). NEXT-SESSION TASK SPEC:
- Touch points (keep host/device bitwise-twinned, RDY_MATH_FN style):
  ApplySourceExplicit host loop (src/swe/swe_petsc.c), the device
  source kernel swejk_rhs_source (swe_jacobian_kokkos.kokkos.cxx), the
  source Jacobian SWESourceJacobian (swe_roe_flux_jacobian_petsc.h,
  shared host+device), SWERHSJacobianP host loop + SWEKokkosJacobianP
  device kernel (dS/dn changes), SWEIJacobianPFriction +
  SWEIFunctionFriction (ARK-IMEX implicit part), and the CEED source
  Q-functions if parity there matters (charter: CEED untouched -- gate
  the change to the PETSc path or coordinate).
- Semantics: with h_anuga_reg_parameter = 0 the new code must be
  BITWISE identical to today (the regularized velocity reduces to q/h)
  -- that is the A/B gate. With h_anuga > 0 re-run the FD gate suite
  (jacobian + adjoint + calibrate ctests pass unchanged since configs
  default h_anuga 0; add a laptop FD-gate run WITH h_anuga > 0 at
  tight tolerances -- the derivative code must differentiate the
  regularization too).
- Then Turning class twin with h_anuga ~ 0.001-0.01 (sweep): expect
  the stall floor to collapse; pick the smallest value that converges
  cleanly and put it in the campaign configs + Wednesday agenda.
- Config plumbing already exists (physics.flow.h_anuga_reg_parameter).

### ANUGA-regularized drag LANDED (2026-08-22 night; RDycore 3110ac50 laptop / 0d9e0d0d PM)

Implementation exactly per the spec above. ComputeSWEManningDrag (new,
swe_roe_flux_petsc.h) is the shared residual drag -- host
ApplySourceExplicit, device swejk_rhs_source, and the ARK-IMEX
SWEIFunctionFriction all call it; SWEFrictionJacobian/SWESourceJacobian
take h_anuga and carry the exact regularization derivatives (host +
device assembly + ARK-IMEX IJacobian); SWEFrictionDN (new) is the
shared dS_fric/dn for SWERHSJacobianP (host), SWEKokkosJacobianP
(device), and SWEIJacobianPFriction. With h_anuga = 0 every touched
path keeps its historical operation sequence VERBATIM (branch on
h_anuga > 0). CEED, semi_implicit, and XQ2018 paths untouched.

Laptop gates (both PASSED):
1. h_anuga = 0 bitwise: ctest adjoint|calibrate|jacobian 14/14;
   adjoint_beuler AND adjoint_arkimex kokkos-type trajectory dumps
   cmp-BITWISE IDENTICAL to a pre-change build at np 1/2, FD-gate log
   lines (dJ/du0, dJ/dn) identical to every digit. (Full ctest: only
   the 6 pre-existing cgns failures + amr_c_np_3_basic, which SEGVs
   identically on the pre-change build -- verified, not ours.)
2. h_anuga > 0 FD gates: new unit twins in test_swe_jacobian (source
   Jacobian across h >> ha / h ~ ha / h < ha / h ~ tiny_h at n = 0.1;
   SWEFrictionDN both branches), a new global FD-coloring-vs-analytic
   twin pair (swe_jacobian_global_anuga_*.yaml: h_anuga = 5 vs h = 10,
   n = 0.1) at 1.7e-8 (gate 1e-6) np 1/2, and driver FD gates
   (h_anuga = 4, n = 0.1): dJ/du0 1.2e-8..4.6e-8, dJ/dn 2.7e-7..5.9e-6
   (gate 1e-5) for {beuler, arkimex} x {host, kokkos types} x np {1,2}.
   J and dJ/dn confirmed to genuinely move under the regularization
   (dJ/dn changes 3.4x at h_anuga = 4), so the gates are not vacuous.

### FINDING (PM): the SECOND discontinuity -- critical-outflow uperp switch (2026-08-22 night)

With the drag fix in, the Turning class twin's early steps now converge
honestly (FNORM_RELATIVE in 5-12 its, where pre-fix they only escaped
by SNORM with a stalled residual) -- but EVERY h_anuga in {0.001, 0.003,
0.01} still died at forward step 4, Newton crawling at lambda = 0.1
(0.9x contraction/it) to a floor of 1.158e-3 and then line-search
collapse with the residual JUMPING 1.158e-3 -> 1.019e-2 at
lambda = 1e-13 (o14_*, o15_diag_anuga01.log). That jump target is
numerically the SAME as the pre-fix diagnosis (1.26e-3 -> 1.01e-2, o9c)
-- same discontinuity, unaffected by the drag regularization. Root
cause CONFIRMED by BC swap: with the outlet's critical-outflow BC
replaced by reflecting (diagnostic only), all 20 forward steps converge
cleanly at the NLCD prior (o16_reflect_anuga01.log). The
CONDITION_CRITICAL_OUTFLOW branch zeroes BOTH states when uperp < 0
(wall) and otherwise imposes the critical ghost -- at the uperp = 0
crossing the flux jumps by the full wet-onto-dry Roe flux, O(g h^2/2)
~ 1e-2 in residual norm. The drag-gate discontinuity was real (it
gated steps 1-3) but the o9 stall was this BC switch. WEDNESDAY AGENDA:
a continuous critical-outflow variant is a scheme decision for the
scientists (smooth blend across uperp = 0, or Froude-limited ghost).

Practical mitigation, MEASURED: the BC pin floor is state-dependent --
rel ~2.7e-5 of the step's initial residual at the NLCD prior (o9d's
snes_rtol 1e-5 just missed it), rel ~1.17e-4 at a TAO trial field
(o18c: floor 5.04e-3 / F0 42.9, lambda pinned at ~1e-6 with the
residual RISING for any larger step). -snes_rtol 1e-3 clears every pin
seen; 1e-4 clears the prior's pin but dies at TAO it 7's trial point
(o18/o18b, and snes_max_it 100 does NOT rescue a lambda = 1e-6 pin).
This is a BC-discontinuity tolerance, not sloppiness: steps away from
the pin converge at 2-8 Newton its.

### h_anuga sweep + 20-it class calibration (o17/o18*, gmres+right ksp rtol 1e-2)

3-TAO-it class twin at snes_rtol 1e-4 (NLCD truth, uniform 0.03 start,
15 classes, 418076 obs cells x 20 times):
- h_anuga 0.001: CLEAN. 140/140 solves converge (28x2, 90x3, 10x4,
  6x6, 6x8 Newton its), J 1.45342e7 -> 7.47493e6, TaoSolve 6.03 s.
- h_anuga 0.003: CLEAN, J-trace nearly identical (7.46462e6) --
  the regularization at this size barely perturbs the physics.
- h_anuga 0.01: truth forward clean (2-6 its) but the calibration
  forward from the uniform-0.03 start hits DIVERGED_FUNCTION_NANORINF
  at Newton it 33 on step 1 -- bigger is NOT safer (weaker drag
  damping lets an iterate excurse to h < 0 -> pow NaN in the flux).
- 20-it calibration at h_anuga 0.001: snes_rtol 1e-4 dies at TAO it 7
  (J already 1.45e7 -> 1.00e5) on a trial-field pin (above); at
  **snes_rtol 1e-3 the full 20 its COMPLETE: J 1.45342e7 -> 6.42e-7,
  EXACT class recovery (rel L2 vs prior 0.0000, max class rel err
  0.0000), TaoSolve 20.7 s** (o18d_cal20_n4.log). The machine-level
  twin convergence is also an end-to-end gradient-quality check at
  this solver tolerance.
- DECISION: NLCD/class-mode recipe = h_anuga_reg_parameter 0.001
  (smallest clean value) + -snes_rtol 1e-3 (BC pins) + gmres+right
  ksp rtol 1e-2. Recorded in beuler_dt1_anuga.yaml and
  beuler_dt1_1hr_anuga.yaml on PM and in campaign-wednesday.md.
  The snes_rtol 1e-3 crutch retires if/when the critical-outflow
  switch is smoothed (Wednesday agenda).

### o26: free-outflow BC lands -- the o22 pin is FIXED (2026-08-24)

Implemented `CONDITION_FREE_OUTFLOW` (yaml `type: free-outflow`): a
transmissive / zero-gradient ghost (qR = qL in primitive space), the
standard "free outflow" of FV SWE codes (ANUGA transmissive, Clawpack
zero-order extrapolation) and exactly option (c) from the design sketch
in next-phase-strategy.md. Checked first, per the plan: RDycore's CEED
path has NO transmissive option -- its `SWEBoundaryFlux_Outflow` is the
same critical ghost with the same q >= 0 switch -- so nothing existed to
match; we added the QFunction there too. The BC is continuous in the
interior state by construction; the Jacobian ghost map is the identity,
wired through host AND device analytic Jacobians (commit 7d2b07fe, all
four solver paths + CEED + docs; default behaviour bitwise unchanged --
new enum value, new switch cases only).

FD gates (laptop, arch-macosx-gnu-rdycore-kokkos-O), all PASSED:
- new global FD-coloring twin pair swe_jacobian_global_freebc_*:
  full-matrix rel err 1.8e-8 (gate 1e-6), np 1/2.
- new driver gates adjoint_{beuler,arkimex}_freeflow.yaml: dJ/du0 and
  dJ/dn vs FD 2e-8..8e-7 (gate 1e-5) across {beuler, arkimex} x
  {host, kokkos vecs} x np {1,2}; arkimex also through the aijkokkos
  DEVICE Jacobian assembly. J differs from the reflecting baseline
  (7.02394 vs 7.03713), so the gates are not vacuous.
- full ctest: only the 7 pre-existing failures (6 cgns, amr_np3).
- CAVEAT (laptop only): adjoint_beuler + -dm_mat_type aijkokkos
  crashes (SEGV/TRAP) for the PRE-EXISTING reflecting baseline too --
  never reaches new code; check on PM at some point. The PM acceptance
  below ran the device Jacobian (baijkokkos) cleanly.

ACCEPTANCE (PM, jobs 57519697/8, debug q, n4 A100): the o22 base
control with ONLY `critical-outflow -> free-outflow` in the yaml
(o26_freeflow.yaml) completes **600/600 steps, zero failures, 598 of
600 solves at 2 Newton its** (1x3, 1x5) in ~6 min wall. The unmodified
o22 base rerun with the SAME new binary still dies at solve 499
(DIVERGED_MAX_IT 50 after 5 solves grinding at 25 its) -- the exact
o22 signature, so the rebuild changed nothing else and the BC swap
alone is the fix. The 25-it grinding phase is entirely absent under
free-outflow. Both jobs then hit a HARMLESS post-forward error: the
2026-08-20 PETSc rebuild rejects `-tao_max_it 0` (o22's "forward-only"
idiom) at TaoSetFromOptions -- use `-tao_max_it 1` or drop the
calibrate flags for future forward-only controls.

CONSEQUENCES:
- Rain-forced NLCD runs are UNBLOCKED. The snes_rtol 1e-3 crutch can
  retire on free-outflow runs (the o26 histogram suggests honest
  convergence; verify on the first long window).
- Long-window configs must switch the outlet yaml to
  `type: free-outflow` (beuler_dt1_*_anuga.yaml etc. on PM still say
  critical-outflow).
- Documented for the scientists in docs/common/input.md with the
  caveats (weak reflections for subcritical outflow; inflow not
  prevented); critical-outflow remains the default -- invite
  correction per the working model.

### o27 + peak-WSE misfit mode: the 1-hr rain-forced window is CLEAN (2026-08-24)

**o27 (PM job 57520315, n4, 15 min wall):** the o22-class run at 6x the
window -- 1-hr rain-forced NLCD classes twin, free-outflow outlet,
beuler dt 1 s, snes_rtol 1e-3 -- completed with **17,198 converged
solves and ZERO failures**: the 3600-step truth forward, the adjoint
sweeps, and a TAO iteration's line-search forwards all clean. The
rain-forced program is unblocked in practice, not just at o26's 600 s.
Cost calibration: the WHOLE tao_max_it-1 run (~4.8 forward passes +
adjoint recomputes) took 15 min at n4 -- far below the old ~1 hr per
TAO-it per simulated hour estimate; budget accordingly.

**Peak-WSE (HWM) misfit mode landed (commit 934d8304).** New
`-adjoint_hwm_file` observation mode in rdycore_adjoint for both
per-cell and NLCD-class calibration: J compares each mark cell's PEAK
sampled WSE against the surveyed mark; the adjoint reuses
AdjointSweepMulti unchanged by placing each mark's residual in the r_k
of its own argmax time (exact gradient a.e.; ties measure-zero).
Below-bed marks are zero-weighted with a printed count; peak-WSE MAE
(the Inunda-comparable number) and the model-dry mark count print at
start and at the TAO solution. `-adjoint_hwm_twin` synthesizes mark
values from the truth forward through the real file path;
`-adjoint_hwm_fd` gates the parameter gradient against central
differences at the all-ones + largest-|g| directions (probe step 1e-3
relative = measured V-curve optimum; smaller steps drown in the forward
solve's noise floor -- 4e-5 at 1e-5 vs 8.6e-7 at 1e-3). Gates: ctests
adjoint_hwm_fd_np_{1,2} at 8.9e-7 (gate 1e-5); classes+hwm smoke at
6.5e-7 with a synthetic 2-class map. Real table:
data/harvey_hwm/turning30m_hwm_obs.txt (108 QC-passed marks) via
make_hwm_obs.py; ALSO staged to PM $SCRATCH/gpu-implicit.

**o28 SUBMITTED (PM job 57521605, regular q, 3 hr):** the
identifiability re-ask -- NLCD-truth twin observed at the 108 real mark
cells' peaks, 1-hr window, 15 TAO its, on a COPY (o28_hwm_twin.txt;
the twin overwrites the table's values -- NEVER point -adjoint_hwm_twin
at the real turning30m_hwm_obs.txt). Compare against o25i (13 gauges /
20 s: rel L2 0.66, worst class 0.81).

### o28: HWM identifiability -- the observable changes the answer (2026-08-24)

PM job 57521605, n4, 100 min wall, 173,575 converged solves, ZERO
failures. NLCD-truth twin observed at the 108 QC-passed real mark
cells' peaks (12 samples over a 1-hr rain-forced window,
free-outflow outlet), uniform 0.03 start, 15 TAO its, beta 1e-4.

J 7.48e4 -> 926 (81x); peak-WSE MAE 0.103 m -> 0.017 m; model-dry
marks 32 -> 19 of 108.

Per-class recovery (rel err): the AREA-DOMINANT classes recover well
-- 22 dev-low 0.7%, 23 dev-med 10.5%, 24 dev-high 8.6%, 81 pasture
1.9%, 82 crops 8.6% -- together 2.43M of 2.93M cells (83% of the
domain in classes recovered to <= 10.5%). Failures concentrate in
small-area classes: 41 deciduous forest pinned at the UPPER bound
(100% err), 52 shrub pinned at the LOWER bound (96%), 95 emergent
wetland 115%, 42 evergreen 52%, 21 dev-open 47%. Unweighted rel L2
0.58; AREA-WEIGHTED rel L2 0.20.

Read against o25i (13 gauges, 20-s window: rel L2 0.66, worst 0.81,
distant classes NEVER MOVED from the start value): with 108 marks and
a 1-hr window the small classes now MOVE -- they are sensitive but
weakly constrained, overshooting to the bounds (semiconvergence along
weak directions; more Tikhonov or longer windows are the obvious
levers) -- while the classes that cover the domain are identified.
The o25i conclusion ("13 gauges cannot constrain 15 classes") does
NOT carry over to the HWM observable: most of the field, by area, is
constrained even by 1 hour of early-transient dynamics.

Caveats for the paper: (1) 1-hr window peaks are early transients,
not flood peaks -- the real 6-12-hr windows should only help; (2) 15
its is not converged (J still falling); (3) twin marks are synthesized
from the truth run (inverse crime -- no representation error), so
this is an identifiability statement, an upper bound on real-data
performance.

Next: the real-data HWM calibration needs a window covering the
actual flood peak (marks record the event maximum) -- window placement
is question Q7 for the scientists, and the long-window cost basis is
now measured (o27).

### o29/o30/o31: wall-clock truths, a robustness hole fixed, and the FIRST REAL-DATA NUMBER (2026-08-24)

**o31 -- the uncalibrated model vs the real survey.** 72-hr rain-forced
forward from the only IC (2017-08-26 18:00) through the Buffalo Bayou
crest, NLCD-prior Manning, free-outflow outlet, beuler dt 1, n4:
**259,200/259,200 implicit solves converged, ZERO failures, 2h05m
wall.** Every one of the 108 QC-passed marks goes wet, and the
uncalibrated model's peak-WSE MAE against the REAL surveyed marks is
**3.41 m** (J 1.24e7). The 1-hr fingerprint (o29g: MAE 1.65 m, 22/108
dry) shows peaks still growing into the window -- crest timing
matters, as expected. Caveats: single hot-start IC, no signed-bias
decomposition yet (o31d dump rerun pending), rain/mesh/datum error
unseparated from roughness error. This is the honest "before" number
to put beside Inunda's calibrated 0.67 m.

**o29 -- measured integrator wall-clocks at 30 m (1-hr forward, n4):**
beuler dt=1: 221 s. Explicit euler dt=0.25: 130 s (exit 0; finiteness
fingerprint run o29f pending -- its first attempt tripped our own
guard: numerics.jacobian: fd is refused with device matrix types, use
analytic). ARK-IMEX dt=0.25: did NOT finish in 100 min => >= 27x
slower than beuler dt1 at this scale; the o19-era 1-km arkimex
numbers do not transfer, needs tuning before quoting. So the honest
implicit-vs-explicit forward factor at 30 m is ~1.7x wall for 4x
fewer steps -- the implicit case rests on robustness (drag stiffness
at trial fields, larger dt headroom), not on this forward ratio.

**o30 -- noisy (0.15 m) HWM twin: found a robustness hole.** Init
with noise: J 361.8, MAE 0.2336 m (clean twin: 0.1034) -- then at TAO
it ~3-4 a trial field hit DIVERGED_MAX_IT(50) and TSStep HARD-ABORTED
the run (exit 91). Fix committed: TSSetErrorIfStepFails disabled;
objectives grade a diverged trial with J=1e30 + zero gradient (line
search backs off); all data-producing forwards explicitly verified
(CheckTruthForward). Rerun pending cert refresh.

**BLOCKED: sshproxy cert expired ~12:30 PT.** Pending PM work, ready
to fire on refresh: (1) o29f explicit fingerprint (yaml fixed to
analytic), (2) o30 rerun with failure-tolerant trials, (3) o31d dump
rerun (-adjoint_hwm_dump o31_marks.txt) for signed bias + per-mark
crest times (window placement for the 12-hr calibration).

### o29f: the explicit path does NOT survive dt 0.25 at 30 m (2026-08-24)

Eval-only forward over the same 1-hr rain-forced window as o29g, on the
same real 108-mark table, n4. The two configs differ in **exactly two
lines** (`diff o27_1hr_freeflow.yaml o29f.yaml`: `temporal` and
`time_step`); mesh, IC, free-outflow BC, ANUGA regularization, rain and
the NLCD prior field are identical.

| run | integrator | dt | result |
| --- | --- | --- | --- |
| o29g | beuler | 1.0 s | J 3.429e6, peak-WSE MAE 1.6479 m, 22/108 marks dry |
| o29f | explicit euler | 0.25 s | **J inf, MAE inf, 108/108 marks dry** |

So dt = 0.25 s is NOT the explicit CFL limit at 30 m -- the explicit run
at that step produces a non-finite state. o32 walked it down: dt 0.125
and dt 0.05 (advective CFL ~0.007) are ALSO non-finite, J inf, 108/108
dry, with the observation frequency scaled so every rung samples peaks
at o29g's 300 s interval.

**This confirms a limit the project already measured and the code already
documents** (see the NOTE in `AddSourceExplicit` in src/swe/swe_petsc.c,
plans/RESULTS-manning-draft.md:273, plans/note-to-team-manning-config.md):
with the dt-FREE drag that the adjoint requires, the binding constraint
is the friction rate, not the CFL --
`dt < 1/tb`, `tb = g n^2 h^{-4/3} |v|`. On the spun-up 30 m state (99.9%
wet, median depth 6 mm) tb reaches 420/s => dt < ~2.4 ms, vs the CFL's
0.25 s; 90k cells violate dt*tb > 1 at 0.25 s. o29f/o32 add the
end-to-end confirmation at three step sizes. I ran the ladder before
grepping for the known result -- the code comment would have predicted
all three outcomes.

Consequences that DO matter:

1. **The o29 wall-clock comparison is void as written.** "Explicit euler
   dt 0.25 = 130 s vs beuler dt 1 = 221 s, so implicit costs ~1.7x"
   timed a run that produces a non-finite state. Against explicit
   stepping of the SAME differentiable RHS, the implicit path wins by
   ~400x in step count, not loses by 1.7x.
2. **The paper's W4 text was wrong** where it called dt = 1 s "4x the
   explicit CFL limit" and inferred a modest implicit-vs-explicit
   factor: that conflates the CFL limit with the stability limit of the
   source method the adjoint actually requires. Both the W4 item and
   red-team item Q6 now state the measurement and the distinction.
3. The genuinely modest comparison -- explicit at 0.25 s with the
   operator-split (`semi_implicit`) friction -- is against a
   NON-DIFFERENTIABLE discretization (it buries dt in the RHS, which is
   why the adjoint cannot use it). Quote it only as such. The
   differentiable equivalent is `ark_imex`, which at 30 m did not
   finish in 100 min (>= 27x beuler dt 1, o29) and needs tuning.

### Mid-event window IC: solved with no conversion machinery (2026-08-24)

The 12-hr calibration window over the crest cannot start at the only IC
(Aug 26 18:00). Investigated the handoff's option (a) and it is simpler
than feared -- no checkpoint->IC-file conversion is needed:

- `-restart <file>` is already a `RDySetup` option, and `ReadCheckpointFile`
  overwrites `u_global` AFTER the yaml IC is applied; the driver stashes
  `u_ic` straight from `u_global` right after `RDySetup`. So a checkpoint
  IS the window's IC.
- Checkpoints are written in NATURAL order, so the window need not run on
  the rank count of the forward that produced it (verified np1 -> np2).
- `DMSetUseNatural(FALSE)` in the adjoint driver does not block checkpoint
  write/read: the natural-SF entry points key off `dm->sfNatural`, which
  survives the flag (only errors when the SF is missing AND useNatural is
  set).

The one real gap was the rain clock: every forward restarts the TS clock
at 0, so a mid-event window replayed rain from the event's first hour.
Fixed by `-adjoint_rain_start_hour <h0>`. Also added
`-adjoint_forward_only` (one forward, then exit) -- without it the
checkpoint on disk is from whichever extra forward ran last (perturbed
IC, optimizer trial), not the window's answer.

Gated as ctests on the rain-forced Houston 1 km mesh (`adjoint_restart_*`,
5 tests, ~2 s): hour 2 of a continuous run vs the same hour restarted
from the 1-hr checkpoint is **bitwise identical** at matched
decomposition and 2.4e-10 max relative across np1->np2; dropping the rain
offset moves it 2.3e-2 relative L2, so the gate discriminates.

### o33: the mid-event window IC path, validated on the PRODUCTION config (2026-08-24)

The `adjoint_restart_*` ctests gate the restart + rain-offset logic on
CPU, on a 2,746-cell mesh. The calibrated run restarts a window on the
GPU at 2.93M cells, so o33 repeats the same three-leg experiment there:
30 m Turning mesh, kokkos vecs + baijkokkos, real MRMS rain, beuler
dt 1, free-outflow, n4.

| leg | final \|u\|_2 | h range |
| --- | --- | --- |
| A: continuous 0-2 h | 7.8231705568e+02 | [4.9016634538e-05, 8.7983908048e+00] |
| B: hour 2 restarted from A's 1-h checkpoint, `-adjoint_rain_start_hour 1` | **7.8231705568e+02** | **[4.9016634538e-05, 8.7983908048e+00]** |
| C: control, same restart with NO rain offset | 7.3088366533e+02 | [1.2901825470e-06, 8.7472719399e+00] |

B reproduces A in all three quantities to every printed digit; the
control C is off by 7% in |u|_2 and 40x in min depth. So on the exact
production configuration a mid-event window is indistinguishable from
the corresponding stretch of a continuous run. Checkpoint write also
works from this driver on the GPU path despite its
`DMSetUseNatural(PETSC_FALSE)` (the natural SF survives the flag).

The calibrated real-data run is no longer blocked on IC machinery.

### o34 (submitted): initial conditions for every candidate window

Repeats the o31 72-hr eval-only forward with HOURLY checkpoints (72
files, ~70 MB each) plus the per-mark dump. Decouples window placement
from IC production: once o31d/o34's crest histogram picks the window,
no further forward is needed, and several placements can be tried for
free (paper Q7). Its "hwm eval" line doubles as a regression check that
the new build reproduces o31's uncalibrated MAE 3.41 m -- both new
options are default-off, so it should match exactly.

### o36: inner-tolerance ladder -- the loose settings did NOT corrupt the forward (2026-08-25)

Mark's concern: `ksp_rtol 1e-2` could be causing a problem that is hard
to see, and `snes_rtol 1e-3` likewise. Both were adopted while the
critical-outflow BC was stalling Newton -- a stall free-outflow removed
-- so neither is paying for anything any more.

**First, the WATCH ITEM from the 2026-08-22 solver decision is not
tripping.** In the completed 72-hr forward at rtol 1e-2, 259,197 of
259,200 steps converge in **2 Newton iterations** (2 steps at 3, 1 at 5;
cap 50). The o30 calibration runs 2-7, plus exactly one pathological
trial field that hit DIVERGED_MAX_IT and was absorbed by the
failure-tolerant machinery. Nothing is visibly stressed.

**Ladder** (1-hr rain-forced window, eval-only vs the real 108 marks, n4):

| rung | ksp / snes | wall | KSP solves | its/solve | total KSP | J | MAE | dry |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A (campaign) | 1e-2 / 1e-3 | 231 s | 7204 | 5.27 | 37,965 | 3.429246e6 | 1.6479 | 22 |
| B | 1e-3 / 1e-3 | **203 s** | **3986** | 7.49 | **29,855** | 3.429268e6 | 1.6479 | 22 |
| C | 1e-4 / 1e-5 | 291 s | 7227 | 9.67 | 69,885 | **3.429274e6** | 1.6479 | 22 |
| D (reference) | 1e-6 / 1e-8 | 348 s | 7581 | 13.96 | 105,832 | **3.429274e6** | 1.6479 | 22 |

Three findings:

1. **C reproduces the converged answer D exactly**, so C is converged to
   printed precision. The campaign setting A is off by **8.2e-6 relative
   in J**, with peak-WSE MAE and dry count identical. The forward science
   answer is insensitive to all of this -- the 3.41 m baseline and the
   crest-censoring result are not tolerance artifacts, at least over
   3,600 steps. o37 repeats the 259,200-step run at C to confirm at scale.
2. **Tightening 1e-2 -> 1e-3 makes the run FASTER** (203 s vs 231 s) on
   21% less linear work. Better linear solves let Newton finish in ONE
   iteration at 3,218 of 3,600 steps instead of two, so the number of KSP
   solves nearly halves (3986 vs 7204) even though each costs more
   iterations. There is no speed argument for 1e-2.
3. **The laptop does not predict Turning.** On the 50-cell free-outflow
   twin, KSP iterations are quantized -- two iterations satisfy both 1e-2
   and 1e-3 (mean 1.97 vs 2.01), both returning the same gradient
   (6.4e-4 adjoint-vs-FD, both FAILING the 1e-5 gate); only 1e-4 adds a
   third iteration and drops the error to 7.2e-6. At Turning the Newton
   behaviour is the opposite (1e-3 needs FEWER Newton iterations), so the
   laptop ladder cannot be used to set the production tolerance. o38
   measures the gradient at production scale.

**Discrepancy to resolve.** `plans/campaign-wednesday.md` and the paper's
open issue 3 both state that "the same configuration at rtol 1e-3 passes
the 1e-5 gates". On the laptop free-outflow beuler twin it does not
(6.4e-4, failing). That claim was measured 2026-08-22, before
free-outflow existed, so it may be configuration-specific rather than
wrong -- o38 settles it at production scale. Do not repeat the claim in
the paper until it does.

**Decision.** Forward/eval work has no gradient in it at all, so its only
requirement is forward accuracy; calibration is where gradient error
lives. o37 (72-hr baseline + fresh hourly checkpoints) and any subsequent
calibration run at **ksp 1e-4 / snes 1e-5**; the checkpoints a calibration
window restarts from must come from the same tolerance the calibration
runs at, which is why o37 regenerates them rather than reusing o34's.

### o37 drainage analysis: the 37 censored marks are a structural drainage failure, not late crests (2026-08-25)

P2 of the Fable brief, run against o37's 73 hourly checkpoints (direct
seeks at the 108 mark cells; script `o37_drainage.py` on PM). The
question: are the marks that never crest in the 72-h window merely
late-cresting (argues for a longer window), or can their cells not drain
at all (argues they are inadmissible for roughness calibration)?

| | censored 37 | genuine-crest 71 (control) |
| --- | --- | --- |
| peak location | ALL at hour 72 (window end) | interior |
| recession from peak | **0.000 -- every single mark** | 52/71 recede > 5% |
| dh/dt over last 12 h | +0.016..+0.060 m/hr, ALL positive | median -0.008 m/hr (draining) |
| mean water column | 0.52 m -> **10.7 m and still rising** | peaks ~1.95 m at h54, then declines |

The control group is already receding in the late window while the
censored cells keep accumulating ~+0.05 m/hr -- that late-window mass is
arriving laterally (the genuine group's decline shows the rain has
effectively ended), i.e. the domain's runoff converges on the downstream
reach and never exits. At the observed fill rates those cells are WEEKS
from cresting: this is not "the window is too short."

Physical reading: the model cannot drain the Buffalo-Bayou/downstream
reach -- either the 30 m mesh's conveyance through the incised channel
is under-resolved or the outlet connectivity is wrong for these cells --
and it piles a mean 10.7 m of standing water there (individual cells to
18 m, vs surveyed h_obs down to ~1 m: marks 27/72 are 11-17 m over).
**No Manning field can repair a reach whose water has no exit**, so the
37 marks are inadmissible for roughness calibration on physical grounds.
Beside that, they are the entire difference between the 3.41 m headline
MAE and 1.51 m on the admissible 71 (bias +1.20 m).

For the scientists (this evidence reframes yesterday's Q1): the
exclusion is no longer "should we drop inconvenient marks?" but "these
sit in a reach the model demonstrably cannot drain; calibrating
roughness against them would absorb a drainage defect into n." The
drainage defect itself is a finding worth reporting -- and worth a look
at the outlet placement/conveyance before any longer-window run.

Units note for reproducers: `turning30m_hwm_obs.txt` column 2 is WSE
(NAVD88 m); the driver converts to water column against the cell bed.
The per-mark dump's h_obs is the converted value. (A first cut of the
drainage script compared raw WSE to model h -- the model-side
classification above never used obs values, so it is unaffected.)

### o40d: the observable-kink hypothesis is REFUTED -- the FD gap lives below the observable (2026-08-25)

The kink-probe instrumentation (commit c51e5063) decomposes the FD gate
exactly over marks, J = J_S + J_flip, counting per probe direction the
marks whose argmax time moves or whose wet/dry state flips, and re-running
the comparison restricted to the non-flipping set (snapshot-reconstructed
FD vs a zero-weighted adjoint re-evaluation; validated on the smooth twin,
where it reproduces the full check to the last digit with zero flips).

Basin scale (debug variant: 30-min window, 108 real marks, classes mode,
converged tolerances ksp 1e-4 / snes 1e-5):

| direction | adjoint g.d | FD | rel err | argmax moved | wet/dry flipped |
| --- | --- | --- | --- | --- | --- |
| ones (all 15 classes) | -1.1256e6 | -8.9248e5 | **2.07e-1** | **0** | **0** |
| e_s (one class) | 3.0650e6 | 3.0721e6 | 2.34e-3 | 0 | 0 |

**Zero observable-level flips, yet the ones direction errs by 21%.** The
smooth-subset error equals the full error because the "smooth" subset is
all 108 marks. Three sharp facts:

1. **Determinism to ~9 digits**: two independent evaluations at the base
   point printed identical J and g.d. Noise is excluded; the ones-direction
   FD difference (dJ ~ 53 on J ~ 3.5e6) is three orders above the
   reproducibility floor. The discrepancy is real and well-resolved.
2. **Error scales with direction support**: domain-wide 2.1e-1 vs
   single-class 2.3e-3 (o38C's coordinate directions: 2.7e-3..6.5e-2).
   A wiring bug has no natural reason to care about sparsity; branch
   switching in the wet/dry DYNAMICS does -- a wider perturbation crosses
   more tiny_h surfaces along the trajectory.
3. **|FD| < |adjoint|** on the ones direction -- the signature of a
   central difference averaging slopes across branch surfaces.

Remaining candidate mechanisms, and the running discriminators:
- (a) dynamics-level wet/dry branch switching: FD error falls ~ eps^1
  as the probe shrinks (fewer surfaces inside the interval);
- (b) genuine curvature (central-difference truncation): falls ~ eps^2;
- (c) a support-correlated defect: flat in eps, systematic in class area.
o39 (eps sweep at converged tolerances, RUNNING) separates (a)/(b) from
(c); o41 (all-15-class coordinate sweep on the 30-min window, gpu4 build
with the direction index printed) measures error vs class area across
four orders of magnitude of support.

Noise-floor arithmetic for reading o39: the base J reproduces to ~1e-7
relative, so the FD noise contribution at probe eps_rel is roughly
0.35/(2 * eps_rel * 0.03 * |g.d|); for the ones direction that is ~5e-3
at eps 1e-3 and ~5e-2 at eps 1e-5 -- the V-curve's left branch is noise,
only the descent from eps 1e-2 -> 1e-3 -> 1e-4 is signal.

Consequence, already in the paper (19c4ed18 wording corrected after this
run): the basin-scale gradient is currently validated by descent
behavior, not by the FD gate, and the paper says so explicitly. What
this does NOT touch: the forward results (o37 baseline, drainage
finding) involve no adjoint; and o28/o30's calibrations descended --
consistent with a gradient that is right up to kink-scale detail.

### o39 (partial): FD is CONVERGED in the probe step and still 7.5% from the adjoint (2026-08-25)

Probe-step ladder at converged tolerances (1-hr window, classes mode,
real marks; o38C supplies the eps 1e-3 point). Ones = all 15 classes
shifted together; coordinates are the 3 largest-|g| classes, same rows
across rungs.

| direction | eps 1e-2 | eps 1e-3 | eps 1e-4 |
| --- | --- | --- | --- |
| ones | 5.244e+0 (FD sign-flips) | 7.635e-2 | 7.470e-2 |
| coord A | 3.587e-2 | 2.668e-3 | (pending) |
| coord B | 7.972e-3 | 1.527e-2 | (pending) |
| coord C | 8.869e-2 | 6.488e-2 | (pending) |

Readings:
- eps 1e-2 (3e-4 absolute in n) is outside the linear regime entirely.
- **The ones-direction FD is stable across a decade** (1.3607e6 at 1e-3,
  1.3586e6 at 1e-4) while the adjoint says 1.2642e6: FD has converged to
  a value 7.5% from the adjoint. Noise bands: ~0.5% at eps 1e-3 (solid),
  ~4.6% at 1e-4 (consistent with flat). So neither pure curvature
  (would fall eps^2) nor sparse kinks (would fall eps^1) explains it.
- The coordinate directions move NON-MONOTONICALLY with eps (one falls
  ~eps, one rises, one is flat) -- characteristic of discrete branch
  surfaces entering/leaving the probe interval, not of a smooth error.

Surviving hypotheses, now sharply posed:
(a) DENSE wet/dry branch structure in the dynamics: so many (cell,step)
    tiny_h crossings inside any reachable interval that FD converges to
    a smoothed (Clarke-type averaged) slope while the adjoint returns
    the exact one-branch derivative -- both stable, persistently apart.
    Property of the objective, not a bug.
(b) A branch-related defect in df/dn or the adjoint accumulation.

Discriminators in flight:
- o41 (all-15-class support sweep, 30-min window): under (a) the gap
  scales with class area; also tests FD superposition
  (sum_s FD(e_s) vs FD(ones)) and carries Jp/Jm for second differences.
- o42 (window ladder 60/300/900 s, debug queue): under (a) the
  contamination ACCUMULATES with trajectory length (a 60-step window
  crosses ~60x fewer surfaces); under (b) the relative gap is
  window-independent. This is the cleanest bug-vs-structure separator.

### o42 (complete): the window-length ladder, ones direction (2026-08-25)

| window | rel FD error | note |
| --- | --- | --- |
| 60 s | 8.923e-5 | inner-solve floor: the gate effectively PASSES |
| 300 s | 8.082e-3 | |
| 900 s | 2.375e-2 | |
| 1800 s | 2.071e-1 | o40d (30-min config) |
| 3600 s | 7.470e-2 | o39/o38C (1-hr config) |

Monotone accumulation through 1800 s; the 3600-s point is a different
J (peaks over a longer horizon), so strict monotonicity across configs
is not expected. Single-class directions stay 1-3 orders cleaner at
every window (e_3: 2.7e-5 at 60 s, 1.7e-5 at 300 s; e_4: 3.0e-3 at
900 s). The verdict stands as committed in 346217b3: the adjoint is
exact (verified at the inner-solve floor with all machinery engaged);
the long-window FD gate measures the objective's dense wet/dry
nonsmoothness, not gradient error. Verification policy: FD-gate on
short windows; descent plus short-window gates on long ones.

### o39/o40/o41 complete: the gradient closure, confirmed three more ways (2026-08-25)

**o39 full eps ladder (1-hr window, converged tolerances), by direction:**

| direction | eps 1e-2 | 1e-3 | 1e-4 | 1e-5 |
| --- | --- | --- | --- | --- |
| ones | 5.24 | 7.64e-2 | 7.47e-2 | 4.38e-2 |
| coord A (class 24) | 3.59e-2 | 2.67e-3 | 4.82e-4 | **8.28e-6 -- PASSES the 1e-5 gate** |
| coord B (class 23) | 7.97e-3 | 1.53e-2 | 1.57e-2 | 5.54e-4 |
| coord C | 8.87e-2 | 6.49e-2 | 6.79e-2 | 2.31e-2 |

Every direction falls toward the gate as the probe shrinks; sparse
directions get clean first. (The e5 rung is meaningful because the
determinism is far better than estimated: o41 resolved a class with
|g| = 249 -- a J-difference of 0.015 on 3.5e6, i.e. ~4e-9 relative --
to 2.4e-5, so reproducibility is ~1e-10/1e-11 relative.)

**o41 all-15-class support sweep (30-min window):** the FD-adjoint gap
per class, mapped to class area, is NOT proportional to area -- it
concentrates in the developed-intensity and wetland classes:

| slot | NLCD | cells | rel gap |
| --- | --- | --- | --- |
| e_1 | 21 developed-open | 178k | 1.10e-1 |
| e_2 | 22 developed-low | 439k | 7.99e-2 |
| e_7 | 42 evergreen | 43k | 2.86e-2 |
| e_3 | 23 developed-med | 928k | 2.60e-2 |
| e_13 | 90 woody wetland | 122k | 6.76e-3 |
| e_4 | 24 developed-high | 552k | 2.34e-3 |
| e_11 | 81 pasture | 428k | 2.48e-4 |
| others | | 12k-86k | 2e-6..2.5e-4 |

What matters is a class's MARGINAL-WETNESS exposure (cells crossing
the depth cutoff during the window), not its size: urban classes flood
progressively, pasture is mostly decisively wet or dry.

**Superposition (from the same o41 data):** the signed per-class gaps
SUM to the domain-wide gap (2.35e5 vs 2.33e5, ~1%). The gap is
additive over classes -- exactly what independent per-cell branch
crossings predict, and what no plausible wiring bug would arrange.

**o40 (full 1-hr probe)** reproduces o38C/o39 with the restricted
machinery engaged: zero observable-level flips, smooth-subset error ==
full error, verdict unchanged.

With the window-length ladder (o42), the mechanism now has four
independent confirmations: eps-convergence per direction,
window-length accumulation, class-support structure with additivity,
and zero observable-level kinks. CLOSED: the adjoint is correct; the
long-window FD gate measures dense wet/dry branch structure in the
objective.

### o44/o45: how much of the error can Manning roughness reach? (2026-08-25)

Uniform scale scan on the production configuration (12-hr cluster-A
window h29-41, 46 real marks, IC = o37 h29 checkpoint, converged
tolerances), evaluating J and peak-WSE MAE at alpha * n_NLCD:

| alpha | n range | J | MAE | dry |
| --- | --- | --- | --- | --- |
| 0.2 | 0.005-0.032 | **DIVERGED_NONLINEAR_SOLVE** | -- | -- |
| 0.3 | 0.008-0.048 | 6.7367e2 | **0.6392** | 1/46 |
| 0.45 | 0.012-0.072 | 7.0380e2 | 0.6614 | 1/46 |
| 0.6 | 0.016-0.096 | 7.4056e2 | 0.6776 | 1/46 |
| 1.0 (NLCD) | 0.027-0.160 | 8.0826e2 | 0.7188 | 0/46 |

**The entire uniform-roughness knob is worth 0.08 m against a 0.72 m
error (11%)** -- and only at alpha 0.3, i.e. roughness at 30% of the
published NLCD values (n as low as 0.008; smooth concrete is ~0.012),
which no one would defend physically. Over a defensible range
(alpha >~ 0.7) it is worth about 3 cm.

Shape: mildly concave, slope steepening toward low alpha (0.103, 0.108,
0.148 m per unit alpha over the three intervals). The limit is NOT
drying (1/46 dry throughout) but numerical: at alpha 0.2 the implicit
solve diverges.

Cross-check: the identifiability twin measured the parameter-induced
peak-WSE signal at ~0.1 m for a large class perturbation -- the same
authority scale, from an independent measurement. So ~85-90% of the
model-vs-survey residual is NOT roughness (rain, DEM, datum,
representation error at 30 m).

Consequences:
1. No Manning calibration against these marks can close the gap to
   Inunda's calibrated 0.67 m (which is on a DIFFERENT mark set --
   never compare the two directly).
2. The paper's claim shifts from "calibration closes the gap" to the
   stronger and more defensible "we can measure, at basin scale, how
   much of a flood model's error a parameter is capable of explaining"
   -- here 11% at the edge of physical plausibility, ~4% within it.
3. The multi-class calibration is still worth running: spatial
   redistribution can beat uniform scaling, and how much it beats it by
   is the quantitative question o43 (running) answers. The alpha curve
   is the one-parameter bar it must clear to justify 15 parameters.
4. Discrepancy stopping at J ~ N/2 ~ 23 is unreachable (J stalls near
   ~670 at best); sigma = 0.15 m is survey error only, and the
   achievable floor is set by structural error an order of magnitude
   larger. Report the floor as measured.

### o47: the peak observable constrains ONE roughness mode, not fifteen (2026-08-25)

Half-scans at alpha 0.45 on the production configuration, splitting the
domain by land-cover group (same window, marks, IC as o44/o45):

| field | J | dJ | MAE | dMAE |
| --- | --- | --- | --- | --- |
| NLCD baseline | 808.3 | -- | 0.7188 | -- |
| developed 21-24 only (71.6% of cells) | 786.8 | -21.5 | 0.7306 | **+0.0118** |
| everything else (28.4%) | 827.5 | +19.2 | 0.7268 | **+0.0080** |
| both (= global alpha 0.45) | 703.8 | **-104.5** | 0.6614 | **-0.0574** |

**Each half alone makes the fit WORSE; together they help by 45x the
sum of the parts.** A strong positive interaction, not two opposing
signs (an intermediate reading of the developed-only point alone, since
refuted by the second half).

Mechanism consistent with all four points: a mark's peak is set by the
PATH-INTEGRATED conveyance from upstream to that mark, and the paths
cross both groups. Speeding part of a path does not drain it, and the
resulting fast/slow junction ponds water -- so partial changes raise
peaks. Only a domain-coherent change lowers them.

**Consequence -- the key design finding.** The peak-WSE observable at
these marks constrains essentially ONE roughness degree of freedom (the
domain-scale, path-integrated conveyance), worth ~0.08 m against a
0.72 m error. It does not carry fifteen classes' worth of information.
This unifies the earlier evidence: o28 identified only the area-dominant
classes (the ones that move the domain mean), and o30 pinned five
classes under realistic noise (the other directions have no signal to
fit, so the optimizer fits noise instead).

Production design follows structurally, not by tuning: calibrate ONE or
TWO parameters rather than fifteen (-adjoint_classes_active, commit
456524dc). o43 (15-class, running) is the control that demonstrates
why -- expect it to descend to roughly the alpha-curve level and then
begin pinning weakly constrained classes.

CAUTION on the alpha-scan bound: it measures the uniform mode, which
this result shows IS the dominant identifiable mode -- so ~0.08 m is a
fair estimate of accessible roughness authority, not merely a lower
bound on one arbitrary direction.

### o48: the calibration was a scaling problem, and it is fixed (2026-08-26)

o43 did zero TAO iterations. The diagnosis in
`plans/PROJECT-STATE-2026-08-26.md` S4 is confirmed and now demonstrated
rather than argued: the failure reproduces in five minutes and the fix
is measured on the production wiring.

**The control.** `o48_smoke.sh` runs the exact production configuration
-- 2.93M cells, GPU types, the 46 real cluster-A marks, the h29
checkpoint, the rain offset -- over a 600-step window instead of 43,200.
Case A is the o43 setup (Manning n as the variable, beta = 1e2, start at
the NLCD prior):

    hwm init: J 7.022047e+02, MAE 0.6630 m, 0 of 46 dry
      0 TAO,  Function value: 702.205,  Residual: 667.459
    class recovery: 0 TAO its, J_final 7.022047e+02

Zero iterations, J_final = J_init, on a window costing five minutes.
The o43 failure was never about the long window or the real data.

**The fix.** Case B is the same start point with
`-adjoint_classes_relative -adjoint_sigma_n 0.015`:

    hwm init: J 7.022047e+02, MAE 0.6630 m, 0 of 46 dry
    objective scaled by 1/J0 = 1.4241e-03; first trial step |g|/J0 = 0.1036 in alpha
      0 TAO,  Function value: 1.,         Residual: 0.103623
      1 TAO,  Function value: 0.970516,   Residual: 0.123666
    hwm final: J 6.815011e+02, MAE 0.6127 m
    class recovery: 1 TAO its, J_final 6.815011e+02

Identical J at the start point to every digit -- alpha = 1 reproduces
the NLCD prior field exactly, so the two cases differ only in the
optimization problem. The first trial step is 0.10 in alpha (a 10%
roughness change) where before it was five orders too large.

**Why it works.** BLMVM's first trial is x - g. With n as the variable
and |g| = 1554 where n ~ 0.05 that lands on the lower bound everywhere,
and uniform alpha 0.2 is measured to diverge (o44), so every trial
failed and the line search gave up. Making the variable alpha_k =
n_k/n_prior_k and the objective J/J(start) puts both at O(1). The bounds
become alpha in [0.3, 3.0] -- a physical statement that also excludes
the divergent region, which a bound of n > 0.005 did not.

**Regularization on principle.** `-adjoint_sigma_n <s>` sets
beta = 1/s^2 from a prior standard deviation on n, and the run reports
sigma_n. Against the measured J(alpha) curve (o44/o45) with
sum_k n_prior_k^2 = 0.1399 over the 15 classes:

| sigma_n | beta | uniform-mode minimum | J_mis | reg | MAE |
| --- | --- | --- | --- | --- | --- |
| 0.010 | 10000 | alpha 0.88 | 790.0 | 9.7 | ~0.708 |
| **0.015** | **4444** | **alpha 0.70** | **757.3** | **28.6** | **~0.690** |
| 0.020 | 2500 | alpha 0.33 (bound) | 678.6 | 79.5 | ~0.643 |

So sigma_n = 0.015 is about the loosest prior that leaves an interior
minimum in the dominant identifiable direction; at 0.020 the problem
runs to the bound again, which is what beta = 1e2 was doing at a
relative roughness change of 13.6. Report sigma_n, never a bare beta.

**Verification** (commit 48485e24, all 21 adjoint/calibration ctests
pass -- 19 previous plus 2 new):
- the alpha gradient passes the central-difference gate at 2.3e-6 (new
  ctest `adjoint_classes_relative_fd_np_1`, eps 1e-4; the default 1e-3
  probe is truncation-limited at 3e-5 and the error scales as eps^2
  across the whole sweep 3e-5..3e-2, which is what says truncation
  rather than a wrong chain rule)
- the dJ/dn recovered from the alpha gradient equals the absolute
  mode's gradient at the same point to 1e-8, the dump's 10-digit
  precision, class by class
- dump/init round-trips exactly and across modes: a relative run's
  dump, read back by either mode, reproduces its J_final to all digits
- `-adjoint_jred_gate` now applies to the classes branch as well, so a
  run that reports a "final" objective equal to its initial one fails
  CI instead of passing quietly

**Also added.** `-adjoint_classes_grad_dump <file>` writes
(code, n, dJ/dn) at the start point before TAO runs. It is the fallback
if the optimizer still struggles -- a line search along -g can then be
run as eval-only forwards (~36 min) instead of trial points (~2 hr) --
and it is a diagnostic in its own right: it says which classes carry
the misfit gradient, which is what `-adjoint_classes_active` needs.

**Caution, from the smoke's one iteration.** Classes 23 and 24 reached
the alpha 0.3 bound on the first accepted step (BLMVM's line search
expanded well past -g), while the objective fell only 3% and the
gradient norm rose. On a 600-step window the peaks are barely resolved,
so this is not a production result -- but it is the expected
over-parameterization behaviour (o28/o30/o47) and the cue to read the
production run's per-class table before believing any per-class number.


### o51: the mesh does scale to 4 nodes -- 2.42x, bit-identical (2026-08-26)

Eval-only forward of the production 43,200-step window at three node
counts, in one interactive allocation:

| ranks | nodes | forward | speedup | node-hr | efficiency |
| --- | --- | --- | --- | --- | --- |
| 4 | 1 | 36m21s | 1.00x | 0.606 | -- |
| 8 | 2 | 21m55s | 1.66x | 0.731 | 83% |
| 16 | 4 | 15m01s | **2.42x** | 1.001 | 61% |

**All three reproduce `J 8.082566e+02, peak-WSE MAE 0.7188 m, 0 of 46
dry` to every printed digit.** That is the result that mattered most:
no decomposition dependence in the peak-WSE observable across a 4x
change in rank count, which also re-validates the argmax bookkeeping
under different partitioning.

This CORRECTS an inference drawn earlier the same day from the b2i
benchmarks, where 64 ranks ran 2.8x slower than 4 and we concluded the
mesh does not scale past one node. It does not scale to 64; it scales
usefully to 16.

Production choice: **n16**, despite 61% parallel efficiency, because
queue wait rather than node-hours is the binding constraint. A
~10-evaluation calibration is 8.3 hr at n16 -- one 12-hr slot with
margin for line-search retries -- against 20 hr at n4, a two-link chain
whose second link may sit pending for hours. The extra ~13 node-hr is
negligible against the ~265 remaining. Forward+adjoint should scale
like the forward, because its cost is dominated by revolve
recomputation, which is forward work.

### o52: the prior is losing to the misfit, and the line search is not the lever (2026-08-26)

One BLMVM iteration on a 7200-step (2 event-hour) window, relative
variables, sigma_n = 0.015, More-Thuente line search. J 741.38 ->
713.45, peak-WSE MAE 0.6837 -> 0.6096, **residual 0.0906 -> 0.1343**.

| NLCD | class | prior n | recovered n | alpha | % cells |
| --- | --- | --- | --- | --- | --- |
| 23 | developed med | 0.1200 | 0.0360 | **0.300** | 31.7% |
| 24 | developed high | 0.1600 | 0.0480 | **0.300** | 18.9% |
| 90 | woody wetland | 0.0980 | 0.0294 | **0.300** | 4.2% |
| 22 | developed low | 0.0900 | 0.2358 | **2.620** | 15.0% |

**69.7% of the domain sits at or against a bound after ONE step**, and
the step direction matches the start-point gradient signs exactly -- so
the direction is right and the magnitude is not.

Decomposing the step at sigma_n = 0.015 (beta 4444): the prior charged
**118.5** units and the misfit paid **146.5** (-19.8%), for a net -27.9.
The Tikhonov term is working; it is simply losing. The weight that
would make this exact step net-worse is beta 5492, i.e.
**sigma_n < 0.0135** -- so 0.015 sits just above the threshold, which
sharpens the earlier "loosest prior with an interior minimum" estimate
from inference into measurement.

**Mechanism.** 22 rises 2.6x while 23 falls to the floor: two adjacent,
spatially interleaved developed classes moving in opposition. For a
path-integrated observable that is a near-null-space direction -- it
barely changes total conveyance while driving both parameters somewhere
indefensible (developed-medium urban at n = 0.036, smoother than
concrete; developed-low at n = 0.236, rougher than forest). This is the
o30 noise-fitting pathology reproduced on **real data at basin scale**
rather than on a synthetic twin, which is a stronger statement than the
paper currently makes.

**Consequence for the production run.** A gentler line search changes
the path, not the minimizer; armijo would walk toward the same place
more slowly. The levers are sigma_n and the number of free parameters,
not `-tao_ls_type`.

CAUTION: 7200 steps is two event-hours. At the production window the
alpha scan puts the ENTIRE uniform authority at ~135 J units while
redistribution bought 146 here, so whether the prior holds at 43,200
steps is genuinely open and is what the production run measures.


### o52 addendum: armijo beats More-Thuente, and the reason is the trade (2026-08-26)

Both line searches, one BLMVM iteration, identical start point
(J 741.38, MAE 0.6837, |g|/J0 0.0906 -- reproduced to every digit):

| | misfit | + prior | = J | residual | pinned at alpha 0.3 | near alpha 3 |
| --- | --- | --- | --- | --- | --- | --- |
| mt | 594.9 | **118.5** | 713.45 | 0.1343 (+48%) | 23, 24, 90 = **54.8%** of cells | 22 = 15.0% |
| armijo | 610.7 | **68.0** | **678.73** | **0.0849 (-6.3%)** | 23, 90 = **35.9%** | none |

More-Thuente expands past -g, and the expansion is a bad trade: it buys
15.8 units of extra misfit for 50.5 units of prior violation, ending at
a HIGHER total objective than the shorter step, with the gradient norm
risen 48% and 69.7% of the domain on or against a bound.
Backtracking-only reaches a lower objective with a falling gradient and
half the pinning. Production uses `-tao_ls_type armijo`.

This corrects the reading recorded above ("the line search changes the
path, not the minimizer"). Asymptotically that stands -- both minimize
the same objective, and if the unconstrained optimum lies past a bound
both will eventually pin. But within the ~10-iteration budget a 12-hour
slot buys, the path is most of what we get, and armijo's is
qualitatively healthier.

### o53: the production start-point gradient (2026-08-26)

Run at n16 on an otherwise-idle interactive allocation while the batch
queue was saturated. `hwm init: J 8.082566e+02, MAE 0.7188` -- exactly
the alpha = 1 reference, confirming again that relative mode's start
point reproduces the NLCD prior field. First trial step
`|g|/J0 = 0.1575` in alpha.

Classes ranked by |dJ/dn| * n_prior, the gradient in the variable the
optimizer actually moves:

| # | NLCD | class | dJ/dn | share | cum | push | rank at 2 h |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 23 | developed med | +755.9 | 28.8% | 28.8% | lower | 1 |
| 2 | 90 | woody wetland | +675.3 | 21.0% | 49.7% | lower | 2 |
| 3 | 81 | pasture | **-898.4** | 10.8% | 60.6% | RAISE | 7 |
| 4 | 22 | developed low | -327.9 | 9.4% | 69.9% | RAISE | 3 |
| 5 | 21 | developed open | -568.3 | 7.2% | 77.1% | RAISE | 5 |
| 6 | 24 | developed high | +135.7 | 6.9% | 84.0% | lower | 4 |

The top two are stable across a 6x change in window length and carry
half the gradient. Below rank 2 the 7200-step proxy is unreliable:
pasture moves 7th -> 3rd (its raw |dJ/dn| is the largest in the field;
it ranked low only because its prior n is small) and developed-high
drops 4th -> 6th. Designing the reduced-parameter run on the proxy
would have picked the wrong third parameter.

**Active set for o50: 23, 90, 81** (60.6% of the gradient). Adding 22
buys 9.4% but reintroduces the 22/23 compensating pair -- two
interleaved developed classes pulling opposite ways, which is nearly
null-space for a path-integrated observable and is what blew up in o52.

Note the gradient is relatively STRONGER at the production window
(|g|/J0 0.1575 vs 0.0906 at two hours), so sigma_n = 0.015 is more
likely to lose here, not less. Expect pinning in the 15-class control;
the defensible number should come from the reduced-parameter run.

