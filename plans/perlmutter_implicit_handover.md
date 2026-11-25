# Handover: testing the fully implicit SWE solver on Perlmutter (CPU and GPU, libCEED)

Branch: `bishtgautam/swe-implicit-ts` (based on `bishtgautam/augmented-roe`)
Plan and laptop results: `plans/swe_implicit_ts_plan.md` (Section 4, "Status")

## 1. What is on the branch

A fully implicit flow TS for first-order Roe SWE (no tracers), selected in the YAML:

```yaml
numerics:
  temporal: beuler   # or cn, bdf  (-ts_type overrides; -ts_bdf_order for bdf)
  riemann : roe
```

- `physics.flow.source.method` defaults to `fully_implicit` (friction `S_f(U)` with no `dt`).
- Supported scope (checked in `ValidateConfig`): `riemann: roe`, `well_balancing: none`,
  first-order (no `second_order`), no sediment/salinity/heat, no adaptive dt.
- Fixed time step by default (`-ts_adapt_type none` is set; options still override).
- Default nonlinear tolerances: `-snes_atol 1e-12 -snes_rtol 1e-10`, max 50 its.

Jacobian (`-rdy_jacobian analytic|fd|mf`, default `analytic`):

| Backend | Where the Jacobian is built | Files |
|---|---|---|
| PETSc (no `-ceed`) | host loop, `MatSetValuesLocal` | `src/operator_jacobian_petsc.c` |
| CEED (`-ceed ...`) | libCEED Q-functions write 3×3 blocks into a COO-values CeedVector; one `MatSetValuesCOO` in the state vector's memory space (device on GPU) | `src/operator_jacobian_ceed.c`, `src/swe/swe_jacobian_ceed.h` |
| both | pointwise derivatives (Roe flux incl. entropy fix and `h_anuga`; bed slope; friction; reflecting / critical-outflow BC maps) | `src/swe/swe_roe_flux_jacobian_ceed.h`, `src/swe/swe_sources_jacobian_ceed.h` |

The CEED path does **not** use `CeedOperatorLinearAssemble`, so any libCEED that RDycore
already builds with works (the preinstalled PETSc 3.24.1 libCEED is fine).

Debug options:
- `-rdy_jacobian_host` — with `-ceed`, build the Jacobian on the host instead (reference).
- `-rdy_jacobian_compare` — with `-ceed`, build both at every Jacobian evaluation, print
  `||J_ceed - J_host||_F / ||J_host||_F`, and error if it exceeds 1e-12. Slow (host copy each time).

When `-ceed /gpu/cuda/*` is used, RDycore already sets `VECCUDA` / `MATAIJCUSPARSE` on the DM
(`src/rdydm.c`), so the Jacobian matrix is a CUSPARSE matrix and the COO values are passed from
device memory.

## 2. Laptop baseline (what to compare against)

macOS, `~/.rdycore-petsc` (petsc_main, debug), CPU backends `/cpu/self/ref|opt`:

| Check | Laptop result |
|---|---|
| `test_swe_roe_jacobian` (unit) | 4/4 pass; 4799/4800 flux-Jacobian columns match FD |
| `-rdy_jacobian_compare`, np 1/2, ref/opt/gen | 5e-17 |
| T1 time-only MMS, L1 rate h | BE 0.98, CN 2.00, BDF2 2.01 |
| T3 space-time MMS (dt ∝ dx), L1 rate h | BE 0.93 (CN 0.94) |
| Newton its/step, BE, sinusoidal MMS, 4096 cells, LU | CFL 1.6: 2.9; CFL 6.6: 2.9; CFL 26: 5.9 |

The default serial linear solver (GMRES + ILU(0)) failed at CFL ≈ 26; LU works.

## 3. Build on Perlmutter

```bash
git clone git@github.com:rdycore/rdycore.git && cd rdycore
git checkout bishtgautam/swe-implicit-ts
git submodule update --init --recursive
```

GPU node build (preinstalled PETSc, config 1 = opt, 64-bit; use config 2 for debug first):

```bash
source config/set_petsc_settings.sh --mach pm-gpu --config 2
mkdir build-pm-gpu-debug && cd build-pm-gpu-debug
cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=cc -DCMAKE_Fortran_COMPILER=ftn ..
ninja
```

CPU node build: same with `--mach pm-cpu` into `build-pm-cpu-*`.

Before anything else, check which libCEED backends exist in that PETSc:

```bash
salloc -N 1 -C gpu -G 4 -q interactive -t 00:30:00 -A <account>
srun -n 1 -G 1 ./driver/rdycore_mms driver/tests/swe_roe/mms_implicit_jacobian_check.yaml -ceed /gpu/cuda/ref -ts_max_steps 1
```

(`/gpu/cuda/gen`, `/gpu/cuda/shared`, `/gpu/cuda/ref` should all be tried; on the laptop the
petsc_main libCEED had no `/cpu/self/gen`, so do not assume a backend exists.)

## 4. Test plan

Run from the build directory (ctest copies the YAML and mesh into
`driver/tests/swe_roe/`). Use `srun` with the GPU binding you normally use, e.g.
`srun -n 4 -G 4 --gpu-bind=single:1`.

### Step 1 — CPU correctness (pm-cpu or the CPU path on a GPU node)

```bash
ctest -R "swe_implicit|test_swe_roe_jacobian" --output-on-failure
```

Expect 20/20 pass (19 implicit ctests + the unit test), same rates as Section 2. Note: in
`ctest -j N`, the existing `swe_roe_ex2b_*chkpt*` tests interfere with each other (they share
output directories); on the laptop they fail at random under `-j6` and pass with `-j1`, on both the
baseline and this branch. That is not caused by this work.

### Step 2 — GPU Jacobian correctness

From `build/driver/tests/swe_roe`:

```bash
for be in /gpu/cuda/ref /gpu/cuda/shared /gpu/cuda/gen; do
  srun -n 1 -G 1 ../../rdycore_mms mms_implicit_jacobian_check.yaml -ceed $be -rdy_jacobian_compare -snes_converged_reason
  srun -n 2 -G 2 ../../rdycore_mms mms_implicit_jacobian_check.yaml -ceed $be -rdy_jacobian_compare -snes_converged_reason
done
```

Expect `Jacobian check: ... = O(1e-16)` (GPU FMA may give up to ~1e-15; the run errors above 1e-12)
and 2–3 Newton iterations per step. If the JIT fails, the error names the Q-function header; the
new headers only use `sqrt`, `pow`, `fabs`, structs returned by value, and `MATERIAL_PROPERTY_MANNINGS`
from `physics_ceed.h`.

### Step 3 — GPU convergence rates

```bash
for m in beuler cn bdf; do
  srun -n 1 -G 1 ../../rdycore_mms mms_implicit_time_$m.yaml -ceed /gpu/cuda/gen
done
srun -n 4 -G 4 ../../rdycore_mms mms_implicit_spacetime_beuler.yaml -ceed /gpu/cuda/gen
```

Expect `PASS` and the Section 2 rates. A linear solver is needed that works with CUSPARSE
matrices; the defaults (GMRES + block Jacobi/ILU) run on the GPU via cuSPARSE ILU. If a solve fails, try
`-pc_type bjacobi -sub_pc_type ilu` explicitly, or `-pc_type gamg`.

### Step 4 — No host↔device traffic in the Jacobian

Use a larger, non-MMS-dominated run (the MMS driver evaluates manufactured sources on the host
with muParser at every stage, which costs host work and copies). For a quick check the MMS case
is still useful: increase `base_refinement` to 4–5 in a copy of `mms_implicit_jacobian_check.yaml`.

```bash
srun -n 1 -G 1 ../../rdycore_mms big.yaml -ceed /gpu/cuda/gen -log_view -log_view_gpu_time
```

In `-log_view` look at `TSJacobianEval`/`SNESJacobianEval`, `MatSetValuesCOO`, `TSFunctionEval`:
the `CpuToGpu` / `GpuToCpu` count columns should be 0 for the Jacobian events (apart from the
first call, which builds the COO pattern on the host). Expected nonzero copies elsewhere: MMS
forcing, and (in the current code) `GetOperatorDomainMaterialProperties` is **not** used on the
CEED Jacobian path, so it should not appear.

### Step 5 — CPU vs GPU performance and solver choice

For a fixed mesh (e.g. `base_refinement` 5–6) and BE at CFL ≈ 5 and 20, record wall time,
Newton its/step, KSP its/step for:

| Run | Options |
|---|---|
| pm-cpu, PETSc backend | `-n 64` (no `-ceed`) |
| pm-cpu, CEED | `-n 64 -ceed /cpu/self/opt` |
| pm-gpu, CEED | `-n 4 -G 4 -ceed /gpu/cuda/gen` |
| explicit Euler reference | same mesh, `temporal: euler`, CFL ≈ 0.5 |

Preconditioners worth trying: `bjacobi`+`ilu` (default), `asm -pc_asm_overlap 1`, `gamg`,
`hypre` (BoomerAMG, if built with CUDA). Large CFL is where ILU(0) failed on the laptop.

## 5. Known limitations / things to watch

- Wet/dry: not exercised. The Jacobian zeroes dry cells (`h ≤ tiny_h`) and uses one-sided
  derivatives at kinks; Newton may leave `h < 0`. Phase 6 in the plan (positivity post-check or
  SNESVI) is not done.
- `h_anuga_regular > 0` is differentiated (unit-tested), but not run in a full simulation.
- The PETSc backend's friction uses `u = hu/h` while CEED uses the desingularized velocity; they
  agree when `h_anuga_regular = 0` (the default).
- Plain Roe uses 3D edge lengths/areas, so a uniform state on a sloped bed is not exactly
  steady (O(slope²)). This is why the time-only MMS uses a flat bed.
- The MMS step count fix: `stop / time_step` is now rounded up with a tolerance (it was truncated,
  which could hang `RDyAdvance`).

## 6. Useful options

```text
-rdy_jacobian analytic|fd|mf     Jacobian type (fd = coloring, mf = JFNK without PC)
-rdy_jacobian_host               CEED runs: assemble the Jacobian on the host
-rdy_jacobian_compare            CEED runs: compare device vs host Jacobian each evaluation
-ts_type beuler|cn|bdf|theta     override the YAML temporal method
-ts_adapt_type basic             re-enable PETSc adaptivity (off by default for implicit)
-snes_monitor -snes_converged_reason -ksp_converged_reason
-log_view -log_view_gpu_time
```
