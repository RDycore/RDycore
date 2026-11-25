# Handover: testing the implicit SWE solver on Perlmutter (CPU and GPU, libCEED)

| | |
|---|---|
| Branch | `bishtgautam/swe-implicit-ts` on `github.com/rdycore/rdycore` (based on `bishtgautam/augmented-roe`) |
| Design notes and laptop history | `plans/swe_implicit_ts_plan.md` |
| Equations | Sections 7 and 8 of `papers/RDycore_SWE_Equations/main.tex` |
| One-command GPU checks | `driver/tests/swe_roe/run_implicit_checks.sh` (Section 4.2) |

**Goal on Perlmutter:** confirm that the implicit solver gives the laptop results on CPU and GPU
nodes (Section 2), that the libCEED Jacobian runs on the GPU with no host↔device copies, and measure
whether implicit steps beat explicit ones in wall time.

---

## 1. What is on the branch

A fully implicit flow TS (backward Euler, Crank–Nicolson or BDF2) for first-order shallow water
equations without tracers, with an analytic Jacobian on both the PETSc and libCEED backends.

```yaml
physics:
  flow:
    mode: swe
    well_balancing: none            # or hydrostatic_reconstruction, augmented_roe
    # h_anuga_reg_parameter: 1e-3   # implicit default; a non-zero value here wins
numerics:
  temporal: beuler                  # or cn, bdf  (-ts_type overrides; -ts_bdf_order for bdf)
  riemann : roe
```

Defaults and limits for implicit runs (checked in `ValidateConfig`, `src/yaml_input.c`):
- `physics.flow.source.method` defaults to `fully_implicit` (friction from the current state, no `dt`).
- `physics.flow.h_anuga_reg_parameter` defaults to `1e-3` (explicit runs keep 0). Needed for Newton at
  moving wet/dry fronts above CFL 1. A value of 0 in the YAML counts as unset.
- Supported: `riemann: roe`; `well_balancing: none | hydrostatic_reconstruction | augmented_roe`;
  first order only; no sediment, salinity or heat; fixed time step (`-ts_adapt_type none` is set).
- Newton: `-snes_atol 1e-12 -snes_rtol 1e-10`, at most 50 iterations; negative depths in a Newton
  iterate are set to a dry state (`-rdy_positivity clip`).

Where the Jacobian is built:

| Backend | How | Files |
|---|---|---|
| PETSc (no `-ceed`) | host loop, `MatSetValuesLocal` | `src/operator_jacobian_petsc.c` |
| libCEED (`-ceed ...`) | Q-functions write each edge's and cell's 3×3 blocks into one COO-values `CeedVector`; the COO pattern is built once on the host; each evaluation is one `CeedOperatorApply` + one `MatSetValuesCOO` in the state vector's memory space (device on GPU) | `src/operator_jacobian_ceed.c`, `src/swe/swe_jacobian_ceed.h` |
| both | pointwise derivatives: Roe, HR and augmented Roe edge fluxes; bed slope; friction; reflecting and critical-outflow boundaries | `src/swe/swe_roe_flux_jacobian_ceed.h`, `src/swe/swe_edge_flux_jacobian_ceed.h`, `src/swe/swe_sources_jacobian_ceed.h` |

The libCEED path does not call `CeedOperatorLinearAssemble`, so the libCEED in the preinstalled
PETSc 3.24.1 is enough. With `-ceed /gpu/cuda/*`, RDycore sets CUDA vectors and an AIJCUSPARSE
matrix on the DM, so the Jacobian is assembled on the device.

---

## 2. Laptop results to compare against

macOS, `~/.rdycore-petsc` (petsc_main, debug build), CPU libCEED backends.

| Check | Result |
|---|---|
| `test_swe_roe_jacobian` (8 unit tests) | all pass; analytic vs finite differences agree in 4799/4800 (Roe), 4739/4800 (HR), 4800/4800 (augmented Roe) columns; the rest sit on kinks |
| libCEED vs host Jacobian (`-rdy_jacobian_compare`), np 1 and 2, ref/opt/gen | 5e-17 to 6e-17 |
| Time-only MMS (uniform state), L1 rate of h | BE 0.98, CN 2.00, BDF2 2.01 |
| Space-time MMS (dt ∝ dx), L1 rate of h, default h_anuga | Roe 0.898, HR 0.891 (PETSc) / 0.872 (libCEED), augmented Roe 0.887; 2.5–2.7 Newton/step |
| Lake at rest (HR, augmented Roe; submerged and island), dt = 0.25 s | preserved to 1e-12; 0–1 Newton/step |
| Dry-bed dam break (Ritter), BE, Roe / HR / augmented Roe | dt 0.02 s (CFL 0.5): 1.04 / 1.04 / 1.09%, 3 Newton/step; dt 0.1 s (CFL 2.5): 1.34 / 1.34 / 1.40%, 5.4–6.0; dt 0.2 s (CFL 5): 1.64 / 1.64 / 1.72%, 8.5–10; min h = 0 |
| Dry-bed dam break, CN / BDF2, plain Roe | dt 0.1 s: 0.93% / 0.94%; dt 0.2 s: CN 0.89%, BDF2 Newton fails at step 18 |
| Wet-bed dam break (Stoker), BE | dt 0.02 / 0.1 / 0.2 s: 1.0 / 1.5 / 2.1%, 3–6 Newton/step |
| Newton/step vs CFL, sinusoidal MMS, 4096 cells, LU | CFL 1.6: 2.9; CFL 6.6: 2.9 (FD Jacobian 3.8); CFL 26: 5.9 |
| `run_implicit_checks.sh /cpu/self` | 22/22 pass in 70 s |

Note: the default serial linear solver (GMRES + ILU(0)) failed at CFL ≈ 26; LU worked.

---

## 3. Build on Perlmutter

Same recipe as `docs/common/installation.md`, on the branch:

```bash
git clone -b bishtgautam/swe-implicit-ts git@github.com:rdycore/rdycore.git rdycore-implicit
cd rdycore-implicit && git submodule update --init --recursive
```

GPU nodes (config 1 = optimized, 64-bit indices; config 2 = debug):

```bash
source config/set_petsc_settings.sh --mach pm-gpu --config 1
cmake -S . -B build-$PETSC_ARCH -DCMAKE_INSTALL_PREFIX=$PWD/build-$PETSC_ARCH -G Ninja
cd build-$PETSC_ARCH && ninja -j8 install
```

CPU nodes: the same with `--mach pm-cpu`, which gives a separate `build-$PETSC_ARCH` directory.

Interactive allocations:

```bash
salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account=<project>_g
salloc --nodes 1 --qos interactive --time 01:00:00 --constraint cpu --account=<project>
```

First, find out which libCEED GPU backends this PETSc has (run in `build-$PETSC_ARCH/driver/tests/swe_roe`).
Don't assume `gen` exists: the laptop's petsc_main libCEED had no `/cpu/self/gen`.

```bash
for be in /gpu/cuda/ref /gpu/cuda/shared /gpu/cuda/gen; do
  srun -N 1 -n 1 -c 32 -G 1 ../../rdycore_mms mms_implicit_jacobian_check.yaml -ceed $be -dm_vec_type cuda > be_check.log 2>&1 \
    && echo "$be: ok" || { echo "$be: failed"; grep -m2 -i "backend\|error" be_check.log; }
done
```

---

## 4. Test plan

Run everything from `build-$PETSC_ARCH/driver/tests/swe_roe`, where ctest has generated the inputs.

### 4.1 CPU correctness (CPU node, or the CPU path on a GPU node)

```bash
cd build-$PETSC_ARCH
ctest -R "swe_implicit|test_swe_roe_jacobian|test_yaml_input" -j 8 --output-on-failure
```

Expect all 113 tests to pass: 110 `swe_implicit_*` tests, `test_swe_roe_jacobian`, and
`test_yaml_input` on 1 and 2 ranks. The rates and errors should match Section 2. The "ceed" ctests here use `-ceed /cpu/self`.

If you run the whole suite with `ctest -j N`, the existing `swe_roe_ex2b_*chkpt*`, `*_ic_file*` and some mesh
tests fail at random: they write to shared output directories. They pass with `-j 1`. This
happens on the unchanged branch too and is not caused by this work.

### 4.2 GPU correctness: one command per backend

```bash
cd build-$PETSC_ARCH/driver/tests/swe_roe
bash ../../../../driver/tests/swe_roe/run_implicit_checks.sh /gpu/cuda/gen \
  "srun -N 1 -c 32 -G 2 --gpu-bind=single:1 -n" -dm_vec_type cuda
```

It runs 22 checks (at most 2 ranks) and prints one PASS/FAIL line each, with logs in
`implicit_checks_gpu_cuda_gen/`:

| Checks | What passes |
|---|---|
| `jacobian_{roe,hr,augroe}_np{1,2}` | libCEED (GPU) Jacobian = host Jacobian to < 1e-12 relative (expect ~1e-16; GPU FMA may give ~1e-15) |
| `mms_time_{beuler,cn,bdf}` | time-only MMS rates above 0.95 / 1.95 / 1.95 |
| `mms_spacetime_beuler_{roe,hr,augroe}` | space-time MMS rates above 0.8 |
| `dam_break_{dry,wet}_{none,hydrostatic_reconstruction,augmented_roe}` | relative L1 depth error ≤ 2% vs Ritter/Stoker and h ≥ 0 |
| `lake_at_rest_{submerged,island}_{hydrostatic_reconstruction,augmented_roe}` | state preserved to 1e-12 |

Repeat with `/gpu/cuda/shared` and `/gpu/cuda/ref` if they exist. The numbers should match the laptop
table to several digits, and the Newton iterations per step should match closely.

If a GPU linear solve fails, add for example `-pc_type bjacobi -sub_pc_type ilu` or `-pc_type gamg` as extra options.
If the libCEED JIT fails, the error names the Q-function header. The new headers use only `sqrt`,
`pow`, `fabs`, small structs passed by value, and `MATERIAL_PROPERTY_MANNINGS` from `physics_ceed.h`.

### 4.3 No host↔device copies in the Jacobian

The MMS driver evaluates its manufactured sources on the host at every stage, which adds copies. So use a
dam break, which has no MMS forcing (here at dt = 0.2 s, i.e. 30 steps):

```bash
sed 's/stop_n: 60/stop_n: 30/' swe_implicit_dam_break_dry_none_np_2_ceed.yaml > big_db.yaml
srun -N 1 -n 1 -c 32 -G 1 ../../rdycore big_db.yaml -ceed /gpu/cuda/gen -dm_vec_type cuda \
  -log_view -log_view_gpu_time > big_db.log 2>&1
```

In `-log_view`, check `TSJacobianEval` / `SNESJacobianEval` and `MatSetValuesCOO`. Their `CpuToGpu` and
`GpuToCpu` counts should be 0, apart from the first evaluation, which builds the COO pattern on the host.
Note anything that does copy; the residual path and the linear solver are the candidates.
On the laptop (`/cpu/self`, this case), a Jacobian evaluation took about 11× a residual evaluation
(7.2 ms vs 0.65 ms per call, 258 vs 352 calls), so the Jacobian kernel is the one to time on the GPU.
For a bigger mesh, use one of the existing large cases (`docs/user/example-cases/dam-break`) with
`numerics.temporal: beuler`.

### 4.4 CPU vs GPU performance and solver choice

Use a fixed, larger mesh (for example the 5120x2560 dam break in `docs/user/example-cases/dam-break`)
with `temporal: beuler` at front CFL ≈ 2, 5 and 20 (scale `stop_n` down from the explicit run).
For each, record wall time, time steps, Newton iterations/step and Krylov iterations/step:

| Run | Options |
|---|---|
| CPU node, PETSc backend | `srun -N 1 -n 64 -c 2 ../../rdycore ... ` |
| CPU node, libCEED | add `-ceed /cpu/self` |
| GPU node, libCEED | `srun -N 1 -n 4 -c 32 -G 4 ... -ceed /gpu/cuda/gen -dm_vec_type cuda` |
| explicit reference | same mesh with `temporal: euler`, CFL ≈ 0.5 |

Preconditioners worth comparing: `bjacobi` + `ilu` (default), `asm -pc_asm_overlap 1`, `gamg`, and
`hypre` (BoomerAMG) if the PETSc build has it. Add `-ksp_converged_reason -snes_converged_reason -log_view`.
Large CFL is where ILU(0) failed on the laptop.

### 4.5 What to bring back

- the output of `run_implicit_checks.sh` for each GPU backend;
- the 4.3 `-log_view` lines for the Jacobian events;
- the 4.4 table (wall time, steps, Newton/step, Krylov/step) for CPU vs GPU vs explicit.

---

## 5. Pitfalls

- **Anything that stops the TS before the final time hangs the run.** `-ts_max_steps`, `-ts_max_time`,
  or a `stop_n` too small for `stop / time_step` makes `RDyAdvance` loop forever. To shorten a run, change
  `stop`/`stop_n` together in the YAML instead. (The branch fixed the `stop / time_step` rounding, but a manual
  cap still loops.)
- **Newton failure stops the run.** Time steps are fixed, and one SNES failure ends the run, since there
  is no adaptive dt yet. If a step fails, lower dt or try a stronger preconditioner.
- **Wet/dry fronts need `h_anuga_reg_parameter > 0`.** It is the implicit default (1e-3). With 0, Newton
  stalls at moving fronts above CFL 1. At CFL 5, plain Roe needs 1e-3 (1e-4 fails); augmented Roe is less
  sensitive. Cost: a small consistency error where h ≈ h_anuga (the 5–15 mm deep MMS rate drops from 0.914 to 0.898).
- **PETSc and libCEED HR residuals differ slightly when h_anuga > 0.** PETSc passes the cell velocities to the Roe
  solver; libCEED rebuilds them from h*(1, u, v). The Jacobian matches each backend, but the two backends give
  slightly different HR answers (space-time MMS rate 0.891 with PETSc vs 0.872 with libCEED). This
predates the branch.
- **BDF2 at CFL 5 failed** on the dry dam break (step 18); not investigated. BE is the robust choice for wet/dry;
  CN is more accurate on the dam break but has no positivity guarantee.
- **`-rdy_positivity vi`** (bound-constrained Newton) failed for augmented Roe; keep the default `clip`.
- **Plain Roe on a sloped bed** uses 3D edge lengths and areas, so a uniform state is not exactly steady
  (O(slope²)). This is why the time-only MMS uses a flat bed.
- **NERSC SSH certificates** last 24 h (`sshproxy.sh -u <user>` on the laptop).

---

## 6. Options

```text
-rdy_jacobian analytic|fd|mf     Jacobian (fd = finite differences with coloring, mf = Jacobian-free, no PC)
-rdy_jacobian_host               libCEED runs: build the Jacobian on the host instead
-rdy_jacobian_compare            libCEED runs: compare the libCEED and host Jacobians at every evaluation;
                                 errors if they differ by more than 1e-12 (slow)
-rdy_positivity clip|vi|none     keep depths >= 0 inside Newton (default clip)
-ts_type beuler|cn|bdf|theta     override numerics.temporal
-ts_adapt_type basic             re-enable PETSc time adaptivity (off by default; BE has no error estimate)
-snes_monitor -snes_converged_reason -ksp_converged_reason -snes_linesearch_monitor
-log_view -log_view_gpu_time
```
