# Campaign plan (2026-08-19 evening)

Guiding constraint: **node-hours are scarce**. So the expensive part of the
science (the adjoint/optimization loop) stays at 1 km, where it is essentially
free, and Perlmutter is used only for 30 m **forward** runs, which need no
trajectory storage.

## Why not do the calibration itself at 30 m

Adjoint cost at 30 m is dominated by trajectory storage, not arithmetic:

| | value |
|---|---|
| state vector (2.93M cells x 3) | 70 MB |
| measured disk-trajectory footprint, ARK-IMEX | **325 MB/step** (stages included) |
| measured 40-step forward+adjoint, disk trajectory, np=6 | **1577 s** (vs 244 s forward-only) |
| 1 h window at dt = 0.25 s | 14,400 steps |
| => 1 h window, solution-only, in RAM | ~1.0 TB (fits on 2-4 CPU nodes) |
| => 12 h window in RAM | ~12 TB (~30 nodes just for the trajectory) |

Disk trajectories are unusable (~6.5x slowdown, 13 GB for 40 steps). PETSc here
is **not built with `--download-revolve`**, so limited-memory checkpointing is
unavailable -- either add it to the Perlmutter build, or hold the whole window
in RAM, which is affordable only for short windows.

Meanwhile the sensitivity study says short windows are exactly what we do not
want: information about n accumulates *superlinearly* within a window
(information per hour of window: 1.4e3 at 1 h, 2.9e3 at 6 h, 5.3e3 at 12 h), so
one 12 h window is worth far more than twelve 1 h windows.

At 1 km the same 12 h window is 1440 steps and the trajectory is ~95 MB. The
science is affordable there and unaffordable at 30 m, so that is where it goes.

## On windowing (cycled assimilation)

Standard 4D-Var practice is short cycled windows, but that is mostly for STATE
estimation, where each cycle is re-anchored by new observations. For PARAMETER
estimation the parameter's effect accumulates with time, which is exactly what
the sensitivity measurement shows -- so chopping into short windows loses
signal. Windowing still helps us in two ways, and we use both:

1. **Storage**: only one window's trajectory need ever be in memory.
2. **Parallelism**: given a reference trajectory, window gradients are
   independent and can be computed concurrently, then summed.

## Tonight, locally (6 cores, no allocation)

1. Build the NLCD 2021 -> Manning map for the domain, on both meshes (18 classes,
   Donghui's table). Deliverable: per-cell prior + class index per cell.
2. 1 km, NLCD-class calibration against the real Harvey gauges: ~18 parameters,
   12 h rising window, MRMS forcing, class map as prior and Tikhonov reference.
   This is the actual science result.
3. Window-length study at 1 km: repeat with 3/6/12/24 h to confirm the 12 h
   choice and find where the return flattens.

## Perlmutter, when it returns (CPU nodes, forward only)

4. **ARK-IMEX vs semi-implicit friction**, Turning_30m, short Harvey window,
   identical settings otherwise. Answers the transfer question: production runs
   semi-implicit, calibration must run ARK-IMEX, and the calibrated n is only
   useful if the two agree. ~2 nodes, ~1-2 h. No adjoint, no trajectory.
5. **Validation forward**: 30 m with the 1 km-calibrated class roughness vs the
   uniform 0.015, both against the 18 gauges. Shows whether the calibrated
   roughness improves the 30 m model. ~2 nodes, ~2 h. No adjoint.

Only after 4 and 5 justify it: a 30 m adjoint on a 1-2 h window, 4 nodes,
requiring a PETSc build with `--download-revolve`.

## CPU, not GPU

Ask for **CPU nodes**. GPU support in RDycore comes through the CEED backend,
but the Jacobian/adjoint path requires the PETSc backend (CEED is explicitly
rejected in `RegisterSWERHSJacobian`), and the implicit friction IFunction is
host-array code. On GPU nodes the calibration path would run on the CPUs anyway
while the GPUs sat idle. The 30 m forward comparisons should also use the PETSc
backend, to be consistent with what the calibration uses.
