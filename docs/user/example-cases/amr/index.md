# Overview

Two small cases that exercise adaptive mesh refinement end to end: the mesh is
refined from the solution itself, coarsened again where the flow has moved on,
and the result is checked against an exact solution. Both run in seconds on a
laptop and are part of `ctest`.

AMR runs under the `rdycore_amr` driver, and the mesh must be simplicial,
because the adapt refines by simplex bisection:

```
rdycore_amr <case>.yaml \
  -refine_criterion jump -refine_at_start -refine_allow_coarsening
```

The transform the adapt needs is set in code -- simplex bisection, with labels
following strata -- because the AMR path cannot work without it, so there is
nothing to pass on the command line. `-dm_plex_transform_*` still overrides it.

On a GPU machine, launch with `--gpu-bind=none` and let PETSc pick the device
by rank:

```
srun -G <ngpus> --gpu-bind=none --cpu-bind=cores -n <nranks> rdycore_amr ...
```

Binding one GPU per rank (`--gpus-per-task=1`, `--gpu-bind=single:1`) hides the
peer GPUs, and GPU-aware MPI's intra-node path is CUDA IPC, which needs every
rank to see its peers': it fails with `cuIpcOpenMemHandle: invalid argument`,
or hangs.

`rdycore_amr -help` lists the rest. The meshes are written by
[`share/meshes/make_swe_tri_mesh.py`](../../../../share/meshes/make_swe_tri_mesh.py),
which needs no gmsh, and the figures below by
[`tools/postprocessing/amr_dam_break.py`](../../../../tools/postprocessing/amr_dam_break.py)
(`error`, `plot`, `panels` and `movie`).

## What drives the refinement

`-refine_criterion jump` marks cells from the state: the indicator of a cell is
the largest jump in water-surface elevation `h + z` across its faces. Free
surface rather than depth, so still water over a varying bed indicates nothing,
and a dam is marked at step 0 before anything moves.

Which cells that selects is a pair of PETSc `VecTagger`s, so it is a runtime
choice; RDycore only sets the defaults, and anything given on the command line
wins:

| option | effect |
| --- | --- |
| `-rdy_amr_refine_vec_tagger_type relative -rdy_amr_refine_vec_tagger_box 0.5,1` | at least half the largest jump (default) |
| `-rdy_amr_refine_vec_tagger_type cdf -rdy_amr_refine_vec_tagger_box 0.95,1` | the worst 5% of cells |
| `-rdy_amr_coarsen_vec_tagger_box 0,0.05` | coarsen the quietest 5%; the default is `0,0.1` (needs `-refine_allow_coarsening`) |
| `-rdy_amr_buffer_layers <n>` | also refine `n` cell layers around each marked cell |
| `-rdy_amr_max_levels <n>` | deepest level a cell may be marked at (default 3) |
| `-rdy_amr_balance 0` | do not repartition at each rebuild (on by default with `-refine_allow_coarsening`) |

`-rdy_amr_max_levels` caps the level a cell may be *marked* at, measured
against the area of the base cell it descends from, so it binds the same way
on a graded mesh as on a uniform one. The mesh can come back one level finer
than the cap: the transform splits a marked triangle into two children or,
where conformity needs a second edge split, into three of areas 1/2, 1/4, 1/4.
`-refine_check_max_level` reports the deepest level after every adapt and fails
if it ever goes further than that.

The default is scale-free on purpose. An absolute threshold disables itself:
halving a cell halves the jump across its faces, so a front that was over the
threshold falls under it after one adapt and the mesh stops following it.

## 2D dam break through a breach

`driver/tests/amr/dam_break_amr.yaml`: a 10 m x 5 m basin with a dam across the
middle pierced by a central breach, 10 m of water upstream and 5 m downstream,
352 triangles to start.

<img src="amr-dam-break.gif" width="70%" height="70%">

The mesh is refined along the dam before the first step, follows the jet
through the breach and the front spreading downstream, and coarsens behind
them: 448 cells at the start, 968 at t = 1 s, 1120 at t = 2 s, 1192 at t = 4 s,
with the count falling again wherever the flow has settled.
The same four instants, side by side:

<img src="amr-dam-break-panels.png" width="60%" height="60%">

## 1D dam break on a 2D mesh

`driver/tests/amr/channel_wave_amr.yaml`: a 50 m x 2 m channel, frictionless,
2 m of water over 1 m, 800 triangles, stopping before either wave reaches an
end wall. Nothing varies across the channel, so this is a wet-bed Riemann
problem run on a 2D mesh, and it has an exact solution -- Stoker's -- which
`amr_dam_break.py error` evaluates independently and compares against.

<img src="amr-channel-stoker.png" width="80%" height="80%">

The computed star depth, the shock position and the rarefaction fan all sit on
the exact solution; the area-weighted L1 depth error is 0.0144 m at t = 4 s on
1184 cells. L1 rather than Linf because the solution has a shock, where a
first-order scheme's pointwise error does not converge.

## Why coarsening matters

Both meshes at t = 4 s, refined on the same criterion; only the lower one is
allowed to coarsen.

<img src="amr-channel-wake.png" width="90%" height="90%">

Refining alone (top, 2788 cells) spends the mesh on the wake: a cell refined at
any adapt stays refined, so the fine region becomes the union of every past
position of the wave, where the solution is flat and the error is near zero.
Measured, 99.6% of its refined cells sit there. Rebuilding the mesh from the
base mesh at every adapt (bottom, 1184 cells) puts them on the shock instead,
for the same error at well under half the cells.

## Load balancing

Refinement concentrates on the feature, so without anything else one rank ends
up owning most of the fine cells while the others wait at the next halo
exchange. `-refine_report_balance` reports the largest number of owned cells on
any rank against the mean, on either side of every adapt.

Each rebuild therefore repartitions the base mesh first, weighted by the number
of cells each base cell is about to carry, and rebuilds from the result; the
adapted mesh is not partitioned directly, because a base cell's descendants
have to stay together for the interpolation and the transfer. Only a rebuild
can do this, so it is on with `-refine_allow_coarsening` and off otherwise.

Measured on a refinement box held in one corner of the dam mesh, four levels
deep, on three ranks: max/mean climbs to 2.85 without it (4564 owned cells on
one rank against a mean of 1599) and stays at 1.12 with it (1792 cells), for
exactly the same sequence of meshes.

## Conservation and the gates

`-refine_check_volume` reports the water volume `sum(h * area)` across every
adapt. The transfer between the two meshes is area-weighted
(`U_new = sum |new ^ old| U_old / |new|`), and what it carries is the free
surface `h + z` rather than the depth, so that a child cell's depth is rebuilt
against its own bed. A triangle's bed is planar and bisection splits it at edge
midpoints, so `sum(z * area)` is the same on both meshes and `sum(h * area)` is
preserved exactly, the only change being the order of the summation: measured at
or below 8.2e-15 relative over every adapt of both cases, on 1 and 3 ranks.

That holds while the domain stays wet, which is what both cases here are. Where
the transferred surface falls below the new bed the depth is clamped at zero,
which adds water, so a transfer that crosses a shoreline is not conservative.
Wetting and drying is the normal case for a flooding code and is not yet
covered by these gates.

The `ctest` suite runs both cases on 1 and 3 ranks with and without coarsening.
Beyond that:

| gate | what it asserts |
| --- | --- |
| `amr_off_is_bitwise` | the adapt machinery with a tagger that marks nothing reproduces the plain `rdycore` driver byte for byte |
| `amr_same_on_any_ranks_{refine,coarsen}` | the same sequence of meshes on 1 rank, on 3, and on 3 without balancing |
| `amr_still_water_{ramp,dem}_c_np_{1,3}` | a lake at rest on a sloping bed survives every adapt exactly, with the bed from the mesh and from a file |
| `amr_graded_max_level_c_np_{1,3}` | on a graded mesh, no cell ends more than one level past `-rdy_amr_max_levels` |
| `amr_balance_improves` | balancing brings max/mean down without changing the mesh, on a case that is still badly balanced without it |
