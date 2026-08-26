#!/bin/bash
#SBATCH -A m1516_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 720
#SBATCH -N 4
#SBATCH -o o50_slurm_%j.out
# o50: the FEW-parameter calibration -- o48's setup with all but a handful of
# land-cover classes frozen at their NLCD prior.
#
#   sbatch o50_calibrate_active.sh <TAO_ITS> <CODES> [INIT_FILE]
#   e.g.  sbatch o50_calibrate_active.sh 3 21,22,23,24
#
# Run this AFTER reading o48's start-point gradient dump (o48_g_<jobid>.txt):
# pick the classes whose |dJ/dn| * n_prior is large, since that product is the
# gradient in the variable actually being optimized. The peak observable at
# these 46 marks constrains about ONE roughness degree of freedom (o47), so
# asking it for fifteen buys nothing and costs the bound-pinning that o28/o30
# measured. Freezing the rest is the structural fix -- fewer, better-determined
# parameters -- rather than leaning on the Tikhonov weight after the fact.
#
# The one-parameter bar this has to clear is the uniform-alpha curve
# (o44/o45/o49): MAE 0.7188 at alpha 1.0, ~0.69 at the sigma_n = 0.015
# optimum near alpha 0.70, 0.6614 at 0.45, 0.6392 at 0.30.
set -u
TAO_ITS=${1:?need TAO_ITS}
CODES=${2:?need comma-separated NLCD codes, e.g. 21,22,23,24}
INIT_FILE=${3:-}

export MPICH_GPU_SUPPORT_ENABLED=1
# o51 measured the production window forward at 36 min (n4, 1 node),
# 21m55s (n8, 2 nodes) and 15m01s (n16, 4 nodes), all three reproducing
# J 8.082566e+02 / MAE 0.7188 to every printed digit -- no decomposition
# dependence. n16 is 2.4x faster at 60% parallel efficiency; we buy the
# wall-clock because it puts a whole calibration inside ONE queue slot,
# and queue wait, not node-hours, is the binding constraint.
RANKS=${RANKS:-16}
NODES=$((RANKS / 4))
cd $SCRATCH/gpu-implicit
ADJ=$HOME/Codes/rdycore-manning/build-claude-gpu6/driver/rdycore_adjoint
RAIN=/global/cfs/cdirs/m4267/shared/data/harvey/spatially-distributed-rainfall/mm-per-hr/mrms/bin

START_HOUR=29
WINDOW_SEC=43200
CKPT=checkpoints_o37/o37.rdycore.r.104400.bin
test -f "$CKPT" || { echo "MISSING $CKPT"; exit 1; }
test -f o43_window.yaml || { echo "MISSING o43_window.yaml"; exit 1; }
strings $ADJ | grep -q adjoint_classes_relative || { echo "STALE BINARY: $ADJ"; exit 1; }

COM="-ts_adapt_type none -snes_max_it 50 -snes_rtol 1e-5 -ksp_max_it 300 -ksp_type gmres -ksp_pc_side right -ksp_rtol 1e-4 -pc_type pbjacobi -dm_vec_type kokkos -dm_mat_type baijkokkos -ts_trajectory_type memory -ts_trajectory_max_cps_ram 400 -adjoint_fd_samples 0"
CAL="-adjoint_calibrate_classes -adjoint_hwm_file turning30m_hwm_obs_clusterA.txt -adjoint_class_file turning30m_class.bin -adjoint_prior_file turning30m_manning.bin -adjoint_obs_freq 300 -adjoint_obs_error 0.15 -adjoint_classes_relative -adjoint_sigma_n 0.015 -adjoint_classes_active $CODES"
DUMP="o50_p_${SLURM_JOB_ID}.txt"
GRAD="o50_g_${SLURM_JOB_ID}.txt"
RESUME=""
if [ -n "$INIT_FILE" ]; then RESUME="-adjoint_classes_init $INIT_FILE"; fi

echo "=== o50 few-parameter calibration start $(date)"
echo "    active classes: $CODES | its this job: $TAO_ITS | start: ${INIT_FILE:-NLCD prior (alpha = 1)}"
srun -N $NODES -n $RANKS -c 32 --cpu-bind=cores -G $RANKS --gpu-bind=none \
  $ADJ o43_window.yaml $COM $CAL $RESUME \
  -restart $CKPT -adjoint_rain_start_hour $START_HOUR \
  -tao_max_it $TAO_ITS -tao_monitor \
  -adjoint_classes_dump $DUMP -adjoint_classes_grad_dump $GRAD \
  -raster_rain_dir $RAIN -raster_rain_start_date 2017,8,26,18,0 > o50_${SLURM_JOB_ID}.log 2>&1
echo "exit=$? $(date)"
grep -E "warm start|class calibration|hwm init|hwm final|TAO,|class recovery" o50_${SLURM_JOB_ID}.log | tail -25
echo "--- per-class table (frozen classes must show rel_err exactly 0.000) ---"
grep -A17 "NLCD  prior_n" o50_${SLURM_JOB_ID}.log | tail -18
echo "--- continue: sbatch o50_calibrate_active.sh <its> $CODES $DUMP"
