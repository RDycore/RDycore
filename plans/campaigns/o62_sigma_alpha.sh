#!/bin/bash
#SBATCH -A m4267_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 360
#SBATCH -N 2
#SBATCH --array=0-3
#SBATCH -o o62_slurm_%A_%a.out
# o62: the calibration under the prior the paper SAYS it uses.
#
#   sbatch o62_sigma_alpha.sh          # four independent 2-node tasks
#   sbatch o62_sigma_alpha.sh          # again: each task resumes from its own dump
#
# Every production calibration so far ran -adjoint_sigma_n 0.015, an ABSOLUTE
# prior width that the driver applies as |n - n_prior|^2 -- which is +/-9% on
# developed-high and +/-56% on barren. The paper's Question 1 and its
# "reformulation" paragraph describe a UNIFORM relative width sigma_alpha
# (+/-15% holds the first step's excursion, +/-30% does not) that no logged
# campaign ever ran. This array runs it, at the two widths the text names, for
# both the fifteen-class problem and the three-class active set the production
# spectrum selects (23, 90, 22 carry the leading eigenvector under either
# prior form).
#
#   task 0: 15 classes, sigma_alpha 0.15     task 2: 3 classes, sigma_alpha 0.15
#   task 1: 15 classes, sigma_alpha 0.30     task 3: 3 classes, sigma_alpha 0.30
#
# What each answers. Tasks 0/1: does +/-15% hold the excursion and +/-30%
# release it -- the claim as written. Tasks 2/3: the candidate headline
# field, calibrated against the prior a hydrologist can actually vouch for;
# the spectrum says +/-30% supports three directions, which is this set.
#
# TWO NODES, NOT FOUR. Measured 2026-08-28: 22 min/forward at n8 vs 15 at n16,
# but the 4-node slot queued 7 h. At ~85 min per TAO iteration (gradient plus
# line-search trials) a 6-hour slot gives 3-4 iterations: enough for the
# excursion question, and the three-class run converged in four on o59. Each
# task rewrites its dump every iteration and a resubmit resumes from it, so
# the wall is a stopping rule, not a lost run. A watcher archives every
# distinct dump as o62_p_<tag>_it<k>.txt so the FIRST step survives -- that
# is the number Question 1 turns on.
#
# The last 25 minutes of each task score the final dump with one eval-only
# forward, because the calibration's own "hwm final" line is lost when the
# wall kills it.
set -u
export MPICH_GPU_SUPPORT_ENABLED=1
cd $SCRATCH/gpu-implicit
ADJ=$HOME/Codes/rdycore-manning/build-claude-gpu9/driver/rdycore_adjoint

# gpu9, not gpu8. gpu8 was linked against a PETSc arch that was replaced on
# 2026-09-03, and the first o62 submission died in 9 s with exit 127
# ("libmuparser.so.2: cannot open shared object file"). gpu9 is built against
# petsc-claude main's arch-perlmutter-opt-gcc-kokkos-cuda after that arch was
# reconfigured with the RDycore package set. The binary has no RPATH, so
# PETSc, muparser, hdf5, libCEED and Kokkos resolve through LD_LIBRARY_PATH --
# which earlier campaigns inherited from the submitting login shell and a job
# submitted over non-interactive ssh does not. Set it here and refuse to run
# if anything is unresolved: a 6-hour slot is not the place to find out.
export LD_LIBRARY_PATH=$HOME/Codes/petsc-claude/arch-perlmutter-opt-gcc-kokkos-cuda/lib:${LD_LIBRARY_PATH:-}
# ("visibility=hidden => not found" is an ldd artifact of the -fvisibility=hidden
# token in PETSc's pkg-config output, not a library: readelf shows no such
# NEEDED entry and the binary loads and runs. Only real libraries count.)
if ldd $ADJ | grep "not found" | grep -qv "visibility=hidden"; then
  echo "UNRESOLVED SHARED LIBRARIES for $ADJ:"; ldd $ADJ | grep "not found" | grep -v "visibility=hidden"; exit 1
fi
RAIN=/global/cfs/cdirs/m4267/shared/data/harvey/spatially-distributed-rainfall/mm-per-hr/mrms/bin
CKPT=checkpoints_o37/o37.rdycore.r.104400.bin

NODES=${SLURM_JOB_NUM_NODES:-2}; RANKS=$((4 * NODES))
RUN="srun -N $NODES -n $RANKS -c 32 --cpu-bind=cores -G $RANKS --gpu-bind=none"
CAL_MIN=300                      # calibration budget; ~25 min left to score

case ${SLURM_ARRAY_TASK_ID:-0} in
  0) TAG=c15_sa0.15; SA=0.15; ACTIVE="" ;;
  1) TAG=c15_sa0.30; SA=0.30; ACTIVE="" ;;
  2) TAG=c3_sa0.15;  SA=0.15; ACTIVE="-adjoint_classes_active 23,90,22" ;;
  3) TAG=c3_sa0.30;  SA=0.30; ACTIVE="-adjoint_classes_active 23,90,22" ;;
  *) echo "bad task id"; exit 1 ;;
esac

test -f o43_window.yaml || { echo "MISSING o43_window.yaml"; exit 1; }
test -f $CKPT            || { echo "MISSING $CKPT"; exit 1; }
strings $ADJ | grep -q adjoint_sigma_alpha || { echo "STALE BINARY (no sigma_alpha): $ADJ"; exit 1; }

COM="-ts_adapt_type none -snes_max_it 50 -snes_rtol 1e-5 -ksp_max_it 300 -ksp_type gmres -ksp_pc_side right -ksp_rtol 1e-4 -pc_type pbjacobi -dm_vec_type kokkos -dm_mat_type baijkokkos -ts_trajectory_type memory -ts_trajectory_max_cps_ram 400 -adjoint_fd_samples 0"
CAL="-adjoint_calibrate_classes -adjoint_hwm_file turning30m_hwm_obs_clusterA.txt -adjoint_class_file turning30m_class.bin -adjoint_prior_file turning30m_manning.bin -adjoint_obs_freq 300 -adjoint_obs_error 0.15 -adjoint_classes_relative -adjoint_sigma_alpha $SA $ACTIVE"
WIN="-restart $CKPT -adjoint_rain_start_hour 29 -raster_rain_dir $RAIN -raster_rain_start_date 2017,8,26,18,0"

DUMP=o62_p_${TAG}.txt; GRAD=o62_g_${TAG}.txt; LOG=o62_${TAG}.log
RESUME=""
if [ -f $DUMP ]; then
  cp $DUMP o62_start_${TAG}.txt; RESUME="-adjoint_classes_init o62_start_${TAG}.txt"
  echo "resuming $TAG from its previous dump"
fi

# archive every distinct dump version: it<k> is the k-th table written this job
( k=0; last=""
  while sleep 30; do
    [ -f $DUMP ] || continue
    cur=$(md5sum < $DUMP)
    if [ "$cur" != "$last" ]; then k=$((k+1)); cp $DUMP o62_p_${TAG}_it${k}.txt; last=$cur; fi
  done ) & WATCH=$!

echo "=== o62 $TAG: sigma_alpha=$SA ${ACTIVE:-all 15 classes} on $NODES nodes, $(date)"
timeout -k 60 ${CAL_MIN}m $RUN $ADJ o43_window.yaml $COM $CAL $RESUME $WIN \
  -tao_max_it 12 -tao_monitor -tao_ls_type armijo \
  -adjoint_classes_dump $DUMP -adjoint_classes_grad_dump $GRAD >> $LOG 2>&1
RC=$?; kill $WATCH 2>/dev/null
echo "calibration exit=$RC (124 = wall budget, dump kept) $(date)"
grep -E "prior sigma|classes active|hwm init|TAO,|hwm final|class recovery" $LOG | tail -14

if [ -f $DUMP ]; then
  echo "=== o62 $TAG: scoring final dump $(date)"
  $RUN $ADJ o43_window.yaml $COM $CAL -adjoint_hwm_eval_only -adjoint_classes_init $DUMP $WIN \
    > o62_score_${TAG}.log 2>&1
  echo "score exit=$? $(date)"; grep -h "hwm eval (class table)" o62_score_${TAG}.log
  echo "--- final class table ---"; cat $DUMP
fi
echo
echo "compare against (absolute sigma_n 0.015):  NLCD prior MAE 0.7188 | 15 classes 0.6116 | 3 classes 0.6274"
