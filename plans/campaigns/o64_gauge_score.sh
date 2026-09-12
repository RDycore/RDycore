#!/bin/bash
#SBATCH -A m4267_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 100
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --array=0-2
#SBATCH -o o64_slurm_%A_%a.out
# o64: the reverse of o63 -- score the MARK-calibrated fields on the gauges.
#
#   sbatch o64_gauge_score.sh        # from a login shell, never from inside a job
#
# o63 calibrates on the above-bed gauge stage and validates on the 46 marks.
# This asks the cheaper converse two weeks earlier: does the field the marks
# chose improve the gauges at all? If the mark-calibrated +/-30% field lowers
# the gauge misfit, the roughness request is the model's and o63's first
# outcome (gauges send the same classes to the floor) is likely; if it raises
# it, the two observables disagree and roughness is compensating for
# different errors in different places.
#
#   task 0: NLCD prior (the reference)
#   task 1: o62 3-class field, sigma_alpha 0.30 (the headline candidate)
#   task 2: o62 15-class field, sigma_alpha 0.30 (the control)
#
# The driver has no gauge eval-only path (-adjoint_hwm_eval_only needs a mark
# table), but relative mode evaluates the start point -- one forward plus one
# adjoint -- before TAO starts, and prints J0 as "objective scaled by 1/J0"
# and in the -adjoint_classes_grad_dump header. A watcher kills the run once
# that line appears, so nothing is spent on TAO's own first evaluation. The
# grad dump is the bonus: the gauge gradient at each field, which says which
# way the gauges would push the three classes before o63 tells us.
#
# J is sum (r/sigma)^2 / 2 over the 134 kept observations at sigma = 0.15 m,
# so RMSE = 0.15 * sqrt(2 J / 134).
set -u
export MPICH_GPU_SUPPORT_ENABLED=1
cd $SCRATCH/gpu-implicit
ADJ=$HOME/Codes/rdycore-manning/build-claude-gpu10/driver/rdycore_adjoint
RAIN=/global/cfs/cdirs/m4267/shared/data/harvey/spatially-distributed-rainfall/mm-per-hr/mrms/bin
CKPT=checkpoints_o37/o37.rdycore.r.104400.bin
OBS=obs_turning_h29_41.txt
export LD_LIBRARY_PATH=$HOME/Codes/petsc-claude/arch-perlmutter-opt-gcc-kokkos-cuda/lib:${LD_LIBRARY_PATH:-}
if ldd $ADJ | grep "not found" | grep -qv "visibility=hidden"; then
  echo "UNRESOLVED SHARED LIBRARIES for $ADJ:"; ldd $ADJ | grep "not found" | grep -v "visibility=hidden"; exit 1
fi
strings $ADJ | grep -q adjoint_obs_above_bed || { echo "STALE BINARY (no obs_above_bed): $ADJ"; exit 1; }

case ${SLURM_ARRAY_TASK_ID:-0} in
  0) TAG=prior;     INIT="" ;;
  1) TAG=c3_sa0.30;  INIT="-adjoint_classes_init o62_p_c3_sa0.30.txt" ;;
  2) TAG=c15_sa0.30; INIT="-adjoint_classes_init o62_p_c15_sa0.30.txt" ;;
  *) echo "bad task id"; exit 1 ;;
esac
for f in o43_window.yaml $CKPT $OBS turning30m_class.bin turning30m_manning.bin; do test -f $f || { echo "MISSING $f"; exit 1; }; done
[ -z "$INIT" ] || test -f ${INIT##* } || { echo "MISSING ${INIT##* }"; exit 1; }

NODES=${SLURM_JOB_NUM_NODES:-2}; RANKS=$((4 * NODES))
RUN="srun -N $NODES -n $RANKS -c 32 --cpu-bind=cores -G $RANKS --gpu-bind=none"

COM="-ts_adapt_type none -snes_max_it 50 -snes_rtol 1e-5 -ksp_max_it 300 -ksp_type gmres -ksp_pc_side right -ksp_rtol 1e-4 -pc_type pbjacobi -dm_vec_type kokkos -dm_mat_type baijkokkos -ts_trajectory_type memory -ts_trajectory_max_cps_ram 400 -adjoint_fd_samples 0"
# same prior and active set as o63, so the J0 here is o63's own start point for task 0
PRIOR="-adjoint_class_file turning30m_class.bin -adjoint_prior_file turning30m_manning.bin -adjoint_classes_relative -adjoint_sigma_alpha 0.30 -adjoint_classes_active 23,90,22"
GAUGE="-adjoint_calibrate_classes -adjoint_obs_file $OBS -adjoint_obs_above_bed -adjoint_obs_error 0.15 $PRIOR"
WIN="-restart $CKPT -adjoint_rain_start_hour 29 -raster_rain_dir $RAIN -raster_rain_start_date 2017,8,26,18,0"

LOG=o64_gauge_${TAG}.log; GRAD=o64_g_gauge_${TAG}.txt
echo "=== o64 $TAG: gauge objective of the ${INIT:-prior} field, $NODES nodes, $(date)"
$RUN $ADJ o43_window.yaml $COM $GAUGE $INIT $WIN -tao_max_it 0 -tao_monitor \
  -adjoint_classes_grad_dump $GRAD > $LOG 2>&1 & PID=$!
# stop once the start-point evaluation has been printed; TAO's own first
# evaluation after that would only repeat it
( while sleep 30; do grep -q "objective scaled by 1/J0" $LOG 2>/dev/null && { sleep 20; kill $PID 2>/dev/null; break; }; done ) & WATCH=$!
wait $PID; RC=$?; kill $WATCH 2>/dev/null
echo "run exit=$RC (143 = killed by the watcher after J0, as intended) $(date)"
grep -E "gauge observations|classes active|prior sigma|objective scaled|classes_init|evaluating" $LOG
echo "--- gauge gradient at this field ---"; cat $GRAD 2>/dev/null
echo
echo "compare: J0 of the prior (task 0) against the mark-calibrated fields (tasks 1, 2); RMSE = 0.15*sqrt(2*J/134)"
