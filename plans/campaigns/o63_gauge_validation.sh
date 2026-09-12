#!/bin/bash
#SBATCH -A m4267_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 360
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -o o63_slurm_%j.out
# o63: calibrate on the gauges, validate on the marks -- Donghui's design,
# accepted at the 2026-09-09 coauthor meeting as the paper's validation.
#
#   sbatch o63_gauge_validation.sh          # resumes from its own dump on resubmit
#
# SUBMIT FROM A LOGIN SHELL, NEVER FROM INSIDE ANOTHER JOB. The first
# submission (58126942, 2026-09-12) was issued by the gpu10 build job -- a
# shared-QOS allocation -- and died in 8 s: "srun: Job step's --cpus-per-task
# value exceeds that of job (32 > 1)". sbatch reads SLURM_* from a parent
# allocation as if they were command-line options, and those outrank the
# #SBATCH lines below. The two task-shape lines above are belt and braces;
# they do not protect against a parent job's environment.
#
# The observable is USGS stage at the 13 rain-driven gauges over the production
# window (event hours 29-41, 15-minute cadence, obs_turning_h29_41.txt), with
# -adjoint_obs_above_bed: an observation counts only while the observed water
# surface is above the cell bed. On this 30 m mesh that keeps the two Buffalo
# Bayou main-stem gauges and the two reservoir gauges for the whole window,
# part of Fulshear, and none of the seven tributary gauges whose cell bed sits
# above even the window's peak stage (the driver prints the exact count). The
# parameterization is the headline one: three classes (23, 90, 22) at the
# uniform relative prior sigma_alpha = 0.30 the coauthors accept.
#
# Validation: the final class table is scored on the 46 cluster-A high-water
# marks by an eval-only forward. Compare against the mark-calibrated fields --
# NLCD prior 0.7188 m; 3 classes on marks 0.6274 m (absolute prior) and the
# o62 sigma_alpha runs when they land. If the gauge-calibrated field predicts
# the marks, that is the strongest result the paper could have; if it does
# not, that is a finding too.
#
# TWO NODES, six hours, ~85 min per TAO iteration; dump rewritten every
# iteration; the last 25 minutes score. Binary is gpu10 (gpu9 + the above-bed
# mask); never rebuild a build dir a queued job launches from.
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
for f in o43_window.yaml $CKPT $OBS turning30m_hwm_obs_clusterA.txt; do test -f $f || { echo "MISSING $f"; exit 1; }; done

NODES=${SLURM_JOB_NUM_NODES:-2}; RANKS=$((4 * NODES))
RUN="srun -N $NODES -n $RANKS -c 32 --cpu-bind=cores -G $RANKS --gpu-bind=none"
CAL_MIN=300
TAG=gauge_c3_sa0.30

COM="-ts_adapt_type none -snes_max_it 50 -snes_rtol 1e-5 -ksp_max_it 300 -ksp_type gmres -ksp_pc_side right -ksp_rtol 1e-4 -pc_type pbjacobi -dm_vec_type kokkos -dm_mat_type baijkokkos -ts_trajectory_type memory -ts_trajectory_max_cps_ram 400 -adjoint_fd_samples 0"
PRIOR="-adjoint_class_file turning30m_class.bin -adjoint_prior_file turning30m_manning.bin -adjoint_classes_relative -adjoint_sigma_alpha 0.30 -adjoint_classes_active 23,90,22"
GAUGE="-adjoint_calibrate_classes -adjoint_obs_file $OBS -adjoint_obs_above_bed -adjoint_obs_error 0.15 $PRIOR"
MARKS="-adjoint_calibrate_classes -adjoint_hwm_file turning30m_hwm_obs_clusterA.txt -adjoint_obs_freq 300 -adjoint_obs_error 0.15 $PRIOR"
WIN="-restart $CKPT -adjoint_rain_start_hour 29 -raster_rain_dir $RAIN -raster_rain_start_date 2017,8,26,18,0"

DUMP=o63_p_${TAG}.txt; GRAD=o63_g_${TAG}.txt; LOG=o63_${TAG}.log
RESUME=""
if [ -f $DUMP ]; then cp $DUMP o63_start_${TAG}.txt; RESUME="-adjoint_classes_init o63_start_${TAG}.txt"; echo "resuming from previous dump"; fi

( k=0; last=""
  while sleep 30; do
    [ -f $DUMP ] || continue
    cur=$(md5sum < $DUMP)
    if [ "$cur" != "$last" ]; then k=$((k+1)); cp $DUMP o63_p_${TAG}_it${k}.txt; last=$cur; fi
  done ) & WATCH=$!

echo "=== o63 $TAG: calibrating classes 23,90,22 on above-bed gauge stage, $NODES nodes, $(date)"
timeout -k 60 ${CAL_MIN}m $RUN $ADJ o43_window.yaml $COM $GAUGE $RESUME $WIN \
  -tao_max_it 12 -tao_monitor -tao_ls_type armijo \
  -adjoint_classes_dump $DUMP -adjoint_classes_grad_dump $GRAD >> $LOG 2>&1
RC=$?; kill $WATCH 2>/dev/null
echo "calibration exit=$RC (124 = wall budget, dump kept) $(date)"
grep -E "gauge observations:|prior sigma|classes active|TAO,|class recovery" $LOG | tail -14

if [ -f $DUMP ]; then
  echo "=== o63 $TAG: VALIDATION -- scoring the gauge-calibrated field on the 46 marks $(date)"
  $RUN $ADJ o43_window.yaml $COM $MARKS -adjoint_hwm_eval_only -adjoint_classes_init $DUMP $WIN > o63_score_${TAG}.log 2>&1
  echo "score exit=$? $(date)"; grep -h "hwm eval (class table)" o63_score_${TAG}.log
  echo "--- gauge-calibrated class table ---"; cat $DUMP
fi
echo
echo "compare (peak-WSE MAE on the 46 marks):  NLCD prior 0.7188 | 3 classes calibrated ON the marks 0.6274 (absolute prior) | o62 sigma_alpha runs"
