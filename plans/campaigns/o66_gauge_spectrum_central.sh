#!/bin/bash
#SBATCH -A m4267_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 40
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --array=0-5
#SBATCH -o o66_slurm_%A_%a.out
# o66: the extra columns the o65 self-checks asked for -- central differences
#      for the classes that matter, and a small step for developed-high.
#
#   sbatch o66_gauge_spectrum_central.sh     # from a LOGIN SHELL only
#
# WHY. o65's one-sided +5% columns reproduce the adjoint gradient within a
# factor ~2 for nine classes but class 24 (developed-high) is 12x off: +5% on
# n_24 raises the Buffalo Bayou at Houston stage by 0.6-1.05 m where every
# other class moves it ~2 cm. Either the response is genuinely that nonlinear
# (a wet/dry or overtopping switch in the downstream reach) or 5% is simply
# past the linear range for the class the Houston cell sits in. Two things
# settle it: a small step (+/-1%) on 24, and -5% columns for the four classes
# the spectrum is built on (22, 23, 24, 90), so the analysis can use central
# differences (o65 +5% with these -5%) and compare against the adjoint.
#
# Tasks: 0: 24 at +1%   1: 24 at -1%   2: 22 at -5%   3: 23 at -5%
#        4: 24 at -5%   5: 90 at -5%
# Each is one forward, ~22 min on 2 nodes. Dumps o65_gauge_col<C>_e<EPS>.txt
# (signed eps in the name) alongside the o65 set; o65_gauge_spectrum.py reads
# them with --central.
set -u
export MPICH_GPU_SUPPORT_ENABLED=1
cd $SCRATCH/gpu-implicit
ADJ=$HOME/Codes/rdycore-manning/build-claude-gpu11/driver/rdycore_adjoint
RAIN=/global/cfs/cdirs/m4267/shared/data/harvey/spatially-distributed-rainfall/mm-per-hr/mrms/bin
CKPT=checkpoints_o37/o37.rdycore.r.104400.bin
OBS=obs_turning_h29_41.txt
export LD_LIBRARY_PATH=$HOME/Codes/petsc-claude/arch-perlmutter-opt-gcc-kokkos-cuda/lib:${LD_LIBRARY_PATH:-}
if ldd $ADJ | grep "not found" | grep -qv "visibility=hidden"; then
  echo "UNRESOLVED SHARED LIBRARIES for $ADJ:"; ldd $ADJ | grep "not found" | grep -v "visibility=hidden"; exit 1
fi
strings $ADJ | grep -q adjoint_obs_model_dump || { echo "STALE BINARY (no obs_model_dump): $ADJ"; exit 1; }
for f in o43_window.yaml o43_p_nlcd.txt $CKPT $OBS turning30m_class.bin turning30m_manning.bin; do test -f $f || { echo "MISSING $f"; exit 1; }; done

T=${SLURM_ARRAY_TASK_ID:-0}
case $T in
  0) C=24; EPS=0.01 ;;
  1) C=24; EPS=-0.01 ;;
  2) C=22; EPS=-0.05 ;;
  3) C=23; EPS=-0.05 ;;
  4) C=24; EPS=-0.05 ;;
  5) C=90; EPS=-0.05 ;;
  *) echo "bad task id $T"; exit 1 ;;
esac
TAG=col${C}_e${EPS}; TABLE=o66_pert_${TAG}.txt
awk -v c="$C" -v e="$EPS" '!/^#/{ if ($1==c) printf "%d %.10g\n", $1, $2*(1+e); else printf "%d %.10g\n", $1, $2 }' o43_p_nlcd.txt > $TABLE
DUMP=o65_gauge_${TAG}.txt; LOG=o65_gauge_${TAG}.log
if [ -s $DUMP ] && [ -s $DUMP.zb ]; then echo "o66 $TAG: dump exists, nothing to do"; exit 0; fi

NODES=${SLURM_JOB_NUM_NODES:-2}; RANKS=$((4 * NODES))
RUN="srun -N $NODES -n $RANKS -c 32 --cpu-bind=cores -G $RANKS --gpu-bind=none"
COM="-ts_adapt_type none -snes_max_it 50 -snes_rtol 1e-5 -ksp_max_it 300 -ksp_type gmres -ksp_pc_side right -ksp_rtol 1e-4 -pc_type pbjacobi -dm_vec_type kokkos -dm_mat_type baijkokkos -ts_trajectory_type memory -ts_trajectory_max_cps_ram 400 -adjoint_fd_samples 0"
PRIOR="-adjoint_class_file turning30m_class.bin -adjoint_prior_file turning30m_manning.bin -adjoint_classes_relative -adjoint_sigma_alpha 0.30"
GAUGE="-adjoint_calibrate_classes -adjoint_obs_file $OBS -adjoint_obs_above_bed -adjoint_obs_error 0.15 $PRIOR"
WIN="-restart $CKPT -adjoint_rain_start_hour 29 -raster_rain_dir $RAIN -raster_rain_start_date 2017,8,26,18,0"

echo "=== o66 $TAG: gauge-series forward of $TABLE (class $C, eps $EPS), $NODES nodes, $(date)"
$RUN $ADJ o43_window.yaml $COM $GAUGE $WIN -adjoint_obs_eval_only -adjoint_classes_init $TABLE \
  -adjoint_obs_model_dump $DUMP > $LOG 2>&1
RC=$?
echo "run exit=$RC $(date)"
grep -E "gauge observations|gauge eval" $LOG
if [ $RC -ne 0 ] || [ ! -s $DUMP ]; then echo "o66 $TAG FAILED (rc=$RC)"; tail -8 $LOG; rm -f $DUMP $DUMP.zb; exit 1; fi
echo "reference: base J 3.131332e+04; o65 +5% columns: 22 3.122705e+04, 23 3.113326e+04, 24 3.825755e+04"
