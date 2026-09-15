#!/bin/bash
#SBATCH -A m4267_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 40
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --array=0-15
#SBATCH -o o65_slurm_%A_%a.out
# o65: the Gauss-Newton spectrum of the GAUGE observable -- the same sixteen
#      forwards as o58/o61 (Sec 6.4), with the modelled stage at the 13 gauges
#      x 48 times written instead of the 46 mark peaks.
#
#   sbatch o65_gauge_spectrum.sh        # from a LOGIN SHELL, never from inside a job
#
# WHY. o63 calibrated three classes on the gauges at the survey grade
# sigma = 0.15 m with 134 autocorrelated records counted as independent, and
# stopped on the 3x bound after one projected step. Whether the gauges
# determine ANY roughness combination at their own error and correlation --
# and how their informative directions sit against the marks' (Emil,
# 2026-09-15) -- is a property of the observation-sensitivity matrix S, not of
# that run. With S in hand every weighting choice (sigma, the above-bed mask,
# per-gauge weights, an AR(1) correlation model, per-gauge demeaning) is a
# 15 x 15 post-processing step; see plans/o63-gauge-weight-audit.md Q5.
#
# WHAT EACH TASK DOES. Task 0 runs the NLCD prior (o43_p_nlcd.txt); task k
# runs the table with class k's Manning n raised by EPS (5%, as o58/o61), the
# same 43,200-step production window from the o37 checkpoint. Each is ONE
# forward (-adjoint_obs_eval_only: no adjoint, no TAO), ~22 min on 2 nodes,
# and writes o65_gauge_<tag>.txt (modelled WSE, obs-table format) and
# o65_gauge_<tag>.txt.zb (gauge cell + bed). The printed J/RMSE uses the
# above-bed mask at sigma 0.15 so task 0 must reproduce o64's J0 = 31313.32
# (RMSE 3.243 m) -- that is the self-check that the binary and inputs are the
# ones the paper's gauge numbers came from.
#
# RESUMABLE. A task whose dump exists and is non-empty exits immediately, so a
# partial array can be resubmitted unchanged.
#
# ANALYSIS (on the laptop, after scp of o65_gauge_*.txt and *.zb):
#   python3 o65_gauge_spectrum.py 0.05 obs_turning_h29_41.txt o65_gauge_base.txt \
#       o65_gauge_col*.txt --sigma-obs 0.15 --sigma-alpha 0.30 \
#       --marks o58_e0.05w43200_pk_base.txt o58_e0.05w43200_pk_col*.txt
set -u
export MPICH_GPU_SUPPORT_ENABLED=1
cd $SCRATCH/gpu-implicit
ADJ=$HOME/Codes/rdycore-manning/build-claude-gpu11/driver/rdycore_adjoint
RAIN=/global/cfs/cdirs/m4267/shared/data/harvey/spatially-distributed-rainfall/mm-per-hr/mrms/bin
CKPT=checkpoints_o37/o37.rdycore.r.104400.bin
OBS=obs_turning_h29_41.txt
EPS=${EPS:-0.05}
export LD_LIBRARY_PATH=$HOME/Codes/petsc-claude/arch-perlmutter-opt-gcc-kokkos-cuda/lib:${LD_LIBRARY_PATH:-}
if ldd $ADJ | grep "not found" | grep -qv "visibility=hidden"; then
  echo "UNRESOLVED SHARED LIBRARIES for $ADJ:"; ldd $ADJ | grep "not found" | grep -v "visibility=hidden"; exit 1
fi
strings $ADJ | grep -q adjoint_obs_model_dump || { echo "STALE BINARY (no obs_model_dump): $ADJ"; exit 1; }
for f in o43_window.yaml o43_p_nlcd.txt $CKPT $OBS turning30m_class.bin turning30m_manning.bin; do test -f $f || { echo "MISSING $f"; exit 1; }; done

T=${SLURM_ARRAY_TASK_ID:-0}
CODES=($(awk '!/^#/{print $1}' o43_p_nlcd.txt))   # 15 NLCD codes in table order
if [ "$T" -eq 0 ]; then
  TAG=base; TABLE=o43_p_nlcd.txt
else
  C=${CODES[$((T-1))]}; TAG=col$C; TABLE=o65_pert_e${EPS}_$C.txt
  awk -v c="$C" -v e="$EPS" '!/^#/{ if ($1==c) printf "%d %.10g\n", $1, $2*(1+e); else printf "%d %.10g\n", $1, $2 }' o43_p_nlcd.txt > $TABLE
fi
DUMP=o65_gauge_${TAG}.txt; LOG=o65_gauge_${TAG}.log
if [ -s $DUMP ] && [ -s $DUMP.zb ]; then echo "o65 $TAG: dump exists, nothing to do"; exit 0; fi

NODES=${SLURM_JOB_NUM_NODES:-2}; RANKS=$((4 * NODES))
RUN="srun -N $NODES -n $RANKS -c 32 --cpu-bind=cores -G $RANKS --gpu-bind=none"

COM="-ts_adapt_type none -snes_max_it 50 -snes_rtol 1e-5 -ksp_max_it 300 -ksp_type gmres -ksp_pc_side right -ksp_rtol 1e-4 -pc_type pbjacobi -dm_vec_type kokkos -dm_mat_type baijkokkos -ts_trajectory_type memory -ts_trajectory_max_cps_ram 400 -adjoint_fd_samples 0"
# same prior, mask and sigma as o63/o64 so the printed J of task 0 is o64's J0
PRIOR="-adjoint_class_file turning30m_class.bin -adjoint_prior_file turning30m_manning.bin -adjoint_classes_relative -adjoint_sigma_alpha 0.30"
GAUGE="-adjoint_calibrate_classes -adjoint_obs_file $OBS -adjoint_obs_above_bed -adjoint_obs_error 0.15 $PRIOR"
WIN="-restart $CKPT -adjoint_rain_start_hour 29 -raster_rain_dir $RAIN -raster_rain_start_date 2017,8,26,18,0"

echo "=== o65 $TAG: gauge-series forward of $TABLE, $NODES nodes, $(date)"
$RUN $ADJ o43_window.yaml $COM $GAUGE $WIN -adjoint_obs_eval_only -adjoint_classes_init $TABLE \
  -adjoint_obs_model_dump $DUMP > $LOG 2>&1
RC=$?
echo "run exit=$RC $(date)"
grep -E "gauge observations|gauge eval" $LOG
if [ $RC -ne 0 ] || [ ! -s $DUMP ]; then echo "o65 $TAG FAILED (rc=$RC); no dump"; tail -8 $LOG; rm -f $DUMP $DUMP.zb; exit 1; fi
echo "dump: $(head -1 $DUMP) gauges x times; $(wc -l < $DUMP) lines"
[ "$TAG" = base ] && echo "self-check: task 0 J must be 3.131332e+04 (o64 prior, RMSE 3.243 m)"
