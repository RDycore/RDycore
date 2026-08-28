#!/bin/bash
#SBATCH -A m1516_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 360
#SBATCH -N 4
#SBATCH -o o61_slurm_%j.out
# o61: the Gauss-Newton spectrum on the PRODUCTION window, plus the one
#      small number o59 Part B never got.
#
#   sbatch o61_spectrum_production.sh
#
# WHY. Section 7.4's spectrum (lambda = 3.03, 0.64, 0.33, ...) is measured on
# a 7,200-step pilot while the calibration it interprets runs on 43,200 steps,
# and the paper states that gap as an open caveat. This closes it: the same 16
# forwards on the same 46 marks, on the window every other number in the paper
# uses. It also measures something the mark-count scaling of Table 11 cannot,
# because that table holds per-mark sensitivity fixed: a longer window raises
# the sensitivity each peak integrates, so lambda should RISE with window
# length. How much is the quantity that turns "what would a longer experiment
# buy?" from speculation into a measurement.
#
# WHAT TO EXPECT. The pilot found one eigenvalue above unity with a gap of
# 4.8 and 2.02 degrees of freedom for signal. If the production window returns
# the same COUNT, Section 7.4 stands as written and the caveat is simply
# deleted. If it returns two supported parameters, the paper's headline claim
# softens from "one" to "one, two on the full window" -- still the same
# argument (the observing system limits it, not the method), but the number in
# the abstract changes and Section 7.4 needs a paragraph. Either outcome is
# publishable; only the silence is not.
#
# PART A first, because it is 15 minutes and completes an existing record:
# o59 Part B calibrated classes 23/90/22 to J 6.821163e+02 but the machine
# died before any MAE was taken. One eval-only forward gets it. Compare
# against the 15-class field's 0.6116 and the NLCD 0.7188.
#
# RESUMABLE. Each forward writes its own peak dump; a column whose dump
# already exists and is non-empty is skipped. If the wall kills this job,
# resubmit it unchanged and it continues where it stopped. That matters
# because the analysis needs all 16 dumps, so losing 15 to a wall would be
# the expensive failure.
#
# ANALYSIS (do NOT omit --sigma-obs; every eigenvalue scales as 1/sigma^2 and
# the driver default of 0.01 vs the correct 0.15 is a factor of 225, enough
# to turn 7 supported directions into 0):
#
#   python3 o58_gauss_newton.py 0.05 o58_e0.05w43200_pk_base.txt \
#       o58_e0.05w43200_pk_col*.txt --sigma-obs 0.15
#
# Read its SELF-CHECKS before its numbers -- in particular the argmax count,
# how many marks moved their peak time across a column. The whole method
# assumes peaks are smooth in alpha where the gradient field is not; if a
# large fraction moved, the Gauss-Newton assembly is sampling a discontinuity
# and the spectrum is not trustworthy.
set -u
cd $SCRATCH/gpu-implicit
ADJ=$HOME/Codes/rdycore-manning/build-claude-gpu8/driver/rdycore_adjoint
RAIN=/global/cfs/cdirs/m4267/shared/data/harvey/spatially-distributed-rainfall/mm-per-hr/mrms/bin
export MPICH_GPU_SUPPORT_ENABLED=1

EPS=${EPS:-0.05}
WINDOW=${WINDOW:-43200}
TAG=${TAG:-e${EPS}w${WINDOW}}

CKPT=checkpoints_o37/o37.rdycore.r.104400.bin
test -f o43_window.yaml || { echo "MISSING o43_window.yaml"; exit 1; }
test -f o43_p_nlcd.txt  || { echo "MISSING o43_p_nlcd.txt";  exit 1; }
test -f "$CKPT"         || { echo "MISSING $CKPT";           exit 1; }
test -x "$ADJ"          || { echo "MISSING $ADJ";            exit 1; }

sed -e "s/^  stop             : .*/  stop             : ${WINDOW}.0/" \
    -e "s/^  coupling_interval: .*/  coupling_interval: ${WINDOW}.0/" \
    o43_window.yaml > o61_window_${WINDOW}.yaml

COM="-ts_adapt_type none -snes_max_it 50 -snes_rtol 1e-5 -ksp_max_it 300 -ksp_type gmres -ksp_pc_side right -ksp_rtol 1e-4 -pc_type pbjacobi -dm_vec_type kokkos -dm_mat_type baijkokkos -ts_trajectory_type memory -ts_trajectory_max_cps_ram 400 -adjoint_fd_samples 0"
EV="-adjoint_calibrate_classes -adjoint_hwm_file turning30m_hwm_obs_clusterA.txt -adjoint_class_file turning30m_class.bin -adjoint_prior_file turning30m_manning.bin -adjoint_obs_freq 300 -adjoint_obs_error 0.15 -adjoint_hwm_eval_only"
RUN="srun -N 4 -n 16 -c 32 --cpu-bind=cores -G 16 --gpu-bind=none"

# ---------------------------------------------------------------------------
# PART A -- the MAE of o59 Part B's three-parameter field
# ---------------------------------------------------------------------------
if [ -f o59_p_57649525.txt ] && [ ! -s o61_p3_score.log ]; then
  echo "=== o61 Part A: scoring the 3-parameter field $(date)"
  $RUN $ADJ o43_window.yaml $COM $EV -adjoint_classes_init o59_p_57649525.txt \
    -restart $CKPT -adjoint_rain_start_hour 29 \
    -raster_rain_dir $RAIN -raster_rain_start_date 2017,8,26,18,0 \
    > o61_p3_score.log 2>&1
  echo "Part A exit=$?"
  grep -h "hwm eval" o61_p3_score.log
  echo "   compare: 15-class 0.6116, NLCD 0.7188, uniform a=0.70 0.6894"
else
  echo "(Part A skipped: no o59_p_57649525.txt, or already scored)"
fi

# ---------------------------------------------------------------------------
# PART B -- 16 forwards for the Gauss-Newton spectrum
# ---------------------------------------------------------------------------
peaks_at () {  # $1 = tag, $2 = class table
  local DUMP=o58_${TAG}_pk_$1.txt
  if [ -s "$DUMP" ]; then echo "  $1 already done, skipping"; return; fi
  $RUN $ADJ o61_window_${WINDOW}.yaml $COM $EV -adjoint_classes_init "$2" \
    -adjoint_hwm_dump $DUMP \
    -restart $CKPT -adjoint_rain_start_hour 29 \
    -raster_rain_dir $RAIN -raster_rain_start_date 2017,8,26,18,0 \
    > o58_${TAG}_$1.log 2>&1
  echo "  $1 exit=$? $(grep -o 'MAE [0-9.]* m' o58_${TAG}_$1.log)"
}

echo "=== o61 spectrum: window ${WINDOW} steps, eps ${EPS}, tag ${TAG} $(date)"
peaks_at base o43_p_nlcd.txt

for C in $(awk '!/^#/{print $1}' o43_p_nlcd.txt); do
  awk -v c="$C" -v e="$EPS" '!/^#/{ if ($1==c) printf "%d %.10g\n", $1, $2*(1+e); else printf "%d %.10g\n", $1, $2 }' \
      o43_p_nlcd.txt > o58_${TAG}_pert_$C.txt
  echo "=== o61 column $C $(date)"
  peaks_at col$C o58_${TAG}_pert_$C.txt
done

echo
echo "=== o61 done $(date). Dumps present: $(ls o58_${TAG}_pk_*.txt 2>/dev/null | wc -l) of 16"
echo "Assemble with (--sigma-obs is REQUIRED):"
echo "    python3 o58_gauss_newton.py $EPS o58_${TAG}_pk_base.txt o58_${TAG}_pk_col*.txt --sigma-obs 0.15"
echo "Pilot for comparison (7,200 steps): lambda = 3.03, 0.64, 0.33, 0.32, 0.21, 0.10, ..."
echo "                                    one above unity, gap 4.8, 2.02 dof for signal"
