#!/bin/bash
#SBATCH -A m1516_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 720
#SBATCH -N 4
#SBATCH -o o59_slurm_%j.out
# o59: calibrate the parameters the data actually supports.
#
#   sbatch o59_spectral_active_set.sh [TAO_ITS] [CODES] [INIT_FILE]
#   default:  sbatch o59_spectral_active_set.sh 12 23,90,22
#
# THE POINT. o58 measured that these 46 marks support ONE roughness
# parameter: the prior-preconditioned Gauss-Newton spectrum is
# 3.03, 0.64, 0.33, 0.32, 0.21, ... with a gap of 4.8 and 2.02 degrees of
# freedom for signal. The 15-class calibration nevertheless ran for nine
# iterations and finished 9.2 prior standard deviations from the NLCD lookup
# with only 2.9% of that displacement in the one constrained direction, having
# moved seven classes about which the marks taught it nothing (posterior =
# prior to <=1%). This run asks the data for what it has.
#
# THE ACTIVE SET IS CHOSEN SPECTRALLY, NOT BY GRADIENT MAGNITUDE. The leading
# eigenvector is 23(-0.81), 90(+0.41), 22(-0.36), 24(-0.18); classes 23, 90
# and 22 carry 95% of its energy, and they are also three of the five classes
# whose posterior uncertainty the marks measurably reduce (36%, 17%, 17%).
# Note this DIFFERS from the ranking by |dJ/dn| * n_prior, which put pasture
# (81) third and woody wetland (90) second-but-by-a-different-margin: the
# gradient says which classes move the objective HERE, the spectrum says which
# the data can DETERMINE. The second is the right criterion for deciding what
# to calibrate, and the two disagree on the third parameter.
#
# WHAT WOULD FALSIFY THE WHOLE PICTURE. If three well-chosen parameters land
# near the 15-class result (J ~ 6.57e2, MAE to be measured in Part A), the
# spectrum is right and twelve of the fifteen parameters were decoration. If
# three do MUCH worse, then the information the 15-class run exploited lives
# outside the leading eigenspace and o58's reading is too narrow -- which
# would matter more than any number in the paper.
#
# Part A scores the converged 15-class field first, because the batch job was
# killed by its wall before printing an MAE and that number is the paper's.
set -u
TAO_ITS=${1:-12}
CODES=${2:-23,90,22}
INIT_FILE=${3:-}

export MPICH_GPU_SUPPORT_ENABLED=1
cd $SCRATCH/gpu-implicit
ADJ=$HOME/Codes/rdycore-manning/build-claude-gpu8/driver/rdycore_adjoint
RAIN=/global/cfs/cdirs/m4267/shared/data/harvey/spatially-distributed-rainfall/mm-per-hr/mrms/bin
RANKS=16; NODES=4

CKPT=checkpoints_o37/o37.rdycore.r.104400.bin
test -f o43_window.yaml || { echo "MISSING o43_window.yaml"; exit 1; }
strings $ADJ | grep -q adjoint_classes_relative || { echo "STALE BINARY: $ADJ"; exit 1; }

COM="-ts_adapt_type none -snes_max_it 50 -snes_rtol 1e-5 -ksp_max_it 300 -ksp_type gmres -ksp_pc_side right -ksp_rtol 1e-4 -pc_type pbjacobi -dm_vec_type kokkos -dm_mat_type baijkokkos -ts_trajectory_type memory -ts_trajectory_max_cps_ram 400 -adjoint_fd_samples 0"
CAL="-adjoint_calibrate_classes -adjoint_hwm_file turning30m_hwm_obs_clusterA.txt -adjoint_class_file turning30m_class.bin -adjoint_prior_file turning30m_manning.bin -adjoint_obs_freq 300 -adjoint_obs_error 0.15 -adjoint_classes_relative -adjoint_sigma_n 0.015"
RUN="srun -N $NODES -n $RANKS -c 32 --cpu-bind=cores -G $RANKS --gpu-bind=none"

# ---- Part A: the MAE of the converged 15-class field (~15 min) -------------
LAST15=$(ls -t o48_p_*.txt 2>/dev/null | head -1)
if [ -n "$LAST15" ]; then
  echo "=== o59 Part A: scoring the converged 15-class field $LAST15 $(date)"
  $RUN $ADJ o43_window.yaml $COM $CAL -adjoint_hwm_eval_only \
    -adjoint_classes_init $LAST15 \
    -restart $CKPT -adjoint_rain_start_hour 29 \
    -raster_rain_dir $RAIN -raster_rain_start_date 2017,8,26,18,0 > o59_score15.log 2>&1
  echo "Part A exit=$? $(date)"; grep -hE "hwm eval" o59_score15.log
else
  echo "(Part A skipped: no o48_p_*.txt found)"
fi

# ---- Part B: the few-parameter calibration --------------------------------
DUMP="o59_p_${SLURM_JOB_ID}.txt"
GRAD="o59_g_${SLURM_JOB_ID}.txt"
RESUME=""; [ -n "$INIT_FILE" ] && RESUME="-adjoint_classes_init $INIT_FILE"

echo
echo "=== o59 Part B: calibrating classes $CODES only $(date)"
echo "    the other twelve are frozen at the NLCD prior, on the spectrum's"
echo "    evidence that the marks cannot determine them"
$RUN $ADJ o43_window.yaml $COM $CAL $RESUME \
  -adjoint_classes_active $CODES \
  -restart $CKPT -adjoint_rain_start_hour 29 \
  -tao_max_it $TAO_ITS -tao_monitor -tao_ls_type armijo \
  -adjoint_classes_dump $DUMP -adjoint_classes_grad_dump $GRAD \
  -raster_rain_dir $RAIN -raster_rain_start_date 2017,8,26,18,0 > o59_${SLURM_JOB_ID}.log 2>&1
echo "Part B exit=$? $(date)"
grep -E "classes active|hwm init|TAO,|hwm final|class recovery" o59_${SLURM_JOB_ID}.log | tail -20
echo "--- per-class table (the 12 frozen must show rel_err exactly 0.000) ---"
grep -A17 "NLCD  prior_n" o59_${SLURM_JOB_ID}.log | tail -18

echo
echo "=== compare against ==="
echo "  NLCD prior              J 8.083e2   MAE 0.7188"
echo "  uniform alpha 0.70      J 7.655e2   MAE 0.6894   (prior-consistent optimum)"
echo "  15 classes, 9 its       J 6.573e2   MAE from Part A above"
echo "  this run, 3 classes     see 'class recovery' line"
echo
echo "If three parameters land near the fifteen-class J, twelve of them were"
echo "decoration and the spectrum called it. If they land far short, the"
echo "information lived outside the leading eigenspace and o58 read too narrow."
