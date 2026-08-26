#!/bin/bash
#SBATCH -A m1516_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 180
#SBATCH -N 1
#SBATCH -o o49_slurm_%j.out
# o49: two more points on the uniform-alpha curve, at the alphas the
# regularized problem is predicted to land on.
#
# The 15-class calibration has to beat a ONE-parameter baseline to justify
# fifteen parameters, and o47 showed the peak observable constrains roughly
# one mode -- so the bar is J(alpha) at the alpha that sigma_n = 0.015 makes
# optimal in the uniform direction (~0.70). o44/o45 measured 0.3, 0.45, 0.6,
# 1.0 but never 0.7 or 0.8, which is exactly the interval that matters here.
# Eval-only forwards, ~36 min each, no adjoint.
set -u
cd $SCRATCH/gpu-implicit
ADJ=$HOME/Codes/rdycore-manning/build-claude-gpu5/driver/rdycore_adjoint
RAIN=/global/cfs/cdirs/m4267/shared/data/harvey/spatially-distributed-rainfall/mm-per-hr/mrms/bin
export MPICH_GPU_SUPPORT_ENABLED=1

CKPT=checkpoints_o37/o37.rdycore.r.104400.bin
COM="-ts_adapt_type none -snes_max_it 50 -snes_rtol 1e-5 -ksp_max_it 300 -ksp_type gmres -ksp_pc_side right -ksp_rtol 1e-4 -pc_type pbjacobi -dm_vec_type kokkos -dm_mat_type baijkokkos -ts_trajectory_type memory -ts_trajectory_max_cps_ram 400 -adjoint_fd_samples 0"

test -f o43_window.yaml || { echo "MISSING o43_window.yaml"; exit 1; }

for A in 0.7 0.8; do
  PRIOR=turning30m_manning_a${A}.bin
  python3 scale_manning.py turning30m_manning.bin $A $PRIOR || exit 1
  echo "=== o49 alpha=$A start $(date)"
  srun -n 4 -c 32 --cpu-bind=cores -G 4 --gpu-bind=none \
    $ADJ o43_window.yaml $COM \
    -adjoint_calibrate_classes -adjoint_hwm_file turning30m_hwm_obs_clusterA.txt \
    -adjoint_class_file turning30m_class.bin -adjoint_prior_file $PRIOR \
    -adjoint_obs_freq 300 -adjoint_obs_error 0.15 -adjoint_hwm_eval_only \
    -restart $CKPT -adjoint_rain_start_hour 29 \
    -raster_rain_dir $RAIN -raster_rain_start_date 2017,8,26,18,0 > o49_a${A}.log 2>&1
  echo "alpha=$A exit=$? $(date)"
  grep -E "hwm eval" o49_a${A}.log
done

echo "=== o49 summary (measured curve: 0.30 J 673.67 MAE 0.6392 | 0.45 703.80 0.6614"
echo "    | 0.60 740.56 0.6776 | 1.00 808.26 0.7188) ==="
for A in 0.7 0.8; do echo -n "alpha=$A: "; grep -h "hwm eval" o49_a${A}.log 2>/dev/null || echo "(not run)"; done
