#!/bin/bash
#SBATCH -A m1516_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 30
#SBATCH -N 1
#SBATCH -o o48_smoke_%j.out
# o48 smoke: exercise -adjoint_classes_relative on the PRODUCTION wiring
# (2.93M cells, GPU types, real mark table, h29 checkpoint, rain offset) over
# a 600-step window instead of 43200, so the whole thing costs minutes rather
# than hours. What it has to show:
#   * alpha = 1 reproduces the NLCD prior field exactly -- the relative run's
#     J at iteration 0 must equal the absolute run's, to every digit
#   * the scaling line prints a first-step |g|/J0 that is O(1) in alpha
#   * TAO accepts iteration 1 (o43's failure was that it accepted none)
set -u
cd $SCRATCH/gpu-implicit
ADJ=$HOME/Codes/rdycore-manning/build-claude-gpu5/driver/rdycore_adjoint
RAIN=/global/cfs/cdirs/m4267/shared/data/harvey/spatially-distributed-rainfall/mm-per-hr/mrms/bin
export MPICH_GPU_SUPPORT_ENABLED=1

CKPT=checkpoints_o37/o37.rdycore.r.104400.bin
sed -e "s/^  stop             : .*/  stop             : 600.0/" \
    -e "s/^  coupling_interval: .*/  coupling_interval: 600.0/" \
    o43_window.yaml > o48_smoke.yaml

COM="-ts_adapt_type none -snes_max_it 50 -snes_rtol 1e-5 -ksp_max_it 300 -ksp_type gmres -ksp_pc_side right -ksp_rtol 1e-4 -pc_type pbjacobi -dm_vec_type kokkos -dm_mat_type baijkokkos -ts_trajectory_type memory -ts_trajectory_max_cps_ram 400 -adjoint_fd_samples 0"
CAL="-adjoint_calibrate_classes -adjoint_hwm_file turning30m_hwm_obs_clusterA.txt -adjoint_class_file turning30m_class.bin -adjoint_prior_file turning30m_manning.bin -adjoint_obs_freq 100 -adjoint_obs_error 0.15"
RUN="srun -n 4 -c 32 --cpu-bind=cores -G 4 --gpu-bind=none"

echo "=== A: ABSOLUTE variables, started at the NLCD prior (the o43 setup) $(date)"
$RUN $ADJ o48_smoke.yaml $COM $CAL -adjoint_beta 1e2 -adjoint_classes_init o43_p_nlcd.txt \
  -restart $CKPT -adjoint_rain_start_hour 29 -tao_max_it 1 -tao_monitor \
  -raster_rain_dir $RAIN -raster_rain_start_date 2017,8,26,18,0 > o48_smoke_abs.log 2>&1
echo "A exit=$? $(date)"
grep -E "hwm init|TAO,|hwm final|class recovery" o48_smoke_abs.log

echo
echo "=== B: RELATIVE variables + sigma_n (the o48 setup) $(date)"
$RUN $ADJ o48_smoke.yaml $COM $CAL -adjoint_classes_relative -adjoint_sigma_n 0.015 \
  -restart $CKPT -adjoint_rain_start_hour 29 -tao_max_it 1 -tao_monitor \
  -adjoint_classes_dump o48_smoke_p.txt -adjoint_classes_grad_dump o48_smoke_g.txt \
  -raster_rain_dir $RAIN -raster_rain_start_date 2017,8,26,18,0 > o48_smoke_rel.log 2>&1
echo "B exit=$? $(date)"
grep -E "class calibration|hwm init|TAO,|hwm final|class recovery" o48_smoke_rel.log
echo "--- start-point gradient ---"
cat o48_smoke_g.txt 2>/dev/null
echo "--- solution (physical n) ---"
cat o48_smoke_p.txt 2>/dev/null
echo
echo "PASS if: the two 'hwm init' J values match, B prints a first-step |g|/J0"
echo "of order 0.01-1, and B's TAO line 1 shows a LOWER function value than 0."
