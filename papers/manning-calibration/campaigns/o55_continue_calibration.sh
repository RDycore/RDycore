#!/bin/bash
# o55: push the calibration further, interactively, because the batch queue
# is not delivering.
#
#   salloc -N 4 -C gpu -q interactive -t 4:00:00 -A m1516_g --gpus 16
#   bash o55_continue_calibration.sh [INIT_FILE]
#
# Continues from o53's iteration-2 parameters (MAE 0.6242, J_total 6.849e2,
# misfit 6.387e2 -- the last verified independently by an eval-only forward,
# which reproduced it to four significant figures).
#
# TWO REASONS THIS IS NOT JUST MORE OF THE SAME:
#
#  1. armijo. o53 ran with BLMVM's default More-Thuente line search, which on
#     the short-window test overshot badly: it bought 15.8 units of extra
#     misfit for 50.5 units of prior violation and ended at a HIGHER total
#     objective than the shorter step, with the gradient norm risen 48% and
#     70% of the domain on a bound. Backtracking-only reached a lower
#     objective with a falling gradient and half the pinning. This is the
#     first time armijo runs on the production window.
#
#  2. Convergence is the top open item in the paper. The reported field is
#     two quasi-Newton iterations in with one class resting on a bound, and
#     "not converged" is the first caveat a reviewer will pick up.
#
# The dump is rewritten after EVERY TAO iteration, so the allocation wall is
# a stopping rule and not a lost run. -tao_max_it is set above what fits.
#
# WHAT TO WATCH, in order of what would change our mind:
#   * the residual. It fell monotonically through o53 (0.1575 -> 0.1483 ->
#     0.1150). If it keeps falling, the calibration is converging normally.
#   * classes reaching alpha 0.3 or 3.0. o53 iteration 1 put 91% of the
#     domain on a bound and iteration 2 pulled six of seven classes back off;
#     that was a first-step transient with no curvature history. A class that
#     STAYS on a bound across several iterations is a real statement that the
#     data wants something the prior will not allow.
#   * J flattening. Stop the chain when it does; a lower J bought by pinned
#     classes is noise-fitting, not skill.
set -u
INIT_FILE=${1:-o53_p.txt}
cd $SCRATCH/gpu-implicit
ADJ=$HOME/Codes/rdycore-manning/build-claude-gpu7/driver/rdycore_adjoint
RAIN=/global/cfs/cdirs/m4267/shared/data/harvey/spatially-distributed-rainfall/mm-per-hr/mrms/bin
export MPICH_GPU_SUPPORT_ENABLED=1

CKPT=checkpoints_o37/o37.rdycore.r.104400.bin
test -f o43_window.yaml || { echo "MISSING o43_window.yaml"; exit 1; }
test -f "$INIT_FILE"   || { echo "MISSING $INIT_FILE"; exit 1; }
strings $ADJ | grep -q adjoint_classes_relative || { echo "STALE BINARY: $ADJ"; exit 1; }

# start from a COPY, so the dump does not overwrite the file we resumed from
cp "$INIT_FILE" o55_start.txt

COM="-ts_adapt_type none -snes_max_it 50 -snes_rtol 1e-5 -ksp_max_it 300 -ksp_type gmres -ksp_pc_side right -ksp_rtol 1e-4 -pc_type pbjacobi -dm_vec_type kokkos -dm_mat_type baijkokkos -ts_trajectory_type memory -ts_trajectory_max_cps_ram 400 -adjoint_fd_samples 0"
CAL="-adjoint_calibrate_classes -adjoint_hwm_file turning30m_hwm_obs_clusterA.txt -adjoint_class_file turning30m_class.bin -adjoint_prior_file turning30m_manning.bin -adjoint_obs_freq 300 -adjoint_obs_error 0.15 -adjoint_classes_relative -adjoint_sigma_n 0.015"

echo "=== o55 start $(date) -- resuming from $INIT_FILE, armijo line search"
srun -N 4 -n 16 -c 32 --cpu-bind=cores -G 16 --gpu-bind=none \
  $ADJ o43_window.yaml $COM $CAL -adjoint_classes_init o55_start.txt \
  -restart $CKPT -adjoint_rain_start_hour 29 \
  -tao_max_it 6 -tao_monitor -tao_ls_type armijo \
  -adjoint_classes_dump o55_p.txt -adjoint_classes_grad_dump o55_g.txt \
  -raster_rain_dir $RAIN -raster_rain_start_date 2017,8,26,18,0 2>&1 | tee o55.log
echo "exit=$? $(date)"

echo "--- iterations ---"
grep -E "warm start|hwm init|TAO,|hwm final|class recovery" o55.log
echo "--- parameters, rewritten every iteration (physical n) ---"
cat o55_p.txt 2>/dev/null
echo
echo "reference: o53 reached J_total 6.849004e+02 (misfit 6.387436e+02),"
echo "peak-WSE MAE 0.6242 m, after 2 iterations with the DEFAULT line search."
echo "Score any new iterate for one forward with:"
echo "  o54-style eval-only + -adjoint_classes_init o55_p.txt"
