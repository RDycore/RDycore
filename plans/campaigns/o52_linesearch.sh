#!/bin/bash
# o52: does BLMVM's line search overshoot into the bounds on this problem?
#
#   salloc -N 1 -C gpu -q interactive -t 4:00:00 -A m1516_g --gpus 4
#   bash o52_linesearch.sh
#
# The 600-step smoke (o48) accepted its first step, which is the fix working
# -- but the step BLMVM took was several times -g: classes 23 and 24 landed
# exactly on the alpha 0.3 bound while the objective fell only 3% and the
# gradient norm ROSE (0.1036 -> 0.1237). A step that lands on a bound and
# makes the gradient worse is a line search overshooting, not a descent.
#
# If that repeats on the production window, the queued 12-hour job spends its
# whole slot producing a pinned answer, and pinning is exactly what we cannot
# distinguish from the real over-parameterization result (o28/o30/o47). So
# measure it first, on a window long enough to mean something but short
# enough to fit an interactive slot: 7200 steps (2 event-hours) instead of
# 43,200, where an objective+gradient is ~20 min instead of ~2 hr.
#
# A: BLMVM's default line search (More-Thuente -- it may EXPAND the step).
# B: -tao_ls_type armijo -- backtracking only, never expands past the initial
#    unit step, which after the relative scaling is the ~10% roughness change
#    we actually want.
#
# Each config: start-point eval + TAO iteration 0 + 1 iteration, so roughly
# 3-4 objective+gradients, ~60-80 min. Both fit a 4-hr slot with margin.
#
# READ THE RESULT AS: how many classes sit at alpha 0.3 or 3.0 after one
# iteration (rel_err 0.700 or 2.000 in the table), and whether the residual
# went DOWN. Fewer pinned classes with a falling residual is the config the
# production job should use.
set -u
cd $SCRATCH/gpu-implicit
ADJ=$HOME/Codes/rdycore-manning/build-claude-gpu6/driver/rdycore_adjoint
RAIN=/global/cfs/cdirs/m4267/shared/data/harvey/spatially-distributed-rainfall/mm-per-hr/mrms/bin
export MPICH_GPU_SUPPORT_ENABLED=1

CKPT=checkpoints_o37/o37.rdycore.r.104400.bin
test -f o43_window.yaml || { echo "MISSING o43_window.yaml"; exit 1; }
test -f "$CKPT" || { echo "MISSING $CKPT"; exit 1; }

# 7200 steps = 2 event-hours from h29. obs_freq 300 keeps 24 peak samples.
sed -e "s/^  stop             : .*/  stop             : 7200.0/" \
    -e "s/^  coupling_interval: .*/  coupling_interval: 7200.0/" \
    o43_window.yaml > o52_window.yaml

COM="-ts_adapt_type none -snes_max_it 50 -snes_rtol 1e-5 -ksp_max_it 300 -ksp_type gmres -ksp_pc_side right -ksp_rtol 1e-4 -pc_type pbjacobi -dm_vec_type kokkos -dm_mat_type baijkokkos -ts_trajectory_type memory -ts_trajectory_max_cps_ram 400 -adjoint_fd_samples 0"
CAL="-adjoint_calibrate_classes -adjoint_hwm_file turning30m_hwm_obs_clusterA.txt -adjoint_class_file turning30m_class.bin -adjoint_prior_file turning30m_manning.bin -adjoint_obs_freq 300 -adjoint_obs_error 0.15 -adjoint_classes_relative -adjoint_sigma_n 0.015"
RUN="srun -n 4 -c 32 --cpu-bind=cores -G 4 --gpu-bind=none"

run_case () {  # $1 = tag, $2... = extra options
  TAG=$1; shift
  echo "=== o52 $TAG start $(date)"
  T0=$(date +%s)
  $RUN $ADJ o52_window.yaml $COM $CAL "$@" \
    -restart $CKPT -adjoint_rain_start_hour 29 \
    -tao_max_it 1 -tao_monitor \
    -adjoint_classes_dump o52_p_${TAG}.txt -adjoint_classes_grad_dump o52_g_${TAG}.txt \
    -raster_rain_dir $RAIN -raster_rain_start_date 2017,8,26,18,0 > o52_${TAG}.log 2>&1
  RC=$?; T1=$(date +%s)
  echo "$TAG exit=$RC elapsed=$(( (T1-T0)/60 ))m $(date)"
  grep -E "objective scaled|hwm init|TAO,|hwm final|class recovery" o52_${TAG}.log
  echo "--- per-class table ($TAG): rel_err 0.700 means the class hit alpha 0.3 ---"
  grep -A17 "NLCD  prior_n" o52_${TAG}.log | tail -18
  echo
}

run_case mt                          # BLMVM default (More-Thuente)
run_case armijo -tao_ls_type armijo  # backtracking only

echo "=== o52 summary ==="
for TAG in mt armijo; do
  echo "--- $TAG"
  grep -E "TAO,|class recovery" o52_${TAG}.log 2>/dev/null
  echo -n "    classes at a bound: "
  grep -A17 "NLCD  prior_n" o52_${TAG}.log 2>/dev/null | awk '$4=="0.700"||$4=="2.000"{printf "%s ", $1} END{print ""}'
done
echo
echo "--- start-point gradient (identical in both; this is the o48 diagnostic,"
echo "    two event-hours in rather than twelve, so treat it as indicative) ---"
cat o52_g_mt.txt 2>/dev/null
