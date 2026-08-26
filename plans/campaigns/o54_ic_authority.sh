#!/bin/bash
# o54: how much of the error can the INITIAL CONDITION explain -- and what is
# the peak-WSE MAE at the calibration's current iterate?
#
#   salloc -N 4 -C gpu -q interactive -t 2:00:00 -A m1516_g --gpus 16
#   bash o54_ic_authority.sh
#
# Six eval-only forwards, ~15 min each at n16, ~90 min total.
#
# ---------------------------------------------------------------------------
# PART A -- the number the calibration cannot report itself
# ---------------------------------------------------------------------------
# A calibration prints its peak-WSE MAE only after TaoSolve returns, so a run
# stopped by the queue wall (or an interactive allocation) never reports one.
# -adjoint_hwm_eval_only now honours -adjoint_classes_init, so the MAE at any
# dumped iterate costs a single forward. Verified on a twin: evaluating a
# dumped solution reproduces the calibration's own J_final to every printed
# digit. o53 reached J 6.849004e+02 after two iterations, down from 8.082566e+02
# at the NLCD prior -- but we have never seen its MAE.
#
# ---------------------------------------------------------------------------
# PART B -- the question the meeting raised
# ---------------------------------------------------------------------------
# Estimating the initial condition was proposed as the next study. The pull is
# obvious: the alpha scan says 85-90% of the model-vs-survey residual is NOT
# roughness, and antecedent state is the natural next suspect. The risk is
# equally obvious: over this window the IC is ~8.8M unknowns (3 dof x 2.93M
# cells) against 46 peak observations, so it will fit the data essentially
# perfectly and mean nothing -- the same lesson as the roughness study, three
# orders of magnitude worse.
#
# So measure authority BEFORE building a control vector, exactly as
# Section 7.4 of the paper does for roughness. -adjoint_ic_scale <a> scales
# the whole restart state (all three dofs, so every velocity is unchanged and
# only antecedent water volume moves). J(a) along that one direction bounds
# what the antecedent state can do for this observable.
#
# HOW TO READ IT. The roughness bar: the entire uniform-n knob is worth 0.08 m
# of a 0.72 m error, and only at physically indefensible roughness; within a
# defensible range, ~3 cm.
#   * If J(a) is FLATTER than that, the IC has less purchase than roughness
#     and IC estimation against peak WSE is not the next study -- whatever a
#     4D-Var would report would be fitting, not information.
#   * If it is much STEEPER, the IC carries real signal, and the next question
#     is the design one (background covariance, positivity of h, wet/dry front
#     motion under an IC control, component scaling across h and hu) -- which
#     is where a careful design pass earns its keep.
#   * Either way this costs 4 forwards, not a development cycle.
#
# CAVEAT to carry into any conclusion: our IC is an o37 model checkpoint, not
# an observation. Scaling it perturbs a state that already contains
# accumulated rainfall and DEM error, and an IC calibration would absorb that
# error rather than correct it -- possibly laundering the downstream drainage
# defect into a "corrected" initial state, the same trap as scoring against
# the 37 marks the model cannot drain.
set -u
cd $SCRATCH/gpu-implicit
ADJ=$HOME/Codes/rdycore-manning/build-claude-gpu7/driver/rdycore_adjoint
RAIN=/global/cfs/cdirs/m4267/shared/data/harvey/spatially-distributed-rainfall/mm-per-hr/mrms/bin
export MPICH_GPU_SUPPORT_ENABLED=1

CKPT=checkpoints_o37/o37.rdycore.r.104400.bin
test -f o43_window.yaml || { echo "MISSING o43_window.yaml"; exit 1; }
test -f "$CKPT" || { echo "MISSING $CKPT"; exit 1; }
strings $ADJ | grep -q adjoint_ic_scale || { echo "STALE BINARY: $ADJ predates -adjoint_ic_scale"; exit 1; }

COM="-ts_adapt_type none -snes_max_it 50 -snes_rtol 1e-5 -ksp_max_it 300 -ksp_type gmres -ksp_pc_side right -ksp_rtol 1e-4 -pc_type pbjacobi -dm_vec_type kokkos -dm_mat_type baijkokkos -ts_trajectory_type memory -ts_trajectory_max_cps_ram 400 -adjoint_fd_samples 0"
EVAL="-adjoint_calibrate_classes -adjoint_hwm_file turning30m_hwm_obs_clusterA.txt -adjoint_class_file turning30m_class.bin -adjoint_prior_file turning30m_manning.bin -adjoint_obs_freq 300 -adjoint_obs_error 0.15 -adjoint_hwm_eval_only"
RUN="srun -N 4 -n 16 -c 32 --cpu-bind=cores -G 16 --gpu-bind=none"

run_eval () {  # $1 = tag, rest = extra options
  TAG=$1; shift
  echo "=== o54 $TAG start $(date)"
  T0=$(date +%s)
  $RUN $ADJ o43_window.yaml $COM $EVAL "$@" \
    -restart $CKPT -adjoint_rain_start_hour 29 \
    -raster_rain_dir $RAIN -raster_rain_start_date 2017,8,26,18,0 > o54_${TAG}.log 2>&1
  RC=$?; T1=$(date +%s)
  echo "$TAG exit=$RC elapsed=$(( (T1-T0)/60 ))m"
  grep -hE "ic scale|hwm eval" o54_${TAG}.log
}

# --- Part A: the MAE at the calibration's current iterate -------------------
if [ -f o53_p.txt ]; then
  run_eval sol_it2 -adjoint_classes_init o53_p.txt
else
  echo "(skipping Part A: no o53_p.txt)"
fi

# --- Part B: the IC authority scan -----------------------------------------
# a = 1 is already measured: J 8.082566e+02, MAE 0.7188, 0/46 dry.
for A in 0.8 0.9 1.1 1.2; do
  run_eval ic${A} -adjoint_ic_scale $A
done

echo
echo "=== o54 summary ==="
echo "reference (a=1, NLCD prior): J 8.082566e+02  MAE 0.7188  0/46 dry"
for T in sol_it2 ic0.8 ic0.9 ic1.1 ic1.2; do
  printf "%-8s " "$T"; grep -h "hwm eval" o54_${T}.log 2>/dev/null || echo "(not run)"
done
echo
echo "ROUGHNESS BAR for comparison: the whole uniform-n knob moves MAE"
echo "0.7188 -> 0.6392 (0.08 m), and only at n = 30% of NLCD values."
echo "Compare the MAE swing per unit relative IC change against that."
