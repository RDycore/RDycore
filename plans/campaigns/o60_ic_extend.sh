#!/bin/bash
#SBATCH -A m1516_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 60
#SBATCH -N 4
#SBATCH -o o60_slurm_%j.out
# o60: source the two IC-scan points the paper already plots.
#
#   sbatch o60_ic_extend.sh
#
# WHY THIS EXISTS. Figure 2 of papers/manning-calibration draws the initial-
# condition curve through a = 0.6 (MAE 0.5818) and a = 0.7 (0.6117), the
# paper's Section 8 quotes 0.5818 in prose, and PROJECT-STATE Section 3
# carries it as a headline row. An audit on 2026-08-27 found no run behind
# them. Precisely what was verified, since PM went down mid-audit: (a) the
# only o54 evaluation logs on $SCRATCH are ic0.8, ic0.9, ic1.1, ic1.2 and
# sol_it2 -- no ic0.6 or ic0.7; (b) o54_ic_authority.sh loops over
# "0.8 0.9 1.1 1.2" and nothing else. NOT verified: a content grep of all
# 209 logs for the values themselves -- three attempts died with the login
# node. So the conclusion is strongly supported but not airtight; if this
# run reproduces the plotted numbers exactly, an earlier ad hoc run
# somewhere is the likely explanation and no harm is done either way. The
# numbers are almost certainly right (they sit exactly on the measured
# curve's continuation) but nothing sources them, so this run makes them
# real before the draft is circulated.
#
# WHAT TO EXPECT, and what to do if it differs. The prediction from the
# existing scan, extrapolating the measured dMAE/da:
#     a = 0.7 -> MAE ~0.612      a = 0.6 -> MAE ~0.582
# If the run reproduces those to ~0.001 m, update RESULTS-gpu-implicit.md
# with the sourced values and nothing in the paper changes.
# If it does NOT, the figure and Section 8 must be corrected -- and note that
# the CLAIM those points support (the IC curve is still falling at a = 0.6,
# i.e. no interior optimum) is what matters, not the exact values. The claim
# survives as long as MAE(0.6) < MAE(0.7) < MAE(0.8) = 0.6434.
#
# CHEAP: two eval-only forwards, ~15 min each at n16. No adjoint, no
# optimizer. The 60-minute slot is 2x the expected need.
#
# NOTE ON THE BINARY: o54 used build-claude-gpu7; this uses gpu8, which
# PROJECT-STATE records as current. The strings gate below fails loudly if
# gpu8 predates -adjoint_ic_scale rather than silently ignoring the flag.
set -u
cd $SCRATCH/gpu-implicit
ADJ=$HOME/Codes/rdycore-manning/build-claude-gpu8/driver/rdycore_adjoint
RAIN=/global/cfs/cdirs/m4267/shared/data/harvey/spatially-distributed-rainfall/mm-per-hr/mrms/bin
export MPICH_GPU_SUPPORT_ENABLED=1

CKPT=checkpoints_o37/o37.rdycore.r.104400.bin
test -f o43_window.yaml || { echo "MISSING o43_window.yaml"; exit 1; }
test -f "$CKPT"         || { echo "MISSING $CKPT"; exit 1; }
test -x "$ADJ"          || { echo "MISSING $ADJ"; exit 1; }
strings $ADJ | grep -q adjoint_ic_scale || { echo "STALE BINARY: $ADJ predates -adjoint_ic_scale"; exit 1; }

COM="-ts_adapt_type none -snes_max_it 50 -snes_rtol 1e-5 -ksp_max_it 300 -ksp_type gmres -ksp_pc_side right -ksp_rtol 1e-4 -pc_type pbjacobi -dm_vec_type kokkos -dm_mat_type baijkokkos -ts_trajectory_type memory -ts_trajectory_max_cps_ram 400 -adjoint_fd_samples 0"
EVAL="-adjoint_calibrate_classes -adjoint_hwm_file turning30m_hwm_obs_clusterA.txt -adjoint_class_file turning30m_class.bin -adjoint_prior_file turning30m_manning.bin -adjoint_obs_freq 300 -adjoint_obs_error 0.15 -adjoint_hwm_eval_only"
RUN="srun -N 4 -n 16 -c 32 --cpu-bind=cores -G 16 --gpu-bind=none"

run_eval () {
  TAG=$1; shift
  echo "=== o60 $TAG start $(date)"
  T0=$(date +%s)
  $RUN $ADJ o43_window.yaml $COM $EVAL "$@" \
    -restart $CKPT -adjoint_rain_start_hour 29 \
    -raster_rain_dir $RAIN -raster_rain_start_date 2017,8,26,18,0 > o60_${TAG}.log 2>&1
  RC=$?; T1=$(date +%s)
  echo "$TAG exit=$RC elapsed=$(( (T1-T0)/60 ))m"
  grep -hE "ic scale|hwm eval" o60_${TAG}.log
}

for A in 0.7 0.6; do
  run_eval ic${A} -adjoint_ic_scale $A
done

echo
echo "=== o60 summary: the IC scan, measured end to end ==="
echo "a=1.2   J 1.025e+03  MAE 0.8174   (o54, sourced)"
echo "a=1.1   J 9.071e+02  MAE 0.7610   (o54, sourced)"
echo "a=1.0   J 8.083e+02  MAE 0.7188   (NLCD prior baseline)"
echo "a=0.9   J 7.284e+02  MAE 0.6796   (o54, sourced)"
echo "a=0.8   J 6.652e+02  MAE 0.6434   (o54, sourced)"
for T in ic0.7 ic0.6; do
  printf "%-8s " "$T"; grep -h "hwm eval" o60_${T}.log 2>/dev/null || echo "(not run)"
done
echo
echo "PAPER VALUES TO CONFIRM: a=0.7 -> 0.6117, a=0.6 -> 0.5818"
echo "CLAIM TO PRESERVE: monotone decreasing, no interior optimum --"
echo "MAE(0.6) < MAE(0.7) < 0.6434. If that holds, the paper stands."
