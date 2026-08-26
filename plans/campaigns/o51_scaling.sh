#!/bin/bash
# o51: how does the production window scale with node count?
#
#   salloc -N 4 -C gpu -q interactive -t 2:00:00 -A m1516_g --gpus 16
#   bash o51_scaling.sh
#
# We know 4 ranks beats 64 by 2.8x on this mesh (b2i_dev_n4 vs b2i_dev_n64,
# same problem, per-rank flops down exactly 16x) -- but n8 and n16 were never
# measured, and the whole calibration schedule rests on the answer. A
# forward+adjoint is ~2 hr at n4 and a 12-hr slot fits ~3 TAO iterations; if
# n16 is even 2x faster, a full calibration becomes one overnight job instead
# of a two-day chain.
#
# Measures the FORWARD only (-adjoint_hwm_eval_only) on the real 43,200-step
# window. That is a fair proxy: TSStep dominates TSAdjointStep 5:1 in the
# device benchmarks (14.4 s vs 3.0 s per 40 steps), and it exercises only the
# long-tested eval-only path, so nothing here can fail for a new-code reason.
#
# FREE CONSISTENCY CHECK: alpha = 1 is the NLCD prior, so every node count
# must reproduce J 8.082566e+02, MAE 0.7188, 0 of 46 dry. A disagreement is a
# decomposition-dependence bug and matters more than the timing.
set -u
cd $SCRATCH/gpu-implicit
ADJ=$HOME/Codes/rdycore-manning/build-claude-gpu6/driver/rdycore_adjoint
RAIN=/global/cfs/cdirs/m4267/shared/data/harvey/spatially-distributed-rainfall/mm-per-hr/mrms/bin
export MPICH_GPU_SUPPORT_ENABLED=1

CKPT=checkpoints_o37/o37.rdycore.r.104400.bin
COM="-ts_adapt_type none -snes_max_it 50 -snes_rtol 1e-5 -ksp_max_it 300 -ksp_type gmres -ksp_pc_side right -ksp_rtol 1e-4 -pc_type pbjacobi -dm_vec_type kokkos -dm_mat_type baijkokkos -ts_trajectory_type memory -ts_trajectory_max_cps_ram 400 -adjoint_fd_samples 0"
CAL="-adjoint_calibrate_classes -adjoint_hwm_file turning30m_hwm_obs_clusterA.txt -adjoint_class_file turning30m_class.bin -adjoint_prior_file turning30m_manning.bin -adjoint_obs_freq 300 -adjoint_obs_error 0.15 -adjoint_hwm_eval_only"

test -f o43_window.yaml || { echo "MISSING o43_window.yaml -- run any o48/o49 job first"; exit 1; }
test -f "$CKPT" || { echo "MISSING $CKPT"; exit 1; }

# n16 first: if the queue eats the allocation, the interesting point is done
for N in 16 8 4; do
  NODES=$((N / 4))
  echo "=== o51 n=$N ($NODES node(s), $N GPUs) start $(date +%s) $(date)"
  T0=$(date +%s)
  srun -N $NODES -n $N -c 32 --cpu-bind=cores -G $N --gpu-bind=none \
    $ADJ o43_window.yaml $COM $CAL \
    -restart $CKPT -adjoint_rain_start_hour 29 \
    -raster_rain_dir $RAIN -raster_rain_start_date 2017,8,26,18,0 > o51_n${N}.log 2>&1
  RC=$?
  T1=$(date +%s)
  echo "n=$N exit=$RC elapsed=$(( (T1-T0)/60 ))m$(( (T1-T0)%60 ))s $(date)"
  grep -E "hwm eval" o51_n${N}.log
done

echo
echo "=== o51 summary (n4 reference: ~36 min, J 8.082566e+02, MAE 0.7188, 0 dry) ==="
for N in 4 8 16; do
  echo -n "n=$N: "; grep -h "hwm eval" o51_n${N}.log 2>/dev/null || echo "(not run)"
done
echo
echo "DECIDE: if n16 is >=2x faster than n4 AND all three J values agree to"
echo "6 digits, switch the o48/o50 job scripts to -N 4 / srun -n 16 -G 16."
echo "If the J values DISAGREE, stop and report -- that is a real bug, not a"
echo "timing result."
