#!/bin/bash
# o58: how many roughness parameters does the observable support -- via the
#      GAUSS-NEWTON Hessian, which is the right object for this problem.
#
#   salloc -N 4 -C gpu -q interactive -t 1:00:00 -A m1516_g --gpus 16
#   bash o58_gauss_newton_spectrum.sh
#
# WHY NOT o56. o56 assembled the full Hessian by differencing the GRADIENT
# field, and its own self-check rejected the result: relative asymmetry 0.45
# and eigenvalues down to -6.8 where an unconstrained direction must return
# exactly 1. The cause is documented in the paper's own nondifferentiability
# catalog. Over a 7200-step window the objective is dense with wet/dry branch
# surfaces; the adjoint returns the exact derivative ON its branch -- the
# directional derivatives from o56's J values matched the adjoint gradient to
# 0.1-4% -- but the gradient FIELD jumps between branches, so differencing it
# over eps = 0.05 samples a discontinuity. Roughly half the differenced signal
# was branch noise. Neither a smaller nor a larger eps rescues that: smaller
# shrinks the signal against the same jumps, larger averages over more of them.
#
# THE RIGHT OBJECT. For a least-squares misfit the Gauss-Newton Hessian is
#
#     H_GN = (1/sigma^2) S^T W S ,   S_ik = d(peak_i)/d(alpha_k)
#
# over the 46 marks and 15 classes. Three reasons it works where o56 did not:
#
#   1. It differences OBSERVATIONS, not gradients. This project has already
#      measured that no mark's argmax moves across an FD probe, so the peak
#      values are smooth in alpha exactly where the gradient field is not.
#      And this script CHECKS that per mark rather than assuming it: the peak
#      dump carries peak_step, so o58_gauss_newton.py reports how many marks
#      changed their peak time in each column.
#   2. It is positive semi-definite by construction. Negative eigenvalues
#      become impossible rather than a symptom, which makes the spectrum
#      interpretable instead of merely plausible.
#   3. It is CHEAPER: 16 forwards, not 16 forward+adjoints. On the 7200-step
#      window at n16 that is ~2.5 min apiece, so this pilot costs ~40 minutes
#      against o56's 100.
#
# Gauss-Newton drops the term involving second derivatives of the model, which
# is the standard and appropriate approximation near a good fit and is what
# defines the Fisher information for this problem -- so the eigenvalue count
# is the information-theoretic answer, not merely a numerical one.
set -u
cd $SCRATCH/gpu-implicit
ADJ=$HOME/Codes/rdycore-manning/build-claude-gpu8/driver/rdycore_adjoint
RAIN=/global/cfs/cdirs/m4267/shared/data/harvey/spatially-distributed-rainfall/mm-per-hr/mrms/bin
export MPICH_GPU_SUPPORT_ENABLED=1
EPS=${EPS:-0.05}
WINDOW=${WINDOW:-7200}

CKPT=checkpoints_o37/o37.rdycore.r.104400.bin
test -f o43_window.yaml || { echo "MISSING o43_window.yaml"; exit 1; }
test -f o43_p_nlcd.txt  || { echo "MISSING o43_p_nlcd.txt"; exit 1; }

sed -e "s/^  stop             : .*/  stop             : ${WINDOW}.0/" \
    -e "s/^  coupling_interval: .*/  coupling_interval: ${WINDOW}.0/" \
    o43_window.yaml > o58_window.yaml

COM="-ts_adapt_type none -snes_max_it 50 -snes_rtol 1e-5 -ksp_max_it 300 -ksp_type gmres -ksp_pc_side right -ksp_rtol 1e-4 -pc_type pbjacobi -dm_vec_type kokkos -dm_mat_type baijkokkos -ts_trajectory_type memory -ts_trajectory_max_cps_ram 400 -adjoint_fd_samples 0"
EV="-adjoint_calibrate_classes -adjoint_hwm_file turning30m_hwm_obs_clusterA.txt -adjoint_class_file turning30m_class.bin -adjoint_prior_file turning30m_manning.bin -adjoint_obs_freq 300 -adjoint_obs_error 0.15 -adjoint_hwm_eval_only"

peaks_at () {  # $1 = tag, $2 = class table
  srun -N 4 -n 16 -c 32 --cpu-bind=cores -G 16 --gpu-bind=none \
    $ADJ o58_window.yaml $COM $EV -adjoint_classes_init "$2" \
    -adjoint_hwm_dump o58_pk_$1.txt \
    -restart $CKPT -adjoint_rain_start_hour 29 \
    -raster_rain_dir $RAIN -raster_rain_start_date 2017,8,26,18,0 > o58_$1.log 2>&1
  echo "  $1 exit=$? $(grep -o 'MAE [0-9.]* m' o58_$1.log)"
}

echo "=== o58 base (NLCD prior), window ${WINDOW} steps, eps ${EPS} $(date)"
peaks_at base o43_p_nlcd.txt

for C in $(awk '!/^#/{print $1}' o43_p_nlcd.txt); do
  awk -v c="$C" -v e="$EPS" '!/^#/{ if ($1==c) printf "%d %.10g\n", $1, $2*(1+e); else printf "%d %.10g\n", $1, $2 }' \
      o43_p_nlcd.txt > o58_pert_$C.txt
  echo "=== o58 column $C $(date)"
  peaks_at col$C o58_pert_$C.txt
done

echo
echo "=== o58 done $(date). Assemble with:"
echo "    python3 o58_gauss_newton.py $EPS o58_pk_base.txt o58_pk_col*.txt"
