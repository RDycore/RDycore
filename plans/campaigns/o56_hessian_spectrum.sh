#!/bin/bash
# o56: how many roughness parameters does the observable actually support?
#      The spectral answer, as a pilot on the short window.
#
#   salloc -N 4 -C gpu -q interactive -t 2:00:00 -A m1516_g --gpus 16
#   bash o56_hessian_spectrum.sh
#
# Every statement this project has made about identifiability -- "the peak-WSE
# observable constrains about ONE roughness degree of freedom" -- rests on
# scans: the uniform-alpha curve, the half-domain split, the classes that pin.
# Those are good evidence and they agree with each other, but they are
# inference from a handful of directions. The eigenvalues of the
# prior-preconditioned Gauss-Newton Hessian answer the same question directly:
# in the basis where the prior is white, each eigenvalue > 1 is a direction in
# which the data says more than the prior does, so counting them IS the number
# of parameters the observable supports. Their eigenvectors say WHICH
# combinations, which the scans cannot.
#
# METHOD. With 15 classes this needs no low-rank machinery: assemble the
# 15x15 Hessian one column at a time by differencing the gradient,
#     H[:,k] ~ ( g(alpha + eps*e_k) - g(alpha) ) / eps,
# which is 16 gradients. -adjoint_classes_grad_only makes each one exactly one
# objective+gradient. On the 7200-step window at n16 that is ~6 min apiece,
# so the pilot costs ~100 minutes; the same thing on the production window
# would cost ~13 hours, which is why this is a pilot. The gradient's leading
# structure was stable across a 6x change in window length (developed-medium
# and woody wetland ranked 1-2 at both), so the eigenvalue COUNT has a fair
# chance of transferring even though the values will not.
#
# eps = 0.05 in alpha: a 5% roughness change, well above the gradient's
# measured noise floor (~1e-6 relative) and small enough to stay in the
# quadratic regime.
#
# The base point is the NLCD prior (alpha = 1), which is where the paper's
# identifiability claims are made.
set -u
cd $SCRATCH/gpu-implicit
ADJ=$HOME/Codes/rdycore-manning/build-claude-gpu8/driver/rdycore_adjoint
RAIN=/global/cfs/cdirs/m4267/shared/data/harvey/spatially-distributed-rainfall/mm-per-hr/mrms/bin
export MPICH_GPU_SUPPORT_ENABLED=1
EPS=0.05

CKPT=checkpoints_o37/o37.rdycore.r.104400.bin
test -f o43_window.yaml || { echo "MISSING o43_window.yaml"; exit 1; }
test -f o43_p_nlcd.txt  || { echo "MISSING o43_p_nlcd.txt (the NLCD prior table)"; exit 1; }
strings $ADJ | grep -q adjoint_classes_grad_only || { echo "STALE BINARY: $ADJ predates -adjoint_classes_grad_only"; exit 1; }

sed -e "s/^  stop             : .*/  stop             : 7200.0/" \
    -e "s/^  coupling_interval: .*/  coupling_interval: 7200.0/" \
    o43_window.yaml > o56_window.yaml

COM="-ts_adapt_type none -snes_max_it 50 -snes_rtol 1e-5 -ksp_max_it 300 -ksp_type gmres -ksp_pc_side right -ksp_rtol 1e-4 -pc_type pbjacobi -dm_vec_type kokkos -dm_mat_type baijkokkos -ts_trajectory_type memory -ts_trajectory_max_cps_ram 400 -adjoint_fd_samples 0"
CAL="-adjoint_calibrate_classes -adjoint_hwm_file turning30m_hwm_obs_clusterA.txt -adjoint_class_file turning30m_class.bin -adjoint_prior_file turning30m_manning.bin -adjoint_obs_freq 300 -adjoint_obs_error 0.15 -adjoint_classes_relative -adjoint_sigma_n 0.015"

CODES=$(awk '!/^#/{print $1}' o43_p_nlcd.txt)

grad_at () {  # $1 = tag, $2 = classes table to evaluate at
  srun -N 4 -n 16 -c 32 --cpu-bind=cores -G 16 --gpu-bind=none \
    $ADJ o56_window.yaml $COM $CAL -adjoint_classes_init "$2" \
    -restart $CKPT -adjoint_rain_start_hour 29 \
    -adjoint_classes_grad_dump o56_g_$1.txt -adjoint_classes_grad_only \
    -raster_rain_dir $RAIN -raster_rain_start_date 2017,8,26,18,0 > o56_$1.log 2>&1
  echo "  $1 exit=$? $(grep -o 'J [0-9.e+]*' o56_$1.log | head -1)"
}

echo "=== o56 base point (NLCD prior) $(date)"
grad_at base o43_p_nlcd.txt

for C in $CODES; do
  # perturb class C by eps in ALPHA, i.e. n_C -> n_C * (1 + eps)
  awk -v c="$C" -v e="$EPS" '!/^#/{ if ($1==c) printf "%d %.10g\n", $1, $2*(1+e); else printf "%d %.10g\n", $1, $2 }' \
      o43_p_nlcd.txt > o56_pert_$C.txt
  echo "=== o56 column $C $(date)"
  grad_at col$C o56_pert_$C.txt
done

echo
echo "=== o56 done $(date). Assemble with:"
echo "    python3 o56_hessian.py $EPS o56_g_base.txt o56_g_col*.txt"
