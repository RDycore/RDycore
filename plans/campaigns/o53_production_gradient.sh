#!/bin/bash
# o53: the production-window gradient, run interactively at n16.
#
#   salloc -N 4 -C gpu -q interactive -t 4:00:00 -A m1516_g --gpus 16
#   bash o53_production_gradient.sh
#
# This is the command that was run by hand on 2026-08-26 to use an
# otherwise-idle 4-node interactive allocation while the batch queue was
# saturated; it is recorded here so the gradient it produced is reproducible.
#
# It is o48_calibrate_rel.sh's configuration exactly -- 12-hour window over
# crest cluster A, h29 checkpoint, rain re-aligned, the 46 real marks,
# relative variables, sigma_n 0.015 -- with -tao_max_it 3, which at ~50 min
# per objective+gradient means the allocation wall stops it rather than the
# iteration count. That is safe: -adjoint_classes_dump rewrites after every
# TAO iteration.
#
# WHAT IT IS FOR: the start-point gradient dump (o53_g.txt) is the artifact
# that aims the reduced-parameter run. Rank the classes by |dJ/dn| * n_prior
# -- the gradient in the variable actually being optimized -- and pass the
# leaders to o50_calibrate_active.sh. Note that the start-point gradient is
# INDEPENDENT of sigma_n, because the Tikhonov term is exactly zero at the
# prior, so this dump is reusable whatever regularization is chosen later.
set -u
cd $SCRATCH/gpu-implicit
ADJ=$HOME/Codes/rdycore-manning/build-claude-gpu6/driver/rdycore_adjoint
RAIN=/global/cfs/cdirs/m4267/shared/data/harvey/spatially-distributed-rainfall/mm-per-hr/mrms/bin
export MPICH_GPU_SUPPORT_ENABLED=1

test -f o43_window.yaml || { echo "MISSING o43_window.yaml -- run any o48 job first"; exit 1; }

srun -N 4 -n 16 -c 32 --cpu-bind=cores -G 16 --gpu-bind=none \
  $ADJ o43_window.yaml \
  -ts_adapt_type none -snes_max_it 50 -snes_rtol 1e-5 -ksp_max_it 300 -ksp_type gmres \
  -ksp_pc_side right -ksp_rtol 1e-4 -pc_type pbjacobi -dm_vec_type kokkos \
  -dm_mat_type baijkokkos -ts_trajectory_type memory -ts_trajectory_max_cps_ram 400 \
  -adjoint_fd_samples 0 -adjoint_calibrate_classes \
  -adjoint_hwm_file turning30m_hwm_obs_clusterA.txt -adjoint_class_file turning30m_class.bin \
  -adjoint_prior_file turning30m_manning.bin -adjoint_obs_freq 300 -adjoint_obs_error 0.15 \
  -adjoint_classes_relative -adjoint_sigma_n 0.015 \
  -restart checkpoints_o37/o37.rdycore.r.104400.bin -adjoint_rain_start_hour 29 \
  -tao_max_it 3 -tao_monitor \
  -adjoint_classes_dump o53_p.txt -adjoint_classes_grad_dump o53_g.txt \
  -raster_rain_dir $RAIN -raster_rain_start_date 2017,8,26,18,0 2>&1 | tee o53_interactive.log

echo "--- start-point gradient (dJ/dn per class) ---"
cat o53_g.txt 2>/dev/null
echo "--- parameters, rewritten after every TAO iteration ---"
cat o53_p.txt 2>/dev/null
