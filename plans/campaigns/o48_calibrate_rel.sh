#!/bin/bash
#SBATCH -A m1516_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 720
#SBATCH -N 4
#SBATCH -o o48_slurm_%j.out
# o48: the real-data calibration, redone with a scaled optimization problem.
#
#   sbatch o48_calibrate_rel.sh <TAO_ITS> [INIT_FILE]
#
# Same physical setup as o43 -- 12-hour window over crest cluster A (event
# hours 29-41), IC from the converged-tolerance hourly checkpoint at h29
# (o37), rain realigned to h29, the 46 REAL cluster-A marks. What changed is
# the OPTIMIZATION problem, because o43 did zero TAO iterations:
#
#  1. -adjoint_classes_relative: the variable is alpha_k = n_k/n_prior_k and
#     the objective is J/J(start), both O(1). BLMVM's first trial point is
#     x - g; with n as the variable and |g| = 1554 that was five orders too
#     large and projected onto the lower bound, where the model provably
#     cannot solve (uniform alpha 0.2 diverges, o44). The bounds are now
#     alpha in [0.3, 3.0]: a physical statement that also keeps the iterates
#     out of the divergent region.
#
#  2. -adjoint_sigma_n 0.015 instead of -adjoint_beta 1e2. beta is
#     dimensional and was set ad hoc everywhere in this project (1e-6 to 1e2
#     -- the spread is the symptom); beta = 1/sigma_n^2 makes it a prior
#     standard deviation on Manning n. Against the measured J(alpha) curve
#     (o44/o45) sigma_n = 0.015 puts the uniform-mode minimum at alpha ~ 0.70
#     (J_mis ~757, reg ~29). sigma_n = 0.020 has NO interior minimum in that
#     direction -- it runs to the bound -- which is what beta = 1e2 was doing
#     at a relative roughness change of 13.6. Report sigma_n, never a bare
#     beta.
#
# EXPECTED OUTCOME, which is the point of the run: descent to roughly the
# alpha-curve level, MAE 0.7188 -> ~0.69, i.e. the calibration converging to
# where the optimizer-independent sensitivity analysis says it must. A larger
# improvement would be surprising and worth distrusting; classes pinning at
# the bounds is the documented over-parameterization result (o28/o30/o47) and
# the cue to rerun with -adjoint_classes_active on the few that carry signal.
#
# COST: an objective+gradient is ~2 hr at n4 and ~50 min at n16 (the cost is
# dominated by revolve RECOMPUTATION, which is forward work, so it scales
# like the forward rather than like the adjoint sweep). The driver's own
# start-point evaluation -- which prints the "before" MAE and writes the
# gradient dump -- is one more. At n16 a 12-hr slot therefore fits roughly a
# dozen TAO iterations, i.e. a whole calibration. TAO_ITS is still set ABOVE
# what fits: the dump is rewritten after every iteration, so the queue wall
# is a stopping rule rather than a lost slot.
# Chain: this job dumps o48_p_<jobid>.txt from iteration 1 onward; pass it
# to the next.
# NO -adjoint_hwm_twin: the table is real data.
set -u
TAO_ITS=${1:?need TAO_ITS}
INIT_FILE=${2:-}   # empty: relative mode starts at alpha = 1, i.e. exactly the
                   # NLCD prior -- no init file needed for the first link

export MPICH_GPU_SUPPORT_ENABLED=1
# o51 measured the production window forward at 36 min (n4, 1 node),
# 21m55s (n8, 2 nodes) and 15m01s (n16, 4 nodes), all three reproducing
# J 8.082566e+02 / MAE 0.7188 to every printed digit -- no decomposition
# dependence. n16 is 2.4x faster at 60% parallel efficiency; we buy the
# wall-clock because it puts a whole calibration inside ONE queue slot,
# and queue wait, not node-hours, is the binding constraint.
RANKS=${RANKS:-16}
NODES=$((RANKS / 4))
cd $SCRATCH/gpu-implicit
ADJ=$HOME/Codes/rdycore-manning/build-claude-gpu6/driver/rdycore_adjoint
RAIN=/global/cfs/cdirs/m4267/shared/data/harvey/spatially-distributed-rainfall/mm-per-hr/mrms/bin

START_HOUR=29
WINDOW_SEC=43200
CKPT=checkpoints_o37/o37.rdycore.r.104400.bin
test -f "$CKPT" || { echo "MISSING $CKPT"; exit 1; }
test -f turning30m_hwm_obs_clusterA.txt || { echo "MISSING cluster-A table"; exit 1; }
strings $ADJ | grep -q adjoint_classes_relative || { echo "STALE BINARY: $ADJ predates -adjoint_classes_relative"; exit 1; }

sed -e "s/^  stop             : .*/  stop             : ${WINDOW_SEC}.0/" \
    -e "s/^  coupling_interval: .*/  coupling_interval: ${WINDOW_SEC}.0/" \
    o31_72hr_freeflow.yaml > o43_window.yaml

COM="-ts_adapt_type none -snes_max_it 50 -snes_rtol 1e-5 -ksp_max_it 300 -ksp_type gmres -ksp_pc_side right -ksp_rtol 1e-4 -pc_type pbjacobi -dm_vec_type kokkos -dm_mat_type baijkokkos -ts_trajectory_type memory -ts_trajectory_max_cps_ram 400 -adjoint_fd_samples 0"
CAL="-adjoint_calibrate_classes -adjoint_hwm_file turning30m_hwm_obs_clusterA.txt -adjoint_class_file turning30m_class.bin -adjoint_prior_file turning30m_manning.bin -adjoint_obs_freq 300 -adjoint_obs_error 0.15 -adjoint_classes_relative -adjoint_sigma_n 0.015"
DUMP="o48_p_${SLURM_JOB_ID}.txt"
GRAD="o48_g_${SLURM_JOB_ID}.txt"
RESUME=""
if [ -n "$INIT_FILE" ]; then RESUME="-adjoint_classes_init $INIT_FILE"; fi

echo "=== o48 scaled calibration start $(date)"
echo "    window: h${START_HOUR}-$((START_HOUR + WINDOW_SEC/3600)) | 46 cluster-A marks | sigma_n 0.015 | sigma 0.15"
echo "    IC: $CKPT | start: ${INIT_FILE:-NLCD prior (alpha = 1)}"
echo "    its this job: $TAO_ITS | dump: $DUMP | grad: $GRAD"
srun -N $NODES -n $RANKS -c 32 --cpu-bind=cores -G $RANKS --gpu-bind=none \
  $ADJ o43_window.yaml $COM $CAL $RESUME \
  -restart $CKPT -adjoint_rain_start_hour $START_HOUR \
  -tao_max_it $TAO_ITS -tao_monitor \
  -adjoint_classes_dump $DUMP -adjoint_classes_grad_dump $GRAD \
  -raster_rain_dir $RAIN -raster_rain_start_date 2017,8,26,18,0 > o48_${SLURM_JOB_ID}.log 2>&1
echo "exit=$? $(date)"
grep -E "warm start|class calibration|hwm init|hwm final|TAO,|class recovery" o48_${SLURM_JOB_ID}.log | tail -25
echo "--- per-class table (recovered_n vs NLCD prior; rel_err IS |alpha-1|) ---"
grep -A17 "NLCD  prior_n" o48_${SLURM_JOB_ID}.log | tail -18
echo "--- start-point gradient (dJ/dn per class) ---"
cat $GRAD 2>/dev/null
echo "--- continue: sbatch o48_calibrate_rel.sh <its> $DUMP"
echo "--- stop the chain when J flattens or classes reach alpha 0.3/3.0 (pinning"
echo "    is noise-fitting in directions the marks do not constrain, o30/o47)"
