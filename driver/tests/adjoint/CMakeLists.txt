#----------------------------------------------------------
# Tests for the adjoint sensitivity driver (rdycore_adjoint)
#----------------------------------------------------------

find_program(H5DIFF h5diff REQUIRED
             PATHS ${PETSC_DIR}/${PETSC_ARCH}/bin)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/adjoint_dam_break.yaml
          ${CMAKE_CURRENT_SOURCE_DIR}/adjoint_dam_break_analytic.yaml
          ${CMAKE_CURRENT_SOURCE_DIR}/adjoint_beuler.yaml
          ${CMAKE_CURRENT_SOURCE_DIR}/adjoint_arkimex.yaml
          ${CMAKE_CURRENT_SOURCE_DIR}/adjoint_beuler_freeflow.yaml
          ${CMAKE_CURRENT_SOURCE_DIR}/adjoint_arkimex_freeflow.yaml
          ${MESH_DIR}/planar_dam_10x5.msh
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# data for the checkpoint/restart consistency gate (rain-forced Houston 1 km)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/adjoint_restart_ref.yaml
          ${CMAKE_CURRENT_SOURCE_DIR}/adjoint_restart_window.yaml
          ${CMAKE_CURRENT_SOURCE_DIR}/adjoint_restart_window_np2.yaml
          ${MESH_DIR}/Houston1km_with_z.exo
          ${CONDITION_DIR}/Houston1km.ic.${PETSC_ID_TYPE}.bin
          ${CONDITION_DIR}/2017-08-26:00-00.${PETSC_ID_TYPE}.bin
          ${CONDITION_DIR}/2017-08-26:01-00.${PETSC_ID_TYPE}.bin
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# NLCD class map + prior for the class-calibration warm-start gate
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/adjoint_classes_houston.yaml
          ${CMAKE_SOURCE_DIR}/data/nlcd/houston1km_class.bin
          ${CMAKE_SOURCE_DIR}/data/nlcd/houston1km_manning.bin
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# dJ/du0 via TSAdjoint vs central finite differences (the driver exits with an
# error if the relative L2 gate fails). The FD-coloring Jacobian is used so
# this validates the adjoint wiring independent of the analytic Jacobian.
add_test(adjoint_dam_break_np_1 ${MPIEXEC} ${MPIEXEC_FLAGS} -n 1 ${adjoint_driver} adjoint_dam_break.yaml)

# two-zone Manning twin calibration: TAO/BLMVM + adjoint gradients recover
# per-region n_true = (0.03, 0.06) from a uniform 0.02 start (gate: 2e-2
# relative per region; observed recovery ~4e-7)
add_test(calibrate_manning_twin_np_1 ${MPIEXEC} ${MPIEXEC_FLAGS} -n 1 ${adjoint_driver} adjoint_dam_break.yaml -adjoint_calibrate)

# same FD gates through the ANALYTIC Jacobian (exact flux + boundary blocks):
# validates increments 2c + 2d end-to-end through the adjoint chain
add_test(adjoint_dam_break_analytic_np_1 ${MPIEXEC} ${MPIEXEC_FLAGS} -n 1 ${adjoint_driver} adjoint_dam_break_analytic.yaml)

# per-cell Manning recovery with multi-time observations + Tikhonov (fast
# CI variant: 8 obs times, 200-iteration cap -> validated 0.199 rel L2;
# the converged setting (20 obs times, beta 1e-5, ~1900 its) reaches 0.080)
add_test(calibrate_manning_percell_np_1 ${MPIEXEC} ${MPIEXEC_FLAGS} -n 1 ${adjoint_driver} adjoint_dam_break_analytic.yaml -adjoint_calibrate_percell -adjoint_obs_stride 1)

# FD gates through fully-implicit BEULER (theta-method adjoint; the exact
# coupled Jacobian drives the SNES solves). Tight inner tolerances: the
# discrete-adjoint gradient is only as exact as the implicit solves.
add_test(adjoint_beuler_np_1 ${MPIEXEC} ${MPIEXEC_FLAGS} -n 1 ${adjoint_driver} adjoint_beuler.yaml -snes_rtol 1e-12 -snes_atol 1e-50 -ksp_rtol 1e-12)

# FD gates through ARK-IMEX with implicit Manning friction (increment 1b):
# fixed steps (the embedded controller otherwise varies the step sequence
# under FD perturbations) and tight inner tolerances
add_test(adjoint_arkimex_np_1 ${MPIEXEC} ${MPIEXEC_FLAGS} -n 1 ${adjoint_driver} adjoint_arkimex.yaml -ts_adapt_type none -snes_rtol 1e-12 -snes_atol 1e-50 -ksp_rtol 1e-12)

# same FD gates with a free-outflow (transmissive) boundary on the bottom
# wall: validates the free-outflow ghost's identity Jacobian map through the
# implicit solves and the adjoint chain
add_test(adjoint_beuler_freeflow_np_1 ${MPIEXEC} ${MPIEXEC_FLAGS} -n 1 ${adjoint_driver} adjoint_beuler_freeflow.yaml -snes_rtol 1e-12 -snes_atol 1e-50 -ksp_rtol 1e-12)
add_test(adjoint_arkimex_freeflow_np_1 ${MPIEXEC} ${MPIEXEC_FLAGS} -n 1 ${adjoint_driver} adjoint_arkimex_freeflow.yaml -ts_adapt_type none -snes_rtol 1e-12 -snes_atol 1e-50 -ksp_rtol 1e-12)

# peak-WSE (high-water-mark) observation mode: a synthetic mark table is
# written from the two-zone truth forward's peaks, read back through the same
# file path real HWM data uses (data/harvey_hwm/), and the per-cell parameter
# gradient of the peak misfit is gated against central differences
# (probes the largest-|g| components; the argmax bookkeeping is the new code
# under test). tao_max_it 1 keeps it cheap -- the FD gate is the pass/fail.
add_test(adjoint_hwm_fd_np_1 ${MPIEXEC} ${MPIEXEC_FLAGS} -n 1 ${adjoint_driver} adjoint_dam_break_analytic.yaml -adjoint_calibrate_gauges -adjoint_gauges_twin -adjoint_hwm_file hwm_twin_np1.txt -adjoint_hwm_twin -adjoint_hwm_fd -adjoint_fd_samples 3 -tao_max_it 1)
add_test(adjoint_hwm_fd_np_2 ${MPIEXEC} ${MPIEXEC_FLAGS} -n 2 ${adjoint_driver} adjoint_dam_break_analytic.yaml -adjoint_calibrate_gauges -adjoint_gauges_twin -adjoint_hwm_file hwm_twin_np2.txt -adjoint_hwm_twin -adjoint_hwm_fd -adjoint_fd_samples 3 -tao_max_it 1)

# gauge-observation calibration (the real-data path): a synthetic WSE table
# is written from a two-zone truth at strided gauge cells, read back through
# the same file format real data uses, and calibrated per-cell. Gate: 2x
# misfit reduction (np_1 and np_2 measure ~10.5x, identical to 6 digits).
add_test(calibrate_manning_gauges_np_1 ${MPIEXEC} ${MPIEXEC_FLAGS} -n 1 ${adjoint_driver} adjoint_dam_break.yaml -adjoint_calibrate_gauges -adjoint_gauges_twin -adjoint_obs_file gauge_twin_obs_np1.txt -adjoint_jred_gate 0.5 -tao_max_it 60)
add_test(calibrate_manning_gauges_np_2 ${MPIEXEC} ${MPIEXEC_FLAGS} -n 2 ${adjoint_driver} adjoint_dam_break.yaml -adjoint_calibrate_gauges -adjoint_gauges_twin -adjoint_obs_file gauge_twin_obs_np2.txt -adjoint_jred_gate 0.5 -tao_max_it 60)

# checkpoint/restart consistency: the mid-event calibration window's IC path.
# A calibration window that does not start at the event's only initial
# condition is started from a checkpoint of a longer forward instead, and its
# rain has to be re-aligned with the event clock because the driver restarts
# the TS clock at 0 for every forward. The gate: hour 2 of a continuous
# 2-hour rain-forced run must equal the same hour computed from the 1-hour
# checkpoint with -adjoint_rain_start_hour 1. Dropping the rain offset moves
# the answer by 2.3e-2 relative L2, so both comparisons below discriminate.
# -adjoint_forward_only is what makes the checkpointed state well defined --
# every other driver mode runs several forwards over the window.
add_test(adjoint_restart_ref_np_1 ${MPIEXEC} ${MPIEXEC_FLAGS} -n 1 ${adjoint_driver} adjoint_restart_ref.yaml -raster_rain_start_date 2017,8,26,0,0 -raster_rain_dir ./ -adjoint_forward_only)

# same decomposition as the reference: the restarted window must reproduce it
# BITWISE (measured: 0 differences over all 8238 dofs)
add_test(adjoint_restart_window_np_1 ${MPIEXEC} ${MPIEXEC_FLAGS} -n 1 ${adjoint_driver} adjoint_restart_window.yaml -raster_rain_start_date 2017,8,26,0,0 -raster_rain_dir ./ -adjoint_forward_only -restart checkpoints/restart_ref.rdycore.r.060.h5 -adjoint_rain_start_hour 1)
set_tests_properties(adjoint_restart_window_np_1 PROPERTIES DEPENDS "adjoint_restart_ref_np_1")

add_test(adjoint_restart_compare_np_1 ${H5DIFF} checkpoints/restart_ref.rdycore.r.120.h5 checkpoints/restart_window.rdycore.r.60.h5 /fields/solution)
set_tests_properties(adjoint_restart_compare_np_1 PROPERTIES DEPENDS "adjoint_restart_window_np_1")

# a DIFFERENT decomposition than the checkpoint's: checkpoints are written in
# natural order, so the calibration window need not use the node count of the
# forward that produced its IC. Only reduction order changes, so this one is
# compared with a relative tolerance (measured: 6 dofs above 1e-10, max 2.4e-10,
# all of them small momentum components).
add_test(adjoint_restart_window_np_2 ${MPIEXEC} ${MPIEXEC_FLAGS} -n 2 ${adjoint_driver} adjoint_restart_window_np2.yaml -raster_rain_start_date 2017,8,26,0,0 -raster_rain_dir ./ -adjoint_forward_only -restart checkpoints/restart_ref.rdycore.r.060.h5 -adjoint_rain_start_hour 1)
set_tests_properties(adjoint_restart_window_np_2 PROPERTIES DEPENDS "adjoint_restart_ref_np_1")

add_test(adjoint_restart_compare_np_2 ${H5DIFF} -p 1e-8 checkpoints/restart_ref.rdycore.r.120.h5 checkpoints/restart_window2.rdycore.r.60.h5 /fields/solution)
set_tests_properties(adjoint_restart_compare_np_2 PROPERTIES DEPENDS "adjoint_restart_window_np_2")

# class-calibration warm start: a production window needs more TAO iterations
# than fit in one queue slot, so a run dumps its parameter vector and the next
# job resumes from it. The resume is exact in the PARAMETERS (measured: the
# warm run's TAO iteration 0 reports the cold run's final objective to 6
# digits, 1466.42); BLMVM's quasi-Newton history does not carry over, which
# on this twin is not a penalty -- chained 2+2 reaches J 115.0 where a single
# cold 4-iteration run reaches 277.2, the restart acting as a rescaling of the
# quasi-Newton model. The gate is that the file round-trips and every class is
# matched by NLCD code (the driver errors if any is missing).
add_test(adjoint_classes_dump_np_1 ${MPIEXEC} ${MPIEXEC_FLAGS} -n 1 ${adjoint_driver} adjoint_classes_houston.yaml -adjoint_calibrate_classes -adjoint_classes_twin -adjoint_class_file houston1km_class.bin -adjoint_prior_file houston1km_manning.bin -adjoint_obs_file classes_twin_obs_np1.txt -adjoint_obs_freq 10 -tao_max_it 2 -adjoint_classes_dump classes_p_np1.txt)

add_test(adjoint_classes_warmstart_np_1 ${MPIEXEC} ${MPIEXEC_FLAGS} -n 1 ${adjoint_driver} adjoint_classes_houston.yaml -adjoint_calibrate_classes -adjoint_classes_twin -adjoint_class_file houston1km_class.bin -adjoint_prior_file houston1km_manning.bin -adjoint_obs_file classes_twin_obs2_np1.txt -adjoint_obs_freq 10 -tao_max_it 1 -adjoint_classes_init classes_p_np1.txt)
set_tests_properties(adjoint_classes_warmstart_np_1 PROPERTIES
  DEPENDS "adjoint_classes_dump_np_1"
  PASS_REGULAR_EXPRESSION "warm start from classes_p_np1.txt \\(15 classes\\)")

# -adjoint_classes_relative: the optimizer's variable is alpha = n/n_prior and
# its objective is J/J(start), both O(1). The chain rule dJ/dalpha = n_prior
# dJ/dn is what this gates, against central differences on alpha (measured
# 2.3e-6 at eps 1e-4; the default 1e-3 probe is truncation-limited here at
# 3e-5 because a step in alpha is a larger step in n than the absolute mode's
# probe, and the error scales as eps^2 through the whole sweep 3e-5..3e-2).
# Cross-checked separately: the dJ/dn recovered from the alpha gradient agrees
# with the absolute mode's gradient at the same point to 1e-8, the dump's
# 10-digit precision.
add_test(adjoint_classes_relative_fd_np_1 ${MPIEXEC} ${MPIEXEC_FLAGS} -n 1 ${adjoint_driver} adjoint_classes_houston.yaml -adjoint_calibrate_classes -adjoint_classes_twin -adjoint_class_file houston1km_class.bin -adjoint_prior_file houston1km_manning.bin -adjoint_obs_file classes_rel_fd_np1.txt -adjoint_obs_freq 10 -tao_max_it 1 -adjoint_classes_relative -adjoint_sigma_n 0.015 -adjoint_hwm_fd -adjoint_fd_samples 3 -adjoint_fd_eps 1e-4)

# and that the scaling does what it is for: BLMVM must actually take steps.
# The first real-data calibration did ZERO iterations -- the unscaled first
# trial point projected onto the bounds and every line-search trial failed --
# so the gate here is a misfit reduction, not just a clean exit. Measured on
# this twin: J 3.101e1 -> 1.263e1 in 3 iterations (0.41x), every iteration
# accepted. -adjoint_sigma_n sets the Tikhonov weight as 1/sigma_n^2.
add_test(adjoint_classes_relative_np_1 ${MPIEXEC} ${MPIEXEC_FLAGS} -n 1 ${adjoint_driver} adjoint_classes_houston.yaml -adjoint_calibrate_classes -adjoint_classes_twin -adjoint_class_file houston1km_class.bin -adjoint_prior_file houston1km_manning.bin -adjoint_obs_file classes_rel_np1.txt -adjoint_obs_freq 10 -tao_max_it 3 -adjoint_classes_relative -adjoint_sigma_n 0.015 -adjoint_jred_gate 0.6)

# -adjoint_sigma_alpha: the prior stated as a fraction of the prior value
# rather than an absolute offset in n. This gate has to be taken at a point
# where alpha != 1, because at the prior BOTH prior forms contribute exactly
# zero to the objective and its gradient, so a check there cannot tell them
# apart -- it warm-starts from the dump of adjoint_classes_dump_np_1 instead.
# Measured: 4.9e-7 for the relative prior and 4.2e-7 for the absolute one at
# the same point (eps 1e-4).
add_test(adjoint_classes_relprior_fd_np_1 ${MPIEXEC} ${MPIEXEC_FLAGS} -n 1 ${adjoint_driver} adjoint_classes_houston.yaml -adjoint_calibrate_classes -adjoint_classes_twin -adjoint_class_file houston1km_class.bin -adjoint_prior_file houston1km_manning.bin -adjoint_obs_file classes_relprior_np1.txt -adjoint_obs_freq 10 -tao_max_it 1 -adjoint_classes_relative -adjoint_classes_init classes_p_np1.txt -adjoint_sigma_alpha 0.3 -adjoint_hwm_fd -adjoint_fd_samples 3 -adjoint_fd_eps 1e-4)
set_tests_properties(adjoint_classes_relprior_fd_np_1 PROPERTIES DEPENDS "adjoint_classes_dump_np_1")
