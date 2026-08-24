#----------------------------------------------------------
# Tests for the adjoint sensitivity driver (rdycore_adjoint)
#----------------------------------------------------------

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/adjoint_dam_break.yaml
          ${CMAKE_CURRENT_SOURCE_DIR}/adjoint_dam_break_analytic.yaml
          ${CMAKE_CURRENT_SOURCE_DIR}/adjoint_beuler.yaml
          ${CMAKE_CURRENT_SOURCE_DIR}/adjoint_arkimex.yaml
          ${CMAKE_CURRENT_SOURCE_DIR}/adjoint_beuler_freeflow.yaml
          ${CMAKE_CURRENT_SOURCE_DIR}/adjoint_arkimex_freeflow.yaml
          ${MESH_DIR}/planar_dam_10x5.msh
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
