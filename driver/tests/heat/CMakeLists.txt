#------------------------------------------------------
# MMS tests for the heat equation operator split
#------------------------------------------------------
#
# Case numbering, used consistently across the inputs and the checker scripts:
#
#   Case 0  lake-at-rest source isolation          heat_ts_mms.yaml
#   Case 1  source-free moving transport           heat_mms_moving_transport.yaml
#   Case 2  moving transport plus a prescribed heat source
#                                                  heat_mms_moving_source.yaml
#   Case 3  spatially varying SWE + heat           heat_mms_conv_study.yaml
#
# (Case 4, MMS verification of the nonlinear atmospheric parameterization, is
# not implemented: the direct_source path replaces HeatQNet(T) rather than
# correcting it, so it cannot verify it.)

set(mms_c_driver ${mms_driver})

# arguments for specific configurations ("basic" config has no options)
set(ceed_args -ceed /cpu/self)

file(COPY
     ${CMAKE_CURRENT_SOURCE_DIR}/heat_ts_mms.yaml
     ${CMAKE_CURRENT_SOURCE_DIR}/heat_mms_moving_transport.yaml
     ${CMAKE_CURRENT_SOURCE_DIR}/heat_mms_moving_source.yaml
     ${CMAKE_CURRENT_SOURCE_DIR}/heat_mms_conv_study.yaml
     ${CMAKE_CURRENT_SOURCE_DIR}/heat_coupling_interval.yaml
     ${MESH_DIR}/mms_triangles_dx1.exo
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Case 0: TS isolation smoke tests for both supported implicit methods, on both backends
foreach(np 1 2)
  foreach(method beuler cn)
    foreach(config basic ceed)
      foreach(lang ${supported_languages})
        add_test(heat_ts_mms_${method}_${lang}_np_${np}_${config} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${mms_${lang}_driver} heat_ts_mms.yaml -heat_ts_type ${method} ${${config}_args})
      endforeach()
    endforeach()
  endforeach()
endforeach()

# The fixed-mesh temporal study verifies first-order backward Euler and
# second-order Crank-Nicolson source sampling in the complete MMS update. It is
# valid here, and only here, because the flow is at rest and the spatial error
# is identically zero.
add_test(NAME heat_ts_mms_temporal_convergence
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_heat_mms_convergence.py
                 --driver ${mms_c_driver} --input ${CMAKE_CURRENT_BINARY_DIR}/heat_ts_mms.yaml)

# Case 1: source-free moving transport. The smoke tests assert that uniform flow
# is preserved and that the manufactured heat source is zero to roundoff; the
# convergence test measures a joint space-time rate and, via an RK4 transport
# integrator, an isolated spatial rate.
foreach(np 1 2)
  foreach(config basic ceed)
    foreach(lang ${supported_languages})
      add_test(heat_mms_moving_transport_${lang}_np_${np}_${config} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${mms_${lang}_driver} heat_mms_moving_transport.yaml ${${config}_args})
    endforeach()
  endforeach()
endforeach()
# The bare runs above only check that nothing crashes -- a single-resolution MMS
# run prints norms rather than judging them -- so the roundoff claims are
# asserted here, across both rank counts and both backends.
foreach(config basic ceed)
  # NOTE: the driver arguments begin with a dash, so they must be passed with
  # NOTE: --driver-args=... rather than as a separate argv entry.
  string(REPLACE ";" " " config_args_str "${${config}_args}")
  add_test(NAME heat_mms_moving_transport_smoke_${config}
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_heat_moving_smoke.py
                   --driver ${mms_c_driver} --input ${CMAKE_CURRENT_BINARY_DIR}/heat_mms_moving_transport.yaml
                   --mpiexec ${MPIEXEC} --source zero "--driver-args=${config_args_str}")
endforeach()
add_test(NAME heat_mms_moving_transport_convergence
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_heat_moving_transport.py
                 --driver ${mms_c_driver} --input ${CMAKE_CURRENT_BINARY_DIR}/heat_mms_moving_transport.yaml)

# Case 2: moving transport plus a prescribed heat source. The convergence test
# measures temporal-plus-splitting order against a same-mesh fine-timestep
# reference; a fixed-mesh study against the exact solution would plateau on the
# spatial error floor instead of converging.
foreach(np 1 2)
  foreach(method beuler cn)
    foreach(config basic ceed)
      foreach(lang ${supported_languages})
        add_test(heat_mms_moving_source_${method}_${lang}_np_${np}_${config} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${mms_${lang}_driver} heat_mms_moving_source.yaml -heat_ts_type ${method} ${${config}_args})
      endforeach()
    endforeach()
  endforeach()
endforeach()
foreach(config basic ceed)
  string(REPLACE ";" " " config_args_str "${${config}_args}")
  foreach(method beuler cn)
    add_test(NAME heat_mms_moving_source_smoke_${method}_${config}
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_heat_moving_smoke.py
                     --driver ${mms_c_driver} --input ${CMAKE_CURRENT_BINARY_DIR}/heat_mms_moving_source.yaml
                     --mpiexec ${MPIEXEC} --source nonzero
                     "--driver-args=-heat_ts_type ${method} ${config_args_str}")
  endforeach()
endforeach()
add_test(NAME heat_mms_moving_source_convergence
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_heat_moving_source.py
                 --driver ${mms_c_driver} --input ${CMAKE_CURRENT_BINARY_DIR}/heat_mms_moving_source.yaml)

# Case 3: SWE + heat joint space-time convergence study, on both backends. The
# CEED variant exercises passive transport of the h*T DOF through the flux and
# source operators.
foreach(np 1 2)
  foreach(config basic ceed)
    foreach(lang ${supported_languages})
      add_test(heat_mms_conv_study_${lang}_np_${np}_${config} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${mms_${lang}_driver} heat_mms_conv_study.yaml ${${config}_args})
    endforeach()
  endforeach()
endforeach()

# the Python checkers run the driver from, and write reference states into, the
# binary directory
set(heat_python_tests
    heat_ts_mms_temporal_convergence
    heat_mms_moving_transport_convergence
    heat_mms_moving_source_convergence)
foreach(config basic ceed)
  list(APPEND heat_python_tests heat_mms_moving_transport_smoke_${config})
  foreach(method beuler cn)
    list(APPEND heat_python_tests heat_mms_moving_source_smoke_${method}_${config})
  endforeach()
endforeach()
set_tests_properties(${heat_python_tests} PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Production coupling regression: the internal transport timestep is shorter
# than RDyAdvance's coupling interval, while the expected heat increment is
# analytic.
add_executable(test_heat_coupling test_heat_coupling.c)
target_link_libraries(test_heat_coupling PRIVATE rdycore_c)
foreach(np 1 2)
  foreach(config basic ceed)
    add_test(heat_coupling_interval_np_${np}_${config} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np}
             ${CMAKE_CURRENT_BINARY_DIR}/test_heat_coupling heat_coupling_interval.yaml ${${config}_args})
  endforeach()
endforeach()
