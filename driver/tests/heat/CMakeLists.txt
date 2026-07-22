#------------------------------------------------------
# MMS tests for the heat equation operator split
#------------------------------------------------------

set(mms_c_driver ${mms_driver})

file(COPY
     ${CMAKE_CURRENT_SOURCE_DIR}/heat_ts_mms.yaml
     ${CMAKE_CURRENT_SOURCE_DIR}/heat_mms_conv_study.yaml
     ${CMAKE_CURRENT_SOURCE_DIR}/heat_coupling_interval.yaml
     ${MESH_DIR}/mms_triangles_dx1.exo
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Case 1: TS isolation smoke tests for both supported implicit methods
foreach(np 1 2)
  foreach(method beuler cn)
    foreach(lang ${supported_languages})
      add_test(heat_ts_mms_${method}_${lang}_np_${np} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${mms_${lang}_driver} heat_ts_mms.yaml -heat_ts_type ${method})
    endforeach()
  endforeach()
endforeach()

# The fixed-mesh temporal study verifies first-order backward Euler and
# second-order Crank-Nicolson source sampling in the complete MMS update.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME heat_ts_mms_temporal_convergence
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_heat_mms_convergence.py
                 --driver ${mms_c_driver} --input ${CMAKE_CURRENT_BINARY_DIR}/heat_ts_mms.yaml)
set_tests_properties(heat_ts_mms_temporal_convergence PROPERTIES
                     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Case 2: SWE + heat spatial convergence study
foreach(np 1 2)
  foreach(lang ${supported_languages})
    add_test(heat_mms_conv_study_${lang}_np_${np} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${mms_${lang}_driver} heat_mms_conv_study.yaml)
  endforeach()
endforeach()

# Production coupling regression: the internal transport timestep is shorter
# than RDyAdvance's coupling interval, while the expected heat increment is
# analytic.
add_executable(test_heat_coupling test_heat_coupling.c)
target_link_libraries(test_heat_coupling PRIVATE rdycore_c)
foreach(np 1 2)
  add_test(heat_coupling_interval_np_${np} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np}
           ${CMAKE_CURRENT_BINARY_DIR}/test_heat_coupling heat_coupling_interval.yaml)
endforeach()
