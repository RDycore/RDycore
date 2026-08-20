# Here we manually define unit tests for the main RDycore library. We can
# discuss ways of setting up tests more easily and automatically.

# We need a mesh for some of our unit tests.
file(COPY ${MESH_DIR}/planar_dam_10x5.msh
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# configs for the global Jacobian assembly test
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/swe_jacobian_global_fd.yaml
          ${CMAKE_CURRENT_SOURCE_DIR}/swe_jacobian_global_analytic.yaml
          ${CMAKE_CURRENT_SOURCE_DIR}/swe_jacobian_global_dam_fd.yaml
          ${CMAKE_CURRENT_SOURCE_DIR}/swe_jacobian_global_dam_analytic.yaml
          ${CMAKE_CURRENT_SOURCE_DIR}/swe_jacobian_global_bc_fd.yaml
          ${CMAKE_CURRENT_SOURCE_DIR}/swe_jacobian_global_bc_analytic.yaml
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

foreach(exe_name
        test_rdyinit
        test_rdymesh
        test_swe_jacobian
        test_swe_jacobian_global
        test_yaml_input)
  set(exe_source ${exe_name}.c)
  add_executable(${exe_name} ${exe_source})
  target_link_libraries(${exe_name} PRIVATE rdycore_c cmocka)
  foreach(np 1 2) # test on 1 and 2 processes
    set(test_name ${exe_name}_np_${np})
    add_test(${test_name} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${CMAKE_CURRENT_BINARY_DIR}/${exe_name})
  endforeach()
endforeach()
