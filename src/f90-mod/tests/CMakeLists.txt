# This is a test of RDycore's ability to couple with E3SM.

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/coupling.yaml ${MESH_DIR}/planar_dam_10x5.msh
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

set(exe_source test_coupling.F90)
add_executable(test_coupling ${exe_source})
target_include_directories(test_coupling PRIVATE ${PROJECT_BINARY_DIR}/src/f90-mod)
target_link_libraries(test_coupling PRIVATE rdycore_f90)
foreach(np 1 2) # test on 1 and 2 processes
  set(test_name test_coupling_np_${np})
  add_test(${test_name} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${CMAKE_CURRENT_BINARY_DIR}/test_coupling coupling.yaml)
endforeach()
