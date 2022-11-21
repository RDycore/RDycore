# Here we manually define unit tests for the main RDycore library. We can
# discuss ways of setting up tests more easily and automatically.

foreach(exe_name
        test_rdyinit
        test_rdyinitnoarguments)
  set(exe_source ${exe_name}.c)
  add_executable(${exe_name} ${exe_source})
  target_link_libraries(${exe_name} PRIVATE ${RDYCORE_LIBRARIES} cmocka)
  target_code_coverage(${exe_name} AUTO ALL)
  foreach(np 1 2 4) # test on 1, 2, and 4 processes
    set(test_name ${exe_name}_np_${np})
    add_test(${test_name} ${PETSC_MPIEXEC} -n ${np} ${CMAKE_CURRENT_BINARY_DIR}/${exe_name})
  endforeach()
endforeach()
