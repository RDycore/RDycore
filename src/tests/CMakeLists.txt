# Here we manually define unit tests for the main RDycore library. We can
# discuss ways of setting up tests more easily and automatically.

add_executable(test_rdyinit test_rdyinit.c)
target_link_libraries(test_rdyinit ${RDYCORE_LIBRARIES} cmocka)
foreach(np 1 2 4) # test on 1, 2, and 4 processes
  add_test(test_rdyinit_np_${np} ${PETSC_MPIEXEC} -n ${np} ${CMAKE_CURRENT_BINARY_DIR}/test_rdyinit)
endforeach()
