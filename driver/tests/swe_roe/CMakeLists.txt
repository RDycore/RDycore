#----------------------------------------------------------
# Tests for the shallow water equations using a roe solver
#----------------------------------------------------------

# ex1
#foreach(np 1 2) # test on 1, 2 processes
#  add_test(swe_roe_ex1_np_${np} ${PETSC_MPIEXEC} -n ${np} ${rdycore_driver} ex1.yaml)
#endforeach()

# ex2b
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/ex2b.yaml
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
generate_msh(planar_dam.geo)
foreach(driver ${rdycore_driver} ${rdycore_f90_driver})
  foreach(np 1 2) # test on 1, 2 processes
    add_test(swe_roe_ex2b_np_${np} ${PETSC_MPIEXEC} -n ${np} ${driver} ex2b.yaml)
    add_test(swe_roe_ex2bf90_np_${np} ${PETSC_MPIEXEC} -n ${np} ${driver} ex2b.yaml)
  endforeach()
endforeach()
