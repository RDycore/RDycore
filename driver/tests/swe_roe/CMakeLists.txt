#----------------------------------------------------------
# Tests for the shallow water equations using a roe solver
#----------------------------------------------------------

# ex1
foreach(np 1 2) # test on 1, 2 processes
  add_test(swe_roe_ex1_np_${np} ${PETSC_MPIEXEC} -n ${np} ${rdycore_driver} ex1.yaml)
endforeach()

# ex2
foreach(np 1 2) # test on 1, 2 processes
  add_test(swe_roe_ex2_np_${np} ${PETSC_MPIEXEC} -n ${np} ${rdycore_driver} ex2.yaml)
endforeach()

