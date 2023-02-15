#----------------------------------------------------------
# Tests for the shallow water equations using a roe solver
#----------------------------------------------------------

# ex2b
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/ex2b.yaml ${MESH_DIR}/planar_dam_10x5.msh
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
foreach(np 1 2) # test on 1, 2 processes
  add_test(swe_roe_ex2b_c_np_${np} ${PETSC_MPIEXEC} ${PETSC_MPIEXEC_FLAGS} -n ${np} ${rdycore_driver} ex2b.yaml)
  add_test(swe_roe_ex2b_f90_np_${np} ${PETSC_MPIEXEC} ${PETSC_MPIEXEC_FLAGS} -n ${np} ${rdycore_f90_driver} ex2b.yaml)
endforeach()
