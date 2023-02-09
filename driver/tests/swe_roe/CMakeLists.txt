#----------------------------------------------------------
# Tests for the shallow water equations using a roe solver
#----------------------------------------------------------

# ex1
#foreach(np 1 2) # test on 1, 2 processes
#  add_test(swe_roe_ex1_np_${np} ${PETSC_MPIEXEC} -n ${np} ${rdycore_driver} ex1.yaml)
#endforeach()

# ex2b
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/ex2b.yaml
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
)
execute_process(COMMAND ${GMSH_EXE} -2 -o ${CMAKE_CURRENT_BINARY_DIR}/planar_dam.msh planar_dam.geo
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
foreach(np 1 2) # test on 1, 2 processes
  add_test(swe_roe_ex2b_np_${np} ${PETSC_MPIEXEC} -n ${np} ${rdycore_driver} ex2b.yaml)
endforeach()
