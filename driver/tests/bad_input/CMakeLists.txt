#--------------------------------------------------------
# Tests that should fail because of bad input parameters
#--------------------------------------------------------

# non-existent boundary test
file (COPY
      ${CMAKE_CURRENT_SOURCE_DIR}/nonexistent_boundary.yaml
      ${MESH_DIR}/Houston1km_with_z.exo
      ${CONDITION_DIR}/Houston1km.ic.${PETSC_ID_TYPE}.bin
      DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_test(nonexistent_boundary ${PETSC_MPIEXEC} ${PETSC_MPIEXEC_FLAGS} -n 1 ${rdycore_driver} nonexistent_boundary.yaml)
set_property(TEST nonexistent_boundary PROPERTY WILL_FAIL TRUE)
