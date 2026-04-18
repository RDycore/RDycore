# Here we manually define unit tests for the main RDycore library. We can
# discuss ways of setting up tests more easily and automatically.

# We need a mesh for some of our unit tests.
file(COPY ${MESH_DIR}/planar_dam_10x5.msh
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Copy the particle integration test YAML into the build directory.
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/planar_dam_particles.yaml
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

foreach(exe_name
        test_rdyinit
        test_rdymesh
        test_yaml_input
        test_particle_swarm)
  set(exe_source ${exe_name}.c)
  add_executable(${exe_name} ${exe_source})
  target_link_libraries(${exe_name} PRIVATE rdycore_c cmocka)
  foreach(np 1 2) # test on 1 and 2 processes
    set(test_name ${exe_name}_np_${np})
    add_test(${test_name} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${CMAKE_CURRENT_BINARY_DIR}/${exe_name})
  endforeach()
endforeach()

# test_particle_advance requires -particles_per_cell 1 to enable the swarm.
add_executable(test_particle_advance test_particle_advance.c)
target_link_libraries(test_particle_advance PRIVATE rdycore_c cmocka)
foreach(np 1 2)
  add_test(test_particle_advance_np_${np}
    ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np}
    ${CMAKE_CURRENT_BINARY_DIR}/test_particle_advance
    -particles_per_cell 1)
endforeach()
