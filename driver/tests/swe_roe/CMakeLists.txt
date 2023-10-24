#----------------------------------------------------------
# Tests for the shallow water equations using a roe solver
#----------------------------------------------------------

# ex2b
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/ex2b.yaml ${MESH_DIR}/planar_dam_10x5.msh
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# arguments for specific configurations ("basic" config has no options)
set(ceed_args -ceed /cpu/self)
set(preload_args -preload)

foreach(np 1 2) # test on 1, 2 processes
  foreach(config basic ceed preload)
    add_test(swe_roe_ex2b_c_np_${np}_${config} ${PETSC_MPIEXEC} ${PETSC_MPIEXEC_FLAGS} -n ${np} ${rdycore_driver} ex2b.yaml ${${config}_args})
    add_test(swe_roe_ex2b_f90_np_${np}_${config} ${PETSC_MPIEXEC} ${PETSC_MPIEXEC_FLAGS} -n ${np} ${rdycore_f90_driver} ex2b.yaml ${${config}_args})
  endforeach()
endforeach()

# Dirichlet BC test
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/ex2b_dirichlet_bc.yaml ${MESH_DIR}/planar_dam_10x5.msh
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
foreach(np 1 2) # test on 1, 2 processes
  foreach(config basic ceed)
    add_test(swe_roe_ex2b_dirichlet_bc_c_np_${np}_${config} ${PETSC_MPIEXEC} ${PETSC_MPIEXEC_FLAGS} -n ${np} ${rdycore_driver} ex2b_dirichlet_bc.yaml ${${config}_args})
    add_test(swe_roe_ex2b_dirichlet_bc_f90_np_${np}_${config} ${PETSC_MPIEXEC} ${PETSC_MPIEXEC_FLAGS} -n ${np} ${rdycore_f90_driver} ex2b_dirichlet_bc.yaml ${${config}_args})
  endforeach()
endforeach()


# Additional tests for CGNS output
foreach(np 1 2) # test on 1, 2 processes
  set(cgns_output_args -ts_monitor_solution cgns:output/ex2b-%d.cgns -viewer_cgns_batch_size 20 -ts_monitor_solution_interval 100)
  foreach(config basic ceed preload)
    add_test(swe_roe_ex2b_cgns_c_np_${np}_${config} ${PETSC_MPIEXEC} ${PETSC_MPIEXEC_FLAGS} -n ${np} ${rdycore_driver} ex2b.yaml ${cgns_output_args} ${${config}_args})
    add_test(swe_roe_ex2b_cgns_f90_np_${np}_${config} ${PETSC_MPIEXEC} ${PETSC_MPIEXEC_FLAGS} -n ${np} ${rdycore_f90_driver} ex2b.yaml ${cgns_output_args} ${${config}_args})
  endforeach()
endforeach()

# ex2b with ICs specified by a file
file(COPY
     ${CMAKE_CURRENT_SOURCE_DIR}/ex2b_ic_file.yaml
     ${MESH_DIR}/DamBreak_grid5x10.exo
     ${CONDITION_DIR}/DamBreak_grid5x10_wetdownstream.IC
     ${MATERIALS_DIR}/manning_grid5x10.bin
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
foreach(np 1 2) # test on 1, 2 processes
  foreach(config basic ceed preload)
    add_test(swe_roe_ex2b_ic_file_c_np_${np}_${config} ${PETSC_MPIEXEC} ${PETSC_MPIEXEC_FLAGS} -n ${np} ${rdycore_driver} ex2b_ic_file.yaml ${${config}_args})
    add_test(swe_roe_ex2b_ic_file_f90_np_${np}_${config} ${PETSC_MPIEXEC} ${PETSC_MPIEXEC_FLAGS} -n ${np} ${rdycore_f90_driver} ex2b_ic_file.yaml ${${config}_args})
  endforeach()
endforeach()

# four_mounds_60x24
file (COPY
      ${CMAKE_CURRENT_SOURCE_DIR}/four_mounds_60x24.yaml
      ${MESH_DIR}/four_mounds_60x24.exo
      ${CONDITION_DIR}/four_mounds_60x24.ic
      ${MATERIALS_DIR}/manning_four_mound_60x24.bin
      DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
foreach(np 1 2)
  foreach(config basic ceed preload)
    add_test(swe_roe_four_mounds_c_np_${np}_${config} ${PETSC_MPIEXEC} ${PETSC_MPIEXEC_FLAGS} -n ${np} ${rdycore_driver} four_mounds_60x24.yaml ${${config}_args})
    add_test(swe_roe_four_mounds_f90_np_${np}_${config} ${PETSC_MPIEXEC} ${PETSC_MPIEXEC_FLAGS} -n ${np} ${rdycore_f90_driver} four_mounds_60x24.yaml ${${config}_args})
  endforeach()
endforeach()
