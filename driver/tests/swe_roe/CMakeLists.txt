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

# Additional tests for CGNS output
foreach(np 1 2) # test on 1, 2 processes
  set(cgns_output_args -ts_monitor_solution cgns:output/ex2b-%d.cgns -viewer_cgns_batch_size 20 -ts_monitor_solution_interval 100)
  add_test(swe_roe_ex2b_cgns_c_np_${np} ${PETSC_MPIEXEC} ${PETSC_MPIEXEC_FLAGS} -n ${np} ${rdycore_driver} ex2b.yaml ${cgns_output_args})
  add_test(swe_roe_ex2b_cgns_c_np_${np}_ceed ${PETSC_MPIEXEC} ${PETSC_MPIEXEC_FLAGS} -n ${np} ${rdycore_driver} ex2b.yaml ${cgns_output_args} -ceed /cpu/self)
  add_test(swe_roe_ex2b_cgns_f90_np_${np} ${PETSC_MPIEXEC} ${PETSC_MPIEXEC_FLAGS} -n ${np} ${rdycore_f90_driver} ex2b.yaml ${cgns_output_args})
endforeach()

# ex2b with ICs specified by a file
file(COPY
     ${CMAKE_CURRENT_SOURCE_DIR}/ex2b_ic_file.yaml
     ${MESH_DIR}/DamBreak_grid5x10.exo
     ${IC_DIR}/DamBreak_grid5x10_wetdownstream.IC
     ${MATERIALS_DIR}/manning_grid5x10.bin
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
foreach(np 1 2) # test on 1, 2 processes
  add_test(swe_roe_ex2b_ic_file_c_np_${np} ${PETSC_MPIEXEC} ${PETSC_MPIEXEC_FLAGS} -n ${np} ${rdycore_driver} ex2b_ic_file.yaml)
  add_test(swe_roe_ex2b_ic_file_f90_np_${np} ${PETSC_MPIEXEC} ${PETSC_MPIEXEC_FLAGS} -n ${np} ${rdycore_f90_driver} ex2b_ic_file.yaml)
  add_test(swe_roe_ex2b_ic_file_f90_np_${np}_ceed ${PETSC_MPIEXEC} ${PETSC_MPIEXEC_FLAGS} -n ${np} ${rdycore_f90_driver} ex2b_ic_file.yaml -ceed /cpu/self)
endforeach()
