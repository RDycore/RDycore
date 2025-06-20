#----------------------------------------------------------
# Tests for the shallow water equations using a roe solver
#----------------------------------------------------------

find_program(H5DIFF h5diff REQUIRED
             PATHS ${PETSC_DIR}/${PETSC_ARCH}/bin)
set(rdycore_c_driver ${rdycore_driver})
set(mms_c_driver ${mms_driver})

# arguments for specific configurations ("basic" config has no options)
set(ceed_args -ceed /cpu/self)
set(preload_args -preload)

# Dirichlet BC test
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/ex2b_dirichlet_bc.yaml ${MESH_DIR}/planar_dam_10x5.msh
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
foreach(np 1 2) # test on 1, 2 processes
  foreach(config basic ceed)
    foreach(lang ${supported_languages})
      add_test(swe_roe_ex2b_dirichlet_bc_${lang}_np_${np}_${config} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${rdycore_${lang}_driver} ex2b_dirichlet_bc.yaml ${${config}_args})
    endforeach()
  endforeach()
endforeach()


# Additional tests for CGNS output
foreach(np 1 2) # test on 1, 2 processes
  set(cgns_output_args -ts_monitor_solution cgns:output/ex2b-%d.cgns -viewer_cgns_batch_size 20 -ts_monitor_solution_interval 100)
  foreach(config basic ceed preload)
    foreach(lang ${supported_languages})
      add_test(swe_roe_ex2b_cgns_${lang}_np_${np}_${config} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${rdycore_${lang}_driver} ex2b.yaml ${cgns_output_args} ${${config}_args})
    endforeach()
  endforeach()
endforeach()

# Checkpoint/restart tests
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/ex2b.yaml ${MESH_DIR}/planar_dam_10x5.msh
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
foreach(np 1 2) # test on 1, 2 processes
  foreach(config basic ceed preload)
    foreach(lang ${supported_languages})
      foreach(chkpt_fmt binary)
        if (chkpt_fmt STREQUAL "binary")
          set(chkpt_sfx "bin")
        else()
          set(chkpt_sfx "h5")
        endif()

        # run the original simulation
        add_test(swe_roe_ex2b_${lang}_chkpt_${chkpt_fmt}_np_${np}_${config} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${rdycore_${lang}_driver} ex2b.yaml ${${config}_args} -checkpoint_format ${chkpt_fmt})

        # stash the original ex2b simulation's final HDF5 output
        add_test(swe_roe_ex2b_reference_${lang}_chkpt_${chkpt_fmt}_np_${np}_${config} mv output/ex2b-0.h5 ex2b-ref.h5)
        set_tests_properties(swe_roe_ex2b_reference_${lang}_chkpt_${chkpt_fmt}_np_${np}_${config}
          PROPERTIES DEPENDS "swe_roe_ex2b_${lang}_chkpt_${chkpt_fmt}_np_${np}_${config}")

        # restart from timestep 100
        add_test(swe_roe_ex2b_restart_${lang}_chkpt_${chkpt_fmt}_np_${np}_${config} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${rdycore_${lang}_driver} ex2b.yaml ${${config}_args} -restart checkpoints/ex2b.rdycore.r.0100.${chkpt_sfx})
        set_tests_properties(swe_roe_ex2b_restart_${lang}_chkpt_${chkpt_fmt}_np_${np}_${config}
          PROPERTIES DEPENDS "swe_roe_ex2b_reference_${lang}_chkpt_${chkpt_fmt}_np_${np}_${config}")

        # compare restarted results with the reference output we stashed
        add_test(swe_roe_ex2b_compare_${lang}_chkpt_${chkpt_fmt}_np_${np}_${config} ${H5DIFF} -d 1e-10 ex2b-ref.h5 output/ex2b-0.h5 "/900 4.500000E-03 hr")
        set_tests_properties(swe_roe_ex2b_compare_${lang}_chkpt_${chkpt_fmt}_np_${np}_${config}
          PROPERTIES DEPENDS "swe_roe_ex2b_restart_${lang}_chkpt_${chkpt_fmt}_np_${np}_${config}")
      endforeach()
    endforeach()
  endforeach()
endforeach()

# ex2b with ICs specified by a file
file(COPY
     ${CMAKE_CURRENT_SOURCE_DIR}/ex2b_ic_file.yaml
     ${MESH_DIR}/DamBreak_grid5x10.exo
     ${CONDITION_DIR}/DamBreak_grid5x10_wetdownstream.ic.${PETSC_ID_TYPE}.bin
     ${MATERIALS_DIR}/manning_grid5x10.${PETSC_ID_TYPE}.bin
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
foreach(np 1 2) # test on 1, 2 processes
  foreach(config basic ceed preload)
    foreach(lang ${supported_languages})
      add_test(swe_roe_ex2b_ic_file_${lang}_np_${np}_${config} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${rdycore_${lang}_driver} ex2b_ic_file.yaml ${${config}_args})
    endforeach()
  endforeach()
endforeach()

# ex2b ensemble test
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/ex2b-ensemble.yaml
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
foreach(lang ${supported_languages})
  foreach(config basic ceed preload)
    add_test(swe_ensemble_${lang}_${config} ${MPIEXEC} ${MPIEXEC_FLAGS} -n 2 ${rdycore_${lang}_driver} ex2b-ensemble.yaml ${${config}_args})
  endforeach()
endforeach()

# four_mounds_60x24
file (COPY
      ${CMAKE_CURRENT_SOURCE_DIR}/four_mounds_60x24.yaml
      ${MESH_DIR}/four_mounds_60x24.exo
      ${CONDITION_DIR}/four_mounds_60x24.ic.${PETSC_ID_TYPE}.bin
      ${MATERIALS_DIR}/manning_four_mound_60x24.${PETSC_ID_TYPE}.bin
      DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
foreach(np 1 2)
  foreach(config basic ceed preload)
    foreach(lang ${supported_languages})
      add_test(swe_roe_four_mounds_${lang}_np_${np}_${config} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${rdycore_${lang}_driver} four_mounds_60x24.yaml ${${config}_args})
    endforeach()
  endforeach()
endforeach()

# houston-harvey test with spatially homogeneous rainfall
file (COPY
      ${CMAKE_CURRENT_SOURCE_DIR}/Houston1km.DirichletBC.yaml
      ${MESH_DIR}/Houston1km_with_z.exo
      ${CONDITION_DIR}/Houston1km.ic.${PETSC_ID_TYPE}.bin
      ${CONDITION_DIR}/Houston1km.bc.${PETSC_ID_TYPE}.bin
      ${CONDITION_DIR}/Houston1km.rain.${PETSC_ID_TYPE}.bin
      DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
foreach(np 1 2)
  foreach(config basic ceed)
    foreach(lang ${supported_languages})
      add_test(houston_${lang}_np_${np}_${config} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${rdycore_${lang}_driver} Houston1km.DirichletBC.yaml -homogeneous_rain_file Houston1km.rain.${PETSC_ID_TYPE}.bin -homogeneous_bc_file Houston1km.bc.${PETSC_ID_TYPE}.bin -temporally_interpolate_bc ${${config}_args})
    endforeach()
  endforeach()
endforeach()

# houston-harvey test with spatially homogeneous rainfall with adaptive timestepping
file (COPY
      ${CMAKE_CURRENT_SOURCE_DIR}/Houston1km.DirichletBC.adaptive_timestep.yaml
      ${MESH_DIR}/Houston1km_with_z.exo
      ${CONDITION_DIR}/Houston1km.ic.${PETSC_ID_TYPE}.bin
      ${CONDITION_DIR}/Houston1km.bc.${PETSC_ID_TYPE}.bin
      ${CONDITION_DIR}/Houston1km.rain.${PETSC_ID_TYPE}.bin
      DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
foreach(np 1 2)
  foreach(config basic)
    foreach(lang ${supported_languages})
      add_test(houston_${lang}_np_${np}_${config} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${rdycore_${lang}_driver} Houston1km.DirichletBC.adaptive_timestep.yaml -homogeneous_rain_file Houston1km.rain.${PETSC_ID_TYPE}.bin -homogeneous_bc_file Houston1km.bc.${PETSC_ID_TYPE}.bin -temporally_interpolate_bc ${${config}_args})
    endforeach()
  endforeach()
endforeach()

# houston-harvey test with spatially heterogeneous rainfall
file (COPY
      ${CONDITION_DIR}/2017-08-26:00-00.${PETSC_ID_TYPE}.bin
      ${CONDITION_DIR}/2017-08-26:01-00.${PETSC_ID_TYPE}.bin
      DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
foreach(np 1)
  foreach(config basic ceed)
    foreach(lang ${supported_languages})
      add_test(houston_hetero_${lang}_np_${np}_${config} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${rdycore_${lang}_driver} Houston1km.DirichletBC.yaml -raster_rain_start_date 2017,8,26,0,0 -raster_rain_dir ./ -raster_rain_write_map_file houston_raster.${PETSC_ID_TYPE}.map ./ ${${config}_args})
    endforeach()
  endforeach()
endforeach()
foreach(np 2)
  foreach(config basic ceed)
    add_test(houston_hetero_c_np_${np}_${config} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${rdycore_driver} Houston1km.DirichletBC.yaml -raster_rain_start_date 2017,8,26,0,0 -raster_rain_dir ./ -raster_rain_read_map_file houston_raster.${PETSC_ID_TYPE}.map ./ ${${config}_args})
  endforeach()
endforeach()

# mms driver (C and Fortran)
file(COPY
     ${CMAKE_CURRENT_SOURCE_DIR}/mms_single_run.yaml
     ${CMAKE_CURRENT_SOURCE_DIR}/mms_conv_study.yaml
     ${MESH_DIR}/mms_triangles_dx1.exo
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
foreach(np 1 2) # test on 1, 2 processes
  foreach(config basic ceed)
    foreach(lang ${supported_languages})
      add_test(swe_mms_single_run_${lang}_np_${np}_${config} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${mms_${lang}_driver} mms_single_run.yaml ${${config}_args})
      add_test(swe_mms_conv_study_${lang}_np_${np}_${config} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${mms_${lang}_driver} mms_conv_study.yaml ${${config}_args})
    endforeach()
  endforeach()
endforeach()

# Exodus II mesh with mixed element types (quads and triangles), IC specified by a binary file, and material property specified by a binary file
file(COPY
     ${CMAKE_CURRENT_SOURCE_DIR}/mixed_elements_ic_file.yaml
     ${MESH_DIR}/DamBreak_grid5x10_mixed_elements.exo
     ${CONDITION_DIR}/DamBreak_grid5x10_mixed_elements_wetdownstream.ic.${PETSC_ID_TYPE}.bin
     ${MATERIALS_DIR}/manning_grid5x10_mixed_elements.${PETSC_ID_TYPE}.bin
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
foreach(np 1 2) # test on 1 processes
  foreach(config basic ceed preload)
    foreach(lang ${supported_languages})
      add_test(swe_roe_mixed_elements_ic_file_${lang}_np_${np}_${config} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${rdycore_${lang}_driver} mixed_elements_ic_file.yaml ${${config}_args})
    endforeach()
  endforeach()
endforeach()

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/simons-bc-${PETSC_ID_TYPE})
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/simons-ss-${PETSC_ID_TYPE})
file(COPY
     ${CMAKE_CURRENT_SOURCE_DIR}/Simons.OceanDirichletBC.yaml
     ${MESH_DIR}/simons.exo
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY
      ${CONDITION_DIR}/simons-bc/2011-08-27:14-00.${PETSC_ID_TYPE}.bin
      ${CONDITION_DIR}/simons-bc/simons_bc_x_y.${PETSC_ID_TYPE}.bin
      DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/simons-bc-${PETSC_ID_TYPE})
file(COPY
      ${CONDITION_DIR}/simons-ss/2011-08-27:14-00.${PETSC_ID_TYPE}.bin
      ${CONDITION_DIR}/simons-ss/simons_forcing_x_y.${PETSC_ID_TYPE}.bin
      DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/simons-ss-${PETSC_ID_TYPE})

foreach(np 1) # test on 1 processes
  foreach(config basic)
    foreach(lang ${supported_languages})
      add_test(simons_write_map_np_${np}_${config}_${lang} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${rdycore_${lang}_driver} Simons.OceanDirichletBC.yaml -unstructured_bc_dir simons-bc-${PETSC_ID_TYPE} -unstructured_bc_start_date 2011,08,27,14,0 -unstructured_bc_mesh_file simons-bc-${PETSC_ID_TYPE}/simons_bc_x_y.${PETSC_ID_TYPE}.bin -unstructured_rain_dir simons-ss-${PETSC_ID_TYPE} -unstructured_rain_start_date 2011,08,27,14,0 -unstructured_rain_mesh_file simons-ss-${PETSC_ID_TYPE}/simons_forcing_x_y.${PETSC_ID_TYPE}.bin -unstructured_bc_write_map_for_debugging -unstructured_rain_write_map_file simons_unstructured_map.${PETSC_ID_TYPE}.bin ${${config}_args})
      add_test(simons_read_map_np_${np}_${config}_${lang} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${rdycore_${lang}_driver} Simons.OceanDirichletBC.yaml -unstructured_bc_dir simons-bc-${PETSC_ID_TYPE} -unstructured_bc_start_date 2011,08,27,14,0 -unstructured_bc_mesh_file simons-bc-${PETSC_ID_TYPE}/simons_bc_x_y.${PETSC_ID_TYPE}.bin -unstructured_rain_dir simons-ss-${PETSC_ID_TYPE} -unstructured_rain_start_date 2011,08,27,14,0 -unstructured_rain_mesh_file simons-ss-${PETSC_ID_TYPE}/simons_forcing_x_y.${PETSC_ID_TYPE}.bin -unstructured_bc_write_map_for_debugging -unstructured_rain_read_map_file simons_unstructured_map.${PETSC_ID_TYPE}.bin ${${config}_args})
    endforeach()
  endforeach()
endforeach()

# Test for specifying multi-homogeneous boundary and source-sink files
file(COPY
     ${CMAKE_CURRENT_SOURCE_DIR}/quad_tri_mesh.yaml
     ${MESH_DIR}/quad_tri_mesh.exo
     ${CONDITION_DIR}/Houston1km.bc.${PETSC_ID_TYPE}.bin
     ${CONDITION_DIR}/Houston1km.rain.${PETSC_ID_TYPE}.bin
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
foreach(np 1) # test on 1 processes
  foreach(config basic ceed preload)
    foreach(lang ${supported_languages})
      add_test(swe_roe_quad_tri_mesh_${lang}_np_${np}_${config} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${rdycore_${lang}_driver} quad_tri_mesh.yaml -homogeneous_rain_files Houston1km.rain.${PETSC_ID_TYPE}.bin,Houston1km.rain.${PETSC_ID_TYPE}.bin -homogeneous_rain_region_ids 1,3 -homogeneous_bc_files Houston1km.bc.${PETSC_ID_TYPE}.bin,Houston1km.bc.${PETSC_ID_TYPE}.bin,Houston1km.bc.${PETSC_ID_TYPE}.bin -homogeneous_bc_region_ids   1,2,4 ${${config}_args})
    endforeach()
  endforeach()
endforeach()

