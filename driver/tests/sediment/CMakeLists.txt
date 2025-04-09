find_program(H5DIFF h5diff REQUIRED
             PATHS ${PETSC_DIR}/${PETSC_ARCH}/bin)
set(rdycore_c_driver ${rdycore_driver})
set(mms_c_driver ${mms_driver})

# arguments for specific configurations ("basic" config has no options)
set(ceed_args -ceed /cpu/self)
set(preload_args -preload)

file(COPY
     ${CMAKE_CURRENT_SOURCE_DIR}/sediment_mms_conv_study.yaml
     ${MESH_DIR}/mms_triangles_dx1.exo
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
foreach(np 2) # test on 1, 2 processes
  foreach(config basic ceed)
    foreach(lang ${supported_languages})
      add_test(sediment_mms_conv_study_${lang}_np_${np}_${config} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${mms_${lang}_driver} sediment_mms_conv_study.yaml ${${config}_args})
    endforeach()
  endforeach()
endforeach()

file(COPY
     ${CMAKE_CURRENT_SOURCE_DIR}/sediment.yaml
     ${MESH_DIR}/DamBreak_grid5x10.exo
     ${CONDITION_DIR}/DamBreak_grid5x10_wetdownstream.ic.${PETSC_ID_TYPE}.bin
     ${MATERIALS_DIR}/manning_grid5x10.${PETSC_ID_TYPE}.bin
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
foreach(np 2) # test on 1, 2 processes
  foreach(config basic ceed)
    foreach(lang ${supported_languages})
      add_test(sediment_${lang}_np_${np}_${config} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${rdycore_${lang}_driver} sediment.yaml ${${config}_args})
    endforeach()
  endforeach()
endforeach()

file(COPY
     ${CMAKE_CURRENT_SOURCE_DIR}/sediment_ic_file.yaml
     ${MESH_DIR}/DamBreak_grid5x10.exo
     ${CONDITION_DIR}/DamBreak_grid5x10_wetdownstream.ic.${PETSC_ID_TYPE}.bin
     ${CONDITION_DIR}/DamBreak_grid5x10_wetdownstream.sediment_ic.${PETSC_ID_TYPE}.bin
     ${MATERIALS_DIR}/manning_grid5x10.${PETSC_ID_TYPE}.bin
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
foreach(np 2) # test on 1, 2 processes
  foreach(config basic ceed)
    foreach(lang c)
      add_test(sediment_ic_file${lang}_np_${np}_${config} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${rdycore_${lang}_driver} sediment_ic_file.yaml ${${config}_args})
    endforeach()
  endforeach()
endforeach()
