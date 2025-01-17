find_program(H5DIFF h5diff REQUIRED
             PATHS ${PETSC_DIR}/${PETSC_ARCH}/bin)
set(rdycore_c_driver ${rdycore_driver})
set(mms_c_driver ${mms_driver})

# arguments for specific configurations ("basic" config has no options)
set(ceed_args -ceed /cpu/self)
set(preload_args -preload)

# mms driver (C and Fortran)
file(COPY
     ${CMAKE_CURRENT_SOURCE_DIR}/sd_mms_conv_study.yaml
     ${MESH_DIR}/mms_triangles_dx1.exo
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
foreach(np 1) # test on 1, 2 processes
  foreach(config basic)
    foreach(lang c f90)
      add_test(mms_conv_study_${lang}_np_${np}_${config} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${mms_${lang}_driver} sd_mms_conv_study.yaml ${${config}_args})
    endforeach()
  endforeach()
endforeach()
