#------------------------------------------------------
# MMS tests for the heat equation operator split
#------------------------------------------------------

set(mms_c_driver ${mms_driver})

file(COPY
     ${CMAKE_CURRENT_SOURCE_DIR}/heat_ts_mms.yaml
     ${CMAKE_CURRENT_SOURCE_DIR}/heat_mms_conv_study.yaml
     ${MESH_DIR}/mms_triangles_dx1.exo
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Case 1: TS isolation — single run, no convergence study
# (temporal convergence is assessed by running at multiple dt values manually)
foreach(np 1 2)
  foreach(lang ${supported_languages})
    add_test(heat_ts_mms_${lang}_np_${np} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${mms_${lang}_driver} heat_ts_mms.yaml)
  endforeach()
endforeach()

# Case 2: SWE + heat spatial convergence study
foreach(np 1 2)
  foreach(lang ${supported_languages})
    add_test(heat_mms_conv_study_${lang}_np_${np} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${mms_${lang}_driver} heat_mms_conv_study.yaml)
  endforeach()
endforeach()
