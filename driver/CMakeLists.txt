# C driver
add_executable(rdycore_exe
  main.c
)
set_target_properties(rdycore_exe PROPERTIES OUTPUT_NAME rdycore)
target_link_libraries(rdycore_exe PRIVATE ${RDYCORE_LIBRARIES})

# MMS C driver
add_executable(rdycore_mms
  mms.c
)
set_target_properties(rdycore_mms PROPERTIES OUTPUT_NAME rdycore_mms)
target_link_libraries(rdycore_mms PRIVATE ${RDYCORE_LIBRARIES})

# AMR C driver
add_executable(rdycore_amr
  amr.c
)
set_target_properties(rdycore_amr PROPERTIES OUTPUT_NAME rdycore_amr)
target_link_libraries(rdycore_amr PRIVATE ${RDYCORE_LIBRARIES})

# MMS Fortran driver
add_executable(rdycore_mms_f90
  mms.F90
)
set_target_properties(rdycore_mms_f90 PROPERTIES OUTPUT_NAME rdycore_mms_f90)
target_include_directories(rdycore_mms_f90 PRIVATE ${PROJECT_BINARY_DIR}/src/f90-mod)
target_link_libraries(rdycore_mms_f90 PRIVATE ${RDYCORE_LIBRARIES})

# Fortran driver
add_executable(rdycore_f90_exe
  main.F90
)
set_target_properties(rdycore_f90_exe PROPERTIES OUTPUT_NAME rdycore_f90)
target_include_directories(rdycore_f90_exe PRIVATE ${PROJECT_BINARY_DIR}/src/f90-mod)
target_link_libraries(rdycore_f90_exe PRIVATE ${RDYCORE_LIBRARIES})
install(
  TARGETS rdycore_exe rdycore_mms rdycore_mms_f90 rdycore_f90_exe
  DESTINATION bin
)

# Tests for the driver.
if (ENABLE_TESTS)
  add_subdirectory(tests)
endif()
