# C driver
add_executable(rdycore_exe
  main.c
)
set_target_properties(rdycore_exe PROPERTIES OUTPUT_NAME rdycore)
target_link_libraries(rdycore_exe PRIVATE rdycore_c)

# MMS C driver
add_executable(rdycore_mms
  mms.c
)
set_target_properties(rdycore_mms PROPERTIES OUTPUT_NAME rdycore_mms)
target_link_libraries(rdycore_mms PRIVATE rdycore_c)

# AMR C driver
add_executable(rdycore_amr
  amr.c
)
set_target_properties(rdycore_amr PROPERTIES OUTPUT_NAME rdycore_amr)
target_link_libraries(rdycore_amr PRIVATE rdycore_c)

# adjoint sensitivity driver
add_executable(rdycore_adjoint
  adjoint_test.c
)
set_target_properties(rdycore_adjoint PROPERTIES OUTPUT_NAME rdycore_adjoint)
target_link_libraries(rdycore_adjoint PRIVATE rdycore_c)

install(
  TARGETS rdycore_exe rdycore_mms rdycore_amr rdycore_adjoint
  DESTINATION bin
)

# Fortran driver
if (ENABLE_FORTRAN)
  add_executable(rdycore_f90_exe main.F90)
  set_target_properties(rdycore_f90_exe PROPERTIES OUTPUT_NAME rdycore_f90)
  target_include_directories(rdycore_f90_exe PRIVATE ${PROJECT_BINARY_DIR}/src/f90-mod)
  target_link_libraries(rdycore_f90_exe PRIVATE rdycore_f90)

  add_executable(rdycore_mms_f90 mms.F90)
  set_target_properties(rdycore_mms_f90 PROPERTIES OUTPUT_NAME rdycore_mms_f90)
  target_include_directories(rdycore_mms_f90 PRIVATE ${PROJECT_BINARY_DIR}/src/f90-mod)
  target_link_libraries(rdycore_mms_f90 PRIVATE rdycore_f90)

  install(
    TARGETS rdycore_mms_f90 rdycore_f90_exe
    DESTINATION bin
  )
endif()

# Tests for the driver.
if (ENABLE_TESTS)
  add_subdirectory(tests)
endif()
