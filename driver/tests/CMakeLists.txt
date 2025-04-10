# Here we define tests for the rdycore driver.

# location of built driver executables
set(rdycore_driver ${CMAKE_CURRENT_BINARY_DIR}/../rdycore)
set(mms_driver ${CMAKE_CURRENT_BINARY_DIR}/../rdycore_mms)
set(amr_driver ${CMAKE_CURRENT_BINARY_DIR}/../rdycore_amr)

add_test(c_driver_usage ${MPIEXEC} ${MPIEXEC_OPTIONS} -n 1 ${rdycore_driver})
set(supported_languages c)

if (ENABLE_FORTRAN)
  set(rdycore_f90_driver ${CMAKE_CURRENT_BINARY_DIR}/../rdycore_f90)
  set(mms_f90_driver ${CMAKE_CURRENT_BINARY_DIR}/../rdycore_mms_f90)
  add_test(f90_driver_usage ${MPIEXEC} ${MPIEXEC_OPTIONS} -n 1 ${rdycore_f90_driver})

  list(APPEND supported_languages f90)
endif()

# sediment dynamics
add_subdirectory(amr)

# shallow water equations with roe riemann solver
add_subdirectory(swe_roe)

# sediment dynamics
add_subdirectory(sediment)

# regression tests for bad input
add_subdirectory(bad_input)
