# Here we define tests for the rdycore driver.

# location of built driver executables
set(rdycore_driver ${CMAKE_CURRENT_BINARY_DIR}/../rdycore)
set(rdycore_f90_driver ${CMAKE_CURRENT_BINARY_DIR}/../rdycore_f90)

# Run the drivers with no arguments to test usage.
foreach(driver ${rdycore_driver} ${rdycore_f90_driver})
  add_test(c_driver_usage   ${PETSC_MPIEXEC} ${PETSC_MPIEXEC_FLAGS} -n 1 ${driver})
  add_test(f90_driver_usage ${PETSC_MPIEXEC} ${PETSC_MPIEXEC_FLAGS} -n 1 ${driver})
endforeach()

# shallow water equations with roe riemann solver
add_subdirectory(swe_roe)
