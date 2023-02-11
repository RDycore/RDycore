# Here we define tests for the rdycore driver.
include(generate_msh)

# location of built driver executables
set(rdycore_driver ${CMAKE_CURRENT_BINARY_DIR}/../rdycore)
set(rdycore_f90_driver ${CMAKE_CURRENT_BINARY_DIR}/../rdycore_f90)

# shallow water equations with roe riemann solver
add_subdirectory(swe_roe)
