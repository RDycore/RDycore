# Here we define tests for the rdycore driver.

# location of built driver executable
set(rdycore_driver ${CMAKE_CURRENT_BINARY_DIR}/../rdycore)

# shallow water equations with roe riemann solver
add_subdirectory(swe_roe)
