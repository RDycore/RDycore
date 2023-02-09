# Here we define tests for the rdycore driver.

# We need gmsh locally to generate some test meshes.
find_program(GMSH_EXE gmsh
             DOC "Gmsh mesh generator")
if (GMSH_EXE STREQUAL "GMSH-NOTFOUND")
  message(FATAL_ERROR "gmsh is required for driver tests but was not found!")
endif()

# location of built driver executable
set(rdycore_driver ${CMAKE_CURRENT_BINARY_DIR}/../rdycore)

# shallow water equations with roe riemann solver
add_subdirectory(swe_roe)
