# This directory includes GMsh mesh geometries that can generate meshes for
# our tests.

if (ENABLE_TESTS)
  # If we enable tests, we'll need gmsh to generate mesh data.
  find_program(GMSH_EXE gmsh
               DOC "Gmsh mesh generator")
  if (GMSH_EXE MATCHES "NOTFOUND")
    message(FATAL_ERROR "\
Gmsh is required for driver tests but was not found! Please install it using \
your favorite package manager or visit https://gmsh.info for more \
information.")
  else()
    message(STATUS "Found Gmsh: ${GMSH_EXE}")
  endif()

  # Generate meshes from geometry files.
  include(generate_msh)
  foreach(geo planar_dam_10x5.geo planar_dam.geo)
    generate_msh(${geo})
  endforeach()

endif()
