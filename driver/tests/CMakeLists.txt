# Here we define tests for the rdycore driver.

# We need gmsh locally to generate some test meshes.
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

# This function generates a (2D) .msh file from a .geo file using Gmsh.
function(generate_msh_file geo_file)
  string(REPLACE ".geo" ".msh" msh_file ${geo_file})
  string(REPLACE "${PROJECT_SOURCE_DIR}/" "" source_dir ${CMAKE_CURRENT_SOURCE_DIR})
  string(REPLACE "${PROJECT_BINARY_DIR}/" "" binary_dir ${CMAKE_CURRENT_BINARY_DIR})
  message(STATUS "Generating ${binary_dir}/${msh_file} from ${source_dir}/${geo_file}")
  execute_process(COMMAND ${GMSH_EXE} -2 -o ${CMAKE_CURRENT_BINARY_DIR}/${msh_file} ${geo_file}
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  OUTPUT_QUIET)
endfunction()

# location of built driver executable
set(rdycore_driver ${CMAKE_CURRENT_BINARY_DIR}/../rdycore)

# shallow water equations with roe riemann solver
add_subdirectory(swe_roe)
