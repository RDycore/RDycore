# This function generates a (2D) .msh file from a .geo file using Gmsh.
function(generate_msh geo_file)
  string(REPLACE ".geo" ".msh" msh_file ${geo_file})
  string(REPLACE "${PROJECT_SOURCE_DIR}/" "" source_dir ${CMAKE_CURRENT_SOURCE_DIR})
  string(REPLACE "${PROJECT_BINARY_DIR}/" "" binary_dir ${CMAKE_CURRENT_BINARY_DIR})
  message(STATUS "Generating ${binary_dir}/${msh_file} from share/meshes/${geo_file}")
  execute_process(COMMAND ${GMSH_EXE} -2 -o ${MESH_DIR}/${msh_file} ${geo_file}
                  WORKING_DIRECTORY ${MESH_GEOMETRY_DIR}
                  OUTPUT_QUIET)
endfunction()


