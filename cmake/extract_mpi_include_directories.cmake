# This function, which accepts no arguments, extracts the MPICC_SHOW variable from RDycore's PETSc
# configuration and parses it into a list of include directories, which it returns in
# directories. We do this to enable clangd to find mpi.h in more situations.
include(extract_petsc_variable)
function(extract_mpi_include_directories directories)
  extract_petsc_variable("MPICC_SHOW" mpicc_show)
  string(REPLACE " " ";" candidates ${mpicc_show})
  foreach(candidate ${candidates})
    string(FIND ${candidate} "-I" flag_pos)
    if (NOT ${flag_pos} EQUAL -1)
      string(SUBSTRING ${candidate} 2 -1 mpi_dir)
      list(APPEND mpi_dirs ${mpi_dir})
    endif()
  endforeach()
  set(${directories} ${mpi_dirs} PARENT_SCOPE)
endfunction()
