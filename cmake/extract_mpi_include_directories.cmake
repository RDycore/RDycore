# This function extracts MPI include directories from RDycore's PETSc
# configuration and returns them in 'directories'.
# We do this to enable clangd to find mpi.h in more situations.
include(extract_petsc_variable)
function(extract_mpi_include_directories directories)
  # Try MPICXX_INCLUDES first (PETSc 3.24+)
  extract_petsc_variable("MPICXX_INCLUDES" mpicxx_includes)
  if (NOT "${mpicxx_includes}" STREQUAL "")
    string(REPLACE " " ";" candidates ${mpicxx_includes})
    foreach(candidate ${candidates})
      string(FIND ${candidate} "-I" flag_pos)
      if (NOT ${flag_pos} EQUAL -1)
        string(SUBSTRING ${candidate} 2 -1 mpi_dir)
        list(APPEND mpi_dirs ${mpi_dir})
      endif()
    endforeach()
    set(${directories} ${mpi_dirs} PARENT_SCOPE)
    return()
  endif()

  # Fall back to MPICC_SHOW (older PETSc versions)
  extract_petsc_variable("MPICC_SHOW" mpicc_show)
  if (NOT "${mpicc_show}" STREQUAL "")
    string(REPLACE " " ";" candidates ${mpicc_show})
    foreach(candidate ${candidates})
      string(FIND ${candidate} "-I" flag_pos)
      if (NOT ${flag_pos} EQUAL -1)
        string(SUBSTRING ${candidate} 2 -1 mpi_dir)
        list(APPEND mpi_dirs ${mpi_dir})
      endif()
    endforeach()
    set(${directories} ${mpi_dirs} PARENT_SCOPE)
    return()
  endif()

  # Neither variable is set when the compiler supplies MPI itself, as the Cray
  # wrappers on Perlmutter and Frontier do. There is nothing to add then, and
  # nothing is lost but an editor hint, so this is not an error.
  message(STATUS "PETSc reports no MPI include directories (the compiler supplies MPI); continuing without them")
  set(${directories} "" PARENT_SCOPE)
endfunction()
