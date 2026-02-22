# This function extracts MPI include directories from RDycore's PETSc
# configuration and returns them in 'directories'.
# We do this to enable clangd to find mpi.h in more situations.
# It tries MPICXX_INCLUDES first (PETSc >= 3.24), then falls back to
# MPICC_SHOW (older PETSc versions).
include(extract_petsc_variable)
function(extract_mpi_include_directories directories)
  # Try MPICXX_INCLUDES first (available in PETSc 3.24+)
  extract_petsc_variable("MPICXX_INCLUDES" mpicxx_includes QUIET)
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
  extract_petsc_variable("MPICC_SHOW" mpicc_show QUIET)
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

  # Neither variable found
  message(WARNING "Could not find MPI include directories from PETSc (neither MPICXX_INCLUDES nor MPICC_SHOW available).")
  set(${directories} "" PARENT_SCOPE)
endfunction()
