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

  # Fall back to the Cray Programming Environment's MPICH_DIR (set by the
  # cray-mpich module), since Cray's compiler wrappers bake MPI in directly
  # and PETSc's petscvariables therefore has no MPI include information.
  if (DEFINED ENV{MPICH_DIR} AND EXISTS "$ENV{MPICH_DIR}/include")
    set(${directories} "$ENV{MPICH_DIR}/include" PARENT_SCOPE)
    return()
  endif()

  # None of the above worked. This only affects IDE/language-server tooling
  # (compile_commands.json), not the actual build, so don't fail the configure.
  message(WARNING "Could not find MPI include directories from PETSc or the environment (skipping).")
  set(${directories} "" PARENT_SCOPE)
endfunction()
