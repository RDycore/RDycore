# This function extracts a variable named varname from the petscvariables file,
# storing its value in the variable var. The resulting value can contain spaces,
# and must be interpreted properly by the caller.
# Pass REQUIRED as an optional third argument to issue a fatal error instead of a warning when the
# variable is not found.
function(extract_petsc_variable varname var)
  # read petscvariables
  file(READ "${PETSC_DIR}/${PETSC_ARCH}/lib/petsc/conf/petscvariables" petscvariables)

  # check for optional REQUIRED argument
  set(required FALSE)
  if (${ARGC} GREATER 2 AND "${ARGV2}" STREQUAL "REQUIRED")
    set(required TRUE)
  endif()

  # find where the variable is set and remove everything preceding its value
  string(FIND ${petscvariables} "\n${varname} = " start)
  if (${start} EQUAL -1)
    if (REQUIRED)
      message(FATAL_ERROR "Could not extract ${varname} from PETSc. Please set ${varname} using -D${varname}=...")
    else()
      message(WARNING "Could not extract ${varname} from PETSc petscvariables (skipping).")
      set(${var} "" PARENT_SCOPE)
      return()
    endif()
  endif()
  string(LENGTH ${varname} varname_length)
  math(EXPR start "${start} + ${varname_length} + 4")
  string(SUBSTRING ${petscvariables} ${start} -1 petsc_var)

  # truncate the value at the first newline encountered
  string(FIND ${petsc_var} "\n" newline)
  string(SUBSTRING ${petsc_var} 0 ${newline} petsc_var)

  message(STATUS "Extracted PETSc variable: ${varname} = ${petsc_var}")
  set(${var} ${petsc_var} PARENT_SCOPE)
endfunction()
