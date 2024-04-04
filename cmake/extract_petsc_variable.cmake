# This function extracts a variable named varname from the petscvariables file,
# storing its value in the variable var. The resulting value can contain spaces,
# and must be interpreted properly by the caller.
function(extract_petsc_variable varname var)
  # read petscvariables
  file(READ "${PETSC_DIR}/${PETSC_ARCH}/lib/petsc/conf/petscvariables" petscvariables)

  # find where the variable is set and remove everything preceding its value
  string(FIND ${petscvariables} "\n${varname} = " start)
  if (${start} EQUAL -1)
    message(FATAL_ERROR "Could not extract ${var_name} from PETSc. Please set ${var_name} using -D${var_name}=...")
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
