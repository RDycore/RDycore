# This function extracts a variable from the petscvariables file
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

  # truncate the value at a space or newline, whichever comes first
  string(FIND ${petsc_var} " " space)
  string(FIND ${petsc_var} "\n" newline)
  if (space LESS newline)
    string(SUBSTRING ${petsc_var} 0 ${space} petsc_var)
  else()
    string(SUBSTRING ${petsc_var} 0 ${newline} petsc_var)
  endif()

  message(STATUS "Extracted PETSc variable: ${varname} = ${petsc_var}")
  set(${var} ${petsc_var} PARENT_SCOPE)
endfunction()
