# Checks that a lake-at-rest run preserved its initial state: the depth and
# momenta in the final output must match those of the initial output to within
# an absolute tolerance.
#
# Usage: cmake -DH5DIFF=<h5diff> -DINITIAL=<file> -DFINAL=<file>
#              -DINITIAL_GROUP=<group> -DFINAL_GROUP=<group> -DTOL=<tol>
#              -P compare_lake_at_rest.cmake
foreach(field Height MomentumX MomentumY)
  execute_process(COMMAND ${H5DIFF} -d ${TOL} ${INITIAL} ${FINAL} "${INITIAL_GROUP}/${field}" "${FINAL_GROUP}/${field}"
                  RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE output)
  if (NOT result EQUAL 0)
    message(FATAL_ERROR "${field} in ${FINAL} differs from ${INITIAL} by more than ${TOL}:\n${output}")
  endif()
endforeach()
