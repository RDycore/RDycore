# Gate (exactness class: bit-identical).
#
# Solution-driven AMR must choose the same mesh however the domain is divided.
# The sequence of adapted cell counts is a discrete object, so it is compared
# exactly: one rank against three, for refinement alone and for refinement with
# coarsening.
#
# What this catches: an error vector indexed by OWNED cell, whose tagged indices
# are then used as mesh points. Those agree only while the owned cells lead the
# chart. A refined mesh with an overlap interleaves the children of ghost cells,
# so from the second adapt onward the wrong cells are refined -- 2036 cells on
# one rank against 3690 on three, with nothing to say so. The adapt indexes by
# local cell for this reason; see the MR description.
#
# Cell counts rather than the cell sets themselves: counts are cheap, and they
# diverged at the very first adapt after the mesh stopped being flat-ordered,
# so they detect the failure that motivated the gate.
#
# Invoked by ctest as:
#   cmake -DRDYCORE_AMR=... -DYAML=... -DMESH=... -DWORK=... -DMPIEXEC=...
#         [-DEXTRA=...] -P amr_same_on_any_ranks.cmake

file(REMOVE_RECURSE ${WORK})
file(MAKE_DIRECTORY ${WORK})
file(COPY ${MESH} ${YAML} DESTINATION ${WORK})
get_filename_component(yaml_name ${YAML} NAME)

separate_arguments(extra_args UNIX_COMMAND "${EXTRA}")

foreach(np 1 3)
  execute_process(
    COMMAND ${MPIEXEC} -n ${np} ${RDYCORE_AMR} ${yaml_name}
            -refine_criterion jump -refine_at_start ${extra_args}
    WORKING_DIRECTORY ${WORK} RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if (NOT rc EQUAL 0)
    message(FATAL_ERROR "the AMR driver failed on ${np} rank(s) (${rc}):\n${out}\n${err}")
  endif()
  # keep only the per-adapt cell counts, in order
  string(REGEX MATCHALL "AMR: [0-9]+ cells" counts "${out}")
  if (counts STREQUAL "")
    message(FATAL_ERROR "no adapt happened on ${np} rank(s), so there is nothing to compare:\n${out}")
  endif()
  set(counts_${np} "${counts}")
endforeach()

if (NOT counts_1 STREQUAL counts_3)
  message(FATAL_ERROR
          "AMR chose a different mesh on three ranks than on one.\n"
          "  1 rank : ${counts_1}\n"
          "  3 ranks: ${counts_3}")
endif()
message(STATUS "AMR cell counts agree on 1 and 3 ranks: ${counts_1}")
