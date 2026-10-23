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
# The third run is three ranks with load balancing turned off. Balancing moves
# the base mesh between ranks before the rebuild, and the mesh it rebuilds must
# not depend on where the cells happen to live -- so the same counts again.
# (On the refine-only mode there is no rebuild and nothing to balance, so that
# run repeats the second; it costs a second and keeps the two modes alike.)
#
# Invoked by ctest as:
#   cmake -DRDYCORE_AMR=... -DYAML=... -DMESH=... -DWORK=... -DMPIEXEC=...
#         [-DEXTRA=...] -P amr_same_on_any_ranks.cmake

file(REMOVE_RECURSE ${WORK})
file(MAKE_DIRECTORY ${WORK})
file(COPY ${MESH} ${YAML} DESTINATION ${WORK})
get_filename_component(yaml_name ${YAML} NAME)

separate_arguments(extra_args UNIX_COMMAND "${EXTRA}")

# label, ranks, extra arguments
set(cases "1 rank" "3 ranks" "3 ranks unbalanced")
set(nps 1 3 3)
set(balance_args "" "" "-rdy_amr_balance 0")

foreach(i RANGE 2)
  list(GET cases ${i} label)
  list(GET nps ${i} np)
  list(GET balance_args ${i} balance_arg)
  separate_arguments(balance_arg UNIX_COMMAND "${balance_arg}")

  execute_process(
    COMMAND ${MPIEXEC} -n ${np} ${RDYCORE_AMR} ${yaml_name}
            -refine_criterion jump -refine_at_start ${extra_args} ${balance_arg}
    WORKING_DIRECTORY ${WORK} RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if (NOT rc EQUAL 0)
    message(FATAL_ERROR "the AMR driver failed on ${label} (${rc}):\n${out}\n${err}")
  endif()
  # keep only the per-adapt cell counts, in order
  string(REGEX MATCHALL "AMR: [0-9]+ cells" counts "${out}")
  if (counts STREQUAL "")
    message(FATAL_ERROR "no adapt happened on ${label}, so there is nothing to compare:\n${out}")
  endif()
  set(counts_${i} "${counts}")
endforeach()

foreach(i 1 2)
  list(GET cases ${i} label)
  if (NOT counts_0 STREQUAL counts_${i})
    message(FATAL_ERROR
            "AMR chose a different mesh on ${label} than on one rank.\n"
            "  1 rank  : ${counts_0}\n"
            "  ${label}: ${counts_${i}}")
  endif()
endforeach()
message(STATUS "AMR cell counts agree on 1 and 3 ranks, balanced and not: ${counts_0}")
