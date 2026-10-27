# Gate for load balancing after an adapt.
#
# Three things have to hold at once, and the third is what stops the first two
# from being satisfiable by doing nothing:
#
#   1. balancing does not change the mesh. The per-adapt global cell counts
#      must be identical with and without it -- it moves cells between ranks
#      and must not move a single boundary of the adapt itself.
#   2. it brings the imbalance down. max owned cells over the mean, over the
#      ranks, must stay under BALANCED_MAX at every adapt.
#   3. the case is still a hard one. Without balancing the same run must reach
#      UNBALANCED_MIN, or there was nothing here to fix and (2) is free.
#
# The case is a refinement box held in one corner of the dam mesh, four levels
# deep, so that one rank ends up owning nearly every fine cell. The
# demonstration cases will not do: their features span the domain, and the
# channel case is already perfectly balanced on four ranks.
#
# Measured on three ranks: without balancing max/mean climbs 1.32, 1.94, 2.54,
# 2.85 and stays there (4564 owned cells on one rank against a mean of 1599);
# with it, 1.24, 1.38, 1.23, 1.12 (1792 against the same mean).
#
# Invoked by ctest as:
#   cmake -DRDYCORE_AMR=... -DYAML=... -DMESH=... -DWORK=... -DMPIEXEC=... -DMPIEXEC_FLAGS=...
#         -DNP=... -P amr_balance_improves.cmake

set(BALANCED_MAX 1.5)
set(UNBALANCED_MIN 2.0)

# PETSc hands us the launcher and its arguments split apart, so put them back
# together: without the flags a runner that needs --oversubscribe or a Cray srun
# launches the wrong layout, or not at all.
separate_arguments(mpiexec_flags UNIX_COMMAND "${MPIEXEC_FLAGS}")

file(REMOVE_RECURSE ${WORK})
file(MAKE_DIRECTORY ${WORK})
file(COPY ${MESH} ${YAML} DESTINATION ${WORK})
get_filename_component(yaml_name ${YAML} NAME)

foreach(balance 0 1)
  execute_process(
    COMMAND ${MPIEXEC} ${mpiexec_flags} -n ${NP} ${RDYCORE_AMR} ${yaml_name}
            -refine_criterion box -refine_box 0,2,0,1 -refine_allow_coarsening
            -rdy_amr_max_levels 4 -refine_report_balance -rdy_amr_balance ${balance}
    WORKING_DIRECTORY ${WORK} RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if (NOT rc EQUAL 0)
    message(FATAL_ERROR "the AMR driver failed with -rdy_amr_balance ${balance} (${rc}):\n${out}\n${err}")
  endif()

  string(REGEX MATCHALL "AMR: [0-9]+ cells after adapt" counts "${out}")
  if (counts STREQUAL "")
    message(FATAL_ERROR "no adapt happened with -rdy_amr_balance ${balance}:\n${out}")
  endif()
  set(counts_${balance} "${counts}")

  # the ratio reported after each adapt, as a list
  string(REGEX MATCHALL "max/mean [0-9.]+" ratios "${out}")
  set(worst 0)
  foreach(r ${ratios})
    string(REGEX REPLACE "max/mean " "" r "${r}")
    if (r GREATER worst)
      set(worst ${r})
    endif()
  endforeach()
  set(worst_${balance} ${worst})
endforeach()

if (NOT counts_0 STREQUAL counts_1)
  message(FATAL_ERROR
          "load balancing changed the mesh, which it must not.\n"
          "  unbalanced: ${counts_0}\n"
          "  balanced  : ${counts_1}")
endif()

if (worst_0 LESS UNBALANCED_MIN)
  message(FATAL_ERROR
          "without balancing this case only reaches max/mean ${worst_0}, under the "
          "${UNBALANCED_MIN} it is supposed to be bad at, so it no longer tests "
          "anything. Find a case that is still badly balanced before trusting the "
          "balanced number below.")
endif()

if (worst_1 GREATER BALANCED_MAX)
  message(FATAL_ERROR
          "load balancing left max/mean at ${worst_1}, over the ${BALANCED_MAX} it "
          "is held to (unbalanced: ${worst_0})")
endif()

message(STATUS "AMR load balancing: worst max/mean ${worst_0} -> ${worst_1}, same mesh either way")
