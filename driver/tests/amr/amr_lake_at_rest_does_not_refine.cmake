# Gate (exactness class: exact degeneration -- a lake at rest has no surface
# jumps at all, so the criterion must mark nothing and the cell count must be
# the base mesh's, exactly).
#
# A lake at rest is the standard well-balancedness benchmark, and it was the one
# state solution-driven AMR could not survive. The indicator is identically zero
# on a flat surface; VecTaggerComputeBoxes_Relative() builds its box as
# min + frac * (max - min), so a uniform indicator gives the degenerate box
# [v, v] and selects EVERY cell. Measured before the fix, on the fully wet ramp:
#
#   800 -> 3200 -> 12800 -> 51200 cells
#   deepest cell 2, then 4, then 6 levels below its base cell (marking limit 3)
#
# which also defeats the refinement cap, because zeroing the indicator on cells
# that have reached max_levels is what is supposed to stop them being marked --
# and once every cell is zeroed, the relative tagger takes them all.
#
# Two arms, and the second is what stops the first being satisfiable by a
# criterion that has stopped working altogether:
#
#   1. the lake at rest must stay at the base mesh's cell count, every adapt;
#   2. the channel wave, same criterion, must still refine.
#
# Invoked by ctest as:
#   cmake -DRDYCORE_AMR=... -DRAMP_YAML=... -DWAVE_YAML=... -DMESH_DIR=...
#         -DWORK=... -DMPIEXEC=... -DMPIEXEC_FLAGS=... -P amr_lake_at_rest_does_not_refine.cmake

# PETSc hands us the launcher and its arguments split apart, so put them back
# together: without the flags a runner that needs --oversubscribe or a Cray srun
# launches the wrong layout, or not at all.
separate_arguments(mpiexec_flags UNIX_COMMAND "${MPIEXEC_FLAGS}")

file(REMOVE_RECURSE ${WORK})
file(MAKE_DIRECTORY ${WORK})
file(COPY ${RAMP_YAML} ${WAVE_YAML} ${MESH_DIR}/ramp_100x4_tri.msh ${MESH_DIR}/channel_100x4_tri.msh
     DESTINATION ${WORK})
get_filename_component(ramp_name ${RAMP_YAML} NAME)
get_filename_component(wave_name ${WAVE_YAML} NAME)

# ---- arm 1: a lake at rest must not refine ----
# -refine_still_water also asserts the water really is at rest, so the gate
# cannot pass because the lake drained and stopped looking flat.
execute_process(
  COMMAND ${MPIEXEC} ${mpiexec_flags} -n 1 ${RDYCORE_AMR} ${ramp_name}
          -dm_plex_gmsh_spacedim 3 -refine_still_water 1.5
          -refine_criterion jump -refine_at_start
  WORKING_DIRECTORY ${WORK} RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if (NOT rc EQUAL 0)
  message(FATAL_ERROR "the AMR driver failed on the lake at rest (${rc}):\n${out}\n${err}")
endif()

string(REGEX MATCHALL "AMR: [0-9]+ cells" counts "${out}")
if (counts STREQUAL "")
  message(FATAL_ERROR "no adapt was attempted on the lake at rest:\n${out}")
endif()

# the ramp mesh is 100 x 4 split into triangles
set(BASE_CELLS 800)
foreach(c ${counts})
  string(REGEX REPLACE "AMR: ([0-9]+) cells" "\\1" n "${c}")
  if (NOT n EQUAL BASE_CELLS)
    message(FATAL_ERROR
            "a lake at rest refined: ${n} cells against the base mesh's ${BASE_CELLS}.\n"
            "  the whole sequence was: ${counts}\n"
            "  a flat surface gives an identically zero indicator, and a relative "
            "VecTagger's box on a uniform vector selects every cell.")
  endif()
endforeach()

# ---- arm 2: the same criterion must still refine something ----
execute_process(
  COMMAND ${MPIEXEC} ${mpiexec_flags} -n 1 ${RDYCORE_AMR} ${wave_name}
          -refine_criterion jump -refine_at_start -refine_check_volume
  WORKING_DIRECTORY ${WORK} RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if (NOT rc EQUAL 0)
  message(FATAL_ERROR "the AMR driver failed on the channel wave (${rc}):\n${out}\n${err}")
endif()

string(REGEX MATCHALL "AMR: [0-9]+ cells" wave_counts "${out}")
set(grew FALSE)
foreach(c ${wave_counts})
  string(REGEX REPLACE "AMR: ([0-9]+) cells" "\\1" n "${c}")
  if (n GREATER BASE_CELLS)
    set(grew TRUE)
  endif()
endforeach()
if (NOT grew)
  message(FATAL_ERROR
          "the jump criterion no longer refines a travelling wave either "
          "(counts: ${wave_counts}), so arm 1 above proves nothing. Fix the "
          "criterion before trusting this gate.")
endif()

message(STATUS "AMR: a lake at rest holds at ${BASE_CELLS} cells; the wave still refines (${wave_counts})")
