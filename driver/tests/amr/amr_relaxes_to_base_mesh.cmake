# Gate (exactness class: exact degeneration -- the rebuilt mesh IS the base
# mesh, cell for cell, and the run has to finish).
#
# A rebuild that refines nothing returns the base mesh itself, which is the one
# case no other gate reaches: the demonstration cases always have a feature
# somewhere, and a lake at rest that never refines never leaves the base mesh
# to come back to it. Here the refinement box sweeps off the end of the channel,
# so the mesh refines, follows the box out, and then relaxes all the way back.
#
# What that used to do: RDyPerformAMR() made the previous mesh the new one's
# coarse DM, so the adapts built a chain new -> old -> ... -> base. When the
# rebuild returns the base mesh, the head of that chain is the mesh already at
# its tail and the chain closes on itself. SNESDestroy() walks it with no cycle
# guard, so the TSDestroy() in the same adapt never returns -- measured as a
# period-3 cycle here (800 -> 1052 -> 1304 -> 800) and a period-12 one on a
# longer run, with no diagnostic in either case. Hence the TIMEOUT: the failure
# this gate is here for is a hang, not a wrong answer.
#
# The counts must rise above the base mesh and come back to it exactly. Without
# the first the box could stop marking anything and the gate would pass on a run
# that never built a chain at all; without the second it would pass on a run
# that never reached a no-op rebuild.
#
# Three ranks as well as one, because load balancing replaces the base mesh
# between adapts, and that is the object the chain closes on.
#
# Invoked by ctest as:
#   cmake -DRDYCORE_AMR=... -DYAML=... -DMESH_DIR=... -DWORK=... -DMPIEXEC=...
#         -P amr_relaxes_to_base_mesh.cmake

file(REMOVE_RECURSE ${WORK})
file(MAKE_DIRECTORY ${WORK})
file(COPY ${YAML} ${MESH_DIR}/ramp_100x4_tri.msh DESTINATION ${WORK})
get_filename_component(yaml_name ${YAML} NAME)

# the ramp mesh is 100 x 4 split into triangles
set(BASE_CELLS 800)

foreach(np 1 3)
  # -refine_box_speed 50 takes the box off the end of the 50 m channel at
  # t = 1.1 s, which leaves the rest of the run to coarsen back; at the 10 m/s
  # the other ramp cases sweep at, the box is still inside the domain at t = 4.
  execute_process(
    COMMAND ${MPIEXEC} -n ${np} ${RDYCORE_AMR} ${yaml_name}
            -dm_plex_gmsh_spacedim 3 -refine_still_water 1.5
            -refine_box -5,5,-1,3 -refine_box_speed 50
            -refine_allow_coarsening -refine_check_volume
    WORKING_DIRECTORY ${WORK} TIMEOUT 120
    RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
  # a timeout is the failure this gate exists for; any other nonzero result is
  # an ordinary failure and must not be reported as the hang, or a run that
  # never started reads as a chain cycle
  if (rc MATCHES "[Tt]imeout")
    message(FATAL_ERROR
            "the AMR driver did not come back on ${np} rank(s): ${rc}.\n"
            "That is the coarse-DM chain closing on itself: the last adapt "
            "reported is the one that hung, inside TSDestroy().\n${out}\n${err}")
  elseif (NOT rc EQUAL 0)
    message(FATAL_ERROR "the AMR driver failed on ${np} rank(s) (${rc}):\n${out}\n${err}")
  endif()

  string(REGEX MATCHALL "AMR: [0-9]+ cells" counts "${out}")
  set(refined FALSE)
  set(relaxed FALSE)
  foreach(c ${counts})
    string(REGEX REPLACE "AMR: ([0-9]+) cells" "\\1" n "${c}")
    if (n GREATER BASE_CELLS)
      set(refined TRUE)
    elseif (refined AND n EQUAL BASE_CELLS)
      set(relaxed TRUE)
    endif()
  endforeach()
  if (NOT refined)
    message(FATAL_ERROR
            "nothing was refined on ${np} rank(s), so no chain was ever built "
            "and the gate proves nothing: ${counts}")
  endif()
  if (NOT relaxed)
    message(FATAL_ERROR
            "the mesh never came back to the base mesh's ${BASE_CELLS} cells on "
            "${np} rank(s), so no rebuild refined nothing: ${counts}")
  endif()
  set(counts_${np} "${counts}")
endforeach()

message(STATUS "AMR: the mesh refines and relaxes back to ${BASE_CELLS} cells: ${counts_1}")
