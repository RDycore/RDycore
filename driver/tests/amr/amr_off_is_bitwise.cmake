# Gate (exactness class: bit-identical).
#
# Running the AMR driver with a tagger box that marks nothing must reproduce the
# plain driver byte for byte on the same configuration: the error indicator and
# the two VecTaggers run at every coupling interval and must leave the answer
# untouched when they select no cell.
#
# What this does NOT cover: when neither tagger selects anything the adapt
# returns before touching the mesh, so the transform, the interpolation matrices
# and the solution transfer never run here. Those are covered by
# amr_still_water_ramp (exact degeneration) and by -refine_check_volume.
#
# Invoked by ctest as:
#   cmake -DRDYCORE=... -DRDYCORE_AMR=... -DYAML=... -DMESH=... -DH5DIFF=...
#         -DWORK=... -P amr_off_is_bitwise.cmake

file(REMOVE_RECURSE ${WORK})
file(MAKE_DIRECTORY ${WORK})
file(COPY ${MESH} DESTINATION ${WORK})

# the output file name follows the config file name, so run the two under
# different names and compare the files they leave behind
file(READ ${YAML} config)
string(REPLACE "output_interval: 50" "output_interval: 2000" config "${config}")
file(WRITE ${WORK}/plain.yaml "${config}")
file(WRITE ${WORK}/amr_off.yaml "${config}")

execute_process(COMMAND ${RDYCORE} plain.yaml WORKING_DIRECTORY ${WORK} RESULT_VARIABLE rc OUTPUT_QUIET)
if (NOT rc EQUAL 0)
  message(FATAL_ERROR "the plain driver failed (${rc})")
endif()

execute_process(
  COMMAND ${RDYCORE_AMR} amr_off.yaml
          -refine_criterion jump
          -rdy_amr_refine_vec_tagger_type absolute -rdy_amr_refine_vec_tagger_box 1e30,1e31
          -options_left
  WORKING_DIRECTORY ${WORK} RESULT_VARIABLE rc OUTPUT_QUIET)
if (NOT rc EQUAL 0)
  message(FATAL_ERROR "the AMR driver failed (${rc})")
endif()

# h5diff exits 0 when two datasets are not even comparable -- different shapes
# report "Not comparable" and still succeed -- so a mesh that was refined when
# it should not have been would slip through an exit-status check. Both
# conditions are asserted.
execute_process(COMMAND ${H5DIFF} -v output/plain-2000.h5 output/amr_off-2000.h5
                WORKING_DIRECTORY ${WORK} RESULT_VARIABLE rc OUTPUT_VARIABLE diff)
if (NOT rc EQUAL 0)
  message(FATAL_ERROR "AMR with nothing marked changed the answer:\n${diff}")
endif()
string(FIND "${diff}" "Not comparable" incomparable)
if (NOT incomparable EQUAL -1)
  message(FATAL_ERROR "AMR with nothing marked produced a different mesh, so the two runs cannot be compared:\n${diff}")
endif()
