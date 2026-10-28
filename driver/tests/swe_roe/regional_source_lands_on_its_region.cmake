# Gate: a regional source puts its water in the region that declares it.
#
# RDySetHomogeneousRegionalWaterSource() takes a region's ID -- the cell-set
# tag the grid gives it, which a config file names as grid_region_id -- and
# finds the region by matching it. Two callers passed an index instead:
# InitSourceConditions() passed its loop counter, and the multi-homogeneous
# forcing path passed a grid ID with one subtracted. An index and an ID are
# different numbers, so a source landed on whichever region happened to carry
# that ID, and the region at index 0 got none at all, because no cell set is
# tagged 0.
#
# Measured on this case, water held at t = 0.005 hours, in m^3:
#
#                                   before            after
#   no sources                      7.91651883108651  7.91651883108651
#   runoff on upstream              7.91651883108651  9.05584914854935
#   runoff on downstream            9.05584914854935  9.28275782018229
#
# Read the first column: declaring runoff on upstream did nothing at all, and
# declaring it on downstream put that water on upstream instead -- which is
# why the "before" downstream number is the "after" upstream one.
#
# Three arms. The first is what fails on the old code; the other two stop it
# being satisfied by a source that reaches every cell whatever region asked.
#
#   1. runoff declared on a region adds water.
#   2. the two regions add different amounts. They are different sizes, so a
#      source spilling across the whole domain would give the same twice.
#   3. runoff on both regions holds more water than runoff on either alone, so
#      the two are not aliasing onto one region.
#
# Comparisons are ordering only: cmake's math() is integer, and these are
# separate runs whose sums differ in the last bits anyway.
#
# Invoked by ctest as:
#   cmake -DRDYCORE=... -DYAML=... -DMESH=... -DWORK=... -DNP=...
#         -DMPIEXEC=... -DMPIEXEC_FLAGS=... -P regional_source_lands_on_its_region.cmake

# PETSc hands us the launcher and its arguments split apart, so put them back
# together: without the flags a runner that needs --oversubscribe or a Cray
# srun launches the wrong layout, or not at all.
separate_arguments(mpiexec_flags UNIX_COMMAND "${MPIEXEC_FLAGS}")

file(REMOVE_RECURSE ${WORK})
file(MAKE_DIRECTORY ${WORK})
file(COPY ${MESH} ${YAML} DESTINATION ${WORK})
get_filename_component(yaml_name ${YAML} NAME)
file(READ ${WORK}/${yaml_name} config)

# Runs one arm and returns the water the domain ends up holding. Every arm uses
# the same mesh and the same run length; only the region carrying the runoff
# changes.
function(run_arm label upstream_src downstream_src out_var)
  file(WRITE ${WORK}/${label}.yaml
       "${config}\nsources:\n  - region: upstream\n    flow: ${upstream_src}\n  - region: downstream\n    flow: ${downstream_src}\n")
  execute_process(
    COMMAND ${MPIEXEC} ${mpiexec_flags} -n ${NP} ${RDYCORE} ${label}.yaml -report_water_volume
    WORKING_DIRECTORY ${WORK} RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if (NOT rc EQUAL 0)
    message(FATAL_ERROR "the driver failed on the ${label} arm (${rc}):\n${out}\n${err}")
  endif()
  string(REGEX MATCHALL "water volume [0-9.eE+-]+" volumes "${out}")
  if (volumes STREQUAL "")
    message(FATAL_ERROR "the ${label} arm reported no water volume, so there is nothing to compare:\n${out}")
  endif()
  list(GET volumes -1 last)
  string(REGEX REPLACE "water volume " "" last "${last}")
  set(${out_var} ${last} PARENT_SCOPE)
endfunction()

run_arm(none       runoff_off runoff_off v_none)
run_arm(upstream   runoff_on  runoff_off v_up)
run_arm(downstream runoff_off runoff_on  v_down)
run_arm(both       runoff_on  runoff_on  v_both)

# ---- arm 1: a source on a region adds water ----
foreach(arm upstream downstream)
  if (arm STREQUAL upstream)
    set(v ${v_up})
  else()
    set(v ${v_down})
  endif()
  if (NOT v GREATER v_none)
    message(FATAL_ERROR
            "runoff declared on ${arm} added no water: ${v} m^3, against ${v_none} with no source at all.\n"
            "  A regional source is addressed by region ID. Passing an index instead names a region that may not exist, "
            "and index 0 never does -- no cell set is tagged 0.")
  endif()
endforeach()

# ---- arm 2: different regions, different amounts ----
if (v_up EQUAL v_down)
  message(FATAL_ERROR
          "runoff on upstream and on downstream each added the same water (${v_up} m^3).\n"
          "  These two regions differ in area, so equal amounts mean the source is reaching cells outside the region that declared it.")
endif()

# ---- arm 3: both regions together hold more than either alone ----
foreach(arm upstream downstream)
  if (arm STREQUAL upstream)
    set(v ${v_up})
  else()
    set(v ${v_down})
  endif()
  if (NOT v_both GREATER v)
    message(FATAL_ERROR
            "runoff on both regions (${v_both} m^3) added no more than runoff on ${arm} alone (${v} m^3).\n"
            "  The two sources are landing on the same region.")
  endif()
endforeach()

message(STATUS
        "regional source: none ${v_none}, upstream ${v_up}, downstream ${v_down}, both ${v_both} m^3")
