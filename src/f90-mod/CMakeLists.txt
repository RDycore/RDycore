# Fortran library
add_library(rdycore_f90
  rdycore.F90
)

install(
  TARGETS rdycore_f90
  DESTINATION lib
)
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/rdycore.mod
  DESTINATION include
)
set(RDYCORE_LIBRARIES rdycore_f90;${RDYCORE_LIBRARIES})

# Pass RDYCORE_LIBRARIES to the parent directory.
set(RDYCORE_LIBRARIES ${RDYCORE_LIBRARIES} PARENT_SCOPE)
