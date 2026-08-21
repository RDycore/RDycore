# Compiles the SWE device-assembly TU (swe_jacobian_kokkos.kokkos.cxx) with
# PETSc's kokkos compile rule, which selects nvcc_wrapper on CUDA builds and
# the plain C++ compiler on host-Kokkos builds. Driven from CMake (see
# src/CMakeLists.txt); RDY_INCLUDES carries the RDycore include paths.
#   make -f kokkos_compile.mk OBJ=<out.o> SRC=<in.kokkos.cxx> RDY_INCLUDES="-I..."
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# reach both the host path (CXXCPPFLAGS <- CXXPPFLAGS) and the CUDA
# nvcc_wrapper path (PETSC_CCPPFLAGS <- CPPFLAGS)
CPPFLAGS += ${RDY_INCLUDES}
CXXPPFLAGS += ${RDY_INCLUDES}

$(OBJ): $(SRC)
	${PETSC_KOKKOSCOMPILE_SINGLE} $(SRC)
