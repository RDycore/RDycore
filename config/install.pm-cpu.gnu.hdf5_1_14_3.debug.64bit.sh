#!/bin/sh

# source modules.pm-cpu.gnu

./configure \
--with-cc=cc \
--with-cxx=CC \
--with-fc=ftn \
--CFLAGS=" -g " \
--CXXFLAGS=" -g " \
--CUDAFLAGS=" -g -Xcompiler -rdynamic " \
--with-fortran-bindings=1 \
--COPTFLAGS="   -O" \
--CXXOPTFLAGS=" -O" \
--FOPTFLAGS="   -O" \
--with-mpiexec="srun -G4" \
--with-batch=0 \
--download-kokkos \
--download-kokkos-kernels \
--download-kokkos-cmake-arguments=-DKokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=OFF \
--with-kokkos-kernels-tpl=0 \
--with-make-np=8 \
--with-64-bit-indices=1 \
--with-netcdf-dir=/opt/cray/pe/netcdf-hdf5parallel/4.9.0.9/gnu/12.3 \
--with-pnetcdf-dir=/opt/cray/pe/parallel-netcdf/1.12.3.9/gnu/12.3 \
--download-hdf5=1 \
--download-parmetis \
--download-metis \
--download-muparser \
--download-zlib \
--download-scalapack \
--download-sowing \
--download-triangle \
--download-exodusii \
--download-libceed \
--download-cgns-commit=HEAD \
--with-debugging=1 \
PETSC_ARCH=pm-cpu-hdf5_1_14_3-debug-64bit-gcc-13-2-1-95934b0d393

