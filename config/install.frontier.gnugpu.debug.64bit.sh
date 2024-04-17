#!/bin/sh

#source modules.pm-gpu.gnugpu

./configure \
--with-cc=/opt/cray/pe/craype/2.7.19/bin/cc \
--with-cxx=/opt/cray/pe/craype/2.7.19/bin/CC \
--with-fc=/opt/cray/pe/craype/2.7.19/bin/ftn \
--with-fortran-bindings=1 \
--with-mpiexec="srun -g 8 --smpiargs=-gpu " \
--with-batch=0 \
--download-kokkos \
--download-kokkos-kernels \
--with-kokkos-kernels-tpl=0 \
--with-make-np=8 \
--with-netcdf-dir=/opt/cray/pe/netcdf-hdf5parallel/4.9.0.1/gnu/9.1 \
--with-pnetcdf-dir=/opt/cray/pe/parallel-netcdf/1.12.3.1/gnu/9.1 \
--with-hdf5-dir=/opt/cray/pe/hdf5-parallel/1.12.2.1/gnu/9.1 \
--with-hip=1 \
--with-hipc=/opt/rocm-5.4.0/bin/hipcc \
--download-parmetis \
--download-metis \
--download-zlib \
--download-scalapack \
--download-sowing \
--download-triangle \
--download-exodusii \
--download-libceed \
--with-debugging=1 \
--with-64-bit-indices=1 \
PETSC_ARCH=frontier-gpu-debug-64bit-gcc-11-2-0-fc288817

