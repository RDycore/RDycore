#!/bin/sh

#source modules.pm-gpu.gnugpu

./configure \
-LDFLAGS='${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a}' \
--with-cc=/opt/cray/pe/craype/2.7.33/bin/cc \
--with-cxx=/opt/cray/pe/craype/2.7.33/bin/CC \
--with-fc=/opt/cray/pe/craype/2.7.33/bin/ftn \
--with-fortran-bindings=1 \
--with-mpiexec="srun -g 8 --smpiargs=-gpu " \
--with-batch=0 \
--with-make-np=8 \
--download-kokkos \
--download-kokkos-kernels \
--with-kokkos-kernels-tpl=0 \
--with-netcdf-dir=/opt/cray/pe/netcdf-hdf5parallel/4.9.0.15/gnu/12.3 \
--with-pnetcdf-dir=/opt/cray/pe/parallel-netcdf/1.12.3.15/gnu/12.3 \
--with-hdf5-dir=/opt/cray/pe/hdf5-parallel/1.14.3.3/gnu/12.3 \
--with-hip=1 \
--with-hipc=/opt/rocm-6.2.4/bin/hipcc \
--download-parmetis \
--download-metis \
--download-muparser \
--download-zlib \
--download-scalapack \
--download-sowing \
--download-triangle \
--download-exodusii \
--download-libceed \
--with-debugging=0 \
--with-64-bit-indices=0 \
PETSC_ARCH=craygnu-mphipcc-opt-32bit-gcc-13-3-0-95934b0d393

