# Hurricane Harvey Flooding Simulation Plan

## Overview
Run the RDycore Houston 1km Hurricane Harvey flooding simulation, which models rain over the domain with a Dirichlet boundary condition using the shallow water equations with a Roe solver.

## Key Configuration
- **Branch**: `bishtgautam/harvey-simulation` (contains fix for model output)
- **PETSc int size**: 32-bit (use `int32` binary files)
- **PETSc**: `PETSC_DIR=/Users/markadams/Codes/petsc PETSC_ARCH=arch-macosx-gnu-g`
- **Build system**: Ninja
- **Build directory**: `build/` (install prefix: `build/`)
- **Simulation config**: [`Houston1km.DirichletBC.yaml`](driver/tests/swe_roe/Houston1km.DirichletBC.yaml)

## PETSc Dependencies Required
PETSc must be configured with the following packages for this simulation:
- `--download-libceed` (GPU/CPU operator library)
- `--download-muparser` (math expression parser for MMS)
- `--download-hdf5` (HDF5 I/O)
- `--download-exodusii` (ExodusII mesh format)
- `--download-netcdf` (NetCDF, required by ExodusII)
- `--download-pnetcdf` (Parallel NetCDF)
- `--download-zlib` (compression, required by NetCDF)

### PETSc Configure Command (with ASan)
```bash
cd /Users/markadams/Codes/petsc
python3 ./configure \
  --with-x=1 --with-debugging=1 --with-strict-petscerrorcode \
  --download-hdf5=1 --download-libceed --download-muparser \
  --download-exodusii --download-netcdf --download-pnetcdf --download-zlib \
  --with-macos-firewall-rules \
  PETSC_ARCH=arch-macosx-gnu-g \
  CFLAGS="-g -Wall -fsanitize=address" \
  CXXFLAGS="-g -Wall -fsanitize=address" \
  LDFLAGS="-fsanitize=address"

make PETSC_DIR=/Users/markadams/Codes/petsc PETSC_ARCH=arch-macosx-gnu-g all
```

**Note**: `LDFLAGS="-fsanitize=address"` is required when using ASan in CFLAGS/CXXFLAGS, otherwise libCEED's shared library link fails with missing `_asan.module_ctor` symbols.

## Simulation Details
- **Physics**: Shallow water equations with Roe Riemann solver
- **Mesh**: [`Houston1km_with_z.exo`](share/meshes/Houston1km_with_z.exo) - 1km resolution Houston area (2746 cells)
- **Final time**: 4200 seconds (70 minutes)
- **Time step**: 30 seconds (140 total steps)
- **Output format**: XDMF, every 600 seconds (8 output files)
- **Manning coefficient**: 0.015 (smooth)
- **Boundary**: Dirichlet outflow BC on boundary ID 1 (2 edges)
- **Courant number**: ~0.50 throughout simulation

### Input Files
- Initial conditions: `Houston1km.ic.int32.bin`
- Rain forcing: `Houston1km.rain.int32.bin` (spatially homogeneous)
- Boundary conditions: `Houston1km.bc.int32.bin`

## Execution Steps

### Step 1: Fetch and Checkout Branch
```bash
git fetch origin
git checkout bishtgautam/harvey-simulation
```

### Step 2: Configure RDycore with Ninja
```bash
mkdir build && cd build
export PETSC_DIR=/Users/markadams/Codes/petsc
export PETSC_ARCH=arch-macosx-gnu-g
cmake .. -G Ninja -DCMAKE_INSTALL_PREFIX=$(pwd) \
  -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx \
  -DENABLE_SANITIZERS=ON
```

**Note**: `-DENABLE_SANITIZERS=ON` is required when PETSc is built with ASan, otherwise the rdycore binary will fail at runtime with `___asan_option_detect_stack_use_after_return` symbol not found.

### Step 3: Build and Install
```bash
cd build
ninja -j4 install
```

### Step 4: Run the Simulation
```bash
cd build/driver/tests/swe_roe
../../../bin/rdycore Houston1km.DirichletBC.yaml \
  -homogeneous_rain_file Houston1km.rain.int32.bin \
  -homogeneous_bc_file Houston1km.bc.int32.bin \
  -temporally_interpolate_bc
```

### Step 5: Generate VisIt File for Visualization
```bash
cd build/driver/tests/swe_roe/output
ls -1 Houston1km.DirichletBC.*.xmf | sort > Houston1km.visit
```

### Step 6: Visualize
Open `Houston1km.visit` in VisIt to view the simulation results.

## Output Files
The simulation produces the following in `build/driver/tests/swe_roe/output/`:
- `Houston1km.DirichletBC-grid.h5` — mesh geometry
- `Houston1km.DirichletBC.2017-08-26.01.00.00.{h5,xmf}` — t=0s
- `Houston1km.DirichletBC.2017-08-26.01.10.00.{h5,xmf}` — t=600s
- `Houston1km.DirichletBC.2017-08-26.01.20.00.{h5,xmf}` — t=1200s
- `Houston1km.DirichletBC.2017-08-26.01.30.00.{h5,xmf}` — t=1800s
- `Houston1km.DirichletBC.2017-08-26.01.40.00.{h5,xmf}` — t=2400s
- `Houston1km.DirichletBC.2017-08-26.01.50.00.{h5,xmf}` — t=3000s
- `Houston1km.DirichletBC.2017-08-26.02.00.00.{h5,xmf}` — t=3600s
- `Houston1km.DirichletBC.2017-08-26.02.10.00.{h5,xmf}` — t=4200s
- `Houston1km.DirichletBC-boundary_fluxes.dat` — boundary flux time series
- `Houston1km.visit` — VisIt index file

## Code Changes Made
- [`cmake/extract_mpi_include_directories.cmake`](cmake/extract_mpi_include_directories.cmake) — Added fallback to query `mpicc --showme:compile` when PETSc's `MPICXX_INCLUDES` and `MPICC_SHOW` variables are not available in `petscvariables`.
