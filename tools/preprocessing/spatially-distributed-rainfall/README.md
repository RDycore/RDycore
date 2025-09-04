# Preprocessing spatially distributed rainfall data

- `daymet/asc`: Contains sample rainfall data in ASCII raster format.

- `convert_asc_rainfall_data_to_dlnd_netcdf_files.m`: Converts ASCII raster format rainfall data
in `daymet/asc ` into netcdf files that can be used by the DLND, data land, of E3SM.

- `convert_asc_rainfall_data_to_petsc_binary_vec.m`: Converts ASCII raster format rainfall data
in PETSc binary file that can be read by RDycore.

- `generate_dlnd.m`: Is called by `convert_asc_rainfall_data_to_dlnd_netcdf_files.m` to generate
netcdf files for DLND.
